"""
video_time_study.py — Video Time Study Analysis Tool

Pipeline:
  1. Extract frames from a video file
  2. User clicks on a part in frame 0 to set an SAM 2 point prompt
  3. SAM 2 video predictor propagates the part mask across all frames
  4. MediaPipe Hands detects hand keypoints on every frame
  5. Per-frame classification:
       VA  — hand keypoint inside the part mask
       SVA — hand within 50px of mask boundary but not inside
       NA  — no hand near mask, or mask absent
  6. Export an Excel report (per-frame sheet + summary sheet)

Usage:
  python video_time_study.py --video VIDEO [options]

Install dependencies first:
  pip install -r requirements.txt
  bash setup_sam2.sh [tiny|small|base_plus|large]
"""

import argparse
import os
import platform
import shutil
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# ClickUI — OpenCV point-selection window
# ---------------------------------------------------------------------------

class ClickUI:
    """Shows frame 0 in an OpenCV window; user clicks a point and presses Enter."""

    WINDOW = "Time Study — Click on the part, then press Enter (Esc to quit)"

    def __init__(self, frame_bgr: np.ndarray):
        self._frame = frame_bgr.copy()
        self._point: tuple[int, int] | None = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._point = (x, y)

    def select_point(self) -> tuple[int, int]:
        """Block until the user clicks a point and presses Enter.

        Returns:
            (x, y) pixel coordinates of the selected point.

        Raises:
            RuntimeError: if the window is closed without selecting a point.
        """
        h, w = self._frame.shape[:2]
        print(f"Frame size: {w}x{h}  — Click on the part to track, then press Enter.")

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW, self._mouse_callback)

        while True:
            display = self._frame.copy()
            if self._point is not None:
                cx, cy = self._point
                cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)
                cv2.circle(display, (cx, cy), 8, (0, 0, 0), 2)
                cv2.putText(
                    display,
                    f"({cx}, {cy}) — Press Enter to confirm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display,
                    "Click on the part to segment",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                )

            cv2.imshow(self.WINDOW, display)
            key = cv2.waitKey(20) & 0xFF

            if key == 13 and self._point is not None:  # Enter
                break
            if key == 27:  # Esc
                cv2.destroyAllWindows()
                raise RuntimeError("No point selected — user pressed Esc.")
            if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("No point selected — window was closed.")

        cv2.destroyAllWindows()
        print(f"Point selected: {self._point}")
        return self._point


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, frames_dir: str) -> tuple[int, float]:
    """Extract all video frames as JPEG files into frames_dir.

    Frames are named 00000.jpg, 00001.jpg, … as required by SAM 2.

    Returns:
        (total_frames, fps)

    Raises:
        FileNotFoundError: if video_path does not exist.
        ValueError: if OpenCV cannot open the video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(frames_dir, exist_ok=True)

    print(f"Extracting {total_frames} frames at {fps:.2f} fps …")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(frames_dir, f"{idx:05d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1
        if idx % 100 == 0:
            print(f"\r  {idx}/{total_frames} frames extracted", end="", flush=True)

    cap.release()
    print(f"\r  {idx} frames extracted.                    ")
    return idx, fps


# ---------------------------------------------------------------------------
# SAM2Tracker — wraps SAM 2 video predictor
# ---------------------------------------------------------------------------

class SAM2Tracker:
    """Wraps the SAM 2 video predictor for mask propagation."""

    _DOWNLOAD_URLS = {
        "sam2_hiera_tiny.pt":      "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "sam2_hiera_small.pt":     "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "sam2_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "sam2_hiera_large.pt":     "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    }

    def __init__(self, checkpoint: str, model_cfg: str, device: str = "cuda"):
        self.device = device
        self._checkpoint = checkpoint
        self._model_cfg = model_cfg
        self._validate_checkpoint()
        self._predictor = self._build_predictor()

    def _validate_checkpoint(self):
        if not os.path.exists(self._checkpoint):
            ckpt_name = os.path.basename(self._checkpoint)
            url = self._DOWNLOAD_URLS.get(ckpt_name, "<checkpoint-url>")
            raise FileNotFoundError(
                f"SAM 2 checkpoint not found: {self._checkpoint}\n"
                f"Download it with:\n"
                f"  mkdir -p checkpoints\n"
                f"  wget {url} -O {self._checkpoint}\n"
                f"Or run:  bash setup_sam2.sh"
            )

    def _build_predictor(self):
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            raise ImportError(
                "SAM 2 is not installed. Install it with:\n"
                "  git clone https://github.com/facebookresearch/segment-anything-2.git\n"
                "  cd segment-anything-2 && pip install -e .\n"
                "Or run:  bash setup_sam2.sh"
            )
        return build_sam2_video_predictor(self._model_cfg, self._checkpoint, device=self.device)

    def init_video(self, frames_dir: str):
        """Initialise SAM 2 inference state for the extracted frames directory."""
        import torch
        with torch.inference_mode():
            state = self._predictor.init_state(video_path=frames_dir)
        return state

    def set_point_prompt(
        self, state, x: int, y: int, obj_id: int = 1
    ) -> np.ndarray:
        """Add a foreground point prompt on frame 0 and return the initial mask.

        Also shows the initial mask overlay for ~1.5 s as a sanity check.

        Returns:
            Binary mask (H×W bool) for frame 0.
        """
        import torch

        with torch.inference_mode():
            _, _, masks = self._predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                points=np.array([[x, y]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )

        mask = (masks[0, 0] > 0).cpu().numpy()

        # Sanity-check preview
        frames_dir = state["images_dir"] if "images_dir" in state else None
        if frames_dir is None:
            # Attempt to locate via internal state dict key used by SAM2
            for key in ("video_path", "frames_path", "images_path"):
                if key in state:
                    frames_dir = state[key]
                    break

        if frames_dir and os.path.isdir(frames_dir):
            frame0_path = os.path.join(frames_dir, "00000.jpg")
            if os.path.exists(frame0_path):
                self._show_mask_preview(cv2.imread(frame0_path), mask, x, y)

        return mask

    @staticmethod
    def _show_mask_preview(frame_bgr: np.ndarray, mask: np.ndarray, x: int, y: int):
        overlay = frame_bgr.copy()
        green = np.zeros_like(overlay)
        green[:, :, 1] = 255  # pure green channel
        alpha = 0.4
        overlay[mask] = (
            alpha * green[mask] + (1 - alpha) * overlay[mask]
        ).astype(np.uint8)
        cv2.circle(overlay, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(
            overlay,
            "Initial mask — closing in 1.5s …",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        win = "SAM 2 — Initial mask preview"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, overlay)
        cv2.waitKey(1500)
        cv2.destroyWindow(win)

    def propagate(self, state):
        """Yield (frame_idx, binary_mask H×W) for every frame.

        Uses streaming so only one mask is in memory at a time.
        """
        import torch

        ctx = torch.inference_mode()
        ctx.__enter__()
        try:
            for frame_idx, obj_ids, masks in self._predictor.propagate_in_video(state):
                mask = (masks[0, 0] > 0).cpu().numpy()
                yield frame_idx, mask
        finally:
            ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# HandDetector — wraps MediaPipe Hands
# ---------------------------------------------------------------------------

class HandDetector:
    """Detects hand keypoints in video frames using MediaPipe Hands."""

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.5):
        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int]]:
        """Return pixel (x, y) coordinates for all detected hand landmarks.

        All 21 landmarks per hand are returned (flattened across hands).
        Returns an empty list when no hands are detected.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return []

        keypoints: list[tuple[int, int]] = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                px = int(np.clip(lm.x * w, 0, w - 1))
                py = int(np.clip(lm.y * h, 0, h - 1))
                keypoints.append((px, py))
        return keypoints

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Distance map + frame classifier
# ---------------------------------------------------------------------------

def compute_dist_map(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (pixels) from each pixel to the nearest True mask pixel.

    Pixels inside the mask have distance 0.
    Pixels outside have positive values.
    If mask is entirely False (no segmentation), returns an all-inf array.
    """
    if not np.any(mask):
        return np.full(mask.shape, np.inf, dtype=np.float32)

    # distanceTransform needs uint8; 0 = "mask present", 1 = "not mask"
    inverted = (~mask).astype(np.uint8)
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist


def classify_frame(
    mask: np.ndarray,
    keypoints: list[tuple[int, int]],
    dist_map: np.ndarray,
    svd_threshold: int = 50,
) -> str:
    """Classify a single frame as VA, SVA, or NA.

    Priority order: VA > SVA > NA.

    Args:
        mask:          Boolean H×W array (True = part present).
        keypoints:     List of (x, y) pixel coords for detected hand landmarks.
        dist_map:      Float32 H×W from compute_dist_map(); inf where mask absent.
        svd_threshold: Max distance (pixels) from mask boundary to count as SVA.

    Returns:
        "VA", "SVA", or "NA".

    Note on indexing convention:
        NumPy arrays are indexed [row, col] = [y, x].  Every pixel access below
        uses mask[y, x] (NOT mask[x, y]).
    """
    if not keypoints:
        return "NA"

    h, w = mask.shape

    best = "NA"
    for (x, y) in keypoints:
        # Bounds check — clamp should have already happened in HandDetector,
        # but guard defensively here as well.
        if not (0 <= x < w and 0 <= y < h):
            continue

        if mask[y, x]:
            return "VA"  # Short-circuit on first VA hit

        if dist_map[y, x] <= svd_threshold:
            best = "SVA"  # Keep looking for VA; might still find one

    return best


# ---------------------------------------------------------------------------
# Report builder and Excel exporter
# ---------------------------------------------------------------------------

def build_report(
    classifications: list[str],
    fps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-frame and summary DataFrames from the list of classifications.

    Args:
        classifications: List of "VA"/"SVA"/"NA" strings, one per frame (0-indexed).
        fps:             Frames per second of the original video.

    Returns:
        (per_frame_df, summary_df)
    """
    frame_duration = 1.0 / fps
    total_frames = len(classifications)

    per_frame_df = pd.DataFrame(
        {
            "Frame Number": range(total_frames),
            "Timestamp (s)": [round(i * frame_duration, 4) for i in range(total_frames)],
            "Classification": classifications,
            "Duration (s)": [round(frame_duration, 6)] * total_frames,
        }
    )

    # Replace any None values (e.g. frames SAM2 skipped) with "NA"
    per_frame_df["Classification"] = per_frame_df["Classification"].fillna("NA")

    counts = per_frame_df["Classification"].value_counts()
    va_frames  = int(counts.get("VA",  0))
    sva_frames = int(counts.get("SVA", 0))
    na_frames  = int(counts.get("NA",  0))

    total_duration = total_frames * frame_duration
    va_time  = va_frames  * frame_duration
    sva_time = sva_frames * frame_duration
    na_time  = na_frames  * frame_duration

    def pct(t: float) -> float:
        return round(t / total_duration * 100, 2) if total_duration > 0 else 0.0

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Total Frames",
                "Video Duration (s)",
                "VA Time (s)",
                "SVA Time (s)",
                "NA Time (s)",
                "VA %",
                "SVA %",
                "NA %",
            ],
            "Value": [
                total_frames,
                round(total_duration, 4),
                round(va_time, 4),
                round(sva_time, 4),
                round(na_time, 4),
                pct(va_time),
                pct(sva_time),
                pct(na_time),
            ],
        }
    )

    return per_frame_df, summary_df


def export_excel(
    per_frame_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Write both DataFrames to an Excel workbook with auto-fitted columns.

    Sheets: "Per-Frame", "Summary".
    """
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        per_frame_df.to_excel(writer, sheet_name="Per-Frame", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        for sheet_name, df in [("Per-Frame", per_frame_df), ("Summary", summary_df)]:
            ws = writer.sheets[sheet_name]
            for col_cells in ws.columns:
                max_len = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in col_cells
                )
                col_letter = col_cells[0].column_letter
                ws.column_dimensions[col_letter].width = max_len + 4


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Video Time Study Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--output", default="report.xlsx", help="Path for Excel output")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/sam2_hiera_large.pt",
        help="SAM 2 model checkpoint (.pt file)",
    )
    p.add_argument(
        "--model-cfg",
        default="sam2_hiera_l.yaml",
        dest="model_cfg",
        help="SAM 2 model config name (e.g. sam2_hiera_l.yaml)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device for SAM 2",
    )
    p.add_argument(
        "--svd-threshold",
        type=int,
        default=50,
        dest="svd_threshold",
        help="SVA boundary distance threshold in pixels",
    )
    p.add_argument(
        "--point",
        default=None,
        help='Skip interactive UI; use this point directly, e.g. --point "960,540"',
    )
    p.add_argument(
        "--frames-dir",
        default="tmp_frames",
        dest="frames_dir",
        help="Temporary directory for extracted JPEG frames",
    )
    p.add_argument(
        "--keep-frames",
        action="store_true",
        dest="keep_frames",
        help="Do not delete temp frames directory after run",
    )
    p.add_argument(
        "--max-hands",
        type=int,
        default=2,
        dest="max_hands",
        help="Maximum number of hands to detect per frame",
    )
    p.add_argument(
        "--hand-confidence",
        type=float,
        default=0.5,
        dest="hand_confidence",
        help="MediaPipe minimum hand detection confidence",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _check_display():
    """Raise if no display is available (headless environment)."""
    if platform.system() != "Windows" and os.environ.get("DISPLAY") is None:
        raise RuntimeError(
            "No display detected ($DISPLAY is not set).\n"
            "Run on a machine with a display, or use --point \"x,y\" to skip the UI."
        )


def main():
    args = parse_args()

    # Validate device
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU.")
                args.device = "cpu"
        except ImportError:
            print("Warning: torch not importable; device forced to cpu.")
            args.device = "cpu"

    frames_dir = args.frames_dir

    try:
        # ------------------------------------------------------------------
        # Step 1: Extract frames
        # ------------------------------------------------------------------
        print("\n[1/5] Extracting frames …")
        total_frames, fps = extract_frames(args.video, frames_dir)

        # ------------------------------------------------------------------
        # Step 2: Get point prompt
        # ------------------------------------------------------------------
        print("\n[2/5] Selecting point prompt …")
        if args.point:
            parts = args.point.split(",")
            click_x, click_y = int(parts[0].strip()), int(parts[1].strip())
            print(f"  Using CLI point: ({click_x}, {click_y})")
        else:
            _check_display()
            frame0_path = os.path.join(frames_dir, "00000.jpg")
            frame0 = cv2.imread(frame0_path)
            if frame0 is None:
                raise RuntimeError(f"Could not read frame: {frame0_path}")
            click_x, click_y = ClickUI(frame0).select_point()

        # ------------------------------------------------------------------
        # Step 3: SAM 2 initialisation + point prompt
        # ------------------------------------------------------------------
        print("\n[3/5] Initialising SAM 2 …")
        tracker = SAM2Tracker(args.checkpoint, args.model_cfg, args.device)
        state = tracker.init_video(frames_dir)
        tracker.set_point_prompt(state, click_x, click_y)
        print("  SAM 2 initialised. Starting propagation …")

        # ------------------------------------------------------------------
        # Steps 4 + 5: Stream propagation, hand detection, classification
        # ------------------------------------------------------------------
        print(f"\n[4/5] Propagating mask + classifying {total_frames} frames …")
        classifications: list[str | None] = [None] * total_frames

        with HandDetector(args.max_hands, args.hand_confidence) as detector:
            for frame_idx, mask in tracker.propagate(state):
                frame_path = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
                if not os.path.exists(frame_path):
                    print(f"\n  Warning: frame {frame_idx} not found, skipping.")
                    classifications[frame_idx] = "NA"
                    continue

                frame_bgr = cv2.imread(frame_path)
                keypoints = detector.detect(frame_bgr)
                dist_map = compute_dist_map(mask)
                classifications[frame_idx] = classify_frame(
                    mask, keypoints, dist_map, args.svd_threshold
                )

                if (frame_idx + 1) % 50 == 0 or frame_idx == total_frames - 1:
                    print(
                        f"\r  {frame_idx + 1}/{total_frames} frames processed",
                        end="",
                        flush=True,
                    )

        print()  # newline after progress output

        # Fill any None slots (shouldn't happen, but be defensive)
        classifications = [c if c is not None else "NA" for c in classifications]

        # ------------------------------------------------------------------
        # Step 6: Build and export Excel report
        # ------------------------------------------------------------------
        print(f"\n[5/5] Exporting report to {args.output} …")
        per_frame_df, summary_df = build_report(classifications, fps)
        export_excel(per_frame_df, summary_df, args.output)

        print(f"\nDone! Report saved to: {os.path.abspath(args.output)}")
        print("\nSummary:")
        for _, row in summary_df.iterrows():
            print(f"  {row['Metric']:25s}: {row['Value']}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as exc:
        # Surface GPU OOM with an actionable message
        try:
            import torch
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                print(
                    "\nGPU out of memory.\n"
                    "Try running with --device cpu, or use a smaller SAM 2 model\n"
                    "(e.g. bash setup_sam2.sh tiny  then --model-cfg sam2_hiera_t.yaml)"
                )
                sys.exit(1)
        except ImportError:
            pass
        raise
    finally:
        if not args.keep_frames and os.path.isdir(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)
            print(f"Temp frames removed: {frames_dir}")


if __name__ == "__main__":
    main()
