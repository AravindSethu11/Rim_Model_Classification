# Video Time Study Analysis Tool

A computer-vision pipeline that automatically classifies manufacturing video footage into **Value-Added (VA)**, **Supplemental Value-Added (SVA)**, and **Non-Value-Added (NA)** activities by tracking a part with SAM 2 and detecting hand contact with MediaPipe Hands.

## How It Works

1. **Frame extraction** — the video is decoded into individual JPEG frames.
2. **Point prompt** — you click on the part in frame 0; the click is used as a SAM 2 foreground point.
3. **Mask propagation** — SAM 2 propagates a segmentation mask of that part across every frame.
4. **Hand detection** — MediaPipe Hands locates hand keypoints on every frame.
5. **Classification** — each frame is labelled using the following priority chain:
   - **VA** — at least one hand keypoint lies *inside* the part mask.
   - **SVA** — at least one hand keypoint is within `--svd-threshold` pixels of the mask boundary (default 50 px).
   - **NA** — no hand is near the part (or no mask / no hand detected).
6. **Excel report** — a two-sheet workbook is written with per-frame detail and a summary.

---

## Requirements

- Python 3.10+
- A CUDA-capable GPU is **strongly recommended** for SAM 2 propagation. CPU mode works but is very slow for long videos.

---

## Installation

### 1. Clone this repository

```bash
git clone <repo-url>
cd Rim_Model_Classification
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install SAM 2 and download a checkpoint

```bash
bash setup_sam2.sh [tiny|small|base_plus|large]
```

Default model: `large`. This script:
- Clones and pip-installs SAM 2 from Meta's GitHub.
- Downloads the selected model checkpoint into `./checkpoints/`.

Alternatively, install manually:

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e . && cd ..

mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
     -O checkpoints/sam2_hiera_large.pt
```

---

## Usage

### Interactive mode (recommended)

```bash
python video_time_study.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/sam2_hiera_large.pt \
    --model-cfg sam2_hiera_l.yaml \
    --output report.xlsx
```

A window opens showing frame 0. Click on the part you want to track and press **Enter** to confirm. Press **Esc** to abort.

### Headless / scripted mode

Pass `--point "x,y"` to skip the interactive window:

```bash
python video_time_study.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/sam2_hiera_large.pt \
    --model-cfg sam2_hiera_l.yaml \
    --point "960,540" \
    --output report.xlsx
```

---

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--video` | *(required)* | Path to the input video file |
| `--output` | `report.xlsx` | Path for the Excel output workbook |
| `--checkpoint` | `checkpoints/sam2_hiera_large.pt` | SAM 2 model checkpoint (`.pt` file) |
| `--model-cfg` | `sam2_hiera_l.yaml` | SAM 2 model config filename |
| `--device` | `cuda` | Compute device: `cuda` or `cpu` |
| `--svd-threshold` | `50` | SVA boundary distance threshold in pixels |
| `--point` | *(none)* | Skip UI; use this point directly, e.g. `"960,540"` |
| `--frames-dir` | `tmp_frames` | Temporary directory for extracted JPEG frames |
| `--keep-frames` | *(flag)* | Keep the frames directory after the run |
| `--max-hands` | `2` | Maximum number of hands MediaPipe detects per frame |
| `--hand-confidence` | `0.5` | MediaPipe minimum hand detection confidence |

---

## Model Size Guide

| Size | Config | Checkpoint | Speed | Accuracy |
|---|---|---|---|---|
| Tiny | `sam2_hiera_t.yaml` | `sam2_hiera_tiny.pt` | Fastest | Lowest |
| Small | `sam2_hiera_s.yaml` | `sam2_hiera_small.pt` | Fast | Good |
| Base+ | `sam2_hiera_b+.yaml` | `sam2_hiera_base_plus.pt` | Moderate | Better |
| Large | `sam2_hiera_l.yaml` | `sam2_hiera_large.pt` | Slowest | Best |

Use `bash setup_sam2.sh tiny` for GPU-constrained or CPU-only environments.

---

## Output Format

The Excel workbook contains two sheets:

### `Per-Frame` sheet

| Frame Number | Timestamp (s) | Classification | Duration (s) |
|---|---|---|---|
| 0 | 0.0000 | NA | 0.033333 |
| 1 | 0.0333 | VA | 0.033333 |
| … | … | … | … |

### `Summary` sheet

| Metric | Value |
|---|---|
| Total Frames | 1800 |
| Video Duration (s) | 60.0 |
| VA Time (s) | 22.5 |
| SVA Time (s) | 8.1 |
| NA Time (s) | 29.4 |
| VA % | 37.5 |
| SVA % | 13.5 |
| NA % | 49.0 |

---

## Tips

- **SAM 2 drift on long videos** — if the mask drifts after many minutes, consider splitting the video into shorter segments and running the tool on each.
- **Low hand detection recall** — lower `--hand-confidence` to `0.3` for difficult industrial footage with unusual hand angles.
- **Variable-frame-rate (VFR) video** — timestamps may be slightly off for VFR sources. Convert to CFR first: `ffmpeg -i input.mp4 -vsync cfr output.mp4`.
- **GPU out of memory** — switch to a smaller model: `bash setup_sam2.sh tiny` and use `--model-cfg sam2_hiera_t.yaml`.
