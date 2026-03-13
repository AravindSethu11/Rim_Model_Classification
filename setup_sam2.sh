#!/usr/bin/env bash
# setup_sam2.sh — Install SAM 2 and download a model checkpoint
# Run once before using video_time_study.py
#
# Usage:
#   bash setup_sam2.sh [tiny|small|base_plus|large]
#   Default model: large

set -e

MODEL="${1:-large}"

case "$MODEL" in
  tiny)
    CFG="sam2_hiera_t.yaml"
    CKPT="sam2_hiera_tiny.pt"
    URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
    ;;
  small)
    CFG="sam2_hiera_s.yaml"
    CKPT="sam2_hiera_small.pt"
    URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
    ;;
  base_plus)
    CFG="sam2_hiera_b+.yaml"
    CKPT="sam2_hiera_base_plus.pt"
    URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
    ;;
  large)
    CFG="sam2_hiera_l.yaml"
    CKPT="sam2_hiera_large.pt"
    URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    ;;
  *)
    echo "Unknown model '$MODEL'. Choose: tiny, small, base_plus, large"
    exit 1
    ;;
esac

echo "=== Installing SAM 2 from source ==="
if [ ! -d "segment-anything-2" ]; then
  git clone https://github.com/facebookresearch/segment-anything-2.git
fi
cd segment-anything-2
pip install -e . --quiet
cd ..

echo "=== Downloading checkpoint: $CKPT ==="
mkdir -p checkpoints
if [ ! -f "checkpoints/$CKPT" ]; then
  wget -q --show-progress "$URL" -O "checkpoints/$CKPT"
  echo "Checkpoint saved to checkpoints/$CKPT"
else
  echo "Checkpoint already exists: checkpoints/$CKPT"
fi

echo ""
echo "=== Done! ==="
echo "Run the tool with:"
echo "  python video_time_study.py --video your_video.mp4 --checkpoint checkpoints/$CKPT --model-cfg $CFG"
echo ""
echo "For CPU-only machines, add: --device cpu"
echo "For headless machines, add: --point \"x,y\""
