"""
make_video.py
-------------
Converts MVTec bottle test images into a demo video for the inspection pipeline.

Creates a realistic inspection scenario:
  - Starts with good (clean) bottles
  - Then shows defective ones: broken_large, broken_small, contamination
  - Mixes them so the demo looks like a real production line

Output: data/test_video.mp4

Run:
    python scripts/make_video.py
"""

import cv2
import glob
import os
import sys
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "bottle" / "test"
OUTPUT = BASE / "data" / "test_video.mp4"

# ─── How many times to repeat each image (controls video length)
REPEAT_GOOD    = 5   # 1 sec each @ 5fps — short, just to show clean state
REPEAT_DEFECT  = 10  # 2 sec each @ 5fps — longer, so YOLO + VLM can process it
FPS = 5              # Low FPS so detections have time to render in demo

# ─── Image sequence: good → broken_large → broken_small → contamination
SEQUENCE = [
    ("good",         REPEAT_GOOD,   "✅ Clean bottles"),
    ("broken_large", REPEAT_DEFECT, "🔴 Large cracks"),
    ("broken_small", REPEAT_DEFECT, "🟠 Small cracks"),
    ("contamination",REPEAT_DEFECT, "🟡 Contamination"),
]

def get_images(folder_name):
    folder = DATA / folder_name
    if not folder.exists():
        print(f"  ⚠️  Folder not found: {folder}")
        return []
    images = sorted(glob.glob(str(folder / "*.png")))
    if not images:
        images = sorted(glob.glob(str(folder / "*.jpg")))
    return images

def main():
    print("=" * 55)
    print("  MVTec Bottle → Demo Video Creator")
    print("=" * 55)

    # Collect all image paths in order
    all_frames = []
    for folder_name, repeat, label in SEQUENCE:
        imgs = get_images(folder_name)
        if not imgs:
            continue
        print(f"  {label}: {len(imgs)} images × {repeat} repeats = {len(imgs) * repeat} frames")
        for img_path in imgs:
            for _ in range(repeat):
                all_frames.append((img_path, label))

    if not all_frames:
        print("\n❌ No images found! Check that data/bottle/ was downloaded correctly.")
        sys.exit(1)

    print(f"\n  Total frames to write: {len(all_frames)}")
    print(f"  Video duration: ~{len(all_frames) / FPS:.1f} seconds @ {FPS}fps\n")

    # Read first image to get dimensions
    first_img = cv2.imread(all_frames[0][0])
    if first_img is None:
        print(f"❌ Cannot read image: {all_frames[0][0]}")
        sys.exit(1)

    h, w = first_img.shape[:2]
    print(f"  Frame size: {w}x{h}")

    # Create output VideoWriter
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w, h)
    )

    if not writer.isOpened():
        print("❌ VideoWriter failed to open. Check OpenCV installation.")
        sys.exit(1)

    # Write frames with a small label overlay
    for i, (img_path, label) in enumerate(all_frames):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Add category label to frame (for visual reference, not for pipeline)
        category = label.split(" ", 1)[-1]
        cv2.putText(
            frame, category,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )

        writer.write(frame)

        # Progress every 50 frames
        if i % 50 == 0:
            print(f"  Writing frame {i+1}/{len(all_frames)}...", end="\r")

    writer.release()
    print(f"\n\n✅ Video saved → {OUTPUT}")
    print(f"   Size: {OUTPUT.stat().st_size / (1024*1024):.1f} MB")
    print(f"   Duration: ~{len(all_frames)/FPS:.0f} seconds")
    print(f"\nNext step:")
    print(f"  python main.py --video data/test_video.mp4 --device cpu --vlm-backend mock")

if __name__ == "__main__":
    main()
