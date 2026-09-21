# 🗄️ SKILL 02 — Dataset Download & Test Video

## Option A — MVTec AD Dataset (Recommended for Report)

### What it is
Industry-standard benchmark for visual anomaly detection. Used in academic papers. Professors will recognize it — great for grades.

### Download
- **URL:** https://www.mvtec.com/company/research/datasets/mvtec-ad
- **Total size:** ~4.9 GB (but you only need 1 category)
- **Free to use** for academic/research purposes

### What to Download (Don't get the whole thing!)
Only download **one** category to save space on your 256 GB drive:

| Category | Size | Best for demo |
|---|---|---|
| `bottle` | ~1.2 GB | ✅ Most visual, easy to show |
| `metal_nut` | ~0.8 GB | Good for industrial look |
| `leather` | ~1.1 GB | Clear scratch patterns |

**Recommendation: Download `bottle`** — the broken glass defects are very visually obvious on video.

### After Download — Folder Structure
```
visual_quality_inspection/
└── data/
    └── mvtec/
        └── bottle/
            ├── train/
            │   └── good/          ← 209 defect-free training images
            └── test/
                ├── good/          ← 83 clean test images
                ├── broken_large/  ← 39 cracked bottle images ← USE THESE
                ├── broken_small/  ← 44 small crack images
                └── contamination/ ← 21 contaminated images
```

---

## Option B — Film Your Own Video (Quickest for Demo)

If MVTec download is too slow or you're short on time:

### What to Film
Use your phone camera. Objects that work well:
- A bottle cap with a scratch (draw with a pen)
- A piece of paper with a tear
- A plastic box with a dent
- Bruised fruit (banana, apple)

### How to Film
1. Place object on a plain white/light-colored surface
2. Film for 30–60 seconds, moving the object slowly
3. Keep lighting consistent (near a window works well)
4. Transfer to laptop via USB cable or AirDrop

### Transfer Location
Save the video as:
```
visual_quality_inspection/data/test_video.mp4
```

---

## Option C — Create Test Video from MVTec Images

After downloading MVTec, run this script to convert images into a video:

Save as `scripts/make_video.py`:
```python
import cv2
import os
import glob

# Point to your MVTec category's test defect folder
images = sorted(glob.glob("data/mvtec/bottle/test/broken_large/*.png"))

if not images:
    print("❌ No images found! Check the path.")
    exit()

frame = cv2.imread(images[0])
h, w = frame.shape[:2]

out = cv2.VideoWriter(
    "data/test_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    5,      # 5 FPS — slow, good for inspection demo
    (w, h)
)

for img_path in images:
    frame = cv2.imread(img_path)
    for _ in range(10):    # repeat each image for 2 seconds
        out.write(frame)

out.release()
print(f"✅ test_video.mp4 created from {len(images)} images!")
```

Run it:
```bash
conda activate ipa
python scripts/make_video.py
```

---

## ✅ Dataset Checklist

| Item | Status | Location |
|---|---|---|
| Test video OR MVTec images | ⬜ TODO | `data/test_video.mp4` or `data/mvtec/` |
| `data/samples/` folder exists | ✅ Already exists | `data/samples/` |

---

## Space Calculator (for 256 GB Mac)
- MVTec bottle category: ~1.2 GB
- BLIP model cache: ~900 MB  
- YOLOv8n weights: ~6 MB
- Generated outputs: ~500 MB
- **Total needed: ~2.7 GB** — well within your storage
