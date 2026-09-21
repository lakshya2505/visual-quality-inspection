# 🎬 SKILL 07 — Demo Day Preparation

## Before Lab Day — Complete Checklist

### 1. Folder Must Contain Everything (No Internet Needed at Lab)

```
visual_quality_inspection/
├── conda_env/           ← export with: conda pack -n ipa -o ipa_env.tar.gz
├── skills/              ← these guide files
├── src/                 ← ✅ written
├── configs/
│   └── config.yaml      ← ✅ written
├── data/
│   └── test_video.mp4   ← ⬜ YOU CREATE THIS
├── outputs/             ← ✅ exists
├── main.py              ← ⬜ YOU CREATE (Skill 05)
├── demo.py              ← ✅ written
├── requirements.txt     ← ✅ written
└── yolov8n.pt           ← ⬜ auto-downloaded (pre-download at home)
```

**BLIP model cache (take on USB or laptop):**
```
~/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base/
```

---

### 2. Pre-Download Everything at Home

```bash
conda activate ipa

# Download YOLO weights
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('✅ YOLO ready')"

# Download BLIP model (~900MB, takes 5-10 minutes first time)
python -c "
from transformers import BlipProcessor, BlipForConditionalGeneration
BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
print('✅ BLIP ready')
"
```

---

### 3. Final Verification Script

Save as `scripts/final_check.py` and run on lab day:

```python
"""Run this at the start of lab day to verify everything works."""
import sys
sys.path.insert(0, ".")

errors = []

# Check 1: Libraries
try:
    import cv2, torch, ultralytics, transformers, pandas, sklearn
    print(f"✅ Libraries OK | PyTorch {torch.__version__} | MPS: {torch.backends.mps.is_available()}")
except ImportError as e:
    errors.append(f"❌ Missing library: {e}")

# Check 2: YOLO weights
import os
if os.path.exists("yolov8n.pt"):
    print("✅ yolov8n.pt found")
else:
    errors.append("❌ yolov8n.pt missing — run: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")

# Check 3: Test video
if os.path.exists("data/test_video.mp4"):
    import cv2
    cap = cv2.VideoCapture("data/test_video.mp4")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"✅ test_video.mp4 found ({frames} frames)")
else:
    errors.append("❌ data/test_video.mp4 missing — see Skill 02")

# Check 4: BLIP model cached
hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
blip_dirs = [d for d in os.listdir(hf_cache) if "blip" in d.lower()] if os.path.exists(hf_cache) else []
if blip_dirs:
    print(f"✅ BLIP model cached: {blip_dirs[0]}")
else:
    errors.append("❌ BLIP not cached — run Skill 04 Step 2 at home with internet")

# Check 5: Quick YOLO inference
try:
    from src.detection.yolo_detector import YOLODefectDetector
    import numpy as np
    det = YOLODefectDetector("yolov8n.pt", device="cpu")  # cpu for quick check
    result = det.detect(np.zeros((480, 640, 3), dtype="uint8"))
    print("✅ YOLO inference works")
except Exception as e:
    errors.append(f"❌ YOLO error: {e}")

# Summary
if errors:
    print("\n⚠️  ISSUES FOUND:")
    for e in errors:
        print(f"  {e}")
else:
    print("\n🎉 ALL CHECKS PASSED — Ready for demo!")
```

```bash
conda activate ipa
python scripts/final_check.py
```

---

## Demo Script for Professor (What to Say)

### Step 1 — Show Input Video
```
"This is our test input video — a product moving past the inspection camera."
→ Open test_video.mp4 in QuickTime Player
```

### Step 2 — Show the Image Preprocessing
```
"Our pipeline applies standard image processing: grayscale, Gaussian blur, 
histogram equalization, and Canny edge detection — these help YOLO detect 
defect boundaries more accurately."
→ Show any preprocessing visualization image
```

### Step 3 — Run the Live Pipeline
```bash
python main.py --video data/test_video.mp4 --device mps --vlm-backend blip
```
```
"YOLO detects defect regions in real-time. Each bounding box shows the class 
and confidence score. The yellow text at the bottom is the VLM's description."
→ Point to the live window
```

### Step 4 — Open the CSV Log
```
"Every detection is automatically logged with timestamp, class, confidence, 
severity, and the natural language description from BLIP."
→ Open outputs/detection_log.csv in Numbers or Excel
```

### Step 5 — Show Metrics
```
"Here are our accuracy metrics — precision, recall, F1 score — shown as 
bar charts and a class distribution plot."
→ Show outputs/analysis_plots.png
```

### Step 6 — Show Output Video
```
"The annotated video is saved with all bounding boxes overlaid."
→ Open outputs/output_annotated.mp4
```

---

## Common Lab Day Issues & Fixes

| Problem | Quick Fix |
|---|---|
| `conda activate ipa` fails | Run `conda init zsh` then restart terminal |
| MPS memory error | Change `--device mps` to `--device cpu` |
| Video won't open | Run `python -c "import cv2; print(cv2.__version__)"` to verify OpenCV |
| BLIP slow (>30s/frame) | Increase `--vlm-interval 60` to run VLM less often |
| No display window | Add `--no-display` flag, video still saves |
| BLIP not cached | Use `--vlm-backend mock` for quick demo |

---

## ✅ Demo Day Checklist

| Item | Status |
|---|---|
| All models downloaded at home | ⬜ TODO |
| `scripts/final_check.py` passes | ⬜ TODO |
| Test video ready | ⬜ TODO |
| `main.py` runs end to end | ⬜ TODO |
| `demo.py` runs without errors | ⬜ TODO |
| Analysis plots generated | ⬜ TODO |
| BLIP cache backed up (USB) | ⬜ TODO (before lab) |
