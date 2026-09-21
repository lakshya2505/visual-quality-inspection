# 🤖 SKILL 03 — YOLO Setup & Integration

## What's Already Done ✅
The file `src/detection/yolo_detector.py` is **fully written** and production-ready. It includes:
- `YOLODefectDetector` class wrapping YOLOv8
- `Defect`, `BoundingBox`, `DetectionResult` data classes
- Image annotation with color-coded bounding boxes
- Defect region cropping for VLM input
- Apple Silicon MPS device support built in

---

## Step 1 — Download YOLOv8 Weights (Automatic)

YOLOv8 weights download automatically on first use. But you can pre-download:

```bash
conda activate ipa
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

This downloads `yolov8n.pt` (~6 MB) to your current directory.

**Model size guide for M4 Mac with 8GB RAM:**

| Model | Size | Speed (M4 CPU) | Recommended? |
|---|---|---|---|
| `yolov8n.pt` | 6 MB | ~30ms/frame | ✅ Best for demo |
| `yolov8s.pt` | 22 MB | ~50ms/frame | Good |
| `yolov8m.pt` | 50 MB | ~100ms/frame | Slow on 8GB |
| `yolov8l.pt` | 87 MB | Very slow | ❌ Skip |

**→ Use `yolov8n.pt` for smooth demo video playback.**

---

## Step 2 — Quick Test (Run This First!)

Save as `scripts/test_yolo.py`:

```python
"""Quick test that YOLO can run on a sample image."""
import cv2
import sys
sys.path.insert(0, ".")
from src.detection.yolo_detector import YOLODefectDetector

detector = YOLODefectDetector(
    model_path="yolov8n.pt",
    confidence_threshold=0.25,
    device="mps",    # ← M4 Mac: use mps for GPU acceleration
)

# Test on the synthetic demo image (requires demo.py to have been run first)
import os
if os.path.exists("data/samples/demo_product.jpg"):
    result = detector.detect_from_path("data/samples/demo_product.jpg")
    print(f"✅ YOLO working! Found {result.defect_count} detections")
    print(f"   Inference time: {result.inference_time_ms:.1f}ms")
else:
    # Create a test image on the fly
    import numpy as np
    test_img = np.ones((480, 640, 3), dtype="uint8") * 120
    result = detector.detect(test_img, "synthetic")
    print(f"✅ YOLO loaded. Model works (0 detections on blank image — expected)")
```

```bash
conda activate ipa
python scripts/test_yolo.py
```

---

## Step 3 — Using the Detector in Your Code

```python
from src.detection.yolo_detector import YOLODefectDetector
import cv2

# Initialize
detector = YOLODefectDetector(
    model_path="yolov8n.pt",
    confidence_threshold=0.25,
    device="mps",        # Use "cpu" if mps causes issues
)

# Run on an image
image = cv2.imread("path/to/image.jpg")
result = detector.detect(image, image_path="path/to/image.jpg")

# Print what was found
print(f"Pass/Fail: {result.pass_fail}")
for defect in result.defects:
    print(f"  {defect.class_name} | conf={defect.confidence:.2f}")

# Get annotated image for display
annotated = detector.annotate_image(image, result)
cv2.imshow("Result", annotated)
cv2.waitKey(0)
```

---

## Step 4 — Device Selection on M4 Mac

Update `configs/config.yaml`:
```yaml
detection:
  device: "mps"    # Apple Silicon GPU
```

Or override at runtime:
```python
# MPS = Apple Silicon GPU (fastest on M4)
# cpu = fallback if MPS causes memory issues with 8GB RAM
detector = YOLODefectDetector(device="mps")
```

> 💡 **If you get MPS memory errors:** Switch to `device="cpu"`. With 8GB shared RAM, MPS can sometimes OOM on large batches.

---

## Step 5 — (Optional) Custom Training on MVTec

Only do this if you have time and want better detection:

```bash
# 1. Annotate your MVTec images with LabelImg
pip install labelImg
labelImg

# 2. Prepare YOLO format labels and dataset.yaml
# (See ipa-project.md Phase 3 for full format)

# 3. Train
yolo detect train data=configs/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640 device=mps
```

Training output goes to `runs/detect/train/`. Best weights: `runs/detect/train/weights/best.pt`

---

## ✅ YOLO Checklist

| Item | Status |
|---|---|
| `yolov8n.pt` downloaded | ⬜ Run Step 1 |
| `src/detection/yolo_detector.py` | ✅ Already written |
| Test script passes | ⬜ Run Step 2 |
| Config updated to `mps` | ⬜ Update config.yaml |
