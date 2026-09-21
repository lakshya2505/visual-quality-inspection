# 🏭 Automated Visual Quality Inspection
## Using YOLO + Vision-Language Models — Full Project Roadmap

> **Project:** Automated Visual Quality Inspection Using YOLO and Vision-Language Models  
> **Subject:** Image Processing  
> **Team:** 2 members (you + AI partner)  
> **Demo Mode:** Pre-recorded video, shown in college computer lab  

---

## 📐 Full System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│              Pre-recorded Video File (.mp4/.avi)                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FRAME EXTRACTION                             │
│         OpenCV reads video → extracts frames one by one         │
│         Resize to 640×640 → Normalize pixel values              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               IMAGE PREPROCESSING (for subject)                 │
│   • Grayscale conversion (cv2.cvtColor)                         │
│   • Gaussian Blur (cv2.GaussianBlur)                            │
│   • Histogram Equalization (cv2.equalizeHist)                   │
│   • Canny Edge Detection (cv2.Canny)  ← show in report         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    YOLO v8 DETECTION                            │
│   • Runs on every frame (fast, ~30ms per frame)                 │
│   • Outputs: Bounding Boxes, Class Labels, Confidence Scores    │
│   • Draws annotations on frame                                  │
└──────────────┬────────────────────────────┬─────────────────────┘
               │                            │
               ▼                            ▼
    Defect Detected ✅               No Defect — skip ⏩
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CROP DEFECT REGION                            │
│      Extract bounding box region from original frame            │
│      Resize crop to 224×224 for VLM input                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                  (every 15 frames only)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│           VISION-LANGUAGE MODEL (BLIP)                          │
│   • Takes cropped defect image as input                         │
│   • Generates natural language description                      │
│   • e.g. "a crack on the surface of the object"                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                               │
│   • Annotated video saved as output_annotated.mp4               │
│   • Live display window during demo                             │
│   • CSV log: frame no., class, confidence, VLM description      │
│   • Accuracy metrics: mAP, Precision, Recall, F1                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 👥 Team Task Division

| Task | Person 1 (You) | Person 2 (AI Partner) |
|---|---|---|
| Environment setup | ✅ | ✅ |
| Dataset download & prep | ✅ | — |
| Preprocessing code | ✅ | — |
| YOLO integration | — | ✅ |
| VLM integration | — | ✅ |
| Main pipeline script | Together | Together |
| Accuracy evaluation | ✅ | — |
| Report / PPT | ✅ | ✅ |
| Demo preparation | Together | Together |

---

## 🗺️ PHASE-BY-PHASE ROADMAP

---

## PHASE 1 — Environment Setup
**Time estimate: 1 day**

### Step 1.1 — Install Python
- Download Python **3.10** (not 3.12, some libraries have issues with it)
- Link: https://www.python.org/downloads/release/python-31011/
- During install: **check "Add Python to PATH"**
- Verify: open terminal → `python --version`

### Step 1.2 — Install Git
- Link: https://git-scm.com/downloads
- Needed to clone repos and manage code between you two

### Step 1.3 — Create a Virtual Environment
```bash
# Navigate to your project folder
mkdir quality_inspection
cd quality_inspection

# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 1.4 — Install All Required Libraries
```bash
pip install ultralytics          # YOLOv8
pip install opencv-python        # video/image processing
pip install torch torchvision    # PyTorch (CPU version)
pip install transformers         # HuggingFace (for BLIP VLM)
pip install Pillow               # image handling
pip install pandas               # for CSV logging
pip install matplotlib           # for accuracy plots
pip install scikit-learn         # for metrics (precision, recall, F1)
```

> **Note for college lab computers:** If you cannot install globally, use `pip install --user <package>` or bring your virtual environment on a USB drive.

### Step 1.5 — Verify Everything Installed
```python
# save as check_install.py and run it
import cv2
import torch
import ultralytics
from transformers import BlipProcessor
import pandas
print("✅ All libraries installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"OpenCV version: {cv2.__version__}")
```

---

## PHASE 2 — Dataset & Test Video
**Time estimate: 1 day**

### Step 2.1 — Download the MVTec AD Dataset (Recommended)
- **What it is:** Industry-standard dataset of defective and non-defective surfaces
- **Download link:** https://www.mvtec.com/company/research/datasets/mvtec-ad
- **Size:** ~4.9 GB (download only 1-2 categories if storage is limited)
- **Recommended categories to download:** `bottle`, `metal_nut`, or `leather`
- **Structure after download:**
```
mvtec/
├── bottle/
│   ├── train/good/         ← defect-free images
│   └── test/
│       ├── good/           ← defect-free test images
│       ├── broken_large/   ← defective images
│       └── broken_small/
```

### Step 2.2 — Alternative: Film Your Own Video (Easiest for Demo)
If MVTec is too large, film a 30–60 second video yourself:
- Use your phone
- Objects to use: cracked bottle cap, torn paper, scratched phone case, bruised fruit
- Move the object slowly in front of the camera
- Transfer to laptop via USB or WhatsApp

### Step 2.3 — Create a Test Video from MVTec Images
If using MVTec images (not video), convert them into a video:
```python
# save as make_video.py
import cv2
import os
import glob

images = glob.glob("mvtec/bottle/test/broken_large/*.png")
images.sort()

frame = cv2.imread(images[0])
h, w = frame.shape[:2]

out = cv2.VideoWriter("test_video.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      5,  # 5 FPS, slow enough to inspect
                      (w, h))

for img_path in images:
    frame = cv2.imread(img_path)
    # repeat each image 10 times to make it 2 seconds each
    for _ in range(10):
        out.write(frame)

out.release()
print("✅ test_video.mp4 created!")
```

---

## PHASE 3 — YOLO Setup & Training
**Time estimate: 2 days**

### Step 3.1 — Understand Your Two Options

| Option | Effort | Accuracy | Recommended? |
|---|---|---|---|
| Use pre-trained YOLOv8 (general objects) | Low | Medium | ✅ For demo |
| Train custom YOLOv8 on MVTec defects | High | High | ✅ For better grades |

**For a college project, do Option A first. Add Option B if time allows.**

### Step 3.2 — Option A: Use Pre-trained YOLOv8

Download happens automatically on first run:
```python
from ultralytics import YOLO

# This downloads yolov8n.pt automatically (~6MB)
model = YOLO("yolov8n.pt")

# Test on one image
results = model("mvtec/bottle/test/broken_large/000.png")
results[0].show()
```

Available model sizes (bigger = more accurate but slower):
- `yolov8n.pt` — Nano (fastest, use this for lab demo)
- `yolov8s.pt` — Small
- `yolov8m.pt` — Medium
- `yolov8l.pt` — Large (most accurate, but slow on CPU)

### Step 3.3 — Option B: Train Custom YOLOv8 on Defect Data

#### Prepare dataset in YOLO format:
```
dataset/
├── images/
│   ├── train/   ← put training images here
│   └── val/     ← put validation images here
└── labels/
    ├── train/   ← .txt annotation files
    └── val/
```

Each `.txt` label file (same name as image):
```
# format: class_id center_x center_y width height (all 0-1 normalized)
0 0.512 0.437 0.234 0.189
```

#### Create dataset.yaml:
```yaml
# dataset.yaml
path: ./dataset
train: images/train
val: images/val

nc: 2  # number of classes
names: ['good', 'defect']
```

#### Annotate your images (free tool):
- **LabelImg:** https://github.com/HumanSignal/labelImg
- Download: `pip install labelImg` then run `labelImg`
- Draw boxes around defects, select class name, save

#### Train the model:
```bash
yolo detect train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Training output will be in `runs/detect/train/` — your best weights are at `runs/detect/train/weights/best.pt`

---

## PHASE 4 — VLM (BLIP) Setup
**Time estimate: 1 day**

### Step 4.1 — What VLM to Use

**Recommended: BLIP (Salesforce)**
- ✅ Runs on CPU (no GPU needed)
- ✅ Small download (~900 MB)
- ✅ Easy to use with HuggingFace
- ✅ Good at describing images in English

**Download link (automatic via code):**
```python
from transformers import BlipProcessor, BlipForConditionalGeneration

# This downloads the model automatically to your cache (~900MB)
# First run will take 5-10 minutes, after that it's cached
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("✅ BLIP model loaded!")
```

> **Tip for college lab:** Download the model at home and copy the cache folder to a USB. Default cache location:
> - Windows: `C:\Users\<you>\.cache\huggingface\hub\`
> - Linux/Mac: `~/.cache/huggingface/hub/`

### Step 4.2 — Test BLIP Alone
```python
# save as test_blip.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_defect(crop_bgr):
    """Takes a BGR numpy array (OpenCV format), returns description string."""
    # Convert BGR to RGB, then to PIL
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    
    # Prepare inputs with a prompt
    inputs = processor(pil_img, "a quality inspection photo showing:", return_tensors="pt")
    
    # Generate description
    output = blip_model.generate(**inputs, max_new_tokens=50)
    description = processor.decode(output[0], skip_special_tokens=True)
    return description

# Test with one image
test_img = cv2.imread("mvtec/bottle/test/broken_large/000.png")
result = describe_defect(test_img)
print(f"BLIP says: {result}")
```

---

## PHASE 5 — Build the Main Pipeline
**Time estimate: 2 days**

### Step 5.1 — Full Working Script

Save this as `main.py` — this is your complete project:

```python
"""
Automated Visual Quality Inspection
Using YOLOv8 + BLIP Vision-Language Model
Image Processing Project
"""

import cv2
import torch
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from datetime import datetime
import os

# ─────────────────────────────────────────────
# CONFIGURATION — change these as needed
# ─────────────────────────────────────────────
VIDEO_PATH = "test_video.mp4"           # your input video
OUTPUT_VIDEO = "output_annotated.mp4"   # where to save result
CSV_LOG = "detection_log.csv"           # detection log file
YOLO_MODEL = "yolov8n.pt"              # or "best.pt" if you trained custom
CONFIDENCE_THRESHOLD = 0.25            # minimum confidence to count as defect
VLM_EVERY_N_FRAMES = 15               # run VLM every N frames (for speed)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
print("Loading YOLOv8...")
yolo = YOLO(YOLO_MODEL)

print("Loading BLIP Vision-Language Model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("✅ Both models loaded.\n")

# ─────────────────────────────────────────────
# VLM DESCRIPTION FUNCTION
# ─────────────────────────────────────────────
def describe_defect(crop_bgr):
    """Generate natural language description of a defect crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown defect"
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    inputs = processor(pil_img, "a quality inspection image showing:", return_tensors="pt")
    with torch.no_grad():
        output = blip.generate(**inputs, max_new_tokens=40)
    return processor.decode(output[0], skip_special_tokens=True)

# ─────────────────────────────────────────────
# IMAGE PREPROCESSING FUNCTION (for report)
# ─────────────────────────────────────────────
def preprocess_frame(frame):
    """Apply image processing techniques — for visualization/report."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    edges = cv2.Canny(equalized, 50, 150)
    return gray, blurred, equalized, edges

# ─────────────────────────────────────────────
# VIDEO SETUP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Could not open video: {VIDEO_PATH}")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps}fps | Total frames: {total}")

out = cv2.VideoWriter(OUTPUT_VIDEO,
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (width, height))

# ─────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────
log_data = []
frame_count = 0
last_description = ""

print("\nProcessing video... Press 'Q' to quit early.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete.")
        break

    # ── YOLO Inference (every frame) ──────────
    results = yolo(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    annotated = results[0].plot()

    # ── VLM Inference (every N frames) ────────
    if frame_count % VLM_EVERY_N_FRAMES == 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id   = int(box.cls[0])
            class_name = yolo.names[class_id]

            # Crop the defect region
            crop = frame[y1:y2, x1:x2]
            description = describe_defect(crop)
            last_description = description

            # Log this detection
            log_data.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "frame": frame_count,
                "class": class_name,
                "confidence": round(confidence, 3),
                "bbox": f"({x1},{y1},{x2},{y2})",
                "vlm_description": description
            })

            print(f"Frame {frame_count:04d} | {class_name} ({confidence:.2f}) | {description}")

    # ── Overlay VLM text on frame ──────────────
    if last_description:
        cv2.putText(annotated, f"VLM: {last_description[:55]}",
                    (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2)

    # ── Frame counter overlay ──────────────────
    cv2.putText(annotated, f"Frame: {frame_count}/{total}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    # ── Write and display ──────────────────────
    out.write(annotated)
    cv2.imshow("Quality Inspection", annotated)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⏹ Stopped early by user.")
        break

# ─────────────────────────────────────────────
# CLEANUP AND SAVE
# ─────────────────────────────────────────────
cap.release()
out.release()
cv2.destroyAllWindows()

# Save detection log
if log_data:
    df = pd.DataFrame(log_data)
    df.to_csv(CSV_LOG, index=False)
    print(f"\n📄 Detection log saved to: {CSV_LOG}")
    print(f"   Total detections logged: {len(log_data)}")
else:
    print("\n⚠️  No defects detected. Try lowering CONFIDENCE_THRESHOLD.")

print(f"🎬 Output video saved to: {OUTPUT_VIDEO}")
```

---

## PHASE 6 — Accuracy Evaluation
**Time estimate: 1 day**

### Step 6.1 — What Metrics to Use

| Metric | What it means | Good value |
|---|---|---|
| **mAP@0.5** | Mean Average Precision at IoU 0.5 | > 0.50 |
| **Precision** | Of all detections, how many were correct | > 0.70 |
| **Recall** | Of all actual defects, how many did we find | > 0.60 |
| **F1 Score** | Balance of precision and recall | > 0.65 |

### Step 6.2 — Evaluate YOLOv8 Automatically

If you trained a custom model, run this:
```bash
yolo detect val model=runs/detect/train/weights/best.pt data=dataset.yaml
```

This prints a full table with mAP, precision, recall for each class.

If using the pre-trained model, evaluate on MVTec test images:
```python
# save as evaluate.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or your best.pt

metrics = model.val(data="dataset.yaml", split="val")

print(f"\n📊 ACCURACY RESULTS")
print(f"mAP@0.5:     {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95:{metrics.box.map:.3f}")
print(f"Precision:   {metrics.box.mp:.3f}")
print(f"Recall:      {metrics.box.mr:.3f}")
```

### Step 6.3 — Plot Confusion Matrix and Graphs

YOLOv8 automatically saves these in `runs/detect/val/`:
- `confusion_matrix.png`
- `PR_curve.png`
- `F1_curve.png`
- `results.png`

**Include these graphs in your report and PPT — professors love them.**

### Step 6.4 — Manual Accuracy Check Script

```python
# save as manual_accuracy.py
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import pandas as pd

# Load your CSV log
df = pd.read_csv("detection_log.csv")

# If you have ground truth labels, compare them
# For demo purposes, set threshold: confidence > 0.5 = defect detected
df["predicted"] = (df["confidence"] > 0.5).astype(int)

# You manually label: 1 = there was actually a defect, 0 = there wasn't
# Create a ground_truth column in your CSV manually for a few rows
# Then:
# ground_truth = df["ground_truth"].tolist()
# predicted    = df["predicted"].tolist()
# print(classification_report(ground_truth, predicted))

print("Detections per class:")
print(df["class"].value_counts())
print(f"\nAverage confidence: {df['confidence'].mean():.3f}")
```

---

## PHASE 7 — Demo Preparation
**Time estimate: 0.5 day**

### Step 7.1 — Folder Structure for Lab Day

Make sure your project folder looks like this before going to the lab:
```
quality_inspection/
├── venv/                        ← your virtual environment (bring this!)
├── main.py                      ← main pipeline
├── evaluate.py                  ← accuracy evaluation
├── test_blip.py                 ← BLIP test script
├── make_video.py                ← video creation utility
├── check_install.py             ← environment check
├── test_video.mp4               ← your input video
├── yolov8n.pt                   ← YOLO weights (auto-downloaded)
├── dataset.yaml                 ← if you trained custom model
└── ~/.cache/huggingface/        ← BLIP model cache (bring this on USB!)
```

> ⚠️ **CRITICAL:** Download all models at home before going to the lab. The lab may not have internet access or may be slow. Cache BLIP model and YOLO weights locally.

### Step 7.2 — Demo Script (What to Say to Professor)

```
1. "This is our input video — you can see a product moving past the camera."
   → Play test_video.mp4 raw

2. "We run this through our pipeline. YOLO detects defect regions in real-time."
   → Run: python main.py
   → Show the live window with bounding boxes

3. "The Vision-Language Model then describes what kind of defect was found."
   → Point to the yellow VLM text at the bottom of the window

4. "Every detection is logged to a CSV file."
   → Open detection_log.csv in Excel

5. "The output annotated video is also saved."
   → Play output_annotated.mp4

6. "Here are our accuracy metrics."
   → Show the confusion matrix and PR curve images
```

### Step 7.3 — Create a Simple Requirements File

```bash
# run this in your project folder
pip freeze > requirements.txt
```

This lets anyone reproduce your environment with:
```bash
pip install -r requirements.txt
```

---

## PHASE 8 — Report & Presentation Points
**Time estimate: 1 day**

### What to Include in Your Report

**Section 1: Introduction**
- Problem statement: manual inspection is slow, error-prone
- Proposed solution: automated AI-based inspection

**Section 2: Literature Review**
- YOLO paper (Redmon et al., 2016)
- BLIP paper (Li et al., 2022)
- MVTec AD dataset paper

**Section 3: Methodology**
- Full architecture diagram (copy from this document)
- Each component explained
- Image preprocessing steps with visual outputs

**Section 4: Implementation**
- Tools and libraries used
- Code snippets for key parts
- Dataset used

**Section 5: Results**
- Screenshots of detection on video
- mAP, Precision, Recall, F1 table
- Confusion matrix image
- Sample VLM descriptions

**Section 6: Conclusion**
- What worked well
- Limitations (VLM is slow, needs GPU for real-time)
- Future work (edge deployment, real-time GPU inference)

---

## 📦 Complete Downloads Checklist

| Item | Where to Get | Size |
|---|---|---|
| Python 3.10 | https://python.org/downloads | ~25 MB |
| Git | https://git-scm.com/downloads | ~50 MB |
| LabelImg (annotation tool) | `pip install labelImg` | ~10 MB |
| MVTec Dataset | https://www.mvtec.com/company/research/datasets/mvtec-ad | ~4.9 GB |
| YOLOv8n weights | Auto-downloaded on first run | ~6 MB |
| BLIP model | Auto-downloaded on first run | ~900 MB |

---

## ⚠️ Common Problems & Fixes

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure venv is activated, reinstall the package |
| Video not opening | Check VIDEO_PATH is correct, try absolute path |
| BLIP download fails | Download at home, copy cache folder to USB |
| Out of memory error | Use `yolov8n.pt` (smallest model), reduce image size |
| VLM very slow | Increase `VLM_EVERY_N_FRAMES` to 30 or 60 |
| No detections showing | Lower `CONFIDENCE_THRESHOLD` to 0.1 |
| Lab has no internet | Pre-download all models, bring on USB drive |

---

## 🗓️ Suggested Weekly Timeline

```
Week 1  │ Phase 1 (Setup) + Phase 2 (Dataset)
Week 2  │ Phase 3 (YOLO) + Phase 4 (BLIP)
Week 3  │ Phase 5 (Main Pipeline) + Phase 6 (Accuracy)
Week 4  │ Phase 7 (Demo Prep) + Phase 8 (Report/PPT)
```

---

## 🔗 All Official Links

- **YOLOv8 Docs:** https://docs.ultralytics.com
- **BLIP on HuggingFace:** https://huggingface.co/Salesforce/blip-image-captioning-base
- **MVTec Dataset:** https://www.mvtec.com/company/research/datasets/mvtec-ad
- **LabelImg Annotation Tool:** https://github.com/HumanSignal/labelImg
- **OpenCV Docs:** https://docs.opencv.org
- **HuggingFace Transformers:** https://huggingface.co/docs/transformers

---

*Roadmap prepared for: Automated Visual Quality Inspection Using YOLO and Vision-Language Models*  
*Last updated: 2026*
