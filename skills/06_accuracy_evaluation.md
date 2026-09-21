# 📊 SKILL 06 — Accuracy Evaluation & Metrics

## What You Need to Show
The project requires these metrics:
- **mAP@0.5** (Mean Average Precision)
- **Precision** — how many detections were correct
- **Recall** — how many actual defects were found
- **F1 Score** — harmonic mean of Precision & Recall

---

## Approach A — Automatic YOLO Evaluation (Best for Report)

If you trained a custom YOLO model on MVTec labels, run:

```bash
conda activate ipa
yolo detect val model=runs/detect/train/weights/best.pt data=configs/dataset.yaml
```

This automatically generates (in `runs/detect/val/`):
- `confusion_matrix.png`
- `PR_curve.png`
- `F1_curve.png`
- `results.png`
- A full mAP table in the terminal

**→ Copy these images directly into your report and PPT!**

---

## Approach B — Evaluate Pre-trained YOLO on MVTec

If using the pre-trained `yolov8n.pt`, create `scripts/evaluate.py`:

```python
"""
evaluate.py — Run YOLO validation metrics on MVTec test set.
Requires: dataset.yaml pointing to your MVTec images with labels.
"""
import sys
sys.path.insert(0, ".")
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # or "runs/detect/train/weights/best.pt"

# Validate on your dataset
metrics = model.val(data="configs/dataset.yaml", split="val", device="mps")

print("\n" + "=" * 45)
print("  📊 ACCURACY RESULTS")
print("=" * 45)
print(f"  mAP@0.5       : {metrics.box.map50:.3f}")
print(f"  mAP@0.5:0.95  : {metrics.box.map:.3f}")
print(f"  Precision     : {metrics.box.mp:.3f}")
print(f"  Recall        : {metrics.box.mr:.3f}")
print("=" * 45)
```

```bash
conda activate ipa
python scripts/evaluate.py
```

---

## Approach C — Manual Analysis from Detection Log (Simplest)

After running `main.py`, analyze `outputs/detection_log.csv`:

```python
# scripts/analyze_log.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("outputs/detection_log.csv")

print(f"Total detections: {len(df)}")
print(f"Average confidence: {df['confidence'].mean():.3f}")
print("\nDetections per class:")
print(df["class"].value_counts())
print("\nSeverity distribution:")
print(df["severity"].value_counts())

# Plot 1: Class distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

df["class"].value_counts().plot(kind="bar", ax=axes[0], color="#4C72B0")
axes[0].set_title("Defects by Class")
axes[0].set_xlabel("Class")
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Confidence histogram
axes[1].hist(df["confidence"], bins=20, color="#DD8452", edgecolor="white")
axes[1].set_title("Confidence Distribution")
axes[1].set_xlabel("Confidence Score")
axes[1].axvline(x=0.5, color="red", linestyle="--", label="0.5 threshold")
axes[1].legend()

# Plot 3: Severity distribution
df["severity"].value_counts().plot(
    kind="pie", ax=axes[2], autopct="%1.0f%%",
    colors=["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
)
axes[2].set_title("Severity Distribution")
axes[2].set_ylabel("")

plt.tight_layout()
plt.savefig("outputs/analysis_plots.png", dpi=150)
print("\n✅ Analysis plots saved to outputs/analysis_plots.png")
plt.show()
```

```bash
conda activate ipa
python scripts/analyze_log.py
```

---

## Approach D — Precision/Recall with Sklearn (Manual Labels)

If you manually labeled some frames:

```python
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import pandas as pd

df = pd.read_csv("outputs/detection_log.csv")

# Predicted: 1 if confidence > 0.5
df["predicted"] = (df["confidence"] > 0.5).astype(int)

# You manually add ground truth column (1=defect, 0=clean)
# df["ground_truth"] = [1, 1, 0, 1, ...]   # add manually for demo frames

# If you have ground truth:
# gt = df["ground_truth"].tolist()
# pred = df["predicted"].tolist()
# print(classification_report(gt, pred, target_names=["clean", "defect"]))
```

---

## What to Include in Your Report (From These Scripts)

| Graph | Source | How to get |
|---|---|---|
| `confusion_matrix.png` | YOLO val | Run Approach B |
| `PR_curve.png` | YOLO val | Run Approach B |
| `F1_curve.png` | YOLO val | Run Approach B |
| Class distribution bar | Log analysis | Run Approach C |
| Confidence histogram | Log analysis | Run Approach C |
| Severity pie chart | Log analysis | Run Approach C |

---

## ✅ Evaluation Checklist

| Item | Status |
|---|---|
| `scripts/evaluate.py` created | ⬜ TODO |
| `scripts/analyze_log.py` created | ⬜ TODO |
| Run pipeline first to get `detection_log.csv` | ⬜ Needs pipeline |
| Plots saved for report | ⬜ TODO |
