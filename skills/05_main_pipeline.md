# ⚙️ SKILL 05 — Main Pipeline & Video Processing

## What's Already Done ✅
- `src/pipeline/inspection_pipeline.py` — Full image inspection pipeline (load → YOLO → crop → VLM → report)
- `demo.py` — Mock demo that works right now without any downloads

---

## The Missing Piece — Video Pipeline

The current `inspection_pipeline.py` handles **individual images**, but the project spec needs a **video processing loop**. We need to create `main.py`.

---

## Step 1 — Create `main.py` (Full Video Pipeline)

Save this as `main.py` in the project root:

```python
"""
main.py
-------
Automated Visual Quality Inspection — Video Pipeline
Uses YOLO + BLIP to process a video file frame by frame.

Run:
    conda activate ipa
    python main.py --video data/test_video.mp4 --device mps
"""

import argparse
import cv2
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, ".")
from src.detection.yolo_detector import YOLODefectDetector
from src.vlm.vlm_analyzer import VLMAnalyzer

# ─────────────────────────────────────────────
# Configuration Defaults
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "yolo_model": "yolov8n.pt",
    "confidence_threshold": 0.25,
    "vlm_every_n_frames": 15,
    "output_video": "outputs/output_annotated.mp4",
    "csv_log": "outputs/detection_log.csv",
    "device": "mps",     # Change to "cpu" if MPS causes issues
}

def parse_args():
    p = argparse.ArgumentParser(description="Visual Quality Inspection — Video Pipeline")
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--device", default=DEFAULT_CONFIG["device"], choices=["cpu", "mps", "cuda"])
    p.add_argument("--conf", type=float, default=DEFAULT_CONFIG["confidence_threshold"])
    p.add_argument("--vlm-interval", type=int, default=DEFAULT_CONFIG["vlm_every_n_frames"])
    p.add_argument("--yolo-model", default=DEFAULT_CONFIG["yolo_model"])
    p.add_argument("--vlm-backend", default="blip", choices=["blip", "mock", "claude", "gpt4v"])
    p.add_argument("--no-display", action="store_true", help="Run headless (no window)")
    return p.parse_args()


def preprocess_frame(frame):
    """Image Processing techniques for the report visualization."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    edges = cv2.Canny(equalized, 50, 150)
    return gray, blurred, equalized, edges


def run_pipeline(args):
    Path("outputs").mkdir(exist_ok=True)

    # ── Load Models ───────────────────────────────
    print("=" * 55)
    print("  Visual Quality Inspection — Video Pipeline")
    print("=" * 55)

    detector = YOLODefectDetector(
        model_path=args.yolo_model,
        confidence_threshold=args.conf,
        device=args.device,
    )

    vlm_device = args.device  # Use same device for BLIP
    analyzer = VLMAnalyzer(backend=args.vlm_backend, device=vlm_device)

    # ── Open Video ────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {args.video}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps}fps | Total frames: {total}")

    out = cv2.VideoWriter(
        DEFAULT_CONFIG["output_video"],
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    # ── Main Loop ─────────────────────────────────
    log_data = []
    frame_count = 0
    last_description = ""

    print(f"\nProcessing... (VLM runs every {args.vlm_interval} frames)")
    print("Press 'Q' in the display window to stop early.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n✅ Video processing complete.")
            break

        # Image preprocessing (for report visuals — computed but not overlaid)
        gray, blurred, equalized, edges = preprocess_frame(frame)

        # YOLO detection on every frame
        result = detector.detect(frame, image_path=f"frame_{frame_count:04d}")
        annotated = detector.annotate_image(frame, result)

        # VLM only every N frames (to keep it fast)
        if frame_count % args.vlm_interval == 0 and result.has_defects:
            crops = detector.crop_defect_regions(frame, result)
            for defect, crop in crops:
                analysis = analyzer.analyze(crop, defect.class_name)
                last_description = analysis.description
                defect.severity = analysis.severity

                log_data.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "frame": frame_count,
                    "class": defect.class_name,
                    "confidence": round(defect.confidence, 3),
                    "severity": analysis.severity,
                    "action": analysis.recommended_action,
                    "bbox": str([round(v) for v in defect.bbox.to_xyxy()]),
                    "vlm_description": analysis.description,
                })

                print(f"  Frame {frame_count:04d} | {defect.class_name} "
                      f"({defect.confidence:.2f}) | {analysis.severity} | {analysis.description[:60]}")

        # Overlay VLM text
        if last_description:
            cv2.putText(
                annotated, f"VLM: {last_description[:60]}",
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )

        # Frame counter
        cv2.putText(
            annotated, f"Frame: {frame_count}/{total}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        out.write(annotated)

        if not args.no_display:
            cv2.imshow("Quality Inspection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹ Stopped by user.")
                break

        frame_count += 1

    # ── Cleanup ───────────────────────────────────
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if log_data:
        df = pd.DataFrame(log_data)
        df.to_csv(DEFAULT_CONFIG["csv_log"], index=False)
        print(f"\n📄 Detection log: {DEFAULT_CONFIG['csv_log']} ({len(log_data)} entries)")
    else:
        print("\n⚠️  No defects logged. Try lowering --conf threshold.")

    print(f"🎬 Output video: {DEFAULT_CONFIG['output_video']}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
```

---

## Step 2 — How to Run the Pipeline

```bash
conda activate ipa

# Standard run with BLIP VLM on M4 Mac
python main.py --video data/test_video.mp4 --device mps --vlm-backend blip

# If MPS gives errors, fall back to CPU
python main.py --video data/test_video.mp4 --device cpu --vlm-backend blip

# Quick test with mock VLM (no model download needed)
python main.py --video data/test_video.mp4 --device cpu --vlm-backend mock

# Headless run (no display window — for terminal-only mode)
python main.py --video data/test_video.mp4 --no-display
```

---

## Step 3 — Run the Existing Mock Demo First

Before adding your video, verify the existing setup works:

```bash
conda activate ipa
python demo.py
```

Expected output:
```
Visual Quality Inspection — Mock Demo
[YOLO] Detected 3 defects: scratch, contamination, dent
[VLM]  Analyzing each defect region...
FINAL VERDICT: FAIL
Check outputs/reports/demo_report.json
```

---

## ✅ Pipeline Checklist

| Item | Status |
|---|---|
| `demo.py` works | ⬜ Run it first |
| `main.py` created | ⬜ Create from Step 1 |
| BLIP backend added to `vlm_analyzer.py` | ⬜ See Skill 04 |
| Test video ready | ⬜ See Skill 02 |
| Full pipeline runs end-to-end | ⬜ Final goal |
