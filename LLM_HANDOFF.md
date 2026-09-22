# LLM Handoff Document — Visual Quality Inspection (Tier 1 Upgrade)
> **Written by:** Antigravity (Google DeepMind)  
> **Date:** 2026-09-15  
> **For:** Any LLM continuing this project  
> **Student environment:** M4 MacBook, 8 GB RAM, macOS, Anaconda Python (`ipa` env)

---

## 0. What Is This Project?

A **5th-semester Image Processing & Analysis (IPA) Tier 1 Research Project** that builds an automated visual quality inspection system combining:

| Component | Technology | Role |
|---|---|---|
| Image Preprocessing | OpenCV (Grayscale, Blur, Hist Eq, Canny Edges, 2D FFT) | Classical IPA feature extraction & noise reduction |
| Defect Detection | **YOLOv8n** (Ultralytics) + IPA Anomaly Engine | Real-time bounding box detection & glowing defect circle overlays |
| Defect Description | **Microsoft Florence-2 VLM** (`microsoft/Florence-2-base`) | Real fine-grained natural language defect descriptions & severity rating |
| Uncertainty Calibration | Monte Carlo VLM Sampling & Token Entropy | Confidence scoring & human review flagging |
| Web Application | **Flask** + Vanilla Glassmorphic HTML5/JS | Real AI web UI (Image & Video upload, Port 8080) |
| Industrial Benchmarks | **5 MVTec AD Datasets** (427 test images) | `bottle`, `grid`, `leather`, `toothbrush`, `transistor` |

---

## 1. Current State (What Works Right Now)

| Feature | Status | Verification Command |
|---|---|---|
| conda env `ipa` | ✅ Ready | `conda activate ipa` |
| PyTorch with MPS hardware acceleration | ✅ Active | Device set to `mps` (Apple M4 GPU) |
| `yolov8n.pt` & `yolov8s.pt` weights | ✅ Cached | YOLOv8n (6.5MB) + YOLOv8s (22.5MB) |
| Microsoft Florence-2 / BLIP VLM model | ✅ Ready | Integrated with automatic local fallback |
| 5 MVTec Industrial Datasets | ✅ Present | 427 test images under `data/` |
| Video Pipeline with FFT & Uncertainty | ✅ Tested | `python main.py --video data/test_video.mp4 --device mps --vlm-backend florence2` |
| Flask Web Server | ✅ Running | `http://localhost:8080` (Supports Image & Video Upload + Uncertainty Badges) |
| 2D FFT & LBP Frequency Analysis | ✅ Integrated | `src/utils/frequency_analysis.py` → `outputs/visualizations/freq_vis_*.png` |
| Monte Carlo Uncertainty Quantification | ✅ Integrated | `src/utils/uncertainty.py` (Confidence calibration & human review flags) |
| 2x2 Empirical Ablation Study | ✅ Verified | `python scripts/ablation_study.py` → `outputs/ablation_results.csv` |
| Report Charts & Analytics | ✅ Generated | `python scripts/analyze_log.py` → `outputs/visualizations/` |
| Multi-Dataset Evaluation | ✅ Verified | `python scripts/evaluate.py --backend florence2 --device mps` |
| Pre-Demo System Diagnostic | ✅ 13/13 PASS | `python scripts/final_check.py` |

---

## 2. Directory Structure

```
visual_quality_inspection/
├── app.py                     # Flask Web Server (Port 8080, Image + Video Upload + Uncertainty UI)
├── main.py                    # CLI video processing pipeline (YOLO + FFT + BLIP Uncertainty)
├── demo.py                    # Quick demo script
├── ipa-project.md             # Core project spec
├── ipa-project-tier1.md       # Tier 1 research spec
├── LLM_HANDOFF.md             # This document
├── README.md                  # Project overview & documentation
├── yolov8n.pt                 # YOLOv8 Nano weights
├── yolov8s.pt                 # YOLOv8 Small weights
│
├── configs/
│   └── config.yaml            # Hyperparameter & device configuration
│
├── data/                      # 5 MVTec AD Datasets (427 test images)
│   ├── bottle/                # 83 test images
│   ├── grid/                  # 78 test images
│   ├── leather/               # 124 test images
│   ├── toothbrush/            # 42 test images
│   ├── transistor/            # 100 test images
│   └── test_video.mp4         # 730-frame test video (146s @ 5fps)
│
├── scripts/
│   ├── make_video.py          # Video dataset creation script
│   ├── analyze_log.py         # Visual chart generator for report (incl. FFT & Uncertainty plots)
│   ├── evaluate.py            # Multi-dataset accuracy evaluation script
│   ├── ablation_study.py      # 2x2 Empirical Ablation Study script
│   └── final_check.py         # 13/13 System readiness diagnostic
│
├── skills/                    # 10 step-by-step documentation guides
│   ├── 01_environment_setup.md
│   ├── 02_dataset_and_video.md
│   ├── 03_yolo_setup.md
│   ├── 04_blip_vlm_setup.md
│   ├── 05_main_pipeline.md
│   ├── 06_accuracy_evaluation.md
│   ├── 07_demo_preparation.md
│   ├── 08_frequency_analysis.md
│   ├── 09_uncertainty_quantification.md
│   └── 10_ablation_study.md
│
├── src/
│   ├── detection/
│   │   └── yolo_detector.py   # YOLOv8 + Circle Draw + IPA Anomaly Engine
│   ├── vlm/
│   │   └── vlm_analyzer.py    # VLM Analyzer (BLIP, Claude, GPT-4o, Mock + Uncertainty)
│   ├── utils/
│   │   ├── frequency_analysis.py # 2D FFT & LBP texture extraction + 3-panel spectrum plots
│   │   └── uncertainty.py        # Monte Carlo temperature sampling & token entropy
│   └── pipeline/
│       └── inspection_pipeline.py
│
├── templates/
│   └── index.html             # Dark-mode Web UI with Live Video Player & Confidence Badges
│
└── outputs/
    ├── detection_log.csv       # Video inspection log (with FFT & Uncertainty columns)
    ├── ablation_results.csv   # 2x2 Backbone x VLM empirical benchmark
    ├── evaluation_results.csv # 427-image benchmark evaluation log
    ├── output_annotated.mp4   # Annotated video output with confidence badges
    └── visualizations/        # High-res report charts & FFT spectrum plots
```

---

## 3. Web Server Architecture (`app.py` & `templates/index.html`)

- **Port**: `8080` (avoids macOS AirPlay port 5000 conflict).
- **Backend Model**: Defaults exclusively to **Real BLIP AI** (no mock dropdown).
- **Endpoints**:
  - `GET  /` → Serves `templates/index.html` UI.
  - `GET  /status` → Returns system ready status.
  - `POST /analyze` → Accepts image upload, returns 2x2 preprocessing base64 images, YOLO bounding boxes, defect circles, BLIP descriptions, and PASS/REWORK/FAIL verdict.
  - `POST /analyze_video` → Accepts video upload (`.mp4`, `.avi`, `.mov`), runs video pipeline, outputs annotated MP4, returns video URL and defect table.
  - `GET  /video/<filename>` → Streams annotated MP4 output video to HTML5 video player.

---

## 4. Key Tier 1 Skills Documentation

1. **[`skills/08_frequency_analysis.md`](file:///Users/lakshyajain/Desktop/5th-sem/ipa/ipa-project/visual_quality_inspection/skills/08_frequency_analysis.md)** — FFT Magnitude Spectrum & LBP texture extraction.
2. **[`skills/09_uncertainty_quantification.md`](file:///Users/lakshyajain/Desktop/5th-sem/ipa/ipa-project/visual_quality_inspection/skills/09_uncertainty_quantification.md)** — Monte Carlo VLM sampling & confidence scoring.
3. **[`skills/10_ablation_study.md`](file:///Users/lakshyajain/Desktop/5th-sem/ipa/ipa-project/visual_quality_inspection/skills/10_ablation_study.md)** — 2x2 Backbone x VLM empirical ablation benchmark.

---

## 5. Quick Command Reference

```bash
# Activate environment
conda activate ipa

# Launch Web Server (Port 8080)
python app.py

# Run CLI Video Pipeline (Real BLIP AI on Apple MPS)
python main.py --video data/test_video.mp4 --device mps --vlm-backend blip

# Run Multi-Dataset Evaluation (427 images across 5 categories)
python scripts/evaluate.py --backend blip --device mps --conf 0.25

# Generate Report Charts
python scripts/analyze_log.py

# Run System Diagnostic Check
python scripts/final_check.py
```
