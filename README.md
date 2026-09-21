# 🏭 Automated Visual Quality Inspection System (Tier 1 Research Upgrade)
### Hybrid YOLOv8 + Salesforce BLIP AI + 2D FFT Frequency Analysis + Uncertainty Quantification

An industrial-grade, research-level **Image Processing & Analysis (IPA)** visual inspection pipeline combining:
1. **Classical Image Processing (IPA)**: Grayscale, Gaussian Blur, Histogram Equalization, Canny Edge Detection & 2D FFT Frequency Spectrum Analysis.
2. **YOLOv8 Real-Time Object Detection**: Hardware-accelerated bounding box detection and glowing red defect circle overlays.
3. **Salesforce BLIP Vision-Language Model**: Real local natural language defect descriptions, severity classification (`low`, `medium`, `high`), and recommended actions.
4. **Uncertainty Quantification**: Monte Carlo VLM sampling for confidence calibration and human-in-the-loop review.
5. **Multi-Dataset Benchmarking**: Evaluated across 5 MVTec AD industrial datasets (**427 test images** across `bottle`, `grid`, `leather`, `toothbrush`, `transistor`).

---

## 📐 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   INDUSTRIAL INPUT LAYER                         │
│         Single Image (.png/.jpg)  OR  Conveyor Belt Video (.mp4) │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│             TIER 1: CLASSICAL IPA PREPROCESSING                  │
│   • Grayscale Conversion (RGB → GRAY)                            │
│   • Gaussian Blur (5x5 kernel, σ=1) — noise suppression          │
│   • Histogram Equalization — illumination correction             │
│   • Canny Edge Detection (50, 150) — structural boundary extraction│
│   ★ 2D Fast Fourier Transform (FFT) — Frequency Spectrum Analysis│
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│             TIER 2: YOLOV8 + DEFECT CIRCLE GENERATOR             │
│   • YOLOv8 Inference on Apple Silicon MPS hardware acceleration   │
│   • Bounding Box Extraction & NMS IoU Filtering                  │
│   • Minimum Enclosing Defect Circle (cx, cy, radius) Calculation │
│   • Glowing Target Circle & Contour Overlay Drawing              │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│             TIER 3: SALESFORCE BLIP VLM (REAL AI)                │
│   • Local Vision-Language Model (`blip-image-captioning-base`)   │
│   • Defect Crop + Prompt → Natural Language Description          │
│   • Keyword Severity Classification: LOW / MEDIUM / HIGH         │
│   ★ Monte Carlo Uncertainty Sampling → Confidence Calibration    │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                      OUTPUT & ANALYTICS                          │
│   • Interactive Dark-Mode Web App (Image + Video Upload, Port 8080)│
│   • Annotated Output Video (`outputs/output_annotated.mp4`)      │
│   • CSV Audit Log (`outputs/detection_log.csv`)                  │
│   • Report Visualizations (`class_distribution`, `defect_timeline`)│
└──────────────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Tier 1 Research Upgrades

### 1. 🌐 2D Fast Fourier Transform (FFT) Frequency Analysis
- Decomposes defect regions into **spatial and frequency domains**.
- High-frequency edge energy ratio determines defect sharpness (cracks vs. surface stains).
- Computes spectral energy, high-frequency energy ratio, and LBP texture entropy.

### 2. 🎯 Monte Carlo Uncertainty Quantification
- Runs VLM sampling across $N=5$ temperature variations.
- Measures token-level entropy and overlap consistency.
- Automatically flags ambiguous predictions as `"Low Confidence ⚠️ — Human Review Required"`.

### 3. 📊 2x2 Empirical Ablation Study
- Benchmarks detection speed vs. precision across multiple model configurations (YOLOv8n vs. YOLOv8s; BLIP vs. ViLT).
- Auto-generates empirical benchmark tables for academic reports.

---

## 📂 Multi-Dataset Benchmark (427 Test Images)

The system is evaluated on 5 MVTec AD industrial datasets:
- 🍼 **`bottle`** (83 images: `broken_large`, `broken_small`, `contamination`, `good`)
- 🕸️ **`grid`** (78 images: `bent`, `broken`, `glue`, `metal_contamination`, `thread`, `good`)
- 🧥 **`leather`** (124 images: `color`, `cut`, `fold`, `glue`, `poke`, `good`)
- 🪥 **`toothbrush`** (42 images: `defective`, `good`)
- ⚡ **`transistor`** (100 images: `bent_lead`, `cut_lead`, `damaged_case`, `misplaced`, `good`)

---

## ⚡ Quick Start Guide

### 1. Environment Setup
```bash
conda activate ipa
pip install -r requirements.txt
```

### 2. Launch Interactive Web Server (Image + Video Upload)
```bash
python app.py
# → Open http://localhost:8080 in your web browser
```

### 3. Run CLI Video Inspection Pipeline
```bash
python main.py --video data/test_video.mp4 --device mps --vlm-backend blip
```

### 4. Run Multi-Dataset Benchmark Evaluation
```bash
python scripts/evaluate.py --backend blip --device mps --conf 0.25
```

### 5. Generate Report Visualizations & Plots
```bash
python scripts/analyze_log.py
# → Output saved to outputs/visualizations/
```

### 6. Run Pre-Demo System Readiness Diagnostic
```bash
python scripts/final_check.py
```

---

## 📁 Repository Structure

```
visual_quality_inspection/
├── app.py                     # Flask web server (Image + Video Upload, Port 8080)
├── main.py                    # CLI video processing pipeline
├── demo.py                    # Mock demo script
├── ipa-project.md             # Core project specification
├── ipa-project-tier1.md       # Tier 1 Research specification
├── LLM_HANDOFF.md             # LLM handoff documentation
│
├── configs/
│   └── config.yaml            # Device and model configurations
│
├── data/                      # 5 MVTec Industrial Datasets
│   ├── bottle/
│   ├── grid/
│   ├── leather/
│   ├── toothbrush/
│   ├── transistor/
│   └── test_video.mp4         # Demo video (730 frames)
│
├── scripts/
│   ├── make_video.py          # Video dataset creation utility
│   ├── analyze_log.py         # Log analysis & visualization chart generator
│   ├── evaluate.py            # Multi-dataset accuracy evaluation script
│   └── final_check.py         # 10/10 Pre-demo health check diagnostic
│
├── skills/                    # Step-by-step documentation guides
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
│   │   └── yolo_detector.py   # YOLOv8 + Circle Draw + IPA Anomaly Detector
│   ├── vlm/
│   │   └── vlm_analyzer.py    # VLM Analyzer (BLIP, Claude, GPT-4o, Mock)
│   └── pipeline/
│       └── inspection_pipeline.py
│
├── templates/
│   └── index.html             # Glassmorphism Web Interface
│
├── outputs/
│   ├── detection_log.csv       # Video pipeline detection log
│   ├── evaluation_results.csv # 427-image benchmark evaluation log
│   ├── output_annotated.mp4   # Output annotated video stream
│   └── visualizations/        # High-res report charts
│
└── yolov8n.pt                 # YOLOv8 nano model weights (6.5 MB)
```

---

## 📊 Citation & Academic Alignment
This project aligns with research papers in **frequency-domain feature extraction**, **vision-language models in industrial inspection**, and **uncertainty quantification in automated quality control**.
