"""
main.py
-------
Automated Visual Quality Inspection — Video Pipeline
YOLO v8 + BLIP Vision-Language Model

Processes a video file frame by frame:
  1. Extracts each frame
  2. Applies image preprocessing (grayscale, blur, equalize, edges)
  3. Runs YOLOv8 to detect defect bounding boxes
  4. Every N frames: crops defects and runs BLIP for natural-language description
  5. Saves annotated output video + CSV detection log

Run:
    conda activate ipa
    python main.py --video data/test_video.mp4 --device mps --vlm-backend blip
    python main.py --video data/test_video.mp4 --device cpu  --vlm-backend mock  # quick test
"""

import argparse
import cv2
import sys
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.detection.yolo_detector import YOLODefectDetector
from src.vlm.vlm_analyzer import VLMAnalyzer
from src.utils.frequency_analysis import extract_frequency_features, save_frequency_visualization


# ─────────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visual Quality Inspection — Video Pipeline (YOLO + BLIP)"
    )
    p.add_argument("--video",        required=True,       help="Path to input video file")
    p.add_argument("--device",       default="mps",       choices=["cpu", "mps", "cuda"],
                   help="Torch device: mps (M4 Mac), cpu, cuda")
    p.add_argument("--conf",         type=float, default=0.25,
                   help="YOLO confidence threshold (default 0.25)")
    p.add_argument("--iou",          type=float, default=0.45,
                   help="YOLO NMS IoU threshold (default 0.45)")
    p.add_argument("--yolo-model",   default="yolov8n.pt",
                   help="YOLO weights file (default yolov8n.pt)")
    p.add_argument("--vlm-backend",  default="blip",
                   choices=["blip", "mock", "claude", "gpt4v"],
                   help="VLM backend to use (default blip)")
    p.add_argument("--vlm-device",   default=None,
                   help="Device for BLIP (defaults to same as --device)")
    p.add_argument("--vlm-interval", type=int, default=15,
                   help="Run VLM every N frames (default 15). Increase to speed up.")
    p.add_argument("--output-video", default="outputs/output_annotated.mp4",
                   help="Path for annotated output video")
    p.add_argument("--output-csv",   default="outputs/detection_log.csv",
                   help="Path for CSV detection log")
    p.add_argument("--no-display",   action="store_true",
                   help="Run headless (no cv2.imshow window)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Image Preprocessing (for IPA subject requirements)
# ─────────────────────────────────────────────────────────────

def preprocess_frame(frame):
    """
    Apply classical image processing steps.
    These outputs are used in the project report to show preprocessing stages.
    Returns: (gray, blurred, equalized, edges)
    """
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    edges     = cv2.Canny(equalized, 50, 150)
    return gray, blurred, equalized, edges


def save_preprocessing_sample(frame, output_dir: Path, frame_num: int):
    """Save a 2x2 grid of preprocessing stages for the report."""
    gray, blurred, equalized, edges = preprocess_frame(frame)

    # Resize all to same size
    h, w = 300, 300
    orig_small  = cv2.resize(frame, (w, h))
    gray_small  = cv2.cvtColor(cv2.resize(gray,      (w, h)), cv2.COLOR_GRAY2BGR)
    eq_small    = cv2.cvtColor(cv2.resize(equalized, (w, h)), cv2.COLOR_GRAY2BGR)
    edge_small  = cv2.cvtColor(cv2.resize(edges,     (w, h)), cv2.COLOR_GRAY2BGR)

    # Add labels
    for img, label in [
        (orig_small, "Original"),
        (gray_small, "Grayscale"),
        (eq_small,   "Hist. Equalized"),
        (edge_small, "Canny Edges"),
    ]:
        cv2.putText(img, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    top    = cv2.hconcat([orig_small, gray_small])
    bottom = cv2.hconcat([eq_small,   edge_small])
    grid   = cv2.vconcat([top, bottom])

    out_path = output_dir / "visualizations" / f"preprocessing_frame{frame_num:04d}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"[Pipeline] Preprocessing sample saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────

def run(args):
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # ── Header ──────────────────────────────────────────────
    print("=" * 60)
    print("  🏭 Visual Quality Inspection — Video Pipeline")
    print("=" * 60)
    print(f"  Video       : {args.video}")
    print(f"  YOLO model  : {args.yolo_model}")
    print(f"  VLM backend : {args.vlm_backend}")
    print(f"  Device      : {args.device}")
    print(f"  VLM every N : {args.vlm_interval} frames")
    print("=" * 60)

    # ── Load YOLO ───────────────────────────────────────────
    detector = YOLODefectDetector(
        model_path=args.yolo_model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )

    # ── Load VLM ────────────────────────────────────────────
    vlm_device = args.vlm_device or args.device
    vlm_kwargs = {"device": vlm_device} if args.vlm_backend == "blip" else {}
    analyzer   = VLMAnalyzer(backend=args.vlm_backend, **vlm_kwargs)

    # ── Open Video ──────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {args.video}")
        sys.exit(1)

    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n  Video info: {width}×{height} @ {fps}fps | {total} frames total\n")

    # ── Output Video Writer ──────────────────────────────────
    out_path = Path(args.output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    # ── Processing Loop ──────────────────────────────────────
    log_rows            = []
    frame_count         = 0
    last_description    = ""
    last_confidence     = "High"
    last_uncertainty    = 0.0
    last_freq_info      = ""
    preprocessing_saved = False
    freq_vis_count      = 0
    pipeline_start      = time.perf_counter()

    print(f"Processing... (VLM every {args.vlm_interval} frames | press Q to stop)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n✅ All frames processed.")
            break

        # Save ONE preprocessing sample for the report (frame 50 or first defect)
        if not preprocessing_saved and frame_count == 50:
            save_preprocessing_sample(frame, output_dir, frame_count)
            preprocessing_saved = True

        # ── YOLO Detection (every frame) ─────────────────────
        det_result = detector.detect(frame, image_path=f"frame_{frame_count:04d}")
        annotated  = detector.annotate_image(frame, det_result)

        # ── VLM + Frequency + Uncertainty Analysis (every N frames) ─
        if frame_count % args.vlm_interval == 0 and det_result.has_defects:
            crops = detector.crop_defect_regions(frame, det_result)
            for defect, crop in crops:
                # 1. 2D FFT & LBP Frequency Feature Extraction
                freq_features = extract_frequency_features(crop)

                # Save 3-panel frequency diagnostic plot for first 5 defect occurrences
                if freq_vis_count < 5:
                    vis_path = output_dir / "visualizations" / f"freq_vis_frame{frame_count:04d}_{defect.class_name}.png"
                    save_frequency_visualization(crop, freq_features, str(vis_path), defect_name=defect.class_name)
                    print(f"  [FFT Spectrum] Saved diagnostic plot → {vis_path}")
                    freq_vis_count += 1

                # 2. Monte Carlo Uncertainty Quantification on VLM
                analysis = analyzer.analyze_with_uncertainty(crop, defect.class_name, n_samples=5)
                last_description = analysis.description
                last_confidence  = analysis.confidence_label
                last_uncertainty = analysis.uncertainty_score
                last_freq_info   = f"HF-Ratio: {freq_features['high_freq_ratio']:.2f} | LBP-Ent: {freq_features['lbp_entropy']:.2f}"
                defect.severity  = analysis.severity

                log_rows.append({
                    "timestamp":            datetime.now().strftime("%H:%M:%S"),
                    "frame":                frame_count,
                    "class":                defect.class_name,
                    "confidence":           round(defect.confidence, 3),
                    "severity":             analysis.severity,
                    "action":               analysis.recommended_action,
                    "bbox":                 str([round(v) for v in defect.bbox.to_xyxy()]),
                    "vlm_description":      analysis.description,
                    # ★ Tier 1 Research Additions:
                    "spectral_energy":      freq_features["spectral_energy"],
                    "high_freq_ratio":      freq_features["high_freq_ratio"],
                    "dominant_freq_radius": freq_features["dominant_freq_radius"],
                    "lbp_entropy":          freq_features["lbp_entropy"],
                    "uncertainty_score":    analysis.uncertainty_score,
                    "confidence_label":     analysis.confidence_label,
                    "flag_human_review":    analysis.flag_human_review,
                })

                review_tag = " [⚠️ HUMAN REVIEW]" if analysis.flag_human_review else ""
                print(
                    f"  Frame {frame_count:04d}/{total} | "
                    f"{defect.class_name:14s} | conf={defect.confidence:.2f} | "
                    f"sev={analysis.severity:7s} | Unc={analysis.uncertainty_score:.2f} ({analysis.confidence_label}){review_tag} | "
                    f"HF={freq_features['high_freq_ratio']:.2f} | {analysis.description[:45]}"
                )

        # ── Overlay VLM text, Uncertainty Badge & Frequency Stats ────
        if last_description:
            # Bottom background banner for readability
            cv2.rectangle(annotated, (0, height - 52), (width, height), (15, 23, 42), -1)
            cv2.line(annotated, (0, height - 52), (width, height - 52), (51, 65, 85), 1)

            # Line 1: VLM Natural Language Caption
            cv2.putText(
                annotated,
                f"VLM: {last_description[:85]}",
                (10, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (241, 245, 249), 1
            )

            # Line 2: Calibrated Confidence Badge & Frequency Metrics
            badge_color = (34, 197, 94) if "High" in last_confidence else (
                (245, 158, 11) if "Medium" in last_confidence else (239, 68, 68)
            )
            badge_text = f"Confidence: {last_confidence} (U={last_uncertainty:.2f})"
            if "Low" in last_confidence:
                badge_text += " [⚠️ REVIEW REQUIRED]"

            cv2.putText(
                annotated,
                badge_text,
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, badge_color, 1
            )

            if last_freq_info:
                cv2.putText(
                    annotated,
                    f"FFT: {last_freq_info}",
                    (width - 320, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (56, 189, 248), 1
                )

        # ── Frame counter ─────────────────────────────────────
        elapsed = time.perf_counter() - pipeline_start
        fps_live = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(
            annotated,
            f"Frame: {frame_count}/{total}  |  {fps_live:.1f} fps",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        # ── Write & display ───────────────────────────────────
        writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Quality Inspection — press Q to stop", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n⏹  Stopped by user.")
                break

        frame_count += 1

    # ── Cleanup ───────────────────────────────────────────────
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    total_time = time.perf_counter() - pipeline_start

    # ── Save CSV log ──────────────────────────────────────────
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if log_rows:
        df = pd.DataFrame(log_rows)
        df.to_csv(str(csv_path), index=False)
        print(f"\n📄 Detection log → {csv_path}  ({len(log_rows)} entries)")
        print(f"   Classes found: {df['class'].value_counts().to_dict()}")
        print(f"   Avg confidence: {df['confidence'].mean():.3f}")
    else:
        print("\n⚠️  No defects logged.")
        print("   Try lowering --conf (e.g. --conf 0.1) or check your video content.")

    print(f"\n🎬 Output video → {out_path}")
    print(f"⏱️  Total time: {total_time:.1f}s | Frames processed: {frame_count}")
    print("\n✅ Pipeline complete!")
    print("\nNext steps:")
    print("  1. Open outputs/output_annotated.mp4 to see annotated video")
    print("  2. Open outputs/detection_log.csv to see the detection log")
    print("  3. Run: python scripts/analyze_log.py  ← generates report charts")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(args)
