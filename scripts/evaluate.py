"""
evaluate.py
-----------
Multi-category evaluation script for Visual Quality Inspection System.
Runs the inspection pipeline across all available MVTec AD datasets in data/
(bottle, grid, leather, toothbrush, transistor) and calculates:
- Per-category & overall Precision, Recall, F1-Score, Accuracy
- Confusion Matrix (PASS vs FAIL)
- Defect Category Breakdown

Usage:
    conda activate ipa
    python scripts/evaluate.py [--backend blip|mock] [--device mps|cpu] [--category bottle|all]
"""

import sys
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.yolo_detector import YOLODefectDetector
from src.vlm.vlm_analyzer import VLMAnalyzer
import cv2

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Inspection Pipeline on MVTec Datasets")
    p.add_argument("--backend", default="blip", choices=["blip", "mock"], help="VLM backend to use")
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"], help="Computation device")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--category", default="all", help="Category to evaluate (e.g. bottle, grid, leather, toothbrush, transistor, or 'all')")
    return p.parse_args()

def evaluate_pipeline(args):
    data_dir = Path("data")
    if args.category.lower() == "all":
        cat_dirs = [d for d in data_dir.iterdir() if d.is_dir() and (d / "test").exists()]
    else:
        target = data_dir / args.category
        cat_dirs = [target] if target.exists() and (target / "test").exists() else []

    if not cat_dirs:
        print(f"❌ No valid dataset categories found under '{data_dir}'. Download MVTec dataset categories first.")
        return

    print("=" * 70)
    print("  🧪 Multi-Category Visual Quality Inspection — Evaluation")
    print("=" * 70)
    print(f"  Categories Found : {[d.name for d in cat_dirs]}")
    print(f"  VLM Backend      : {args.backend}")
    print(f"  Device           : {args.device}")
    print(f"  YOLO Conf        : {args.conf}")
    print("=" * 70 + "\n")

    # Load models
    print("[1/2] Loading YOLOv8 Detector...")
    detector = YOLODefectDetector(model_path="yolov8n.pt", confidence_threshold=args.conf, device=args.device)

    print(f"[2/2] Loading VLM Analyzer ({args.backend})...")
    vlm_kwargs = {"device": args.device} if args.backend == "blip" else {}
    analyzer = VLMAnalyzer(backend=args.backend, **vlm_kwargs)
    print("✅ Models loaded successfully.\n")

    y_true_all = []
    y_pred_all = []
    records = []

    start_time = time.time()
    total_images = 0

    print("Running evaluation across dataset categories...\n")

    for dataset_dir in cat_dirs:
        dataset_name = dataset_dir.name
        test_dir = dataset_dir / "test"
        sub_cats = [d for d in test_dir.iterdir() if d.is_dir()]

        print(f"\n📂 Dataset Category: [{dataset_name.upper()}] ({len(sub_cats)} sub-types)")

        for cat_dir in sorted(sub_cats):
            defect_type = cat_dir.name
            is_good = (defect_type == "good")
            ground_truth = 0 if is_good else 1  # 0 = PASS/GOOD, 1 = FAIL/DEFECT

            image_files = sorted(list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg")))
            print(f"  • {defect_type:20s}: {len(image_files):3d} images (GT: {'PASS' if is_good else 'DEFECT'})")

            for img_path in image_files:
                total_images += 1
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                det_result = detector.detect(img, image_path=img_path.name)
                predicted = 1 if det_result.has_defects else 0
                vlm_desc = ""
                max_sev = "low"

                if det_result.has_defects:
                    crops = detector.crop_defect_regions(img, det_result)
                    for defect, crop in crops:
                        analysis = analyzer.analyze(crop, defect.class_name)
                        defect.severity = analysis.severity
                        vlm_desc = analysis.description
                        max_sev = analysis.severity

                y_true_all.append(ground_truth)
                y_pred_all.append(predicted)

                records.append({
                    "dataset": dataset_name,
                    "filename": img_path.name,
                    "sub_category": defect_type,
                    "ground_truth": "PASS" if ground_truth == 0 else "DEFECT",
                    "predicted": "PASS" if predicted == 0 else "DEFECT",
                    "correct": ground_truth == predicted,
                    "defect_count": det_result.defect_count,
                    "max_severity": max_sev if det_result.has_defects else "N/A",
                    "vlm_description": vlm_desc
                })

    elapsed_time = time.time() - start_time
    df_results = pd.DataFrame(records)

    # Calculate global metrics
    acc = accuracy_score(y_true_all, y_pred_all)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='binary', zero_division=0)
    cm = confusion_matrix(y_true_all, y_pred_all)

    print("\n" + "=" * 70)
    print(" 📊 MULTI-DATASET OVERALL EVALUATION REPORT")
    print("=" * 70)
    print(f" Total Categories Evaluated  : {len(cat_dirs)}")
    print(f" Total Test Images Evaluated : {total_images}")
    print(f" Total Evaluation Time      : {elapsed_time:.2f}s ({elapsed_time/total_images*1000:.1f} ms/img)")
    print("-" * 70)
    print(f" Overall Accuracy  : {acc*100:.2f}%")
    print(f" Overall Precision : {prec*100:.2f}%")
    print(f" Overall Recall    : {rec*100:.2f}%")
    print(f" Overall F1-Score  : {f1*100:.2f}%")
    print("-" * 70)
    print(" Overall Confusion Matrix (Ground Truth vs Predicted):")
    print(f"               Pred PASS   Pred DEFECT")
    print(f"  Actual PASS    {cm[0][0]:^9d}  {cm[0][1]:^11d}")
    print(f"  Actual DEFECT  {cm[1][0]:^9d}  {cm[1][1]:^11d}")
    print("-" * 70)
    
    # Print per-dataset accuracy
    print("\n  Per-Dataset Accuracy Breakdown:")
    for ds_name, grp in df_results.groupby("dataset"):
        ds_acc = grp['correct'].mean() * 100
        print(f"    • {ds_name:12s} : {ds_acc:6.2f}% accuracy ({len(grp)} images)")
    print("=" * 70)

    # Save evaluation report to CSV
    out_csv = Path("outputs/evaluation_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_csv, index=False)
    print(f"\n📄 Saved evaluation detailed log → {out_csv}")
    print("✅ Multi-dataset evaluation complete!")

if __name__ == "__main__":
    args = parse_args()
    evaluate_pipeline(args)
