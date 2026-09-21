"""
demo.py  —  Quick-start demonstration
--------------------------------------
Runs the full pipeline on a synthetic test image so you can verify
your setup without any API keys or real product images.

Run:
    python demo.py
"""

import cv2
import numpy as np
import json
from pathlib import Path

# ── Create a synthetic "product with defect" image ──────────────────
def create_synthetic_image(save_path: str = "data/samples/demo_product.jpg"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Simulate a grey metallic surface
    img = np.ones((480, 640, 3), dtype=np.uint8) * 180

    # Add texture noise
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    # Draw product boundary
    cv2.rectangle(img, (50, 50), (590, 430), (100, 100, 100), 3)

    # Simulate a scratch defect (diagonal line)
    cv2.line(img, (200, 150), (280, 250), (60, 60, 60), 3)
    cv2.line(img, (202, 150), (282, 250), (40, 40, 40), 1)

    # Simulate a dent/shadow
    cv2.ellipse(img, (420, 300), (40, 25), 30, 0, 360, (130, 130, 130), -1)
    cv2.ellipse(img, (415, 295), (35, 20), 30, 0, 360, (155, 155, 155), -1)

    # Simulate contamination (dark blob)
    cv2.circle(img, (350, 180), 18, (80, 70, 60), -1)

    cv2.imwrite(save_path, img)
    print(f"[Demo] Synthetic test image saved → {save_path}")
    return save_path


# ── Mock pipeline run (no YOLO download needed for this demo) ────────
def run_mock_demo():
    """
    Demonstrates the data structures and report format without
    requiring YOLO weights or API keys.
    """
    print("\n" + "=" * 60)
    print("  Visual Quality Inspection — Mock Demo")
    print("=" * 60)

    # Create a fake detection result
    from src.detection.yolo_detector import (
        DetectionResult, Defect, BoundingBox
    )
    from src.vlm.vlm_analyzer import VLMAnalyzer
    from src.pipeline.inspection_pipeline import InspectionReport

    import time

    # Simulate YOLO output
    fake_defects = [
        Defect(
            class_id=0,
            class_name="scratch",
            confidence=0.87,
            bbox=BoundingBox(195, 145, 290, 260),
        ),
        Defect(
            class_id=3,
            class_name="contamination",
            confidence=0.72,
            bbox=BoundingBox(330, 160, 375, 205),
        ),
        Defect(
            class_id=1,
            class_name="dent",
            confidence=0.65,
            bbox=BoundingBox(378, 272, 462, 328),
        ),
    ]

    detection_result = DetectionResult(
        image_path="data/samples/demo_product.jpg",
        image_shape=(480, 640, 3),
        defects=fake_defects,
        inference_time_ms=23.4,
        pass_fail="FAIL",
    )

    print(f"\n[YOLO] Detected {detection_result.defect_count} defects:")
    for d in detection_result.defects:
        print(f"  • {d.class_name:20s}  conf={d.confidence:.2f}  "
              f"bbox={[round(v) for v in d.bbox.to_xyxy()]}")

    # Simulate VLM analysis with mock backend
    print("\n[VLM]  Analyzing each defect region...")
    analyzer = VLMAnalyzer(backend="mock")

    img = create_synthetic_image()
    image = cv2.imread(img)
    from src.detection.yolo_detector import YOLODefectDetector

    # Use a lightweight mock crop approach since we don't have YOLO here
    vlm_analyses = []
    for defect in fake_defects:
        analysis = analyzer.analyze(image, defect.class_name)
        defect.severity = analysis.severity
        vlm_analyses.append(analysis)
        print(f"  • {defect.class_name:20s}  severity={analysis.severity:8s}  "
              f"action={analysis.recommended_action}")

    # Build report
    report = InspectionReport(
        image_path="data/samples/demo_product.jpg",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        detection_result=detection_result,
        vlm_analyses=vlm_analyses,
        total_time_ms=152.7,
    )

    print("\n" + "─" * 60)
    print(f"  FINAL VERDICT:  {report.pass_fail}")
    print(f"  Max Severity :  {report._max_severity()}")
    print(f"  Total Time   :  {report.total_time_ms:.0f}ms")
    print("─" * 60)

    # Save report
    report.save_json("outputs/reports/demo_report.json")
    print("\n[Demo] Full JSON report:")
    print(report.to_json())
    print("\n[Demo] Complete! Check outputs/reports/demo_report.json")


if __name__ == "__main__":
    run_mock_demo()
