"""
inspection_pipeline.py
-----------------------
End-to-end Visual Quality Inspection Pipeline.

Flow:
    Image → YOLO Detection → Crop Defects → VLM Analysis → Report

Usage:
    pipeline = InspectionPipeline(
        yolo_model="yolov8n.pt",
        vlm_backend="mock",       # switch to "claude" or "gpt4v" in production
    )
    report = pipeline.inspect("path/to/product_image.jpg")
    print(report.summary())
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

# Relative imports (works when running from project root)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.detection.yolo_detector import YOLODefectDetector, DetectionResult, Defect
from src.vlm.vlm_analyzer import VLMAnalyzer, VLMAnalysis


# ─────────────────────────────────────────────
# Report Data Structure
# ─────────────────────────────────────────────

@dataclass
class InspectionReport:
    """Full inspection result combining YOLO detections and VLM analyses."""

    image_path: str
    timestamp: str
    detection_result: DetectionResult
    vlm_analyses: List[VLMAnalysis] = field(default_factory=list)
    total_time_ms: float = 0.0

    @property
    def pass_fail(self) -> str:
        """
        Final pass/fail:
          - PASS  → no defects
          - REWORK → medium/low defects only
          - FAIL  → any high or critical defect
        """
        if not self.detection_result.has_defects:
            return "PASS"
        max_severity = self._max_severity()
        if max_severity in ("high", "critical"):
            return "FAIL"
        return "REWORK"

    def _max_severity(self) -> str:
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3, "unknown": -1}
        defects = self.detection_result.defects
        if not defects:
            return "none"
        return max(defects, key=lambda d: order.get(d.severity, -1)).severity

    def summary(self) -> dict:
        return {
            "image": self.image_path,
            "timestamp": self.timestamp,
            "verdict": self.pass_fail,
            "defect_count": self.detection_result.defect_count,
            "max_severity": self._max_severity(),
            "total_time_ms": round(self.total_time_ms, 2),
            "defects": [
                {
                    "class": d.class_name,
                    "confidence": round(d.confidence, 3),
                    "severity": d.severity,
                    "bbox": d.bbox.to_xyxy(),
                    "vlm_analysis": a.to_dict() if a else None,
                }
                for d, a in zip(
                    self.detection_result.defects,
                    self.vlm_analyses or [None] * self.detection_result.defect_count,
                )
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.summary(), indent=indent)

    def save_json(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(self.to_json())
        print(f"[Report] Saved JSON report → {output_path}")


# ─────────────────────────────────────────────
# Inspection Pipeline
# ─────────────────────────────────────────────

class InspectionPipeline:
    """
    Chains YOLO detection and VLM analysis into a single callable.

    Parameters
    ----------
    yolo_model : str
        Path to YOLO weights (.pt). Use 'yolov8n.pt' to download pretrained.
    vlm_backend : str
        One of 'mock', 'claude', 'gpt4v', 'llava'.
    confidence_threshold : float
        YOLO minimum confidence (default 0.25).
    device : str
        Torch device ('cpu', 'cuda', 'mps').
    save_annotated : bool
        If True, saves annotated images alongside JSON reports.
    output_dir : str
        Directory to save reports and annotated images.
    vlm_kwargs : dict
        Extra keyword arguments passed to the VLM backend constructor.
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        vlm_backend: str = "mock",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
        save_annotated: bool = True,
        output_dir: str = "outputs",
        vlm_kwargs: Optional[dict] = None,
    ):
        print("=" * 55)
        print("  Initializing Visual Quality Inspection Pipeline")
        print("=" * 55)

        self.output_dir = Path(output_dir)
        self.save_annotated = save_annotated

        # 1. YOLO detector
        self.detector = YOLODefectDetector(
            model_path=yolo_model,
            confidence_threshold=confidence_threshold,
            device=device,
        )

        # 2. VLM analyzer
        self.analyzer = VLMAnalyzer(backend=vlm_backend, **(vlm_kwargs or {}))

        print("=" * 55)
        print("  Pipeline ready. Call pipeline.inspect(image_path)")
        print("=" * 55)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def inspect(self, image_path: str) -> InspectionReport:
        """
        Full pipeline for a single image.

        Returns an InspectionReport with detection + VLM analysis.
        """
        start = time.perf_counter()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        print(f"\n[Pipeline] Inspecting: {image_path}")

        # Step 1: Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        # Step 2: YOLO detection
        detection_result = self.detector.detect(image, image_path=image_path)
        print(
            f"[Pipeline] YOLO found {detection_result.defect_count} defect(s) "
            f"in {detection_result.inference_time_ms:.1f}ms"
        )

        # Step 3: Crop defect regions
        vlm_analyses: List[VLMAnalysis] = []
        if detection_result.has_defects:
            crops = self.detector.crop_defect_regions(image, detection_result)
            print(f"[Pipeline] Running VLM analysis on {len(crops)} crop(s)...")
            vlm_analyses = self.analyzer.analyze_batch(crops)

        # Step 4: Build report
        total_ms = (time.perf_counter() - start) * 1000
        report = InspectionReport(
            image_path=image_path,
            timestamp=timestamp,
            detection_result=detection_result,
            vlm_analyses=vlm_analyses,
            total_time_ms=total_ms,
        )

        # Step 5: Save outputs
        self._save_outputs(image, report)

        verdict_color = {"PASS": "\033[92m", "REWORK": "\033[93m", "FAIL": "\033[91m"}
        reset = "\033[0m"
        color = verdict_color.get(report.pass_fail, "")
        print(
            f"[Pipeline] Verdict: {color}{report.pass_fail}{reset} "
            f"| Total time: {total_ms:.0f}ms"
        )
        return report

    def inspect_directory(
        self,
        directory: str,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> List[InspectionReport]:
        """Inspect all images in a directory. Returns list of reports."""
        dir_path = Path(directory)
        image_paths = [
            str(p) for p in dir_path.iterdir()
            if p.suffix.lower() in extensions
        ]
        print(f"\n[Pipeline] Found {len(image_paths)} images in '{directory}'")
        reports = []
        for path in sorted(image_paths):
            try:
                reports.append(self.inspect(path))
            except Exception as e:
                print(f"[Pipeline] ERROR on {path}: {e}")
        self._save_batch_summary(reports, directory)
        return reports

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _save_outputs(self, image: np.ndarray, report: InspectionReport):
        stem = Path(report.image_path).stem

        # JSON report
        json_path = self.output_dir / "reports" / f"{stem}_report.json"
        report.save_json(str(json_path))

        # Annotated image
        if self.save_annotated and report.detection_result.has_defects:
            annotated = self.detector.annotate_image(
                image, report.detection_result
            )
            img_path = self.output_dir / "visualizations" / f"{stem}_annotated.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_path), annotated)
            print(f"[Pipeline] Annotated image → {img_path}")

    def _save_batch_summary(self, reports: List[InspectionReport], directory: str):
        summary = {
            "directory": directory,
            "total_inspected": len(reports),
            "pass": sum(1 for r in reports if r.pass_fail == "PASS"),
            "rework": sum(1 for r in reports if r.pass_fail == "REWORK"),
            "fail": sum(1 for r in reports if r.pass_fail == "FAIL"),
            "results": [r.summary() for r in reports],
        }
        out_path = self.output_dir / "reports" / "batch_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Pipeline] Batch summary → {out_path}")
        print(
            f"  PASS: {summary['pass']}  |  "
            f"REWORK: {summary['rework']}  |  "
            f"FAIL: {summary['fail']}"
        )
