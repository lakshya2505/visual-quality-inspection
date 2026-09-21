"""
uncertainty.py
--------------
Monte Carlo Temperature Sampling & Token Entropy for VLM Uncertainty Quantification.

Research Basis: Vision-Language Models can generate plausible-sounding defect descriptions
even when image crops are ambiguous, noisy, or borderline. Monte Carlo temperature sampling
measures description consistency across stochastic variations (tau in [0.7, 1.0]).
Detections with high uncertainty (U >= 0.55) are flagged for Human-in-the-Loop review.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image


def compute_token_jaccard(ref_text: str, hyp_text: str) -> float:
    """
    Compute unigram token Jaccard similarity between reference text and sampled hypothesis text.
    """
    s1 = set(ref_text.lower().split())
    s2 = set(hyp_text.lower().split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def quantify_vlm_uncertainty(
    crop_bgr: np.ndarray,
    processor,
    model,
    prompt: str = "a quality inspection photo showing a defect:",
    n_samples: int = 5,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Execute Monte Carlo temperature sampling on BLIP VLM to quantify generation uncertainty.

    Args:
        crop_bgr: Defect crop image (BGR numpy array)
        processor: HuggingFace BlipProcessor
        model: HuggingFace BlipForConditionalGeneration
        prompt: Conditional caption prompt
        n_samples: Total MC sampling iterations (1 greedy + (n_samples-1) stochastic)
        device: PyTorch device ("mps", "cpu", "cuda")

    Returns:
        dict containing:
            - description (str): Deterministic (greedy) description
            - samples (List[str]): List of all sampled descriptions
            - uncertainty_score (float): Value in [0.0, 1.0] (0 = deterministic/certain, 1 = chaotic)
            - confidence_label (str): "High", "Medium", or "Low ⚠️"
            - flag_human_review (bool): True if uncertainty >= 0.55
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "description": "Unknown defect (empty crop)",
            "samples": [],
            "uncertainty_score": 1.0,
            "confidence_label": "Low ⚠️",
            "flag_human_review": True
        }

    import torch

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    inputs = processor(pil_img, prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    descriptions: List[str] = []

    with torch.no_grad():
        for i in range(n_samples):
            if i == 0:
                # Deterministic greedy decoding
                output_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
            else:
                # Stochastic sampling with temperature scaling
                temp = 0.7 + (i * 0.08)  # 0.78, 0.86, 0.94, 1.02
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.9
                )
            desc = processor.decode(output_ids[0], skip_special_tokens=True)
            descriptions.append(desc.strip())

    greedy_desc = descriptions[0] if descriptions else "Quality inspection analysis"

    # Compute pairwise consistency between greedy baseline and stochastic samples
    overlaps = [compute_token_jaccard(greedy_desc, sample) for sample in descriptions[1:]]
    avg_overlap = float(np.mean(overlaps)) if overlaps else 1.0
    uncertainty_score = round(float(np.clip(1.0 - avg_overlap, 0.0, 1.0)), 4)

    # Calibrate confidence levels
    if uncertainty_score < 0.25:
        confidence_label = "High"
    elif uncertainty_score < 0.55:
        confidence_label = "Medium"
    else:
        confidence_label = "Low ⚠️"

    flag_human_review = bool(uncertainty_score >= 0.55)

    return {
        "description": greedy_desc,
        "samples": descriptions,
        "uncertainty_score": uncertainty_score,
        "confidence_label": confidence_label,
        "flag_human_review": flag_human_review
    }


def quantify_mock_uncertainty(defect_class: str) -> Dict[str, Any]:
    """Fallback uncertainty generator for Mock VLM backend."""
    scores = {
        "crack": 0.12,
        "scratch": 0.18,
        "dent": 0.22,
        "contamination": 0.38,
        "corrosion": 0.62,
    }
    score = scores.get(defect_class.lower(), 0.30)
    label = "High" if score < 0.25 else ("Medium" if score < 0.55 else "Low ⚠️")
    return {
        "description": f"[MOCK] Detected {defect_class} pattern.",
        "samples": [f"[MOCK] Sample {i} for {defect_class}" for i in range(5)],
        "uncertainty_score": score,
        "confidence_label": label,
        "flag_human_review": score >= 0.55
    }
