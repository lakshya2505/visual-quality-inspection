# Skill 09 — Uncertainty Quantification & Confidence Calibration

> **Tier 1 Research Component:** Monte Carlo VLM Sampling & Token Entropy  
> **Subject Alignment:** Trustworthy & Explainable AI in Manufacturing

---

## 1. Concept & Rationale

Vision-Language Models (VLMs) like BLIP can sometimes produce confident-sounding descriptions even when an image crop is blurry, ambiguous, or contains edge noise.

In an industrial deployment:
- **High-Confidence Predictions**: Handled automatically by the pipeline.
- **Low-Confidence Predictions**: Flagged for **Human-in-the-Loop Review** to avoid shipping defective products.

### Method: Monte Carlo Temperature Sampling
1. For each detected defect crop, run BLIP VLM inference $N=5$ times with slight decoding temperature variations ($\tau \in [0.7, 1.0]$).
2. Measure token-level Jaccard overlap consistency between deterministic greedy output $D_0$ and stochastic sampled outputs $D_{1..N}$.
3. Compute **Uncertainty Score** $U \in [0, 1.0]$:
   $$U = 1.0 - \frac{1}{N-1} \sum_{i=1}^{N-1} \text{JaccardOverlap}(D_0, D_i)$$
4. Assign confidence rating:
   - $U < 0.25$ $\rightarrow$ **HIGH CONFIDENCE** (Automatic Dispatch)
   - $0.25 \le U < 0.55$ $\rightarrow$ **MEDIUM CONFIDENCE** (Log Warning)
   - $U \ge 0.55$ $\rightarrow$ **LOW CONFIDENCE ⚠️** (Flagged for Human Review)

---

## 2. Implementation Code (`src/utils/uncertainty.py`)

```python
import torch
import numpy as np
import cv2
from PIL import Image

def analyze_vlm_uncertainty(crop_bgr: np.ndarray, processor, blip_model, n_samples: int = 5) -> dict:
    """
    Run Monte Carlo temperature sampling to quantify VLM output uncertainty.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "description": "unknown defect",
            "uncertainty_score": 1.0,
            "confidence_label": "Low ⚠️",
            "flag_human_review": True
        }

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    inputs = processor(pil_img, "a quality inspection photo showing:", return_tensors="pt")

    descriptions = []

    with torch.no_grad():
        for i in range(n_samples):
            if i == 0:
                # Deterministic greedy decoding
                output = blip_model.generate(**inputs, max_new_tokens=40, do_sample=False)
            else:
                # Stochastic sampling with temperature scaling
                temp = 0.7 + (i * 0.08)
                output = blip_model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.9
                )
            desc = processor.decode(output[0], skip_special_tokens=True)
            descriptions.append(desc)

    greedy_desc = descriptions[0]

    # Compute Jaccard token overlap
    def jaccard_similarity(text1, text2):
        s1 = set(text1.lower().split())
        s2 = set(text2.lower().split())
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    overlaps = [jaccard_similarity(greedy_desc, d) for d in descriptions[1:]]
    avg_overlap = np.mean(overlaps) if overlaps else 1.0
    uncertainty_score = round(float(1.0 - avg_overlap), 4)

    if uncertainty_score < 0.25:
        label = "High"
    elif uncertainty_score < 0.55:
        label = "Medium"
    else:
        label = "Low ⚠️"

    return {
        "description": greedy_desc,
        "samples": descriptions,
        "uncertainty_score": uncertainty_score,
        "confidence_label": label,
        "flag_human_review": uncertainty_score >= 0.55
    }
```

---

## 3. Report Section: Explainability & Safety

In your report:
> *"To ensure reliability in industrial production, we implement Monte Carlo uncertainty quantification on VLM outputs. By measuring token-level entropy across stochastic temperature samples, the system quantifies description variance. Detections with uncertainty score $U \ge 0.55$ are automatically routed to human quality control operators, bridging automated AI inference with human-in-the-loop safety."*
