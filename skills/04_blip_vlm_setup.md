# 🧠 SKILL 04 — Vision-Language Model Setup (Microsoft Florence-2 & BLIP)

## Why Microsoft Florence-2 for Visual Quality Inspection?
- ✅ **Fine-Grained Industrial Perception** — Purpose-built 2024 vision foundation model from Microsoft Research.
- ✅ **Eliminates Web-Caption Hallucinations** — Unlike generic web-captioners, Florence-2 excels at dense visual grounding and `<MORE_DETAILED_CAPTION>` parsing.
- ✅ **100% Free of Cost** — Zero API keys, zero subscription costs.
- ✅ **Runs Locally** — Operates on Apple Silicon MPS hardware acceleration and CPU.
- ✅ **Multi-Backend Fallback Resilience** — Automated graceful fallback to local cached BLIP model (`Salesforce/blip-image-captioning-base`).

---

## Supported Local VLM Backends

`src/vlm/vlm_analyzer.py` supports:
1. `florence2` → Microsoft Florence-2 (`microsoft/Florence-2-base`) — **Primary Recommended**
2. `blip`      → Salesforce BLIP (`Salesforce/blip-image-captioning-base`) — **Local Cached Fallback**
3. `claude`    → Anthropic Claude Vision
4. `gpt4v`     → OpenAI GPT-4o Vision
5. `mock`      → Deterministic Offline Mock for unit testing

---

## 1. Florence-2 Backend Implementation

```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(device)
model.eval()

# Defect analysis prompt
task_prompt = "<MORE_DETAILED_CAPTION>"
inputs = processor(text=task_prompt, images=pil_img, return_tensors="pt").to(device)

with torch.no_grad():
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=128,
        num_beams=3
    )

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed = processor.post_process_generation(generated_text, task=task_prompt, image_size=(pil_img.width, pil_img.height))
description = parsed["<MORE_DETAILED_CAPTION>"]
```

---

## 2. Using VLMAnalyzer in Python

```python
from src.vlm.vlm_analyzer import VLMAnalyzer

# Initialize Florence-2 analyzer
analyzer = VLMAnalyzer(backend="florence2", model_id="microsoft/Florence-2-base")

# Analyze cropped defect image
analysis = analyzer.analyze_with_uncertainty(crop_bgr, defect_class="scratch", n_samples=3)
print(f"Description : {analysis.description}")
print(f"Severity    : {analysis.severity}")
print(f"Action      : {analysis.recommended_action}")
print(f"Uncertainty : {analysis.uncertainty_score} ({analysis.confidence_label})")
```

---

## 3. Performance & Resource Footprint

| Model | Parameters | Weights Size | Device Latency (MPS) | Primary Advantage |
|---|---|---|---|---|
| **Florence-2 Base** | 230M | ~900 MB | ~0.8–1.5s / crop | Fine-grained geometric & defect grounding |
| **BLIP Base** | 224M | ~900 MB | ~1.0–2.0s / crop | Cached offline baseline |

---

## ✅ VLM Checklist

| Item | Status |
|---|---|
| Implement `Florence2Backend` in `vlm_analyzer.py` | ✅ DONE |
| Register `florence2` in `VLMAnalyzer.BACKENDS` | ✅ DONE |
| Configure `app.py` with Florence-2 default | ✅ DONE |
| Monte Carlo Uncertainty Calibration | ✅ DONE |
| Graceful Offline Local Fallback | ✅ DONE |

