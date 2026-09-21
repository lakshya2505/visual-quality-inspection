# 🧠 SKILL 04 — BLIP VLM Setup (Chosen Backend)

## Why BLIP for Your M4 Mac?
- ✅ **No API key needed** — fully local
- ✅ **~900 MB** — fits in your 256 GB storage easily
- ✅ **CPU + MPS compatible** — works on Apple Silicon
- ✅ **First download at home** — cache it before lab day

---

## The Current Code Situation

The `src/vlm/vlm_analyzer.py` currently supports: `claude`, `gpt4v`, `llava`, `mock`.

**BLIP is NOT yet implemented** in `vlm_analyzer.py` — we need to add a BLIP backend.

---

## Step 1 — Add BLIP Backend to vlm_analyzer.py

Add this class to `src/vlm/vlm_analyzer.py` (after the `MockVLMBackend` class, before `VLMAnalyzer`):

```python
# ─────────────────────────────────────────────
# BLIP Backend (Local, No API Key)
# ─────────────────────────────────────────────

class BLIPBackend(VLMBackend):
    """
    Runs Salesforce BLIP locally using HuggingFace Transformers.

    Requirements:
        pip install transformers accelerate Pillow
        First run downloads ~900MB model to ~/.cache/huggingface/hub/
        
    M4 Mac: Use device="mps" for faster inference via Metal.
    8GB RAM: device="cpu" is safe, device="mps" is faster but uses shared memory.
    """

    def __init__(
        self,
        model_id: str = "Salesforce/blip-image-captioning-base",
        device: str = "cpu",     # Use "mps" on M4 Mac for speed
    ):
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            from PIL import Image

            self.torch = torch
            self.Image = Image
            self.device = device

            print(f"[VLM] Loading BLIP model (first run downloads ~900MB)...")
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(model_id)
            self.model = self.model.to(device)
            self.model.eval()
            print(f"[VLM] BLIP model loaded on {device}.")

        except ImportError as e:
            raise ImportError(
                f"Install required packages: pip install transformers accelerate Pillow\n{e}"
            )

    def analyze_image(
        self, image: np.ndarray, defect_class: str, context: Optional[str] = None
    ) -> VLMAnalysis:
        """
        Takes a BGR numpy array (OpenCV format), returns a VLMAnalysis.
        BLIP generates free-form captions, which we wrap into our structure.
        """
        from PIL import Image as PILImage
        import torch

        # Convert BGR → RGB → PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        # Build a defect-aware prompt
        prompt = f"a quality inspection photo showing a {defect_class} defect:"

        inputs = self.processor(pil_img, prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=60)

        description = self.processor.decode(output[0], skip_special_tokens=True)

        # Map to our structured format (BLIP gives text, not JSON)
        severity = self._infer_severity(description, defect_class)
        action = "rework" if severity in ("high", "critical") else (
            "inspect_further" if severity == "medium" else "pass"
        )

        return VLMAnalysis(
            defect_class=defect_class,
            severity=severity,
            description=description,
            recommended_action=action,
            confidence_text="BLIP image captioning model output",
            raw_response=description,
        )

    def _infer_severity(self, description: str, defect_class: str) -> str:
        """Heuristic severity from BLIP description text."""
        desc_lower = description.lower()
        if any(w in desc_lower for w in ("large", "severe", "deep", "broken", "major")):
            return "high"
        if any(w in desc_lower for w in ("small", "minor", "slight", "tiny")):
            return "low"
        # Class-based defaults
        class_severity = {
            "crack": "high", "corrosion": "critical",
            "dent": "medium", "scratch": "low", "contamination": "medium",
        }
        return class_severity.get(defect_class.lower(), "medium")
```

Then update the `BACKENDS` dictionary in `VLMAnalyzer`:

```python
BACKENDS = {
    "claude": ClaudeVisionBackend,
    "gpt4v": GPT4VisionBackend,
    "llava": LLaVABackend,
    "blip": BLIPBackend,      # ← ADD THIS LINE
    "mock": MockVLMBackend,
}
```

And update `configs/config.yaml`:
```yaml
vlm:
  backend: "blip"    # ← change from "mock"
```

---

## Step 2 — Download BLIP Model (Do This at Home!)

```bash
conda activate ipa
python -c "
from transformers import BlipProcessor, BlipForConditionalGeneration
print('Downloading BLIP model (~900MB)...')
proc = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
print('✅ BLIP downloaded and cached!')
"
```

**Cache location (for USB backup before lab day):**
```
~/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base/
```

---

## Step 3 — Test BLIP Alone

Save as `scripts/test_blip.py`:

```python
"""Test BLIP works on a sample image."""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from src.vlm.vlm_analyzer import VLMAnalyzer

# Create a simple test image
test_img = np.ones((224, 224, 3), dtype="uint8") * 128

# Test with BLIP
analyzer = VLMAnalyzer(backend="blip", device="cpu")  # or device="mps"
analysis = analyzer.analyze(test_img, defect_class="scratch")
print(f"Description : {analysis.description}")
print(f"Severity    : {analysis.severity}")
print(f"Action      : {analysis.recommended_action}")
```

```bash
conda activate ipa
python scripts/test_blip.py
```

---

## Step 4 — Performance on M4 Mac

| Device | BLIP inference time | Notes |
|---|---|---|
| `cpu` | ~3-5 seconds/crop | Safe, always works |
| `mps` | ~1-2 seconds/crop | Faster, uses GPU |

For the pipeline, BLIP runs **every 15 frames** (configurable), so even 3-5 sec/crop is fine for demo.

---

## ⚠️ Lab Day Checklist

```bash
# Pack the BLIP cache for offline use:
cp -r ~/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base/ /Volumes/USB/
```

At lab, restore:
```bash
cp -r /Volumes/USB/models--Salesforce--blip-image-captioning-base/ ~/.cache/huggingface/hub/
```

---

## ✅ BLIP Checklist

| Item | Status |
|---|---|
| Add `BLIPBackend` class to `vlm_analyzer.py` | ⬜ TODO |
| Update `BACKENDS` dict in `VLMAnalyzer` | ⬜ TODO |
| Update `configs/config.yaml` to `backend: "blip"` | ⬜ TODO |
| Download BLIP model at home | ⬜ TODO |
| Test with `scripts/test_blip.py` | ⬜ TODO |
| Backup cache to USB | ⬜ TODO (before lab) |
