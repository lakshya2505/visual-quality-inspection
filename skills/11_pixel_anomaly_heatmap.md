# Skill 11: Pixel Anomaly Heatmap & Interactive Side-by-Side Difference Slider

## 🎯 Overview & Purpose
In industrial defect inspection, bounding boxes only highlight regions of interest. High-precision quality control requires **pixel-level anomaly localization** and an interactive **before-and-after comparison slider** that allows quality engineers to inspect micro-defects seamlessly against normal baseline textures.

---

## 🔬 Core Concepts & Mathematical Foundation

### 1. High-Pass Spatial Frequency Residuals
Defects on homogeneous or repetitive industrial surfaces create local high-frequency disturbances.
1. **Gaussian Difference ($\text{DoG}$):**
   $$\text{Residual}(x, y) = |I(x, y) - G_\sigma(I(x, y))|$$
2. **Morphological Gradient:**
   $$\text{Grad}(x, y) = \text{Dilation}(I) - \text{Erosion}(I)$$
3. **Normalized Anomaly Fusion:**
   $$A(x, y) = \alpha \cdot \text{DoG}(x, y) + (1 - \alpha) \cdot \text{Grad}(x, y)$$

### 2. Thermal Anomaly Map Rendering
- The continuous anomaly matrix $A(x, y) \in [0, 1]$ is normalized and color-mapped using `cv2.COLORMAP_JET` or `cv2.COLORMAP_TURBO`.
- Alpha blending with the original grayscale image provides industrial context:
  $$I_{\text{overlay}} = \beta \cdot I_{\text{orig}} + (1 - \beta) \cdot I_{\text{heatmap}}$$

---

## 🛠️ Implementation Architecture

### 1. Backend Generator (`src/utils/heatmap_generator.py`)
```python
import cv2
import numpy as np

def generate_anomaly_heatmap(image_bgr, blur_ksize=21, colormap=cv2.COLORMAP_JET):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    diff = cv2.absdiff(gray, blurred)
    
    # Contrast stretching
    diff_norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(diff_norm, colormap)
    
    # Alpha blend with original
    blended = cv2.addWeighted(image_bgr, 0.45, heatmap, 0.55, 0)
    return diff_norm, heatmap, blended
```

### 2. Frontend Side-by-Side Difference Slider (Vanilla CSS + JS)
- Container with two overlapping image layers (`position: absolute; overflow: hidden;`).
- Interactive divider handle draggable horizontally across the image.
- Synchronized zoom and pan support for micro-inspection.

---

## 🧪 Verification & Acceptance Criteria
- [ ] Anomaly heatmaps generate in $< 20\text{ms}$ per standard $1024 \times 1024$ frame.
- [ ] Slider responds smoothly at $60\text{fps}$ with zero jitter.
- [ ] Micro-scratches, pits, and texture anomalies light up with warm colors (red/yellow) while pristine background stays cool (deep blue).
