# Skill 08 — Spatial & Frequency Domain Feature Extraction (2D FFT & LBP)

> **Tier 1 Research Component:** Frequency-Domain Analysis for Industrial Quality Control  
> **Subject Alignment:** Image Processing & Analysis (IPA) — Fourier Transform & Texture Descriptors

---

## 1. Concept & Rationale

In industrial visual inspection, surface defects possess unique **frequency signatures**:
- **High-Frequency Defects** (Cracks, sharp scratches, cut edges): Produce energy at the outer periphery of the 2D FFT magnitude spectrum.
- **Low-Frequency Defects** (Contamination blobs, stains, color discolouration): Produce energy concentrated near the center DC component.

By applying **2D Fast Fourier Transform (FFT)** and **Local Binary Patterns (LBP)** to the cropped defect Region of Interest (ROI), we extract quantitative features beyond simple bounding boxes:
1. **Total Spectral Energy**: $\sum |F(u, v)|^2$
2. **High-Frequency Energy Ratio**: Fraction of energy outside the central low-frequency radius.
3. **Dominant Frequency Radius**: Peak magnitude distance from the DC spectrum center.
4. **LBP Texture Entropy**: Micro-texture variation descriptor.

---

## 2. Implementation Script (`src/utils/frequency_analysis.py`)

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def extract_frequency_features(crop_bgr: np.ndarray) -> dict:
    """
    Extract spatial and 2D FFT frequency domain features from a defect crop.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "spectral_energy": 0.0,
            "high_freq_ratio": 0.0,
            "dominant_freq_radius": 0,
            "lbp_entropy": 0.0
        }

    # Normalize size for fair frequency comparison
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    # 1. 2D Fast Fourier Transform
    fft = np.fft.fft2(gray.astype(np.float32))
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    log_magnitude = 20 * np.log10(magnitude + 1.0)

    # 2. Total Spectral Energy
    spectral_energy = np.sum(magnitude ** 2)

    # 3. Frequency Ring Masking (Low vs High Frequency Ratio)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    radius = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_r = min(cx, cy)

    inner_mask = radius < (max_r * 0.3)
    outer_mask = radius >= (max_r * 0.3)

    low_freq_energy  = np.sum(magnitude[inner_mask] ** 2)
    high_freq_energy = np.sum(magnitude[outer_mask] ** 2)
    high_freq_ratio  = high_freq_energy / (spectral_energy + 1e-10)

    # 4. Dominant Frequency Radius
    radial_profile = []
    for r in range(1, max_r):
        ring_mask = (radius >= r) & (radius < r + 1)
        if np.any(ring_mask):
            radial_profile.append(np.mean(magnitude[ring_mask]))
        else:
            radial_profile.append(0.0)
    
    dominant_freq_radius = int(np.argmax(radial_profile) + 1)

    # 5. Local Binary Pattern (LBP) Texture Entropy
    lbp = compute_lbp(gray)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-10)
    lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))

    return {
        "spectral_energy": round(float(spectral_energy), 2),
        "high_freq_ratio": round(float(high_freq_ratio), 4),
        "dominant_freq_radius": dominant_freq_radius,
        "lbp_entropy": round(float(lbp_entropy), 4),
        "log_magnitude": log_magnitude
    }

def compute_lbp(gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = gray[i, j]
            code = 0
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                ni = i - int(round(radius * np.sin(angle)))
                nj = j + int(round(radius * np.cos(angle)))
                if 0 <= ni < h and 0 <= nj < w:
                    code |= (gray[ni, nj] >= center) << k
            lbp[i, j] = code
    return lbp

def save_fft_visualization(crop_bgr: np.ndarray, features: dict, out_path: str):
    """
    Generate 3-panel report plot: Defect Crop, FFT Magnitude Spectrum, Radial Frequency Profile.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    log_mag = features.get("log_magnitude", np.zeros((128, 128)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Defect ROI (Spatial)")
    axes[0].axis("off")

    axes[1].imshow(log_mag, cmap="hot")
    axes[1].set_title(f"FFT Spectrum\nHF Ratio: {features['high_freq_ratio']:.3f}")
    axes[1].axis("off")

    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    radius = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_r = min(cx, cy)
    profile = [np.mean(log_mag[radius.astype(int) == r]) for r in range(1, max_r)]

    axes[2].plot(profile, color="cyan")
    axes[2].axvline(features["dominant_freq_radius"], color="red", linestyle="--", label="Dominant Freq")
    axes[2].set_title("Radial Frequency Profile")
    axes[2].set_xlabel("Frequency Radius")
    axes[2].set_ylabel("Magnitude")
    axes[2].legend()

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
```

---

## 3. Integration into Report

In your IPA project report:
> *"To characterize defect types beyond bounding box spatial coordinates, we compute 2D Fast Fourier Transform (FFT) feature representations on cropped ROIs. Sharp structural defects (scratches, cracks) exhibit high-frequency energy at the periphery of the log magnitude spectrum (`high_freq_ratio > 0.65`), whereas smooth contamination blobs show energy concentrated near the spectrum center."*
