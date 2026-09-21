"""
frequency_analysis.py
---------------------
Spatial & 2D FFT Frequency Domain Feature Extraction for Defect ROIs.

Subject Alignment: Image Processing & Analysis (IPA) — Fourier Transform & Texture Descriptors.
Research Basis: Defects such as sharp cracks/scratches exhibit high-frequency energy concentrated
at the periphery of the 2D FFT spectrum, whereas smooth contamination and stains produce low-frequency
clusters near the DC spectrum center.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any


def compute_lbp(gray_img: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """
    Compute Local Binary Pattern (LBP) texture descriptor without external heavy dependencies.
    """
    h, w = gray_img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = gray_img[i, j]
            code = 0
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                ni = i - int(round(radius * np.sin(angle)))
                nj = j + int(round(radius * np.cos(angle)))
                if 0 <= ni < h and 0 <= nj < w:
                    code |= (gray_img[ni, nj] >= center) << k
            lbp[i, j] = code
    return lbp


def extract_frequency_features(crop_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Compute spatial and 2D FFT frequency domain features from a defect crop.
    
    Returns:
        dict containing:
            - spectral_energy (float): Total energy in magnitude spectrum
            - high_freq_ratio (float): Ratio of high-frequency energy to total spectral energy
            - dominant_freq_radius (int): Peak radial frequency distance from DC center
            - lbp_entropy (float): Shannon entropy of Local Binary Pattern texture
            - log_magnitude (np.ndarray): 2D log magnitude spectrum for visualization
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "spectral_energy": 0.0,
            "high_freq_ratio": 0.0,
            "dominant_freq_radius": 0,
            "lbp_entropy": 0.0,
            "log_magnitude": np.zeros((128, 128), dtype=np.float32)
        }

    # Normalize dimensions for fair frequency comparison
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

    # ── SPATIAL DOMAIN: Local Binary Pattern (LBP) Texture ───────
    lbp = compute_lbp(gray)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-10)
    lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))

    # ── FREQUENCY DOMAIN: 2D Fast Fourier Transform ───────────────
    fft = np.fft.fft2(gray.astype(np.float32))
    fft_shifted = np.fft.fftshift(fft)  # Center DC component
    magnitude = np.abs(fft_shifted)
    log_magnitude = 20 * np.log10(magnitude + 1.0)  # Log scale for dynamic range

    # Total spectral energy
    spectral_energy = np.sum(magnitude ** 2)

    # Frequency Ring Masks (Low vs High frequency separation)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    radius = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r = min(cx, cy)

    inner_mask = radius < (max_r * 0.3)
    outer_mask = radius >= (max_r * 0.3)

    low_freq_energy = np.sum(magnitude[inner_mask] ** 2)
    high_freq_energy = np.sum(magnitude[outer_mask] ** 2)
    high_freq_ratio = high_freq_energy / (spectral_energy + 1e-10)

    # Radial profile and dominant frequency radius
    radial_profile = []
    for r in range(1, max_r):
        ring_mask = (radius >= r) & (radius < r + 1)
        ring_values = magnitude[ring_mask]
        if len(ring_values) > 0:
            radial_profile.append(np.mean(ring_values))
        else:
            radial_profile.append(0.0)

    radial_profile = np.array(radial_profile)
    dominant_freq_radius = int(np.argmax(radial_profile) + 1) if len(radial_profile) > 0 else 0

    return {
        "spectral_energy": round(float(spectral_energy), 2),
        "high_freq_ratio": round(float(high_freq_ratio), 4),
        "dominant_freq_radius": dominant_freq_radius,
        "lbp_entropy": round(float(lbp_entropy), 4),
        "log_magnitude": log_magnitude
    }


def save_frequency_visualization(crop_bgr: np.ndarray, features: Dict[str, Any], out_path: str, defect_name: str = ""):
    """
    Save 3-panel diagnostic visualization:
      [1] Spatial Defect Crop
      [2] 2D FFT Magnitude Spectrum
      [3] Radial Frequency Profile with Dominant Peak
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    log_mag = features.get("log_magnitude", np.zeros((128, 128)))

    # Compute radial mean profile of log magnitude for clean curve
    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    radius = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cx, cy)
    profile = [
        np.mean(log_mag[radius == r])
        for r in range(1, max_r)
        if np.any(radius == r)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#0e1726")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Panel 1: Defect ROI (Spatial)
    axes[0].imshow(gray, cmap="gray")
    title1 = f"Defect ROI (Spatial)\n{defect_name.capitalize()}" if defect_name else "Defect ROI (Spatial)"
    axes[0].set_title(title1, color="#38bdf8", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    # Panel 2: 2D FFT Log Magnitude Spectrum
    im2 = axes[1].imshow(log_mag, cmap="inferno")
    axes[1].set_title(
        f"2D FFT Spectrum\nHF Ratio: {features['high_freq_ratio']:.3f} | Energy: {features['spectral_energy']:.1e}",
        color="#38bdf8", fontsize=10, fontweight="bold", pad=8
    )
    axes[1].axis("off")

    # Panel 3: Radial Frequency Profile
    axes[2].plot(profile, color="#38bdf8", lw=2, label="Radial Mean")
    axes[2].axvline(
        features["dominant_freq_radius"],
        color="#ef4444", linestyle="--", lw=1.8,
        label=f"Peak Radius: {features['dominant_freq_radius']}px"
    )
    axes[2].set_title("Radial Frequency Profile", color="#38bdf8", fontsize=11, fontweight="bold", pad=8)
    axes[2].set_xlabel("Frequency (Distance from DC Center)", color="#94a3b8", fontsize=9)
    axes[2].set_ylabel("Mean Log Magnitude", color="#94a3b8", fontsize=9)
    axes[2].legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=8)
    axes[2].grid(True, linestyle=":", alpha=0.3, color="#64748b")

    plt.tight_layout()
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_file), dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def generate_frequency_b64(crop_bgr: np.ndarray, features: Dict[str, Any], defect_name: str = "") -> str:
    """
    Generate 3-panel 2D FFT diagnostic visualization directly to base64 string.
    """
    import io, base64

    if crop_bgr is None or crop_bgr.size == 0:
        return ""

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    log_mag = features.get("log_magnitude", np.zeros((128, 128)))

    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    radius = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cx, cy)
    profile = [
        np.mean(log_mag[radius == r])
        for r in range(1, max_r)
        if np.any(radius == r)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), facecolor="#060c14")
    for ax in axes:
        ax.set_facecolor("#0a1524")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#1e3a5f")

    # Panel 1: Defect ROI (Spatial)
    axes[0].imshow(gray, cmap="gray")
    label_txt = defect_name.capitalize() if defect_name else "Region of Interest"
    axes[0].set_title(f"Spatial ROI\n{label_txt}", color="#00d4ff", fontsize=10, fontweight="bold", pad=6)
    axes[0].axis("off")

    # Panel 2: 2D FFT Log Magnitude Spectrum
    axes[1].imshow(log_mag, cmap="inferno")
    hf_ratio = features.get("high_freq_ratio", 0.0)
    energy = features.get("spectral_energy", 0.0)
    axes[1].set_title(
        f"2D FFT Spectrum |F(u,v)|\nHF Ratio: {hf_ratio:.3f} | E: {energy:.1e}",
        color="#00d4ff", fontsize=9.5, fontweight="bold", pad=6
    )
    axes[1].axis("off")

    # Panel 3: Radial Frequency Profile
    axes[2].plot(profile, color="#00d4ff", lw=2, label="Radial Mean E(r)")
    dom_rad = features.get("dominant_freq_radius", 1)
    axes[2].axvline(
        dom_rad,
        color="#ff4466", linestyle="--", lw=1.6,
        label=f"Peak: r={dom_rad}px"
    )
    lbp_h = features.get("lbp_entropy", 0.0)
    axes[2].set_title(f"Radial Profile (LBP H={lbp_h:.2f})", color="#00d4ff", fontsize=10, fontweight="bold", pad=6)
    axes[2].set_xlabel("Frequency (Radius r from DC)", color="#64748b", fontsize=8)
    axes[2].set_ylabel("Mean Log Magnitude", color="#64748b", fontsize=8)
    axes[2].legend(facecolor="#0a1524", edgecolor="#1e3a5f", labelcolor="white", fontsize=7.5, loc="upper right")
    axes[2].grid(True, linestyle=":", alpha=0.35, color="#1e293b")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
