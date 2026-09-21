import cv2
import numpy as np

def generate_anomaly_heatmap(image_bgr, blur_ksize=25, colormap=cv2.COLORMAP_JET):
    """
    Computes spatial frequency residual & Difference of Gaussians (DoG)
    to highlight micro-defects, scratches, and texture anomalies as a thermal heatmap.
    Also produces a downsampled 128x128 normalized heightmap grid for 3D WebGL rendering.
    
    Returns:
        diff_norm (np.ndarray): Grayscale anomaly intensity (0-255)
        heatmap (np.ndarray): Thermal color anomaly map (BGR)
        blended (np.ndarray): Original image blended with thermal heatmap (BGR)
        heightmap_grid (list of lists): 128x128 float array in [0.0, 1.0] for 3D displacement
        ra_roughness (float): Approximate surface roughness average index
    """
    # 1. Convert to grayscale
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr.copy()
        
    h, w = gray.shape[:2]
    
    # 2. Multi-scale Gaussian Difference (DoG) & Morphological High-pass
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    diff = cv2.absdiff(gray, blurred)
    
    # Morphological gradient for crisp micro-crack boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    
    # Combine high-pass difference and morphological gradient
    combined = cv2.addWeighted(diff, 0.65, morph_grad, 0.35, 0)
    
    # Contrast normalization
    diff_norm = cv2.normalize(combined, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # 3. Apply thermal colormap (Jet / Turbo)
    heatmap = cv2.applyColorMap(diff_norm, colormap)
    
    # 4. Blend with original image
    if len(image_bgr.shape) == 2:
        orig_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    else:
        orig_bgr = image_bgr.copy()
        
    blended = cv2.addWeighted(orig_bgr, 0.40, heatmap, 0.60, 0)
    
    # 5. Generate 128x128 Heightmap Grid for 3D Topography Viewer
    # Combines normalized intensity contour with high-frequency anomaly spikes
    norm_gray = gray.astype(np.float32) / 255.0
    norm_anomaly = diff_norm.astype(np.float32) / 255.0
    
    # Elevation: base geometric shading + micro-roughness anomaly spikes
    elevation = 0.45 * norm_gray + 0.55 * norm_anomaly
    
    # Downsample to 128x128 grid for efficient WebGL mesh vertex manipulation
    grid_res = 128
    elevation_128 = cv2.resize(elevation, (grid_res, grid_res), interpolation=cv2.INTER_AREA)
    elevation_128 = cv2.normalize(elevation_128, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    
    # Calculate Ra (roughness average indicator)
    mean_val = np.mean(elevation_128)
    ra_roughness = float(np.mean(np.abs(elevation_128 - mean_val)) * 100.0)
    
    heightmap_grid = elevation_128.tolist()
    
    return diff_norm, heatmap, blended, heightmap_grid, round(ra_roughness, 2)
