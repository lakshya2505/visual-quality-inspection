# Skill 12: Interactive 3D Surface Roughness Viewer with Three.js

## 🎯 Overview & Purpose
Physical surface metrology (e.g. profilometers, laser interferometers) evaluates surface topography ($R_a$, $R_z$ roughness metrics). This skill details how to convert 2D photometric intensity, FFT high-frequency energy, and anomaly gradients into an **interactive 3D mesh surface** inside the browser using **Three.js** and WebGL.

---

## 🔬 Core Principles & 3D Topography Construction

### 1. Depth & Elevation Estimation from 2D Gradient
Using the photometric luminance equation and anomaly intensity:
$$Z(x, y) = s \cdot \left[ (1 - \lambda) \cdot \frac{I(x, y)}{255} + \lambda \cdot \frac{A(x, y)}{255} \right]$$
Where:
- $I(x, y)$ is the grayscale pixel intensity (base contour).
- $A(x, y)$ is the high-frequency / anomaly residual (micro-roughness spikes).
- $s$ is the user-adjustable vertical scale factor (depth exaggeration).
- $\lambda \in [0, 1]$ is the anomaly weighting parameter.

### 2. 3D WebGL Mesh Pipeline in Three.js
1. **Geometry:** `THREE.PlaneGeometry(width, height, segmentsX, segmentsY)` with $128 \times 128$ or $256 \times 256$ vertices.
2. **Displacement Texture / Vertex Height:** Load elevation map into vertex attributes or displacement map.
3. **Lighting & Shadows:** Directional grazing industrial light + Ambient light to accentuate surface micro-defects (dents, ridges, cracks).
4. **Interactive Controls:** `THREE.OrbitControls` for fluid 360° rotation, tilt, and zoom.

---

## 🛠️ Dashboard Integration Guide

### 1. Backend Heightmap API
- Returns low-resolution ($128 \times 128$ or $256 \times 256$) normalized heightmap as base64 PNG or 2D float array.

### 2. Three.js Client Canvas Component
```javascript
// Initialization
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

// Mesh creation with dynamic height displacement
const geometry = new THREE.PlaneGeometry(10, 10, 128, 128);
const material = new THREE.MeshStandardMaterial({
    roughness: 0.3,
    metalness: 0.7,
    wireframe: false,
    color: 0x38bdf8
});
const surfaceMesh = new THREE.Mesh(geometry, material);
surfaceMesh.rotation.x = -Math.PI / 3;
scene.add(surfaceMesh);

// User Controls: Elevation Exaggeration, Wireframe Mode, Lighting Angle
```

---

## 🧪 Verification & Acceptance Criteria
- [ ] 3D Canvas renders inside the web dashboard at stable $\ge 60\text{fps}$.
- [ ] Rotation, Pan, and Zoom respond smoothly via mouse or trackpad.
- [ ] Surface roughness spikes and crack indentations correspond accurately to the 2D visual defect coordinates.
