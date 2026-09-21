# 🛠️ SKILL 01 — Environment Setup (M4 Mac)

## Machine Specs
- **Laptop:** Apple M4 Mac
- **RAM:** 8 GB
- **Storage:** 256 GB
- **OS:** macOS
- **Python found:** 3.12.7 via Anaconda (`/opt/anaconda3`)

> ⚠️ Project spec says Python 3.10, but 3.12.7 is already installed via Anaconda and works fine with all required packages. We'll use a dedicated conda environment to stay clean.

---

## Step A — Create a Dedicated Conda Environment

Open Terminal and run **exactly** these commands one by one:

```bash
# 1. Create a new isolated environment named "ipa" with Python 3.10
conda create -n ipa python=3.10 -y

# 2. Activate it (do this EVERY TIME you open a new terminal for this project)
conda activate ipa

# 3. Verify
python --version   # should print Python 3.10.x
```

> 💡 **Why conda over venv?** You already have Anaconda. Conda handles Apple Silicon native builds better for PyTorch/OpenCV.

---

## Step B — Install PyTorch (Apple Silicon MPS)

```bash
# Install PyTorch with MPS (Metal Performance Shaders) support for M4 Mac
pip install torch torchvision torchaudio
```

> ✅ On M4 Mac, PyTorch automatically supports MPS acceleration. This means YOLO and BLIP will use your GPU cores (Neural Engine pathway) — much faster than pure CPU!

---

## Step C — Install All Project Libraries

```bash
pip install ultralytics          # YOLOv8 — auto-downloads weights
pip install opencv-python        # Video + image processing
pip install transformers         # HuggingFace (for BLIP)
pip install Pillow               # PIL image support
pip install pandas               # CSV logging
pip install matplotlib           # Accuracy plots
pip install scikit-learn         # Precision, Recall, F1
pip install accelerate           # Speeds up HuggingFace models
pip install PyYAML               # Config file reading
pip install seaborn              # Better plots
pip install tqdm                 # Progress bars
```

One-liner alternative (paste all at once):
```bash
pip install ultralytics opencv-python transformers Pillow pandas matplotlib scikit-learn accelerate PyYAML seaborn tqdm
```

---

## Step D — Verify Everything

Save this as `check_install.py` in the project root and run `python check_install.py`:

```python
import cv2
import torch
import ultralytics
from transformers import BlipProcessor
import pandas
import sklearn
print("✅ All libraries installed successfully!")
print(f"PyTorch version : {torch.__version__}")
print(f"OpenCV version  : {cv2.__version__}")
print(f"MPS available   : {torch.backends.mps.is_available()}")  # Should be True on M4
print(f"Device to use   : {'mps' if torch.backends.mps.is_available() else 'cpu'}")
```

Expected output:
```
✅ All libraries installed successfully!
PyTorch version : 2.x.x
OpenCV version  : 4.x.x
MPS available   : True
Device to use   : mps
```

---

## Step E — Update `configs/config.yaml` After Setup

Change the `device` field from `cpu` to `mps`:
```yaml
detection:
  device: "mps"    # ← change this from "cpu"
```

This tells YOLO to use Apple Silicon GPU cores.

---

## What to Run (in order)

| # | Command | Time |
|---|---|---|
| 1 | `conda create -n ipa python=3.10 -y` | ~2 min |
| 2 | `conda activate ipa` | instant |
| 3 | `pip install torch torchvision torchaudio` | ~5 min |
| 4 | `pip install ultralytics opencv-python transformers ...` | ~5 min |
| 5 | `python check_install.py` | ~30 sec |

---

## ✅ Done When
- `check_install.py` prints all green checkmarks
- `MPS available: True` shows in output
- `conda activate ipa` works every time you open terminal
