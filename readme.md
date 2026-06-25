# MRI Compression Environment Setup

## Create Environment

```bash
conda create -n mri_codec python=3.10 -y
conda activate mri_codec
```

---

## Core Dependencies

```bash
pip install numpy pillow glymur imageio imageio[ffmpeg] tqdm

conda install -c conda-forge ffmpeg -y
conda install -c conda-forge openjpeg -y
```

### Verify Installation

```bash
python -c "import glymur; print(glymur.version.openjpeg_version)"

ffmpeg -version
```

---

## PyTorch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install torchmetrics
```

---

## MRI & Evaluation Libraries

```bash
pip install ismrmrd matplotlib pandas h5py
pip install "torchmetrics[image]"
```

---

## LPIPS Check

### Option 1

```bash
python -c "
import lpips
lpips.LPIPS(net='vgg')
"
```

### Option 2

```bash
python -c "
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
LearnedPerceptualImagePatchSimilarity(net_type='vgg')
"
```

### Locate Downloaded VGG Weights

```bash
find ~/.cache -type f | grep -i vgg
```

---

# 2D SPIHT Installation

## Clone Repository

```bash
git clone https://github.com/theAdamColton/spiht.git
cd spiht
```

## Create Environment

```bash
conda create -n spiht python=3.10 -y
conda activate spiht
```

## Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Install Rust

```bash
curl https://sh.rustup.rs -sSf | sh
source ~/.cargo/env
```

## Build SPIHT Extension

```bash
pip install maturin networkx

maturin develop --release
```

## Verify Rust Installation

```bash
rustc --version
cargo --version
```

## Run Demo

```bash
python demonstrate.py
```

### Important Fix

Inside `demonstrate.py`, change:

```python
color_space='IPT'
```

---

## Additional Packages (Optional)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install matplotlib ismrmrd pandas torchmetrics
```

---

# H.266 / VVC Support

The default Conda FFmpeg build does **not** include VVC (`libvvenc`).

Download a static FFmpeg build with H.266 support:

```bash
# Navigate to home directory
cd ~

# Download latest Linux GPL build
wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz

# Extract
tar -xf ffmpeg-master-latest-linux64-gpl.tar.xz
```

---

## Verify H.266 Support

```bash
cd ~/ffmpeg-master-latest-linux64-gpl

./bin/ffmpeg -encoders | grep -i vvenc
```

Expected output:

```text
libvvenc H.266 / VVC (codec vvc)
```

---

## FFmpeg Path

Use the static FFmpeg binary in Python:

```python
FFMPEG = "/home/subhankar/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"
```

instead of:

```python
"ffmpeg"
```

for H.266 experiments.

---

## Check Available VVC Encoder Options

```bash
~/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg \
-h encoder=libvvenc
```

Useful options:

* `preset`: faster, fast, medium, slow, slower
* `qp`
* `tier`
* `level`

Default preset:

```text
medium
```
