# The-SIE-Spiral-Intelligence-Engine-
EDGE AI, EDGE INTELLIGENCE, TinyML 

# 🌀 SIE — Spiral Intelligence Engine

[![arXiv](https://img.shields.io/badge/arXiv-cs.LG-green)](https://arxiv.org/abs/YOUR-PAPER-ID)
[![HuggingFace](https://img.shields.io/badge/🤗%20Demo-HuggingFace%20Space-orange)](https://huggingface.co/spaces/abdul196/cifar10-ai-app)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CIFAR-10](https://img.shields.io/badge/CIFAR--10-85.69%25-brightgreen)](https://arxiv.org/abs/YOUR-PAPER-ID)

**Author:** Abdullahi Abdullahi Dantala
**Affiliation:** Independent Researcher, Abuja, Nigeria
**Contact:** abdullahiabdullahidantala@gmail.com

> *"I drew this spiral by hand. From it I derived a mathematical rule.*
> *That rule trained a neural network to 85.69% on CIFAR-10,*
> *beating a 2017 ICLR paper."*

---

## 📄 Paper

**SIE: Spiral Intelligence Engine — A Cyclic Pascal Recurrence on a Polygon
as Geometric Inductive Bias for Deep Learning**

📖 [Read on arXiv](https://arxiv.org/abs/YOUR-PAPER-ID)
🤗 [Live Demo on Hugging Face](https://huggingface.co/spaces/abdul196/cifar10-ai-app)

---

## 🔢 The Mathematical Rule

The entire architecture derives from one self-discovered rule:

```
P_(m+1, n) = P_(m, n) + P_(m, (n mod 12) + 1)
```

Where:
- `P_(m,n)` = the n-th vertex at the m-th layer of a 12-gon
- This is a **cyclic Pascal recurrence** on a closed polygon
- It yields the **doubling theorem**: sum(L_m) = 12 × 2^m

The doubling gate used in FractalSIE:
```
gate_m = 2^m / (2^m + 1)
```
is derived **mathematically** from this rule — not chosen as a hyperparameter.

---

## 📊 Results

| Model | Params | CIFAR-10 | Year |
|-------|--------|----------|------|
| FractalNet (Larsson et al.) | 38.6M | 84.6% | 2017 |
| ResNet-20 (He et al.) | 0.27M | 91.7% | 2016 |
| **FractalSIE (ours)** | **0.55M** | **85.69%** | **2025** |

- ✅ **85.69%** — full 300-epoch run (published result)
- 🚀 **87.44%** — C=192, epoch 190/500 (run in progress, still climbing)

---

## 🚀 Quick Start

### Run the demo
```bash
pip install gradio torch torchvision
python app.py
```

### Run the experiment
Open in Google Colab or Kaggle:

```python
# Install
pip install torch torchvision

# Import the model
from fractal_sie import FractalSIE

# Load trained weights
model = FractalSIE(C=128, n_layers=16, n_cls=10)
model.load_state_dict(torch.load('sie_best.pt'))
model.eval()
```

---

## 📁 Repository Structure

```
SIE/
├── app.py                    # Hugging Face Gradio demo
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── notebooks/
│   └── SIE_CIFAR10_Experiment.ipynb   # Complete Kaggle experiment
├── results/
│   ├── sie_final_results.png          # Training curves (C=128, 300ep)
│   └── sie_results.json               # All training metrics
└── paper/
    └── SIE_arXiv_Paper.pdf            # Published paper
```

---

## ⚙️ Architecture

```
Input image (32×32×3)
        ↓
    Stem (Conv-3×3, GroupNorm-8, GELU)
        ↓
    [SpiralBlock × gate_m]  × 16     ← ONE block, shared
    gate_m = 2^m / (2^m + 1)         ← from doubling rule
        ↓
    Down (Conv stride-2, GroupNorm, GELU)
        ↓
    Global Average Pool
        ↓
    Head (Linear → 10 classes)
```

**Parameter count:** ~0.55M (C=128) | ~1.2M (C=192)

---

## 🔬 Experimental Configuration

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-10 (50K train / 10K test) |
| Optimiser | AdamW (lr=3×10⁻⁴, wd=0.05) |
| Scheduler | CosineAnnealingLR (T_max=EPOCHS) |
| Loss | CrossEntropy (label_smoothing=0.1) |
| Epochs | 300 (published) / 500 (in progress) |
| Hardware | 2× NVIDIA T4 (Kaggle free tier) |
| Channels C | 128 (published) / 192 (in progress) |

---

## 🌍 Future Applications

| Domain | Application | Dataset needed |
|--------|-------------|----------------|
| 🌾 Agriculture | Crop disease, satellite monitoring | PlantVillage |
| 🏥 Healthcare | Malaria, X-ray, skin lesion | NIH ChestX-ray14 |
| 🏦 Finance | Fraud detection (graph path) | Transaction graphs |
| 🚗 Transportation | Vehicle classification | Nigerian road data |
| 🛍 Retail | Product recognition | Custom retail data |
| 🔒 Security | Object detection, edge AI | Surveillance data |

---

## 📝 Citation

```bibtex
@article{dantala2025sie,
  title   = {SIE: Spiral Intelligence Engine --- A Cyclic Pascal Recurrence
             on a Polygon as Geometric Inductive Bias for Deep Learning},
  author  = {Dantala, Abdullahi Abdullahi},
  journal = {arXiv preprint arXiv:YOUR-PAPER-ID},
  year    = {2025}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*From a hand-drawn spiral on purple paper in Abuja, Nigeria*
*to a published geometric deep learning architecture.*
*Insha Allah.*
