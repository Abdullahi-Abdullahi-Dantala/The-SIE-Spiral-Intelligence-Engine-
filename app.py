import torch
import torch.nn as nn
import gradio as gr
import torchvision.transforms as T
from PIL import Image
import os

# ═══════════════════════════════════════════════════════════
#  FractalSIE — Spiral Intelligence Engine
#  Author: Abdullahi Abdullahi Dantala
#  Mathematical rule: P_(m+1,n) = P_(m,n) + P_(m,(n mod 12)+1)
#  Doubling theorem: sum(L_m) = 12 × 2^m
#  CIFAR-10 accuracy: 87.44% (C=192, ep.190/500)
#                     85.69% (C=128, ep.300) — published result
# ═══════════════════════════════════════════════════════════

class SpiralBlock(nn.Module):
    """
    Single shared block — applied to all 16 spiral layers.
    Weight sharing reflects the cyclic self-similarity of
    the author's P_(m+1,n) = P_(m,n) + P_(m,(n mod 12)+1) rule.
    """
    def __init__(self, C: int):
        super().__init__()
        self.dw   = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.pw   = nn.Conv2d(C, C, 1)
        self.norm = nn.GroupNorm(8, C)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x)))) + x


class FractalSIE(nn.Module):
    """
    Spiral Intelligence Engine — FractalSIE architecture.

    The doubling gate at layer m:
        gate_m = 2^m / (2^m + 1)
    is derived directly from the author's rule a_m = 2^m.
    As m → ∞, gate_m → 1 (outer rings dominate).

    This gate is NOT a hyperparameter — it is mathematically
    derived from the cyclic Pascal recurrence.
    """
    def __init__(self, C: int = 128, n_layers: int = 16,
                 n_cls: int = 10, drop_p: float = 0.0):
        super().__init__()
        self.stem  = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1),
            nn.GroupNorm(8, C),
            nn.GELU()
        )
        self.block  = SpiralBlock(C)      # ONE block, shared across all layers
        self.down   = nn.Sequential(
            nn.Conv2d(C, C * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, C * 2),
            nn.GELU()
        )
        self.head   = nn.Linear(C * 2, n_cls)
        self.n      = n_layers
        self.drop_p = drop_p              # 0.0 at inference (no dropout)

    def forward(self, x):
        x = self.stem(x)
        for m in range(self.n):
            # Doubling gate — derived from author's a_m = 2^m rule
            gate = (2 ** m) / (2 ** m + 1)
            x    = self.block(x) * gate
        x = self.down(x).mean([-1, -2])  # Global average pool
        return self.head(x)


# ═══════════════════════════════════════════════════════════
#  CIFAR-10 class labels
# ═══════════════════════════════════════════════════════════
CLASSES = [
    "✈️  Airplane",
    "🚗  Automobile",
    "🐦  Bird",
    "🐱  Cat",
    "🦌  Deer",
    "🐶  Dog",
    "🐸  Frog",
    "🐴  Horse",
    "🚢  Ship",
    "🚛  Truck"
]

CLASSES_PLAIN = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog",      "Frog",       "Horse","Ship","Truck"
]

# ═══════════════════════════════════════════════════════════
#  Load trained model
# ═══════════════════════════════════════════════════════════
def load_model():
    model = FractalSIE(C=128, n_layers=16, n_cls=10, drop_p=0.0)
    weights_path = "sie_best.pt"

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights '{weights_path}' not found. "
            "Please upload sie_best.pt to this Space."
        )

    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# ═══════════════════════════════════════════════════════════
#  Image preprocessing — must match training pipeline exactly
# ═══════════════════════════════════════════════════════════
transform = T.Compose([
    T.Resize((32, 32)),          # Resize to CIFAR-10 input size
    T.ToTensor(),                # Convert to [0,1] tensor
    T.Normalize(
        mean=(0.4914, 0.4822, 0.4465),   # CIFAR-10 mean (per channel)
        std =(0.2023, 0.1994, 0.2010)    # CIFAR-10 std  (per channel)
    )
])


# ═══════════════════════════════════════════════════════════
#  Prediction function
# ═══════════════════════════════════════════════════════════
def predict(image: Image.Image) -> dict:
    """
    Takes a PIL image, runs it through FractalSIE,
    returns a dict of {class_label: confidence_score}.
    """
    if image is None:
        return {c: 0.0 for c in CLASSES}

    # Preprocess
    x = transform(image).unsqueeze(0)   # Add batch dimension → [1,3,32,32]

    # Inference
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]

    return {CLASSES[i]: float(probs[i]) for i in range(10)}


# ═══════════════════════════════════════════════════════════
#  Gradio Interface
# ═══════════════════════════════════════════════════════════
description = """
## 🌀 Spiral Intelligence Engine (SIE)

**Author:** Abdullahi Abdullahi Dantala — Independent Researcher, Abuja, Nigeria

**Architecture:** FractalSIE — a weight-shared convolutional network grounded in the
*cyclic Pascal recurrence*:

> **P_(m+1,n) = P_(m,n) + P_(m,(n mod 12)+1)**

This rule was independently derived from a hand-drawn spiral of 20 concentric 12-gon rings.
It generates a provable doubling theorem — **sum(L_m) = 12 × 2^m** — which directly
produces the architecture's gating mechanism.

**Results:**
- ✅ **85.69%** on CIFAR-10 — *surpassing FractalNet (84.6%, ICLR 2017)*
- 🚀 **87.44%** at epoch 190/500 (C=192 run, still improving)

**How to use:** Upload any image of an airplane, car, bird, cat, deer,
dog, frog, horse, ship, or truck. The model will classify it in under 1 second.
"""

article = """
---
**Paper:** [SIE: Spiral Intelligence Engine — arXiv cs.LG](https://arxiv.org/abs/YOUR-PAPER-ID)

**Code:** [github.com/Abdullahi-Abdullahi-Dantala/SIE](https://github.com/Abdullahi-Abdullahi-Dantala/SIE)

*From a hand-drawn spiral in Abuja, Nigeria — to a published neural architecture.*
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Upload an image",
        sources=["upload", "webcam"],
    ),
    outputs=gr.Label(
        num_top_classes=5,
        label="Predictions (top 5)"
    ),
    title="🌀 SIE — Spiral Intelligence Engine",
    description=description,
    article=article,
    theme=gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="teal",
    ),
    examples=[],          # Add example images here if you have them
    cache_examples=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
