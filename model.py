import io
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0, mobilenet_v2
import timm
from PIL import Image

# ==============================
# 1. CONFIG
# ==============================

CLASS_NAMES = [
    "Amrapali",
    "Banana",
    "Bandigori",
    "Brunei King",
    "Harivanga",
    "Himsagar",
    "Kacha Mitha",
    "Surjapuri",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same as Kaggle (val_tf)
IMG_SIZE = 224
PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Must match training
MODEL_NAMES = ["DeiT-Tiny", "Swin-Tiny", "EfficientNetB0", "MobileNetV2"]
WEIGHTED_ENSEMBLE: Dict[str, float] = {
    "DeiT-Tiny": 0.45,
    "Swin-Tiny": 0.35,
    "EfficientNetB0": 0.20,
    "MobileNetV2": 0.0,
}

DEFAULT_BUNDLE_PATH = "models/Hybrid-MangoLeaf_bundle.pth"

# ==============================
# 2. MODEL FACTORY (same logic as Kaggle)
# ==============================

def _num_features_from_module(mod: nn.Module):
    """
    Try to find a reasonable in_features for classifier replacement.
    Copied from your Kaggle code.
    """
    for attr in ["classifier", "fc", "head", "linear"]:
        if hasattr(mod, attr):
            m = getattr(mod, attr)
            if isinstance(m, nn.Sequential):
                for layer in reversed(m):
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
            elif isinstance(m, nn.Linear):
                return m.in_features
            try:
                if hasattr(m, '__getitem__') and isinstance(m[1], nn.Linear):
                    return m[1].in_features
            except Exception:
                pass

    for cand in ["last_channel", "in_features"]:
        if hasattr(mod, cand):
            return getattr(mod, cand)

    return None


def get_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    Recreate exactly the same backbone architectures you trained on Kaggle.
    """
    if name == "EfficientNetB0":
        try:
            m = efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception:
            m = efficientnet_b0(pretrained=pretrained)

        nf = _num_features_from_module(m)
        if nf is None:
            raise RuntimeError("Couldn't detect classifier in EfficientNetB0")

        if hasattr(m, "classifier") and isinstance(m.classifier, nn.Sequential):
            m.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(nf, num_classes),
            )
        else:
            m.classifier = nn.Linear(nf, num_classes)

    elif name == "MobileNetV2":
        try:
            m = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception:
            m = mobilenet_v2(pretrained=pretrained)

        nf = _num_features_from_module(m)
        if nf is None:
            raise RuntimeError("Couldn't detect classifier in MobileNetV2")

        m.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(nf, num_classes),
        )

    elif name == "DeiT-Tiny":
        m = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    elif name == "Swin-Tiny":
        m = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    else:
        raise ValueError(f"Unknown model name: {name}")

    return m


# ==============================
# 3. HYBRID / ENSEMBLE MODEL
# ==============================

class HybridMangoLeafEnsemble(nn.Module):
    """
    Holds 4 classifiers and combines their predictions
    using WEIGHTED_ENSEMBLE (probability-level fusion).
    """

    def __init__(
        self,
        class_names: List[str],
        model_names: List[str],
        weights_dict: Dict[str, float],
    ):
        super().__init__()

        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model_names = model_names

        # Build all 4 models
        models_dict = {}
        for name in self.model_names:
            models_dict[name] = get_model(name, self.num_classes, pretrained=False)
        self.models_dict = nn.ModuleDict(models_dict)

        # Prepare ensemble weights (same order as model_names)
        w_list = []
        for name in self.model_names:
            w_list.append(float(weights_dict.get(name, 0.0)))
        w = torch.tensor(w_list, dtype=torch.float32)
        if w.sum() <= 0:
            w = torch.ones_like(w)
        w = w / (w.sum() + 1e-12)
        self.register_buffer("weights", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W]
        Returns: [B, num_classes] probabilities after weighted ensemble.
        """
        per_model_probs = []

        for name in self.model_names:
            logits = self.models_dict[name](x)
            prob = F.softmax(logits, dim=1)
            per_model_probs.append(prob)

        # [num_models, B, num_classes]
        probs_stack = torch.stack(per_model_probs, dim=0)

        # weights: [num_models] -> [num_models,1,1]
        w = self.weights.view(-1, 1, 1)

        # Weighted sum over models -> [B, num_classes]
        weighted_probs = (w * probs_stack).sum(dim=0)

        return weighted_probs  # already probabilities


# ==============================
# 4. LOAD MODEL FROM BUNDLE
# ==============================

def load_model(weights_path: str = DEFAULT_BUNDLE_PATH):
    """
    This replaces your old:
        model = models.hybridmodel(weights=None)
    and loads the full hybrid ensemble from Hybrid-MangoLeaf_bundle.pth

    - Must be called once at app startup.
    - Returns a ready-to-use nn.Module in eval mode.
    """
    global CLASS_NAMES

    bundle = torch.load(weights_path, map_location=DEVICE)

    # Classes from bundle (match your folder names on Kaggle)
    classes = bundle.get("classes", None)
    if classes is not None and len(classes) > 0:
        CLASS_NAMES = classes  # update global to keep predict() simple

    model_names = bundle.get("model_names", MODEL_NAMES)
    weights_dict = bundle.get("weights", WEIGHTED_ENSEMBLE)

    # Build ensemble arch
    ensemble = HybridMangoLeafEnsemble(
        class_names=CLASS_NAMES,
        model_names=model_names,
        weights_dict=weights_dict,
    )

    # Load state_dict for each backbone from bundle['states']
    states: Dict[str, Dict[str, torch.Tensor]] = bundle["states"]
    for name, state_dict in states.items():
        if name in ensemble.models_dict:
            ensemble.models_dict[name].load_state_dict(state_dict)
        else:
            print(f"[Warning] State dict for '{name}' in bundle but no model with this name in ensemble.")

    ensemble.to(DEVICE)
    ensemble.eval()
    return ensemble


# ==============================
# 5. PREDICTION HELPER
# ==============================

def predict_image_bytes(model, image_bytes):
    """
    Same interface as before:
        predicted_class, confidence = predict_image_bytes(model, image_bytes)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = model(x)                    # [1, num_classes], already softmaxed
        top_prob, top_idx = torch.max(probs, dim=1)

    predicted_class = CLASS_NAMES[top_idx.item()]
    confidence = float(top_prob.item())

    return predicted_class, confidence
