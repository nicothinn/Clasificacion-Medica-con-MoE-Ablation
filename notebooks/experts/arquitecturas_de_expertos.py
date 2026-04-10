"""
Arquitecturas y utilidades de carga de expertos para MoE.

Colab: sube/copia este archivo a Drive en:
    /content/drive/MyDrive/PROYECTO_MOE_VISION/code/arquitecturas_de_expertos.py

Antes de importar, añade esa carpeta a sys.path (ver constante COLAB_DRIVE_CODE_DIR).

Ejemplo en notebook:
    import sys
    sys.path.insert(0, "/content/drive/MyDrive/PROYECTO_MOE_VISION/code")
    from arquitecturas_de_expertos import load_all_experts_from_drive, COLAB_DRIVE_CODE_DIR
    experts, info = load_all_experts_from_drive("/content/drive/MyDrive/PROYECTO_MOE_VISION/03_Weights")
"""

import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import timm
import torchvision.models as models
from torchvision.models.video import R3D_18_Weights, r3d_18

# Carpeta en Google Drive donde vive el código compartido (incluye este .py)
COLAB_DRIVE_CODE_DIR = "/content/drive/MyDrive/PROYECTO_MOE_VISION/code"


def prepend_colab_code_dir():
    """Inserta COLAB_DRIVE_CODE_DIR al inicio de sys.path si la carpeta existe (útil tras import ya resuelto)."""
    import sys

    if os.path.isdir(COLAB_DRIVE_CODE_DIR) and COLAB_DRIVE_CODE_DIR not in sys.path:
        sys.path.insert(0, COLAB_DRIVE_CODE_DIR)


# ============================================================================
# Experto 1 (NIH) - Swin-Tiny (NIH_ChestXray_Swin_Tiny_Training.ipynb, 5 clases)
# ============================================================================
class SwinNIHClassifier(nn.Module):
    """Misma envoltura que SwinClassifier en el notebook de entrenamiento (timm)."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


# ============================================================================
# Experto 3 (Osteo) - VGG16-BN
# ============================================================================
def build_vgg16_bn(num_classes: int = 5, pretrained: bool = True):
    model = models.vgg16_bn(weights="IMAGENET1K_V1" if pretrained else None)
    old_conv = model.features[0]
    new_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        new_conv.bias.copy_(old_conv.bias)
    model.features[0] = new_conv
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


# ============================================================================
# Expertos 4 y 5 (3D) - R3D18
# ============================================================================
class R3D18Expert(nn.Module):
    """
    Arquitectura base compartida para:
    - Exp4 (LUNA16)
    - Exp5 (Pancreas)
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        base = r3d_18(weights=weights)
        old_conv = base.stem[0]
        stem_conv = nn.Conv3d(
            1,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        with torch.no_grad():
            stem_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        base.stem[0] = stem_conv
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.head(feat)


# ============================================================================
# Factory y carga de pesos
# ============================================================================
EXPERT_SPECS = {
    1: {"name": "exp1_nih", "num_classes": 5, "arch": "swin_tiny_nih"},
    2: {"name": "exp2_isic", "num_classes": 9, "arch": "efficientnet_b3"},
    3: {"name": "exp3_osteo", "num_classes": 5, "arch": "vgg16_bn"},
    4: {"name": "exp4_luna16", "num_classes": 2, "arch": "r3d18"},
    5: {"name": "exp5_pancreas", "num_classes": 2, "arch": "r3d18"},  # R3D18 implicito
}


def build_expert(expert_id: int, pretrained_backbone: bool = True):
    spec = EXPERT_SPECS[int(expert_id)]
    arch = spec["arch"]
    num_classes = spec["num_classes"]

    if arch == "swin_tiny_nih":
        return SwinNIHClassifier(num_classes=num_classes, pretrained=pretrained_backbone)
    if arch == "efficientnet_b3":
        return timm.create_model("efficientnet_b3", pretrained=pretrained_backbone, num_classes=num_classes)
    if arch == "vgg16_bn":
        return build_vgg16_bn(num_classes=num_classes, pretrained=pretrained_backbone)
    if arch == "r3d18":
        return R3D18Expert(num_classes=num_classes, pretrained=pretrained_backbone)
    raise ValueError(f"Arquitectura no soportada: {arch}")


def _default_checkpoint_candidates(weights_dir: str) -> Dict[int, List[str]]:
    """
    Candidatos por experto para tolerar nombres antiguos/nuevos de checkpoints.
    """
    return {
        1: [
            os.path.join(weights_dir, "Experts_2D", "MaxViT_NIH_5cls.pth"),
            os.path.join(weights_dir, "MaxViT_NIH_5cls.pth"),
            os.path.join(weights_dir, "exp1_NIH_SwinTiny_best.pth"),
            os.path.join(weights_dir, "exp1_NIH_LungMaxViT_best.pth"),
        ],
        2: [
            os.path.join(weights_dir, "exp2_ISIC_EfficientNetB3_best.pth"),
        ],
        3: [
            os.path.join(weights_dir, "exp3_Osteo_VGG16BN_best.pth"),
        ],
        4: [
            os.path.join(weights_dir, "exp4_LUNA16_3D_best.pth"),
            os.path.join(weights_dir, "r3d18_luna16_best_V2.pth"),
        ],
        5: [
            os.path.join(weights_dir, "exp5_Pancreas_3D_best.pth"),
            os.path.join(weights_dir, "r3d18_pancreas_best_V2.pth"),
        ],
    }


def resolve_checkpoint(candidates: List[str]) -> str:
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def load_weights(model: nn.Module, ckpt_path: str, map_location: str = "cpu", strict: bool = False):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False, "checkpoint no encontrado"
    raw = torch.load(ckpt_path, map_location=map_location)
    if isinstance(raw, dict):
        if "state_dict" in raw:
            state = raw["state_dict"]
        elif "model_state_dict" in raw:
            state = raw["model_state_dict"]
        elif "model" in raw and isinstance(raw["model"], dict):
            state = raw["model"]
        else:
            state = raw
    else:
        state = raw
    model.load_state_dict(state, strict=strict)
    return True, "ok"


def freeze_and_eval(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def load_all_experts_from_drive(
    weights_dir: str,
    device: str = "cpu",
    strict: bool = False,
    pretrained_backbone: bool = False,
) -> Tuple[Dict[int, nn.Module], Dict[int, dict]]:
    """
    Crea y carga los 5 expertos con pesos desde Drive.

    Retorna:
      experts: dict[int, nn.Module]
      info: dict[int, {arch, ckpt, loaded, params}]
    """
    candidates = _default_checkpoint_candidates(weights_dir)
    experts = {}
    info = {}

    for eid in sorted(EXPERT_SPECS.keys()):
        model = build_expert(eid, pretrained_backbone=pretrained_backbone)
        ckpt_path = resolve_checkpoint(candidates[eid])
        loaded, msg = load_weights(model, ckpt_path, map_location="cpu", strict=strict)
        model = freeze_and_eval(model).to(device)
        experts[eid] = model
        info[eid] = {
            "name": EXPERT_SPECS[eid]["name"],
            "arch": EXPERT_SPECS[eid]["arch"],
            "ckpt": ckpt_path,
            "loaded": loaded,
            "message": msg,
            "params": int(sum(p.numel() for p in model.parameters())),
        }

    return experts, info


def print_expert_load_report(info: Dict[int, dict]):
    print("=== Expert Load Report ===")
    for eid in sorted(info.keys()):
        row = info[eid]
        status = "OK" if row["loaded"] else "MISSING"
        print(
            f"Exp{eid} | {row['name']} | arch={row['arch']} | {status} | "
            f"params={row['params']:,} | ckpt={row['ckpt'] or 'N/A'}"
        )


