"""
mock_models.py — Modelos simulados para desarrollo del Dashboard.

Cada Mock replica EXACTAMENTE la misma firma de entrada/salida que
el modelo real. Cuando existan los checkpoints (.pth), solo hay que
reemplazar la funcion `load_models()` en moe_inference.py.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ======================================================================
#  Informacion de Expertos (constante compartida)
# ======================================================================
EXPERT_INFO = {
    0: {
        "name": "Expert 1 — NIH (Profesor)",
        "arch": "LungMaxViT",
        "dataset": "NIH ChestX-ray 14",
        "num_classes": 14,
        "class_names": [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
            "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", 
            "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax"
        ],
        "is_3d": False,
    },
    1: {
        "name": "Expert 2 — ISIC 2019",
        "arch": "EfficientNet-B3",
        "dataset": "ISIC 2019",
        "num_classes": 8,
        "class_names": [
            "Melanoma", "Melanocytic nevus", "BCC", "Actinic keratosis",
            "Benign keratosis", "Dermatofibroma", "Vascular lesion",
            "SCC",
        ],
        "is_3d": False,
    },
    2: {
        "name": "Expert 3 — Osteoarthritis",
        "arch": "VGG-16 BN",
        "dataset": "Knee Osteoarthritis",
        "num_classes": 5,
        "class_names": ["KL Grade 0", "KL Grade 1", "KL Grade 2", "KL Grade 3", "KL Grade 4"],
        "is_3d": False,
    },
    3: {
        "name": "Expert 4 — LUNA16",
        "arch": "DCSwinB-Style 3D",
        "dataset": "LUNA16 / LIDC-IDRI",
        "num_classes": 2,
        "class_names": ["Benigno", "Maligno"],
        "is_3d": True,
    },
    4: {
        "name": "Expert 5 — Pancreas",
        "arch": "R3D-18",
        "dataset": "Pancreatic Cancer CT",
        "num_classes": 2,
        "class_names": ["Non-PDAC", "PDAC"],
        "is_3d": True,
    },
}


# ======================================================================
#  Mock Router (ViT-Tiny)
# ======================================================================
class MockVisionRouter(nn.Module):
    """
    Simula el VisionRouter.
    Input:  tensor preprocesado (2D o 3D)
    Output: (gating_logits [1,5], cls_token [1,192], attn_weights [H, N, N])
    """

    def __init__(self, embed_dim=192, num_experts=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        # Pesos ficticios para que el Router tenga preferencias estables
        self._bias_2d = torch.tensor([0.30, 0.28, 0.25, 0.09, 0.08])
        self._bias_3d = torch.tensor([0.05, 0.05, 0.05, 0.42, 0.43])

    def forward(self, tensor):
        """
        Args:
            tensor: [C, H, W] para 2D  o  [1, D, H, W] para 3D
        Returns:
            gating_logits: [1, 5]
            cls_token:     [1, 192]
            attn_weights:  [num_heads, seq_len, seq_len]  (simulado)
        """
        time.sleep(0.02)  # simular latencia realista

        is_3d = tensor.ndim == 4  # [1, D, H, W]

        # Gating scores con ruido controlado
        base = self._bias_3d if is_3d else self._bias_2d
        noise = torch.randn(self.num_experts) * 0.05
        gating_logits = (base + noise).unsqueeze(0)  # [1, 5]

        # CLS token simulado
        cls_token = torch.randn(1, self.embed_dim)

        # Attention map simulado (para heatmap)
        if is_3d:
            spatial = tensor.shape[1]  # D
            seq_len = max(8, spatial // 8) ** 3 + 1
        else:
            seq_len = (tensor.shape[1] // 16) * (tensor.shape[2] // 16) + 1

        num_heads = 3
        attn = torch.rand(num_heads, seq_len, seq_len)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        return gating_logits, cls_token, attn


# ======================================================================
#  Mock Expertos
# ======================================================================
class MockExpert(nn.Module):
    """
    Experto generico simulado.
    Recibe la imagen/volumen y devuelve logits del numero correcto de clases.
    """

    def __init__(self, expert_id: int):
        super().__init__()
        info = EXPERT_INFO[expert_id]
        self.expert_id = expert_id
        self.num_classes = info["num_classes"]
        self.is_3d = info["is_3d"]

    def forward(self, tensor):
        """
        Args:
            tensor: [C, H, W] o [1, D, H, W]
        Returns:
            logits: [1, num_classes]
        """
        time.sleep(0.01)  # simular latencia
        # Generar logits con una clase dominante para que la prediccion sea interesante
        logits = torch.randn(1, self.num_classes) * 0.5
        dominant = torch.randint(0, self.num_classes, (1,)).item()
        logits[0, dominant] += 2.0  # hacer que una clase domine
        return logits


# ======================================================================
#  Funcion de carga (punto de intercambio Mock / Real)
# ======================================================================
def load_models(use_mock=True, checkpoint_dir=None, device="cpu"):
    """
    Carga todos los modelos del sistema MoE.

    Cuando use_mock=True:  retorna modelos simulados.
    Cuando use_mock=False: carga desde checkpoints .pth reales.

    Returns:
        router:  nn.Module (VisionRouter o MockVisionRouter)
        experts: dict {expert_id: nn.Module}
    """
    if use_mock:
        router = MockVisionRouter()
        experts = {i: MockExpert(i) for i in range(5)}
        return router, experts

    # ----- Carga real (placeholder para cuando existan los .pth) -----
    # from models.vision_router import VisionRouter
    # router = VisionRouter(...)
    # router.load_state_dict(torch.load(f"{checkpoint_dir}/router.pth"))
    #
    # experts = {}
    # for i in range(5):
    #     model = create_expert(i)  # ConvNeXt, EfficientNet, VGG, R3D, Swin3D
    #     model.load_state_dict(torch.load(f"{checkpoint_dir}/expert_{i}.pth"))
    #     experts[i] = model
    #
    # return router.to(device), {k: v.to(device) for k, v in experts.items()}
    raise NotImplementedError("Carga real de checkpoints aun no implementada.")
