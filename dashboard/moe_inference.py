"""
moe_inference.py — Motor de Inferencia del Sistema MoE.

Orquesta todo el pipeline de inferencia:
  1. Preprocesamiento adaptativo (2D/3D)
  2. Router ViT → gating scores + CLS token + attention map
  3. Deteccion OOD via entropia
  4. Experto activado (Top-1) → logits de clase
  5. Generacion de heatmap
  6. Empaquetado de resultados

Usa @st.cache_resource para cargar modelos UNA sola vez en memoria.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mock_models import load_models, EXPERT_INFO
from preprocessing import AdaptivePreprocessor, get_display_image
from heatmap_utils import generate_mock_heatmap
from ood_utils import detect_ood


@dataclass
class InferenceResult:
    """Resultado completo de una inferencia del sistema MoE."""

    # --- Preprocesamiento ---
    original_shape: tuple = ()
    processed_shape: tuple = ()
    is_3d: bool = False

    # --- Router ---
    gating_scores: np.ndarray = field(default_factory=lambda: np.zeros(5))
    expert_id: int = 0
    expert_name: str = ""
    expert_arch: str = ""
    expert_dataset: str = ""

    # --- Experto ---
    class_label: str = ""
    class_index: int = 0
    confidence: float = 0.0
    all_class_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    class_names: list = field(default_factory=list)

    # --- Heatmap ---
    heatmap_image: Optional[np.ndarray] = None   # RGB [H,W,3]
    display_image: Optional[Image.Image] = None   # Imagen original para mostrar

    # --- OOD ---
    is_ood: bool = False
    entropy: float = 0.0
    ood_threshold: float = 0.0
    ood_message: str = ""
    router_confidence: float = 0.0

    # --- Timing ---
    preprocess_ms: float = 0.0
    router_ms: float = 0.0
    expert_ms: float = 0.0
    total_ms: float = 0.0


class MoEInferenceEngine:
    """
    Motor de inferencia completo del sistema Mixture of Experts.

    Carga los modelos (mock o reales) y ejecuta el pipeline completo
    de inferencia sobre un archivo subido por el usuario.
    """

    def __init__(self, use_mock=True, checkpoint_dir=None, device="cpu"):
        self.device = device
        self.preprocessor = AdaptivePreprocessor()

        # Cargar modelos
        self.router, self.experts = load_models(
            use_mock=use_mock,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )

        # Modo evaluacion
        self.router.eval()
        for exp in self.experts.values():
            exp.eval()

        self.use_mock = use_mock

    @torch.no_grad()
    def run(self, uploaded_file) -> InferenceResult:
        """
        Ejecuta el pipeline completo de inferencia.

        Args:
            uploaded_file: archivo subido via st.file_uploader

        Returns:
            InferenceResult con todos los campos poblados
        """
        result = InferenceResult()
        t_start = time.perf_counter()

        # ---- 1. Preprocesamiento ----
        t0 = time.perf_counter()
        tensor, original_shape, is_3d = self.preprocessor.process_uploaded_file(
            uploaded_file
        )
        result.preprocess_ms = (time.perf_counter() - t0) * 1000
        result.original_shape = original_shape
        result.processed_shape = tuple(tensor.shape)
        result.is_3d = is_3d

        # Imagen para mostrar en el dashboard
        uploaded_file.seek(0)
        result.display_image = get_display_image(uploaded_file, tensor, is_3d)

        # ---- 2. Router ----
        t0 = time.perf_counter()
        gating_logits, cls_token, attn_weights = self.router(tensor)
        result.router_ms = (time.perf_counter() - t0) * 1000

        # Softmax de los gating scores
        gating_probs = F.softmax(gating_logits, dim=-1).squeeze(0)
        result.gating_scores = gating_probs.numpy()

        # Seleccion Top-1
        expert_id = int(gating_probs.argmax().item())
        result.expert_id = expert_id

        # Info del experto seleccionado
        info = EXPERT_INFO[expert_id]
        result.expert_name = info["name"]
        result.expert_arch = info["arch"]
        result.expert_dataset = info["dataset"]
        result.class_names = info["class_names"]

        # ---- 3. Deteccion OOD ----
        ood_result = detect_ood(gating_probs)
        result.is_ood = ood_result["is_ood"]
        result.entropy = ood_result["entropy"]
        result.ood_threshold = ood_result["threshold"]
        result.ood_message = ood_result["message"]
        result.router_confidence = ood_result["confidence"]

        # ---- 4. Experto activado ----
        t0 = time.perf_counter()
        expert = self.experts[expert_id]
        expert_logits = expert(tensor)  # [1, num_classes]
        result.expert_ms = (time.perf_counter() - t0) * 1000

        # Probabilidades de clase
        class_probs = F.softmax(expert_logits, dim=-1).squeeze(0).numpy()
        result.all_class_probs = class_probs
        result.class_index = int(class_probs.argmax())
        result.confidence = float(class_probs.max())
        result.class_label = info["class_names"][result.class_index]

        # ---- 5. Heatmap ----
        if result.display_image is not None:
            if self.use_mock:
                result.heatmap_image = generate_mock_heatmap(
                    result.display_image, attn_weights
                )
            else:
                # Futuro: usar generate_real_heatmap con attn_weights reales
                result.heatmap_image = generate_mock_heatmap(
                    result.display_image, attn_weights
                )

        # ---- 6. Timing total ----
        result.total_ms = (time.perf_counter() - t_start) * 1000

        return result
