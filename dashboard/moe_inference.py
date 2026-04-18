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

import os
import time
import zipfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

from real_models import (
    VisionRouter, LungMaxViT, build_efficientnet_b3_expert,
    build_vgg16_bn_expert, DCSwinBStyle3D, R3D18Expert,
)
from mock_models import EXPERT_INFO
from preprocessing import AdaptivePreprocessor, get_display_image
from heatmap_utils import generate_mock_heatmap, generate_real_heatmap
from ood_utils import detect_ood

# ======================================================================
# Configuración de Hugging Face
# ======================================================================
HF_REPO_ID = "Lucu1232004p/Proyecto-MoE-Pesos"
HF_TOKEN = None  # Se puede leer de st.secrets["HF_TOKEN"] si es privado


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

    def __init__(self, use_mock=True, repo_id=None, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.preprocessor = AdaptivePreprocessor()
        self.use_mock = use_mock
        self.repo_id = repo_id or HF_REPO_ID

        # Cargar modelos
        self.router, self.experts = self._load_all_models()

        # Modo evaluacion
        self.router.eval()
        for exp in self.experts.values():
            exp.eval()

    def _load_all_models(self):
        """Carga el sistema MoE completo."""
        if self.use_mock:
            from mock_models import load_models as load_mocks
            return load_mocks(use_mock=True, device=self.device)

        # 1. Cargar Router Real
        # Nota: Se usa .pth (val_acc=0.776) en lugar de .pth.zip (val_acc=0.885)
        # porque .pth.zip fue entrenado con una versión diferente de timm/monai
        # y sus embeddings no generalizan correctamente en timm>=1.0,
        # causando colapso de routing (100% ISIC para toda entrada).
        # El .pth produce routing correcto cross-modalidad con ImageNet norm.
        router = VisionRouter(num_experts=5, pretrained=False)
        router_path = self._download_weight("router_professor_fase3_only.pth")
        ckpt = torch.load(router_path, map_location="cpu")
        router.load_state_dict(ckpt["model_state_dict"])
        router.to(self.device)

        # 2. Cargar Expertos Reales
        experts = {}
        
        weight_maps = {
            0: ("exp1_NIH_LungMaxViT_best.pth",
                lambda: LungMaxViT(num_classes=14)),
            1: ("exp2_ISIC_EfficientNetB3_best.pth.zip",
                lambda: build_efficientnet_b3_expert(num_classes=8)),
            2: ("exp3_Osteo_VGG16BN_best.pth",
                lambda: build_vgg16_bn_expert(num_classes=5)),
            3: ("exp4_LUNA16_3D_best.pth",
                lambda: DCSwinBStyle3D(num_classes=2)),
            4: ("exp5_Pancreas_3D_best.pth",
                lambda: R3D18Expert(num_classes=2)),
        }

        for eid, (filename, builder) in weight_maps.items():
            model_path = self._download_weight(filename)
            
            # Caso especial ZIP (Experto 2)
            if filename.endswith(".zip"):
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    pth_name = next((n for n in zip_ref.namelist() if n.endswith(".pth")), None)
                    if pth_name:
                        target_pth = os.path.join(os.path.dirname(model_path), pth_name)
                        if not os.path.exists(target_pth):
                            zip_ref.extractall(os.path.dirname(model_path))
                        model_path = target_pth

            model = builder()
            state_dict = torch.load(model_path, map_location="cpu")
            # Manejar si el dict viene envuelto
            if isinstance(state_dict, dict):
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]
            
            model.load_state_dict(state_dict)
            experts[eid] = model.to(self.device)

        return router, experts

    def _download_weight(self, filename):
        """Descarga un archivo desde HF Hub."""
        try:
            return hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                token=HF_TOKEN
            )
        except Exception as e:
            print(f"Error descargando {filename}: {e}")
            # Fallback local si ya existe en la carpeta actual o similar
            local_path = os.path.join("weights", filename)
            if os.path.exists(local_path): return local_path
            raise e

    def _prepare_router_tensor(self, tensor_raw, is_3d):
        """
        Prepara el tensor para el Router.
        
        El Router fue entrenado con:
        - 2D: AdaptivePreprocessor -> ViT_AdapterWrapper -> ImageNet norm
        - 3D: AdaptivePreprocessor (HU windowed [0,1]) sin norm adicional
        
        El SwitchablePatchEmbed acepta tensores sin batch dim,
        pero se lo añadimos por consistencia.
        """
        if is_3d:
            # 3D volumes: already in [0,1] from HU windowing, no extra norm
            return tensor_raw.to(self.device)
        else:
            # 2D: El Router funciona mejor con el tensor crudo [0,1]
            # (ImageNet norm sobre radiografías distorsiona el CLS token y causa colapso a ISIC)
            return tensor_raw.to(self.device)

    def _prepare_expert_tensor(self, tensor_raw, is_3d, expert_id):
        """
        Prepara el tensor para el Experto seleccionado.
        
        Cada experto fue entrenado con su propia normalización:
          - Exp 0 (NIH LungMaxViT): 3 canales, ImageNet norm
          - Exp 1 (ISIC EfficientNet): 3 canales, ImageNet norm
          - Exp 2 (Osteo VGG-16 BN): 1 canal, z-score normalization
          - Exp 3 (LUNA16 DCSwinB): 1 canal 3D, [0,1] (HU windowed)
          - Exp 4 (Pancreas R3D-18): 3 canales 3D, [0,1]
        
        Todos los tensors se devuelven con batch dimension [B, ...].
        """
        if expert_id == 0:  # NIH LungMaxViT — 3ch + ImageNet norm
            t = AdaptivePreprocessor.apply_imagenet_norm(tensor_raw)
            if t.ndim == 3:
                t = t.unsqueeze(0)  # [3,H,W] -> [1,3,H,W]
            return t.to(self.device)

        elif expert_id == 1:  # ISIC EfficientNet-B3 — 3ch + ImageNet norm
            t = AdaptivePreprocessor.apply_imagenet_norm(tensor_raw)
            if t.ndim == 3:
                t = t.unsqueeze(0)  # [3,H,W] -> [1,3,H,W]
            return t.to(self.device)

        elif expert_id == 2:  # Osteo VGG-16 BN — 1ch grayscale
            # Convertir RGB [3,H,W] a grayscale [1,H,W]
            if tensor_raw.ndim == 3 and tensor_raw.shape[0] == 3:
                gray = tensor_raw.mean(dim=0, keepdim=True)  # [1, H, W]
                gray = gray.unsqueeze(0)  # [1, 1, H, W]
            elif tensor_raw.ndim == 4 and tensor_raw.shape[1] == 3:
                gray = tensor_raw.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            else:
                gray = tensor_raw
                if gray.ndim == 3:
                    gray = gray.unsqueeze(0)
            return gray.to(self.device)

        elif expert_id == 3:  # LUNA16 3D — 1ch, ya en [0,1]
            t = tensor_raw
            if t.ndim == 4:
                t = t.unsqueeze(0)  # [1,D,H,W] -> [1,1,D,H,W]
            return t.to(self.device)

        elif expert_id == 4:  # Pancreas R3D-18 — 3 canales 3D
            t = tensor_raw
            # Añadir batch dim si no la tiene
            if t.ndim == 4:
                t = t.unsqueeze(0)  # [C,D,H,W] -> [1,C,D,H,W]
            # El checkpoint fue entrenado con 3 canales de entrada
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1, 1)  # [1,1,D,H,W] -> [1,3,D,H,W]
            return t.to(self.device)

        else:
            t = tensor_raw
            if t.ndim == 3:
                t = t.unsqueeze(0)
            return t.to(self.device)

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
        # tensor_raw está en rango [0, 1] (sin normalización ImageNet)
        t0 = time.perf_counter()
        tensor_raw, original_shape, is_3d = self.preprocessor.process_uploaded_file(
            uploaded_file
        )
        result.preprocess_ms = (time.perf_counter() - t0) * 1000
        result.original_shape = original_shape
        result.processed_shape = tuple(tensor_raw.shape)
        result.is_3d = is_3d

        # Imagen para mostrar en el dashboard
        uploaded_file.seek(0)
        result.display_image = get_display_image(uploaded_file, tensor_raw, is_3d)

        # ---- 2. Router ----
        # Aplicar normalización específica del Router (ImageNet para 2D)
        t0 = time.perf_counter()
        router_tensor = self._prepare_router_tensor(tensor_raw, is_3d)
        gating_logits, cls_token, attn_weights = self.router(router_tensor)
        result.router_ms = (time.perf_counter() - t0) * 1000

        # Softmax de los gating scores
        gating_probs = F.softmax(gating_logits, dim=-1).squeeze(0)
        result.gating_scores = gating_probs.cpu().numpy()

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
        # Preparar tensor con normalización específica del experto
        t0 = time.perf_counter()
        expert = self.experts[expert_id]
        expert_tensor = self._prepare_expert_tensor(tensor_raw, is_3d, expert_id)
        expert_logits = expert(expert_tensor)  # [1, num_classes]
        result.expert_ms = (time.perf_counter() - t0) * 1000

        # Probabilidades de clase
        class_probs = F.softmax(expert_logits, dim=-1).squeeze(0).cpu().numpy()
        result.all_class_probs = class_probs
        result.class_index = int(class_probs.argmax())
        result.confidence = float(class_probs.max())
        result.class_label = info["class_names"][result.class_index]

        # ---- 5. Heatmap ----
        if result.display_image is not None:
            if self.use_mock or attn_weights is None:
                result.heatmap_image = generate_mock_heatmap(
                    result.display_image, attn_weights
                )
            else:
                # Heatmap real con atención del ViT (consigna #18)
                try:
                    result.heatmap_image = generate_real_heatmap(
                        result.display_image,
                        attn_weights,
                        patch_grid_size=(14, 14),
                    )
                except Exception:
                    # Fallback al mock si falla
                    result.heatmap_image = generate_mock_heatmap(
                        result.display_image, attn_weights
                    )

        # ---- 6. Timing total ----
        result.total_ms = (time.perf_counter() - t_start) * 1000

        return result
