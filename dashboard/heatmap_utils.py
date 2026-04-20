"""
heatmap_utils.py — Generacion de Attention Heatmaps para el Dashboard.

Funcionalidad obligatoria #18 de la consigna:
  "Attention Heatmap del Router ViT: mapa de calor sobre la imagen original."

Cuando se usen modelos Mock:
  - Genera un heatmap gaussiano simulado pero visualmente convincente.

Cuando se usen modelos reales (futuro):
  - Extraera los attention weights del ultimo bloque de Self-Attention
    del ViT usando PyTorch forward hooks.
"""

import numpy as np
import cv2
from PIL import Image


def generate_mock_heatmap(image_pil, attn_weights=None, seed=42):
    """
    Genera un heatmap simulado gaussiano sobre la imagen original.

    El heatmap simula la atencion del Router ViT: una zona focal donde
    el modelo 'mira' para tomar su decision de routing.

    Args:
        image_pil:    PIL.Image de la imagen original (RGB)
        attn_weights: tensor de attention weights (opcional, del MockRouter)
        seed:         semilla para reproducibilidad

    Returns:
        heatmap_overlay: numpy array RGB [H, W, 3] con el heatmap superpuesto
    """
    rng = np.random.RandomState(seed)

    # Convertir imagen a numpy
    img_np = np.array(image_pil.resize((224, 224)))
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    h, w = img_np.shape[:2]

    # Crear heatmap gaussiano con centro semi-aleatorio
    # (centrado en el tercio central para simular atencion medica)
    cx = rng.randint(w // 4, 3 * w // 4)
    cy = rng.randint(h // 4, 3 * h // 4)
    sigma_x = w / (3 + rng.random() * 2)
    sigma_y = h / (3 + rng.random() * 2)

    y_grid, x_grid = np.mgrid[0:h, 0:w]
    gaussian = np.exp(
        -(((x_grid - cx) ** 2) / (2 * sigma_x ** 2)
          + ((y_grid - cy) ** 2) / (2 * sigma_y ** 2))
    )

    # Agregar un segundo foco mas pequeno para hacerlo mas realista
    cx2 = rng.randint(w // 3, 2 * w // 3)
    cy2 = rng.randint(h // 3, 2 * h // 3)
    gaussian2 = np.exp(
        -(((x_grid - cx2) ** 2) / (2 * (sigma_x * 0.5) ** 2)
          + ((y_grid - cy2) ** 2) / (2 * (sigma_y * 0.5) ** 2))
    ) * 0.6

    heatmap = gaussian + gaussian2
    heatmap = heatmap / heatmap.max()  # normalizar a [0, 1]

    return overlay_heatmap(img_np, heatmap)


def generate_real_heatmap(image_pil, attn_weights, patch_grid_size):
    """
    Genera un heatmap REAL a partir de los attention weights del ViT.

    Esta funcion se usara cuando se tengan los modelos entrenados reales.
    Extrae la atencion del CLS token (fila 0) hacia todos los patches,
    la reorganiza en una grilla espacial y la interpola al tamano
    de la imagen original.

    Args:
        image_pil:       PIL.Image original
        attn_weights:    tensor [num_heads, seq_len, seq_len] del ultimo bloque
        patch_grid_size: tuple (rows, cols) de la grilla de patches

    Returns:
        heatmap_overlay: numpy array RGB [H, W, 3]
    """
    img_np = np.array(image_pil.resize((224, 224)))
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    # Soportar tensores con batch dimension [B, H, S, S] o sin ella [H, S, S]
    if attn_weights.ndim == 4:
        attn_weights = attn_weights[0]  # usar el primer sample del batch

    # Promediar sobre las cabezas de atencion
    attn = attn_weights.mean(dim=0)  # [seq_len, seq_len]

    # Extraer atencion del CLS token (fila 0) hacia todos los patches
    # Excluir la posicion 0 (auto-atencion del CLS consigo mismo)
    cls_attn = attn[0, 1:].detach().cpu().numpy()  # [num_patches]

    # Reorganizar en grilla espacial
    num_patches = patch_grid_size[0] * patch_grid_size[1]
    cls_attn = cls_attn[:num_patches]  # truncar si hace falta
    attn_map = cls_attn.reshape(patch_grid_size)

    # Normalizar
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Interpolar al tamano de la imagen con interpolacion bicubica
    h, w = img_np.shape[:2]
    attn_map_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # Filtro Gaussiano para suavizar las manchas agresivamente
    attn_map_blur = cv2.GaussianBlur(attn_map_resized, (75, 75), 0)

    # Normalizacion Min-Max a [0, 1] estricto
    attn_map_blur = (attn_map_blur - attn_map_blur.min()) / (attn_map_blur.max() - attn_map_blur.min() + 1e-8)

    return overlay_heatmap(img_np, attn_map_blur)


def generate_real_heatmap_3d(image_pil, attn_weights, patch_grid_size_3d=(8, 8, 8)):
    """
    Genera un heatmap REAL para volúmenes 3D realizando slice matching.
    Extrae el corte central del volumen de atención 3D y lo mapea al corte 2D de la imagen original.
    """
    img_np = np.array(image_pil)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    if attn_weights.ndim == 4:
        attn_weights = attn_weights[0]

    attn = attn_weights.mean(dim=0)
    cls_attn = attn[0, 1:].detach().cpu().numpy()

    num_patches = patch_grid_size_3d[0] * patch_grid_size_3d[1] * patch_grid_size_3d[2]
    cls_attn = cls_attn[:num_patches]
    attn_vol = cls_attn.reshape(patch_grid_size_3d)  # [Z, Y, X]

    # 1. Selección de Corte (Slice Matching)
    z_index = patch_grid_size_3d[0] // 2
    attn_slice = attn_vol[z_index]  # Corte 2D de atención (ej. 8x8)

    # 2. Redimensionamiento 2D a 2D al tamaño de la imagen original mostrada
    h, w = img_np.shape[:2]
    attn_slice_resized = cv2.resize(attn_slice, (w, h), interpolation=cv2.INTER_CUBIC)

    # 3. Suavizado 2D agresivo (manchas continuas)
    attn_map_blur = cv2.GaussianBlur(attn_slice_resized, (75, 75), 0)

    # 4. Normalización estricta y fusión
    attn_map_blur = (attn_map_blur - attn_map_blur.min()) / (attn_map_blur.max() - attn_map_blur.min() + 1e-8)

    return overlay_heatmap(img_np, attn_map_blur, alpha=0.5)


def overlay_heatmap(image_np, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Superpone un mapa de calor sobre una imagen.

    Args:
        image_np: numpy array RGB [H, W, 3] con valores 0-255
        heatmap:  numpy array [H, W] con valores 0.0-1.0
        alpha:    transparencia del heatmap (0=invisible, 1=opaco)
        colormap: colormap de OpenCV (default: JET)

    Returns:
        blended: numpy array RGB [H, W, 3] con valores 0-255
    """
    # Convertir heatmap a uint8 y aplicar colormap
    heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    image_float = image_np.astype(np.float32)
    heat_float = heatmap_colored.astype(np.float32)
    blended = (1 - alpha) * image_float + alpha * heat_float
    blended = blended.clip(0, 255).astype(np.uint8)

    return blended
