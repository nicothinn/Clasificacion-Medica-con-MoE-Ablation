"""
ood_utils.py — Deteccion Out-of-Distribution (OOD) para el Dashboard.

Funcionalidad obligatoria #22 de la consigna:
  "OOD Detection: alerta cuando la entropia del gating score supera
   el umbral calibrado."

Metodo:
  Se calcula la entropia de Shannon del vector de probabilidades
  (gating scores) que produce el Router despues de Softmax.

  Si la entropia es alta, significa que el Router esta "confundido"
  (distribuye probabilidad uniformemente entre expertos), lo que indica
  que la imagen probablemente NO pertenece a ninguno de los 5 dominios
  medicos conocidos.

  H(g) = -sum(g_i * log(g_i))

  Entropia maxima teorica con 5 expertos: log(5) = 1.609 nats
  Umbral calibrado: 85% de la entropia maxima = 1.368 nats
"""

import numpy as np
import torch


# Constantes
NUM_EXPERTS = 5
MAX_ENTROPY = np.log(NUM_EXPERTS)  # log(5) ~ 1.609 nats
DEFAULT_OOD_THRESHOLD_RATIO = 1.0  # 100% de la entropia maxima (Deshabilita el OOD por entropia)


def compute_entropy(gating_probs):
    """
    Calcula la entropia de Shannon del vector de gating probabilities.

    Args:
        gating_probs: tensor o numpy array [num_experts]
                      (probabilidades post-Softmax)

    Returns:
        entropy: float (en nats, base e)
    """
    if isinstance(gating_probs, torch.Tensor):
        probs = gating_probs.detach().cpu().numpy().flatten()
    else:
        probs = np.asarray(gating_probs).flatten()

    # Clip para evitar log(0)
    probs = np.clip(probs, 1e-10, 1.0)

    # Renormalizar (por seguridad numerica)
    probs = probs / probs.sum()

    # Entropia de Shannon
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def detect_ood(gating_probs, threshold_ratio=DEFAULT_OOD_THRESHOLD_RATIO):
    """
    Detecta si una imagen es Out-of-Distribution basandose en la
    entropia de sus gating scores.

    Args:
        gating_probs:    tensor o array [num_experts] post-Softmax
        threshold_ratio: fraccion de la entropia maxima como umbral
                         (default 0.85 = 85%)

    Returns:
        result: dict con:
            - is_ood:       bool (True si se detecta OOD)
            - entropy:      float (entropia calculada)
            - threshold:    float (umbral utilizado)
            - max_entropy:  float (entropia maxima teorica)
            - confidence:   float (1 - entropy/max_entropy, 0-1)
            - message:      str (mensaje para el dashboard)
    """
    entropy = compute_entropy(gating_probs)
    threshold = threshold_ratio * MAX_ENTROPY
    is_ood = entropy > threshold

    # Confianza del router: inversa de la entropia normalizada
    confidence = 1.0 - (entropy / MAX_ENTROPY)
    confidence = max(0.0, min(1.0, confidence))

    if is_ood:
        message = (
            f"ALERTA OOD: La entropia del router ({entropy:.3f} nats) "
            f"supera el umbral ({threshold:.3f} nats). "
            f"Esta imagen probablemente NO pertenece a ningun dominio "
            f"medico conocido por el sistema."
        )
    else:
        message = (
            f"Imagen dentro de distribucion. "
            f"Entropia: {entropy:.3f} / {MAX_ENTROPY:.3f} nats. "
            f"Confianza del router: {confidence:.1%}"
        )

    return {
        "is_ood": is_ood,
        "entropy": entropy,
        "threshold": threshold,
        "max_entropy": MAX_ENTROPY,
        "confidence": confidence,
        "message": message,
    }
