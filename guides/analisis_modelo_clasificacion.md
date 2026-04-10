# 🦴 Análisis del Paper Osteo-NeT y Recomendaciones de Arquitectura

## Contexto del Proyecto

| Elemento | Detalle |
|---|---|
| **Tarea** | Clasificación de Osteoartritis de Rodilla (KL Grades 0–4) |
| **Datos** | Radiografías X-ray preprocesadas (224×224 px, MHA → CLAHE → MONAI) |
| **Referencia** | Paper: *Osteo-NeT (VGG-16 + ResNet-50 transfer learning, acc 92.17%)* |
| **Recomendación del Profesor** | Exp. 3: `VGG-16 BN`, `ResNet-34`, `Inception-V3` — *"La tarea es relativamente simple, modelos livianos suelen funcionar bien"* |

---

## 1. Qué SÍ Podemos Aprovechar del Paper Osteo-NeT

### 1.1 Preprocesamiento ✅ (Ya implementado en nuestro pipeline)

El paper usa técnicas que **ya tenemos implementadas**:

| Técnica del Paper | Estado en Nuestro Notebook |
|---|---|
| Eliminación de ruido (Median Filter) | ✅ CLAHE adaptativo (Fase 3) |
| Histogram Equalization (CLAHE clip=2.0, tile=8×8) | ✅ Implementado con parámetros adaptativos por imagen |
| Resize 512×512 | ✅ Resize a 224×224 (compatible) |
| Data Augmentation (Rotación ±20°, shifts) | ✅ MONAI pipeline (sin rotación, se eliminó) |
| Entropía para filtrado de ruido | ✅ Fase 1.7 implementada |
| Split bilateral de radiografías dobles | ✅ Fase 1.7 implementada |

> ⭐ **Ventaja**: Nuestro pipeline de preprocesamiento es **más sofisticado** que el del paper, porque usa métricas G-CLAHE para validar la mejora cuantitativamente.

### 1.2 Estrategia de Entrenamiento ✅ Usar

- **Optimizador Adam** con LR muy bajo: `lr=0.00001` para fine-tuning. Nuestra configuración MONAI ya usa Adam; afinar a `1e-5`.
- **Batch size = 32**: Compatible con Colab y suficiente para generalización.
- **Epochs = 30**: Razonable; con EarlyStopping podemos detener antes.
- **Binary Cross-Entropy**: El paper usa BCE porque clasifica 2 clases. Nuestro problema es **multi-clase** (KL 0–4): usar **Categorical Cross-Entropy** o **Focal Loss** (especialmente si hay desbalanceo de clases).

### 1.3 Métricas de Evaluación ✅ Usar

Exactamente las mismas que el paper evalúa:

```
Precision, Recall, F1-Score, Specificity, AUC-ROC
Confusion Matrix (normalizada)
Grad-CAM para visualización de predicciones
```

### 1.4 Grad-CAM ✅ Muy Recomendado

El paper usa **Grad-CAM** para confirmar que el modelo mira las regiones correctas (espacio articular, osteofitos). **Debemos implementarlo** — es una herramienta clave de interpretabilidad exigida en presentaciones académicas.

---

## 2. Qué NO Usar del Paper

| Elemento del Paper | Por Qué No |
|---|---|
| Sequential CNN base model puro | Performance limitada (90.95%), reemplazada por TL |
| Arquitectura completa de VGG-16 sin BatchNorm | El profesor indica `VGG-16 BN` (con BatchNorm); más estable |
| ResNet-50 | El profesor indica `ResNet-34` que es más ligero y suficiente |
| Resize a 512×512 | Innecesario; 224×224 es estándar para todas las redes pre-entrenadas |
| Sin EarlyStopping / LR Scheduler | Implementar ambos para evitar overfitting |

---

## 3. Arquitectura Recomendada

### 3.1 Decisión Principal: Seguir la Recomendación del Profesor

> *"La tarea es relativamente simple — modelos livianos suelen funcionar bien"*

**Orden de preferencia:**

```
1º VGG-16 BN         → Simple, probado en OA, BatchNorm lo hace más estable
2º ResNet-34          → Ligero, residual connections, buen balance performance/costo
3º Inception-V3       → Multi-escala, pero más pesado; sería opción de comparación
```

### 3.2 Estrategia: Fine-Tuning en 2 Fases

```
FASE A — Feature Extraction (Congelar backbone)
  - Congelar todos los pesos del backbone (VGG-16 o ResNet-34)
  - Solo entrenar el clasificador custom
  - Epochs: 5-10, LR: 1e-3
  - Objetivo: Aclimatar el clasificador a los datos de OA

FASE B — Fine-Tuning (Descongelar últimas capas)
  - Descongelar las últimas 2-3 capas del backbone
  - LR: 1e-5 (muy bajo)
  - Epochs: 20-30 con EarlyStopping
  - Objetivo: Afinar features para radiografías
```

### 3.3 Cabeza Clasificadora Custom

```python
# Después del backbone pre-entrenado:
Sequential(
    GlobalAveragePooling2D(),   # Más robusto que Flatten
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),               # Paper usa 0.5; mantenerlo
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(5, activation='softmax')  # 5 clases KL 0-4
)
```

### 3.4 Comparación de Opciones

| Arquitectura | Params | Acc Esperada (OA) | Costo Colab | Recomendado |
|---|---|---|---|---|
| VGG-16 BN | ~138M | ~90-92% | Medio | ⭐⭐⭐ |
| ResNet-34 | ~21M | ~89-91% | Bajo | ⭐⭐⭐ |
| ResNet-50 | ~25M | ~90-92% | Bajo-Medio | ⭐⭐ |
| Inception-V3 | ~24M | ~90-92% | Medio | ⭐⭐ |
| EfficientNet-B0 | ~5M | ~89-91% | Muy Bajo | ⭐⭐⭐ (bonus) |

---

## 4. Hiperparámetros Recomendados

```python
HYPERPARAMS = {
    "img_size":       224,
    "batch_size":     32,
    "epochs_phase_a": 10,
    "epochs_phase_b": 30,
    "lr_phase_a":     1e-3,
    "lr_phase_b":     1e-5,
    "optimizer":      "Adam",
    "loss":           "FocalLoss",   # Por desbalanceo de clases KL
    "dropout":        0.5,
    "early_stopping": {"patience": 5, "restore_best_weights": True},
    "lr_scheduler":   "ReduceLROnPlateau (factor=0.3, patience=3)"
}
```

### ¿Por qué Focal Loss en lugar de Cross-Entropy?

El dataset KL tiene desbalanceo (Grado 2 y 3 son más frecuentes). Focal Loss penaliza más los ejemplos fáciles y fuerza al modelo a aprender los ejemplos difíciles (Grados 0 y 4 que son los extremos con menos datos).

```
Focal Loss: FL(p) = -(1 - p)^γ * log(p)
γ = 2 es el valor estándar
```

---

## 5. Data Augmentation Recomendada (MONAI)

| Transformación | Justificación |
|---|---|
| ❌ Rotación | Eliminada (recomendación clínica: orientación anatómica importa) |
| ✅ Horizontal Flip | Simula rodilla izquierda/derecha |
| ✅ Brightness/Contrast Jitter | Simula variaciones de exposición en X-ray |
| ✅ Zoom/Scale (±10%) | Simula distancias de adquisición |
| ✅ Gaussian Noise | Simula ruido de sensor |
| ✅ Random Crop + Resize | Mejora robustez |
| ❌ Vertical Flip | No tiene sentido clínico en rodillas |

---

## 6. Métricas y Evaluación

### Tabla de Métricas a Reportar

```
Por clase (KL 0, 1, 2, 3, 4):
  - Precision / Recall / F1-Score / Support

Global:
  - Macro-F1 (trata todas las clases por igual → detecta clases raras)
  - Weighted-F1 (pondera por support → refleja performance general)
  - AUC-ROC por clase (One-vs-Rest)
  - Accuracy general

Visual:
  - Confusion Matrix (normalizada)
  - Grad-CAM en ejemplos de cada grado KL
  - Curvas de entrenamiento (loss/acc por época)
```

---

## 7. Plan de Implementación Sugerido

```
Notebook 05: Entrenamiento del Modelo de Clasificación
│
├── 5.1 Cargar datos preprocesados (del pipeline existente)
├── 5.2 Definir arquitectura (VGG-16 BN o ResNet-34)
├── 5.3 FASE A: Feature Extraction (10 épocas, LR=1e-3)
│   └── Graficar curvas de loss/acc
├── 5.4 FASE B: Fine-Tuning (30 épocas, LR=1e-5)
│   └── EarlyStopping + ReduceLROnPlateau
├── 5.5 Evaluación completa
│   └── Métricas por clase + Confusion Matrix
├── 5.6 Grad-CAM visualizations
│   └── 5 ejemplos por grado KL
└── 5.7 Comparación de modelos (VGG-16 BN vs ResNet-34)
```

---

## 8. Resumen de Decisiones

| Decisión | Elección | Justificación |
|---|---|---|
| **Modelo base** | VGG-16 BN | Probado en OA, BatchNorm = más estable, recomendado por profesor |
| **Modelo alternativo** | ResNet-34 | Más ligero, residual learning |
| **Estrategia** | Fine-tuning 2 fases | Evita catastrophic forgetting |
| **Loss** | Focal Loss | Desbalanceo de clases KL |
| **Preprocesamiento** | Pipeline existente (CLAHE + MONAI) | Ya funciona y está validado |
| **Interpretabilidad** | Grad-CAM | Requerido para validación clínica |
| **Sin rotación** | ✅ Confirmado | Orientación anatómica es clínicamente relevante |

---

*Generado el 2026-03-21. Basado en: Paper Osteo-NeT (VGG-16/ResNet-50, 92.17%), recomendación del profesor (Exp. 3: modelos livianos), pipeline existente del proyecto.*
