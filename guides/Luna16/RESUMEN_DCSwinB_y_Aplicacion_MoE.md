# Resumen — DCSwinB: Deep Learning for Lung Cancer Classification (CT Images)

> **Fuente:** [Deep learning-based lung cancer classification of CT images](https://pmc.ncbi.nlm.nih.gov/articles/PMC12210548/)
> **Dataset:** LUNA16 / LIDC-IDRI
> **Tarea:** Clasificación binaria de nódulos pulmonares — Benigno vs Maligno

---

## 1. Problema que resuelven

Los radiólogos sin asistencia CAD tienen tasas de **falsos positivos entre 51% y 83.2%**, con sensibilidades de 94–96%. El reto es clasificar nódulos pulmonares en CT con alta precisión, capturando tanto **características locales** (bordes, texturas del nódulo) como **dependencias globales** (contexto del tejido circundante).

Los modelos anteriores fallan porque:
- Las CNNs capturan bien lo local pero tienen campo receptivo limitado (no ven contexto global).
- Los ViTs capturan bien lo global pero pierden detalles locales finos y son computacionalmente caros.

---

## 2. Solución: DCSwinB (Dual CNN Swin Branch)

### Arquitectura general

```
Input CT ROI (64×64×64) 
        │
   Conv1×1 split (canal)
  ┌──────┴──────┐
  │ Rama ViT   │ Rama CNN
  │ Swin-Tiny  │ Conv3×3 + MaxPool
  │ + Conv-MLP │
  └──────┬──────┘
         │  Concatenar canales
     Clasificador final
```

La clave está en **dos ramas paralelas** que procesan la mitad de los canales cada una:

| Rama | Qué captura | Operaciones |
|------|------------|-------------|
| **Swin Transformer + Conv-MLP** | Dependencias globales + contexto largo | W-MSA → MLP → DW_Conv → MLP → SW-MSA |
| **CNN (Conv3×3 + MaxPool)** | Características locales, bordes, texturas | 3×3 Conv → MaxPool |

### Innovación central: Conv-MLP

El módulo Conv-MLP **inserta una convolución depthwise 3×3** entre dos MLPs dentro del bloque Swin Transformer. Esto:
- **Refuerza conexiones entre ventanas adyacentes** (el Swin original las trata independientemente).
- Captura interacciones espaciales locales dentro del espacio de atención global.
- Es eficiente: DW_Conv usa solo `3×3×C` parámetros (vs `3×3×C×C` de conv estándar).

### Jerarquía del modelo

4 etapas con `[2, 2, 6, 2]` bloques Swin Transformer. Entre etapas: **Patch Merging** con capas lineales para downsamplear. Patch inicial: `n=4`, `C=96` canales.

---

## 3. Preprocesamiento del Dataset (lo que realmente hicieron)

### Filtrado de datos
- Excluyen scans con: slice thickness > 3mm, slices faltantes, spacing inconsistente.
- Excluyen nódulos < 3mm (clínicamente insignificantes).

### Pipeline de preprocesamiento
```
1. Resamplear → spacing isótropo 1×1×1 mm (trilinear interpolation)
2. Clip HU → [-1000, 400]
3. Normalización → MinMax a [0.0, 1.0]
4. Extraer ROI → cubo 64×64×64 vóxeles centrado en el centroide del nódulo
```

### Etiquetado binario
| Score promedio radiológico | Etiqueta |
|---------------------------|---------|
| < 3 (promedio de 4 radiólogos) | Benigno (0) |
| ≥ 3 | Maligno (1) |
| = 3 exacto | **Excluido** (ambigüedad) |

**Resultado tras filtrado:** 554 benignos + 450 malignos = **1,004 nódulos** de los 1,186 totales.

### Augmentation durante training
- Rotaciones aleatorias ±15° en cada eje.
- Escala aleatoria entre 0.9–1.1×.
- Traslaciones ±5 vóxeles en cada eje.
- Flip horizontal/vertical con p=0.5.

---

## 4. Split y Validación

- **Split a nivel de PACIENTE** (no por nódulo) → evita data leakage.
- Ratio: **80% train / 10% val / 10% test**.
- **10-fold cross-validation** con splits por paciente y muestreo estratificado por clase.
- Métricas reportadas = promedio de los 10 folds.

---

## 5. Hiperparámetros de Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Épocas | 300 |
| Batch size | 8 |
| Optimizer | SGD (momentum=0.9) |
| LR inicial | 0.01 |
| LR decay | → 0.001 (época 60) → 0.0001 (época 120) |
| Weight decay (L2) | 0.0001 |
| Dropout | 0.5 |
| GPU | NVIDIA RTX 4060 16 GB |
| Tiempo de training | ~12 horas (10-fold) |

---

## 6. Resultados (LUNA16-K dataset — mejor resultado)

| Modelo | Accuracy | Recall | Specificity | AUC | F1 |
|--------|----------|--------|-------------|-----|----|
| VGG16 | 82.35% | 81.64% | 79.52% | 0.79 | 81.64% |
| ResNet50 | 83.36% | 81.96% | 80.88% | 0.80 | 81.96% |
| DenseNet | 84.00% | 81.96% | 81.31% | 0.81 | 81.96% |
| Swin-T | 84.35% | 84.02% | 85.55% | 0.84 | 84.02% |
| ConvNeXt | 85.58% | 84.96% | 85.45% | 0.85 | 84.96% |
| SpikingResformer | 89.96% | 88.96% | 89.19% | 0.89 | 88.96% |
| **DCSwinB** | **90.96%** | **90.56%** | **89.65%** | **0.94** | **90.56%** |

### Ablation study — impacto de Conv-MLP

| Variante | Accuracy | Recall | AUC |
|---------|----------|--------|-----|
| Swin-Tiny (baseline) | 87.94% | 85.56% | 0.92 |
| DCSwinB sin Conv-MLP | 88.56% | 87.02% | 0.93 |
| **DCSwinB con Conv-MLP** | **90.96%** | **90.56%** | **0.94** |

→ Conv-MLP añade **+2.4% accuracy** y **+3.5% recall** sobre DCSwinB sin él.

### Eficiencia computacional vs Swin-T

- **−19.8% parámetros**
- **−24.4% FLOPs**
- **16% más rápido** en inferencia (−2.6 ms/imagen)

---

## 7. Limitaciones que reconocen

1. **Mayor uso de memoria** por la arquitectura dual-branch.
2. **Dependencia del preentrenamiento** en LUNA16 — puede degradarse en otros dominios.
3. **Generalización incierta** a CTs de diferente calidad, ruido o resolución.
4. El modelo aún opera en **2D** (pese a que los datos son 3D) — trabajo futuro.

---
---

# Cómo nosotros podemos implementar los mejores métodos de este paper

> Esta sección adapta las lecciones del paper al **Experto 4 (LUNA16 3D)** del sistema MoE del proyecto, respetando las restricciones del `consigna.md`.

## Restricciones del proyecto (no negociables)

| Requisito | Valor |
|-----------|-------|
| Resize | **64×64×64** ✅ (igual al paper) |
| Normalización HU | **[-1000, 400]** ✅ (igual al paper) |
| Gradient checkpointing | **Obligatorio** |
| Arquitectura guía | ViViT-Tiny / R3D-18 / MC3-18 |
| Clases | 2 (nódulo / no nódulo) |
| Output del experto | Probabilidades para el router MoE |

---

## A. Preprocesamiento — copiar exactamente del paper

El paper valida este pipeline en 1,004 nódulos con resultados SOTA:

```python
# Paso 1: Resamplear a 1×1×1 mm (trilinear)
resample_to_isotropic(img, spacing=(1.0, 1.0, 1.0))

# Paso 2: Clip HU
img_clipped = np.clip(arr, -1000.0, 400.0)

# Paso 3: MinMax normalize
img_norm = (img_clipped - (-1000.0)) / (400.0 - (-1000.0))  # → [0, 1]

# Paso 4: Extraer ROI centrada en el nódulo
roi = extract_cube(img_norm, center_voxel, size=64)  # 64×64×64
```

> [!IMPORTANT]
> El paper **resamplea a 1×1×1 mm ANTES de extraer el ROI de 64³**. Esto es distinto a solo hacer resize. El resampleo preserva el tamaño físico real del nódulo en vóxeles.

---

## B. Etiquetado — estrategia del paper (más limpia que candidatos)

En lugar de usar `candidates_V2.csv` (muchos falsos negativos), usar `annotations.csv`:

```python
# Cargar anotaciones
annotations = pd.read_csv('annotations.csv')
# seriesuid, coordX, coordY, coordZ, diameter_mm

# Cruzar con malignancy scores del LIDC-IDRI
# Si malignancy_avg >= 3 → label = 1 (maligno)
# Si malignancy_avg < 3  → label = 0 (benigno)
# Si malignancy_avg == 3 → EXCLUIR (ambigüedad)
```

Dataset resultante esperado: ~1,004 nódulos (554 benignos + 450 malignos).

---

## C. Arquitectura del Experto — opciones por restricción de hardware

El paper usa DCSwinB (híbrido Swin + CNN). Para el proyecto, las opciones en orden de **recomendación**:

### Opción 1 (✅ Recomendada): R3D-18 + adaptaciones del paper
```python
import torchvision.models.video as vm

# Cargar R3D-18 preentrenado
model = vm.r3d_18(pretrained=True)
model.fc = nn.Linear(512, 2)

# Activar gradient checkpointing (obligatorio)
model.layer3 = torch.utils.checkpoint.checkpoint_sequential(model.layer3, segments=2)
model.layer4 = torch.utils.checkpoint.checkpoint_sequential(model.layer4, segments=2)
```

**Por qué R3D-18:** Es 3D nativo (como los datos), fácil de implementar, compatible con gradient checkpointing, y en la tabla del paper ConvNeXt (primo de R3D) alcanza 85.58%. Con gradient checkpointing cabe en 12 GB.

### Opción 2 (💡 Avanzada): Swin-Tiny 3D (más cercana al paper)
```python
import timm

# Swin Transformer adaptado a 3D (requiere monai o implementación custom)
from monai.networks.nets import SwinUNETR
# o usar: timm.create_model('swin_tiny_patch4_window7_224', ...)
# adaptado con inflación 3D
```

> [!NOTE]
> El paper usa Swin-Tiny como backbone del DCSwinB. Si se implementa Swin-Tiny 3D con gradient checkpointing, se replican las condiciones del paper directamente (AUC 0.92 solo con Swin-T base).

### Opción 3 (⚡ Mínima viable): MC3-18
```python
model = vm.mc3_18(pretrained=True)  # Mixed Convolutions 3D
model.fc = nn.Linear(512, 2)
```

---

## D. Augmentation 3D — copiar del paper

```python
import monai.transforms as mt

train_transforms = mt.Compose([
    mt.RandRotate90d(keys=['image'], prob=0.5, spatial_axes=(0, 1, 2)),
    mt.RandRotated(keys=['image'], range_x=0.26, range_y=0.26, range_z=0.26, prob=0.5),
    # range = 15° = 0.26 rad
    mt.RandZoomd(keys=['image'], min_zoom=0.9, max_zoom=1.1, prob=0.5),
    mt.RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.3),
    mt.RandFlipd(keys=['image'], spatial_axis=0, prob=0.5),
    mt.RandFlipd(keys=['image'], spatial_axis=1, prob=0.5),
    mt.RandFlipd(keys=['image'], spatial_axis=2, prob=0.5),
])
```

---

## E. Split y validación — exactamente como el paper

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=10)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=patient_ids)):
    # patient_ids evita data leakage entre nódulos del mismo paciente
    ...
```

> [!WARNING]
> El paper hace split **por paciente**, no por nódulo. Un paciente puede tener múltiples nódulos — si no se agrupa por paciente, ocurre data leakage severo.

---

## F. Hiperparámetros adaptados al proyecto

| Parámetro | Paper original | Adaptación para Colab/MoE |
|-----------|---------------|--------------------------|
| Épocas | 300 | 50–100 (Fase 1 expertos) |
| Optimizer | SGD (momentum=0.9) | AdamW (más estable con LR bajo) |
| LR | 0.01 → 0.001 → 0.0001 | 1e-3 → reducir con ReduceLROnPlateau |
| Batch size | 8 | 4–8 (con gradient accumulation ×4) |
| Weight decay | 0.0001 | 0.0001 |
| Dropout | 0.5 | 0.3–0.5 |
| Loss | No especifica | BCEWithLogitsLoss o CrossEntropyLoss |
| Precision | FP32 | **FP16 (AMP)** — obligatorio |
| Gradient checkpoint | No usa | **Obligatorio** para expertos 3D |

---

## G. Loss function — para imbalance leve (554 vs 450)

El desbalance en LUNA16 es leve (1.23:1). Se puede usar:

```python
# Opción simple (recomendada por el equilibrio):
criterion = nn.CrossEntropyLoss()

# Opción con peso si el desbalance empeora:
weights = torch.tensor([450/1004, 554/1004])  # inversamente proporcional
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
```

---

## H. Métricas a monitorear (alineadas con el paper)

```python
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Métricas del paper para reportar en el informe final:
# Accuracy, Recall (Sensitivity), Specificity, AUC, Precision, F1
# Target del proyecto: F1-Macro > 0.65 (aceptable), > 0.65 (full)
```

---

## I. Resumen de decisiones para el EDA (lo que hacer en el notebook)

| Decisión | Opción elegida | Justificación |
|---------|---------------|--------------|
| **Formato salida** | `.mha` 64³ | Consistencia con resto del pipeline MoE |
| **Normalización** | Clip [-1000,400] → MinMax [0,1] | Validado en el paper SOTA |
| **Resampleo** | 1×1×1 mm antes de extraer ROI | Preserva tamaño físico real del nódulo |
| **Etiquetado** | Score radiológico ≥3 = maligno | Más limpio que candidates_V2.csv |
| **Excluir score=3** | Sí | Elimina ambigüedad, mejora señal |
| **Split** | Por paciente (GroupKFold) | Evita data leakage |

> [!TIP]
> El EDA debe generar el CSV `processed_labels.csv` con columnas: `seriesuid, mha_path, label (0/1), malignancy_avg, diameter_mm`. Este CSV es la entrada directa del notebook de training del Experto 4.
