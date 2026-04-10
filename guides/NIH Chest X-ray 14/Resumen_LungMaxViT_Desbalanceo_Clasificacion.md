# 📋 Resumen: LungMaxViT — Clasificación Multilabel de Enfermedades Pulmonares con Datos Desbalanceados
>
> **Fuente:** *"Explainable Hybrid Transformer for Multi-Classification of Lung Disease Using Chest X-Rays"*  
> Fu, Lin, Du, Tavares & Liang — Scientific Reports (2025) | DOI: 10.1038/s41598-025-90607-x  
> **Datasets:** COVID-QU-Ex (3 clases) y **ChestX-ray14** (15 clases, 112K imágenes)

---

## 1. 🔥 El Problema: ¿Qué tan grave era el desbalanceo?

El paper trabaja principalmente con **ChestX-ray14**, cuyo desbalanceo es uno de los más extremos en la literatura:

| Clase | N° Muestras |
|---|---|
| **No Finding** (clase dominante) | **60,361** |
| Infiltration | 9,547 |
| Atelectasis | 4,215 |
| Effusion | 3,955 |
| Nodule | 2,705 |
| Pneumothorax | 2,194 |
| Mass | 2,139 |
| Consolidation | 1,310 |
| Pleural Thickening | 1,126 |
| Cardiomegaly | 1,093 |
| Emphysema | 892 |
| Fibrosis | 727 |
| Edema | 628 |
| Pneumonia | 322 |
| **Hernia** (clase minoritaria) | **110** |

> [!CAUTION]
> La ratio de desbalanceo es de ~549:1 entre `No Finding` y `Hernia`. Además, el problema es **multiclase** (no multilabel puro), lo que significa que el desbalanceo afecta directamente al aprendizaje de la función de pérdida estándar.

El dataset COVID-QU-Ex, por contraste, está **relativamente balanceado** (~11K imágenes por clase), lo que explica por qué los resultados en ese dataset son mucho mejores que en ChestX-ray14.

---

## 2. 🛠️ Estrategias Aplicadas para Tratar el Desbalanceo

### 2.1 — CLAHE (Contrast Limited Adaptive Histogram Equalization) ⭐ Clave

Esta fue la técnica de preprocesamiento más importante del paper para las clases raras.

**¿Cómo funciona?**

1. Divide la imagen en regiones pequeñas (*tiles*)
2. Calcula el histograma de intensidades dentro de cada región independientemente
3. Realiza ecualización de histograma *local* (no global) con un límite de contraste
4. Une las regiones mediante interpolación bilineal

**¿Por qué ayuda con el desbalanceo?**  
Las enfermedades raras (como Hernia o Fibrosis) tienen señales muy sutiles en la radiografía. CLAHE hace visibles esos detalles que de otro modo el modelo ignoraría, dando al modelo más "información útil" de los pocos ejemplos disponibles. El paper menciona explícitamente que Hernia obtuvo AUC=0.997 en parte gracias al preprocesamiento.

```
Antes CLAHE: features de enfermedades raras difíciles de ver
Después CLAHE: mejor visibilidad de nódulos, texturas, sombras
                → el modelo puede aprender de menos ejemplos
```

### 2.2 — Gaussian Filtering / Denoising

Se aplicó filtrado Gaussiano para eliminar ruido de las imágenes:

- Fuentes de ruido en radiografías: artefactos de equipos, calibración incorrecta, escáneres de baja calidad
- Al reducir el ruido, las características patológicas relevantes (especialmente de clases minoritarias) son más discernibles
- **Tradeoff importante:** denoising excesivo puede borrar lesiones sutiles → se usó Gaussian Filter con parámetros conservadores

> [!WARNING]
> El paper reconoce que durante el denoising "ciertas características detalladas de la imagen podrían perderse, especialmente lesiones sutiles". Es una técnica de doble filo cuando hay pocas muestras de una clase.

### 2.3 — Data Augmentation en el Pipeline de Entrenamiento

Se aplicó augmentation específicamente durante el entrenamiento para aumentar la variabilidad en clases con pocas muestras:

| Técnica | Valor/Configuración |
|---|---|
| **Rescaling** (zoom in/out) | ratio 5/6 a 1/6 |
| **Zoom range** | 0.75 – 0.95 |
| **Rotation range** | 1 (ligera rotación) |
| **Horizontal flip** | ✅ Activado |
| **Resize final** | 224×224 pixels |

**Flipping en detalle:** Se realizaron flips horizontales y verticales aleatorios para simular que el modelo "ve" pulmones desde diferentes perspectivas, reduciendo su sensibilidad a la orientación.

### 2.4 — Selección de Arquitectura como Estrategia Implícita

El paper argumenta que usar MaxViT (y su mejora LungMaxViT) como backbone es en sí una estrategia para lidiar con el desbalanceo, porque:

- El **mecanismo de atención multi-eje** (local block + global grid) permite capturar features tanto locales (lesiones pequeñas en clases raras) como globales (patrones de distribución pulmonar)
- La atención **adapta sus pesos dinámicamente** según el contenido de la imagen, lo que lo hace más robusto a ejemplos atípicos de clases minoritarias que modelos CNN puros

> [!NOTE]
> A diferencia del paper anterior (Liz et al., 2022), este paper **NO usa** Weighted Loss ni WeightedRandomSampler explícitamente. La estrategia principal contra el desbalanceo es el preprocesamiento (CLAHE + denoising) y la arquitectura más potente.

---

## 3. 🏗️ Arquitecturas Evaluadas

Se compararon 5 modelos con Transfer Learning desde ImageNet-1K:

| Modelo | Tipo | Fortaleza principal |
|---|---|---|
| **ResNet50** | CNN puro | Baseline clásico, skip connections |
| **MobileNetV2** | CNN eficiente | Depth-wise separable convolutions, bajo costo |
| **ViT** | Transformer puro | Self-attention global, necesita mucho pre-entrenamiento |
| **MaxViT** | Híbrido CNN+Transformer | Atención local (block) + global (grid), complejidad lineal |
| **LungMaxViT** ✅ | Híbrido mejorado | MaxViT + bloque CNN inicial + SE block |

### La arquitectura ganadora: LungMaxViT

Consiste en 3 bloques principales encadenados:

```
Imagen 224×224
      ↓
[1] Initial State Block (CNN)
    → 3 segmentos conv: Conv2d → BatchNorm2d → GELU
    → stride=2 para reducir dimensiones
      ↓
[2] SE Block (Squeeze-and-Excitation)
    → Global Average Pooling → comprime a 1 canal
    → Reduce channels de 64 a 16 → amplifica features detalladas
    → Recalibra inter-channel relationships adaptativamente
      ↓
[3] MaxViT Blocks (Transformer Multi-Axis)
    → MBConv (Mobile Inverted Bottleneck) → positional encoding implícito
    → Block Attention (atención LOCAL en ventanas pequeñas)
    → Grid Attention (atención GLOBAL dilatada)
      ↓
Clasificación final (15 clases para ChestX-ray14)
```

**Innovación clave del SE Block:** Reduce canales de 64→16 antes de recalibrarlos. Esto **amplifica los pesos de features finas y detalladas**, que son justamente las que corresponden a enfermedades raras con señales sutiles.

### Configuración de entrenamiento

| Parámetro | Valor |
|---|---|
| Optimizer | **SGD** (no Adam) |
| Learning rate | 0.001 |
| Momentum | 0.9 |
| Épocas | 150 |
| Batch size | 32 |
| Image size | 224×224 |
| Pre-entrenamiento | ImageNet-1K |
| Fine-tuning | Completo (todos los pesos) |

> [!NOTE]
> Usan **SGD con momentum** en lugar de Adam. El paper no justifica explícitamente esta elección, pero SGD con momentum suele generalizar mejor cuando se tiene un dataset desbalanceado porque no adapta el learning rate por parámetro (lo cual puede sobre-ajustar rápidamente a la clase mayoritaria).

---

## 4. 📊 Resultados: ¿Qué modelo fue el mejor?

### ChestX-ray14 (dataset con desbalanceo severo)

| Modelo | AUC Global | F1-score Global |
|---|---|---|
| **LungMaxViT** | **0.932** | **0.707** |
| MaxViT | 0.926 | — |
| MobileNetV2 | 0.876 | — |
| ViT | 0.874 | — |
| ResNet50 | 0.873 | — |

### Por clase en ChestX-ray14 (comparación LungMaxViT vs MaxViT)

| Clase | N° Muestras | AUC MaxViT | AUC LungMaxViT | Mejora |
|---|---|---|---|---|
| Hernia | 110 | 0.99 | **0.997** | +0.7% |
| Edema | 628 | 0.99 | **0.994** | +0.4% |
| Emphysema | 892 | 0.98 | **0.989** | +0.9% |
| Cardiomegaly | 1,093 | 0.98 | **0.990** | +1.0% |
| Consolidation | 1,310 | 0.95 | **0.958** | +0.8% |
| Nodule | 2,705 | 0.87 | **0.881** | **+1.1%** ← mayor mejora relativa |
| Infiltration | 9,547 | 0.81 | **0.816** | +0.6% |
| No Finding | 60,361 | 0.81 | **0.820** | +1.0% |

### COVID-QU-Ex (dataset balanceado — referencia)

| Modelo | Accuracy | AUC | F1-score |
|---|---|---|---|
| **LungMaxViT** | **96.8%** | **98.3%** | **96.7%** |
| MaxViT | 96.5% | — | 96.5% |
| MobileNetV2 | 95.7% | — | 95.3% |
| ViT | 94.6% | — | 94.4% |
| ResNet50 | 94.5% | — | 94.4% |

### Comparación con el estado del arte en ChestX-ray14

| Modelo | Mean AUC |
|---|---|
| **LungMaxViT (este paper)** | **0.932** |
| Z-Net | 0.858 |
| Guan et al. | 0.822 |
| CheXNet | 0.818 |
| Wang et al. (DenseNet-121) | 0.813 |
| Thorax-Net | 0.787 |

---

## 5. 📉 Limitaciones Identificadas — El Desbalanceo que Persiste

A pesar de las mejoras, el paper reconoce que el desbalanceo no se resolvió completamente:

- **`Infiltration`** (9,547 muestras): F1-score = 0.45 — pobre a pesar de ser la clase más frecuente después de `No Finding`
- **`No Finding`** (60,361 muestras): F1-score = 0.46 — el modelo tiene dificultades para discriminar casos normales
- **`Atelectasis`** (4,215 muestras): F1-score = 0.54 — rendimiento mediocre

> [!WARNING]
> La paradoja del desbalanceo en este paper: las clases con menos muestras (Hernia, Pneumonia, Emphysema) obtienen **mejores AUC** que las clases de tamaño mediano. Esto sugiere que las enfermedades raras tienen patrones más distintivos y fáciles de aprender, mientras que enfermedades comunes como Infiltration o Atelectasis tienen features que se solapan con otras clases.

---

## 6. 🔑 ¿Qué Diferenció los Mejores Resultados? Jerarquía de Impacto

```
Impacto en AUC / F1 global (aprox.)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Arquitectura: MaxViT → LungMaxViT            +0.6 pts AUC
   (MaxViT: 0.926 → LungMaxViT: 0.932)
   █████████████████████████████████

2. CLAHE + Gaussian Denoising                   Alta influencia en clases raras
   (Hernia AUC: 0.997 gracias a preprocesamiento)
   ██████████████████████████

3. Transfer Learning desde ImageNet-1K          Base fundamental
   (sin TL, modelos como ViT no convergen bien)
   ████████████████████

4. Data Augmentation (flip + zoom + rotación)   Mejora moderada
   ██████████

5. SGD vs Adam                                  Diferencia sutil pero estabilizadora
   ████
```

### Tabla resumen de decisiones clave

| Técnica | ¿Usada? | Impacto en desbalanceo |
|---|---|---|
| **CLAHE** | ✅ | Alto — hace visibles features de clases raras |
| **Gaussian Denoising** | ✅ | Medio — reduce ruido que oculta lesiones pequeñas |
| **Horizontal/Vertical Flip** | ✅ | Medio — aumenta variabilidad artificial |
| **Zoom + Rotation Augmentation** | ✅ | Medio — mejora generalización |
| **SE Block (canal-wise attention)** | ✅ | Alto — amplifica features finas de clases raras |
| **Weighted Loss** | ❌ No usada | — (usaron CE estándar + SGD) |
| **WeightedRandomSampler** | ❌ No usada | — |
| **Ensemble** | ❌ No usado | — (modelo individual) |
| **Segmentación pulmonar U-Net** | ❌ No usada | — (a diferencia de Liz et al.) |
| **Transfer Learning ImageNet** | ✅ | Fundamental — punto de partida robusto |

---

## 7. 🔬 Explainability: Grad-CAM como Validación del Aprendizaje

El paper valida que el modelo aprendió correctamente mediante **Grad-CAM**:

1. Calcula gradientes de la clase objetivo respecto a la última capa convolucional
2. Genera mapas de activación ponderados por esos gradientes
3. Superpone el heatmap sobre la radiografía original

**¿Qué validó?**

- Para **COVID-19**: el modelo focaliza su atención dentro del área pulmonar (correcto)
- Para las **14 enfermedades de ChestX-ray14**: el foco coincide con las regiones de lesión conocidas
- Cuando el modelo se equivoca en clases desbalanceadas, el Grad-CAM muestra que atiende regiones incorrectas o el fondo de la imagen

> [!TIP]
> Grad-CAM es útil no solo para explicabilidad médica, sino como **diagnóstico del desbalanceo**: si el modelo focaliza en zonas irrelevantes para clases raras, indica que no tuvo suficientes ejemplos para aprender la señal correcta.

---

## 8. 💡 Comparación con el Paper Anterior y Aplicabilidad al Proyecto

### Comparación directa: Liz et al. (2022) vs Fu et al. (2025)

| Aspecto | Liz et al. (PadChest) | Fu et al. (ChestX-ray14) |
|---|---|---|
| **Dataset** | PadChest, 174 clases, 160K imgs | ChestX-ray14, 15 clases, 112K imgs |
| **Tipo de problema** | Multilabel (múltiples etiquetas/imagen) | Multiclase (una etiqueta predominante) |
| **Desbalanceo ratio** | 1:172 | 1:549 |
| **Weighted Loss** | ✅ Sí (principal estrategia) | ❌ No |
| **Sampler especial** | ❌ No | ❌ No |
| **Segmentación pulmonar** | ✅ U-Net (crítico) | ❌ No |
| **CLAHE** | ❌ No | ✅ Principal estrategia |
| **Ensemble** | ✅ CTP (5 modelos) | ❌ Modelo individual |
| **Arquitectura** | CNNs clásicas (DenseNet, EfficientNet) | MaxViT híbrido (Transformer) |
| **Mejor AUC** | 0.840 (CTP ensemble) | **0.932** (LungMaxViT solo) |

### Lo que este paper aporta a nuestro contexto (NIH ChestX-ray 14)

1. **LungMaxViT es la arquitectura más directamente aplicable a nuestro proyecto.** El nuestro usa MaxViT-Tiny — esta es su versión mejorada con el Initial State Block + SE Block.

2. **CLAHE adaptativo es más importante que la función de pérdida** en escenarios donde las features de las clases raras son sutiles. Nuestro notebook NIH ya implementó CLAHE adaptativo — es la decisión correcta.

3. **SGD con momentum puede ser una alternativa a AdamW** para mayor estabilidad en datasets desbalanceados, aunque en práctica ambos funcionan si el LR está bien calibrado.

4. **La falta de Weighted Loss se compensa con mejor preprocessing.** Este paper sugiere que si el preprocesamiento es excelente (CLAHE + denoising), se puede prescindir de técnicas de balanceo en la función de pérdida.

5. **Hernia (110 muestras) logra AUC=0.997.** Esto demuestra que con la arquitectura y preprocesamiento correctos, incluso clases con ~100 muestras pueden clasificarse bien en datasets de este tipo.

---

*Documento generado para el Proyecto 2 — Expert NIH Chest X-ray | Abril 2026*
