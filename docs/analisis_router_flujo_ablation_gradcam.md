# Análisis técnico: flujo del router, límites del entrenamiento conjunto con ablation, e interpretación Grad-CAM NIH vs Osteo

Este documento enlaza tres piezas: el pipeline de `05_Router_Profesor_Fase3_Solo.ipynb`, los requisitos de la consigna (MoE + routers estadísticos + ablation), y el diagnóstico visual de `06_GradCAM_Diagnostico_NIH_vs_Osteo.ipynb`. Los valores numéricos concretos (épocas, `ALPHA_AUX`, umbrales de entropía) dependen de la celda de hiperparámetros que ejecutes; aquí se describe la **lógica** del sistema.

---

## 1. Flujo de datos y transformaciones (notebook 05)

### 1.1 Entrada única, sin metadatos de modalidad

El diseño del proyecto exige que el sistema reciba solo la imagen o el volumen (formatos soportados por el preprocesador). La **modalidad 2D vs 3D** se infiere de la forma del tensor tras cargar el archivo: rank 4 para 2D, rank 5 para 3D.

### 1.2 `AdaptivePreprocessor`

Función central: homogeneizar resolución y rango numérico antes del backbone del router.

| Rama | Operación típica | Notas |
|------|-------------------|--------|
| 2D (PNG, JPG, JPEG) | `Resize` a 224×224, `ToTensor` | Entrada al `VisionRouter` como tensores `[C,H,W]` en el batch del router. |
| 2D desde `.mha` / arrays 2D | Normalización min–max o escala, interpolación a 224×224, repetición a 3 canales si hace falta | Alineado con rutas mixtas del proyecto. |
| 3D (`.mhd`, NIfTI, etc.) | Ventana HU típica `[-1000, 400]`, clip, normalización a `[0,1]`, `trilinear` a 64³ | Salida `[1, D, H, W]` para el pipeline 3D del router. |

Para el **experto LUNA (R3D-18)** el volumen en `[0,1]` se puede convertir a 3 canales con estadísticas **Kinetics** antes del forward del experto (`luna_1ch_to_kinetics_3ch`), coherente con el notebook de entrenamiento 3D del experto 4.

### 1.3 Construcción del dataset del router

1. **Índice de etiquetas de tarea** por dataset (`build_dataset_label_index`), necesario cuando el loader incluye etiquetas de tarea para `L_task`.
2. **Filtrado NIH** hacia las clases que usa el experto NIH en el MoE (subconjunto multietiqueta del CSV).
3. **`WeightedRandomSampler`**: pesos por muestra para **equilibrar dominios** (no confundir con la métrica de balanceo del router; aquí se trata de muestreo).
4. **Opcional**: filtro de **entropía de Shannon** en NIH y Osteo para descartar muestras de muy baja o muy alta entropía (ruido o casos extremos), según umbrales definidos en la celda de entrenamiento.

### 1.4 Arquitectura del router (Visión)

- **`SwitchablePatchEmbed`**: parches 2D (MONAI) y 3D (MONAI), dimensión de embedding alineada con ViT-Tiny, **token CLS** y **embedding posicional** sobre la secuencia rellenada (padding por batch).
- **`VisionRouter`**: `vit_tiny_patch16_224` (timm) con `patch_embed` sustituido por identidad; el forward aplica solo los **bloques ViT** sobre la secuencia ya embebida; la cabeza `router_head` es lineal sobre el **CLS** final.

### 1.5 Configuración de entrenamiento (fases y pérdidas)

| Componente | Rol |
|--------------|-----|
| `L_routing` | `CrossEntropy` entre logits del router y el **índice de experto** (dominio de origen). |
| `L_task` | Solo cuando hay expertos en GPU y el batch trae **etiqueta de tarea** compatible: logits del experto elegido (ponderados por el gating) vs etiqueta de tarea. **NIH multietiqueta** queda fuera de este término en el código actual (conjunto `_SINGLE_LABEL_DATASETS`). |
| `L_aux` | Pérdida de balanceo estilo Switch Transformer: `N * sum_i f_i * P_i` con `f_i` frecuencia de experto elegido en el batch y `P_i` media de probabilidades. |
| `alpha_aux` | Peso de `L_aux` (valor en celda de hiperparámetros, p. ej. orden `1e-2`). |
| Warmup | Primeras épocas **sin** `L_task` (solo routing + aux) para estabilizar el gating antes del feedback de expertos. |
| Fases | Típicamente: (1) cabeza del router o backbone parcial; (2) fine-tune con router más completo y, si aplica, solo cabezas de expertos con LR bajo. |

### 1.6 Balanceo: qué se mide y qué se optimiza

- **Métrica de desbalance**: ratio `max(f_i) / min(f_i)` sobre las frecuencias de enrutamiento por experto (train y validación); objetivo de proyecto cercano a **≤ 1.30** cuando la consigna lo exige.
- **Entropía** de las probabilidades del router: control de incertidumbre (no es lo mismo que el ratio anterior).
- El **sampler** equilibra **cuántas muestras por dominio entran al batch**; `L_aux` empuja a que **las decisiones del router** no colapsen en un solo experto.

---

## 2. Por qué no se entrena “todo junto”: expertos + ablation + cabezas estadísticas en un solo paso

La consigna pide, en el sistema completo:

1. Un router **deep learning** (ViT / CvT / Swin) que produzca embeddings.
2. **Tres routers estadísticos** (GMM, Naive Bayes, k-NN) sobre **los mismos embeddings**.
3. Un **ablation study** comparando los cuatro mecanismos.

El notebook **`05_Router_Profesor_Fase3_Solo`** está planteado de forma **acotada**: entrena el **VisionRouter con pérdida de routing + aux + (opcional) feedback de expertos**, y declara explícitamente **sin ablation** en el propio flujo. Eso no es un fallo: son **problemas distintos** que compiten por tiempo, VRAM y diseño experimental.

### 2.1 Razones técnicas

| Motivo | Explicación |
|--------|-------------|
| **Dos objetivos distintos** | Entrenar el ViT-router con gradientes end-to-end es un problema de **clasificación / gating supervisado**. Ajustar GMM, NB o k-NN sobre embeddings es **estimación de densidad o clasificación en espacio fijo**; no requiere backprop a través del mismo loop que el MoE completo. |
| **Embeddings “congelados” para estadísticos** | Lo habitual en ablation es: (1) extraer `z = CLS` (o pooling) para todo el conjunto; (2) **congelar** el backbone; (3) entrenar o ajustar GMM / NB / k-NN **offline** o en celdas separadas. Mezclarlo en el mismo `fit` que el MoE multiplica complejidad y depuración. |
| **Validación del router** | En 05, la validación rápida usa **cabeza lineal sobre embeddings CLS cacheados** (`unified_cls_tokens.npz`). Las cabezas estadísticas necesitan **otra tabla de resultados** (log-loss, accuracy de routing por método). |
| **VRAM y tiempo** | Expertos 3D + router + cinco cabezas + acumulación: el clúster de 12 GB por GPU se satura; la consigna ya exige FP16, acumulación y checkpointing para expertos 3D. |
| **NIH multietiqueta** | `L_task` no se aplica igual que en dominios single-label; el entrenamiento “profesor” ya filtra ese caso en el código. Añadir ablation encima no simplifica el gráfico de dependencias. |

### 2.2 Cómo encaja con el informe (recomendación de presentación)

- **Sección A**: Pipeline MoE + entrenamiento del router (notebook 05).
- **Sección B**: Extracción de embeddings unificados + **entrenamiento/evaluación offline** de GMM, Naive Bayes y k-NN (y FAISS si aplica), con la **misma** definición de `z`.
- **Sección C**: Tabla comparativa (routing accuracy, ratio de balanceo, tiempo de inferencia) = **ablation formal**.

Así se cumple la consigna sin forzar un único bucle de optimización imposible de depurar.

---

## 3. Resultados e interpretación: `06_GradCAM_Diagnostico_NIH_vs_Osteo`

### 3.1 Propósito del notebook

Comparar, sobre **las mismas radiografías**, qué regiones activa:

- el **experto NIH** (Swin-Tiny, multietiqueta), y  
- el **experto Osteo** (VGG16-BN, rodilla),

usando **Grad-CAM** (`pytorch_grad_cam`) con capas objetivo alineadas a cada arquitectura.

El notebook reutiliza **rutas de datos y pesos** coherentes con el proyecto (`DATASET_ROOTS`, `WEIGHTS_DIR`). La versión alineada con el router usa el **mismo criterio de entrada** (resize, normalización por modalidad) que el flujo del router en la iteración en que se diseñó el diagnóstico: el objetivo es que **ni el NIH ni el Osteo** vean una representación “distinta” solo porque el script de diagnóstico simplifique el preprocesado; si el router y los expertos se entrenaron con un pipeline global (ImageNet, CLAHE, entropía, etc.), el diagnóstico debe **replicarlo** para que la comparación sea justa.

### 3.2 Qué muestran los mapas (lectura típica)

Cuando el **router** confunde Osteo con NIH (o al revés), los expertos especializados pueden seguir dando **predicciones de su dominio** sobre la imagen “equivocada”:

- **NIH sobre rodilla**: a menudo aparecen activaciones **puntuales** en zonas de **baja densidad entre huesos**; el modelo de tórax interpreta patrones tipo “líquido / pleura” en estructuras que en rodilla son **espacio articular**.
- **Osteo sobre tórax**: a veces el mapa se reparte en **bordes y hombros** o bandas horizontales; la VGG busca “ranuras” tipo rodilla en contraste global que en tórax corresponden a **diafragma o costillas**.

Eso **no** contradice que el router use un CLS global: los expertos siguen siendo **detectores de textura y forma local** entrenados en su dominio. La confusión del router aparece porque **en espacio de embedding** (y a veces en apariencia radiológica) hay **solapamiento** entre patrones de distintas modalidades cuando solo se mira intensidad y borde.

### 3.3 Vínculo con el MoE

El diagnóstico Grad-CAM **no entrena** el router; **explica** por qué dos expertos 2D pueden “verse bien” en sus propias métricas pero el **enrutador** necesita más épocas, mejor balanceo (`L_aux`, sampler), o representaciones más separables (fine-tune, más datos, o backbone distinto). Sirve como evidencia cualitativa para el reporte junto a matrices de confusión del router.

---

## 4. Lectura rápida en cadena

```text
ZIP / Drive → AdaptivePreprocessor → tensores 2D/3D homogéneos
                    ↓
         VisionRouter (ViT-Tiny + CLS) → logits de experto
                    ↓
    L_routing + α·L_aux + (opcional) L_task [single-label domains]
                    ↓
    Métricas: val_acc routing, ratio max/min, entropía, CM 5×5
                    ↓
    (Otro bloque de trabajo) Embeddings → GMM / NB / k-NN → ablation

Diagnóstico paralelo: mismas imágenes → Grad-CAM NIH vs Osteo → interpretación de confusión dominio-cruzado
```

---

## 5. Referencias internas del repositorio

| Artefacto | Uso |
|-----------|-----|
| `consigna.md` | MoE, preprocesado por dataset, routers estadísticos, ablation, balanceo. |
| `05_Router_Profesor_Fase3_Solo.ipynb` | Entrenamiento router + expertos congelados + métricas. |
| `06_GradCAM_Diagnostico_NIH_vs_Osteo.ipynb` | Comparación visual expertos NIH vs Osteo sobre mismas entradas. |

---

*Documento generado para el reporte técnico del proyecto; ajustar números de épocas y rutas de checkpoint a la versión exacta del notebook que ejecutes.*
