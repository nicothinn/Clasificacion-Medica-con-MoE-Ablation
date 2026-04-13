# 💡 Tips y Advertencias del Profesor (Video 3)

Este es el análisis y plan maestro ultra detallado, consolidando absolutamente toda la información: tus clases, la asesoría magistral (donde el profesor reveló detalles técnicos críticos que no están en el PDF) y el documento oficial del proyecto.

El profesor fue sumamente enfático en la asesoría: **si el flujo de datos entre el Router y los Expertos se hace mal, se pierden horas de computación y el proyecto fracasa.**

A continuación, tienes la radiografía completa del proyecto, dividida en la arquitectura real, los *insights* ocultos, y el análisis de riesgos y estrategias.

---

# 🚀 PLAN MAESTRO: Proyecto Final MoE + Ablation Study

## 1. LA ARQUITECTURA REAL (El "Secreto" de la Asesoría)

El mayor punto de confusión en la asesoría fue cómo se comunican los datos. El profesor aclaró que existen **DOS CAMINOS PARALELOS** para la información. Entender esto es el 50% del proyecto:

### Camino A: El Router (Procesa EMBEDDINGS, no imágenes)

- **El Problema:** El Router (Vision Transformer) va a recibir un *batch mixto* (imágenes 2D de [224x224] y volúmenes 3D de [64x64x64]). Si le metes eso crudo, el modelo colapsa por inconsistencia dimensional.
- **La Solución:** Hay que crear un **Preprocesador Adaptativo**. Este algoritmo detecta el `rank` del tensor (rank=4 para 2D, rank=5 para 3D). Convierte ambas modalidades en *parches* y luego en un tensor homogéneo de **192 dimensiones (embeddings/tokens)**.
- **El Router:** El ViT-Tiny recibe tensores `[Batch, 192]`. El profesor recomienda usar **4 u 8 cabezas de atención (attention heads)**, no más, para no reventar la VRAM.
- **La Salida:** El Router pasa por una capa Softmax y genera un vector de tamaño 5 (Logits). Este vector actúa como un **"Gatillo" (Switch)** que dice: *"Para esta imagen, enciendan el Experto 3 y apaguen los demás"*.

### Camino B: Los Expertos (Procesan IMÁGENES CRUDAS, no embeddings)

- **La entrada:** A los expertos **NO** les llegan los embeddings de 192 dimensiones. A ellos les llega la imagen 2D o el volumen 3D preprocesado con CLAHE.
- **La ejecución:** Gracias al "gatillo" del Router, la imagen solo pasa por **un experto a la vez** durante la inferencia. Si enviaran la imagen a los 5 expertos al tiempo, consumirían 5 veces más GPU y el sistema haría OOM (*Out Of Memory*).
- **Modelos:** Usar arquitecturas pre-entrenadas eficientes (EfficientNet-B0/B3, ConvNeXt-Tiny). **Tip del profesor:** Agreguen 1 o 2 capas convolucionales propias antes del clasificador final para hacer un *fine-tuning* verdadero hacia imágenes médicas.

---

## 2. INSIGHTS CRÍTICOS DEL PROFESOR (Lo que da el 5.0)

1. **El Preprocesamiento Espacial es innegociable:** El profesor se burló de los enfoques "vainilla" (solo hacer *resize* y *flip*). Para sacar un F1-Score alto en rayos X (NIH / Osteoarthritis), **tienen que aplicar CLAHE** (Contrast Limited Adaptive Histogram Equalization) combinado con correcciones Gamma. Si la red no puede diferenciar el hueso del tejido pulmonar por el bajo contraste, jamás encontrará un nódulo.
2. **Unidades Hounsfield (HU) en 3D (Corrección al PDF):** El PDF dice normalizar los volúmenes LUNA16 de [-1000, 400]. El profesor en la asesoría corrigió a un estudiante indicando que un rango de **[-1200, 600]** funciona mucho mejor para capturar aire (pulmones) y tejido óseo simultáneamente.
3. **Pérdida Auxiliar (Load Balancing Loss):** Es obligatorio implementar esto. Si el Router descubre que el Experto 2 es muy bueno y le manda el 100% de las imágenes (Expert Collapse), el proyecto tiene una **penalización del 40% de la nota**. La relación entre el experto más usado y el menos usado no puede superar 1.30 (`max(fi)/min(fi) < 1.30`).
4. **Cero Metadatos:** El sistema debe saber si es un pulmón, una rodilla o un lunar **solo mirando los píxeles**. Si le pasan un *string* o etiqueta al modelo para ayudarle a enrutar, les quitan el 20% de la nota.

---

## 3. EL ABLATION STUDY (La Ciencia del Proyecto)

El proyecto requiere comparar el Router ViT de Deep Learning contra 3 métodos estadísticos clásicos usando los mismos *CLS tokens* (embeddings guardados de la Fase 0).

- **ViT + Linear (Baseline):** Se entrena con la red. Consume GPU. Usa la *Auxiliary Loss*.
- **ViT + GMM (Gaussian Mixture Model):** No usa gradientes. Se ajusta con *Expectation-Maximization*. El profe advierte: Si usan matriz de covarianza `full`, son ~184K parámetros. Si falla, cambien a `diag`.
- **ViT + Naive Bayes:** Asume independencia de las 192 dimensiones. Extremadamente rápido.
- **ViT + k-NN (FAISS):** Guarda todos los embeddings de entrenamiento. Usa distancia coseno. El profe advierte sobre la "maldición de la dimensionalidad"; si k-NN no rinde bien, **apliquen un PCA para reducir las 192 dimensiones a 32 antes de usar FAISS**. *(Poner esto en el reporte suma muchísimos puntos).*

---

## 4. ANÁLISIS DE RIESGOS, BENEFICIOS Y ESTRATEGIAS

### 🔴 RIESGOS (Cuellos de Botella)

1. **Out of Memory (OOM) en la GPU:** Meter el Router ViT y 5 expertos (incluyendo 2 volumétricos en 3D) reventará la VRAM (12GB disponibles en su clúster).
2. **Cuello de botella en el Disco Duro:** Leer miles de imágenes y volúmenes simultáneamente durante el entrenamiento ralentiza todo. (Un estudiante reportó este error en la asesoría).
3. **Colapso del Router (Expert Collapse):** Que el modelo decida ignorar la pérdida auxiliar y mande todo a la clase mayoritaria (ej. NIH ChestX-ray que tiene 112k imágenes).

### 🧠 ESTRATEGIAS DE MITIGACIÓN (Plan de Acción)

1. **Para la Memoria (VRAM):**
    - Usar `Mixed Precision (FP16)` con `torch.cuda.amp.autocast`.
    - Usar **Gradient Accumulation**: En vez de un batch size de 32 (que explotaría la GPU), usen un batch size de 8 y acumulen gradientes 4 veces (`loss.backward()` pero `optimizer.step()` cada 4 iteraciones).
    - Usar **Gradient Checkpointing** obligatoriamente en los expertos 3D.
2. **Para el Disco Duro (I/O Bottleneck):**
    - Conviertan los archivos DICOM/NIfTI a formato **.MHA**. El profesor lo recomendó enfáticamente porque elimina la metadata médica inútil y acelera la carga.
    - Usar servicios Cloud: Como discutieron en la asesoría, usar *RunPod* o *Hetzner* ($1.5 USD la hora por una RTX 3090/4090) para el entrenamiento final global.
3. **Para el Balanceo de Carga:**
    - Usar un **DataLoader Proporcional**. No le pasen los datos crudos, o el dataset de 112k imágenes aplastará al de 281 volúmenes de páncreas. El DataLoader debe generar baches con porcentajes equilibrados de las 5 clases.

---

## 5. CRONOGRAMA DE EJECUCIÓN OPTIMIZADO (4 Semanas)

- **Semana 1 (Preprocesamiento y Expertos Base):**
  - Crear el `AdaptivePreprocessor` (Detector dinámico de Rank 4 o Rank 5).
  - Aplicar CLAHE y transformación a `.MHA`.
  - Entrenar los 3 expertos 2D de manera totalmente independiente y guardar los pesos (`.pth`).
- **Semana 2 (Expertos 3D y Fase 0 de Ablation):**
  - Entrenar los 2 expertos 3D (Pancreas y Pulmón) con Gradient Checkpointing.
  - Conectar el Backbone ViT congelado y hacer una pasada a todo el dataset para **guardar los CLS Tokens (192d) en disco como tensores Numpy**.
- **Semana 3 (Ablation Study y Ensamble):**
  - *Trabajo en CPU:* Ejecutar GMM, Naive Bayes y FAISS sobre los tokens guardados. Llenar la tabla del reporte técnico.
  - *Trabajo en GPU:* Ensamblar el Router ViT con la *Auxiliary Loss*. Entrenar la lógica de ruteo (el switch) conectando los expertos pre-entrenados.
- **Semana 4 (Dashboard y Reporte ABET):**
  - Crear la interfaz en Streamlit (carga de imagen, detección de modalidad automática, heatmap de atención del ViT y activación visual del experto seleccionado).
  - Redactar el reporte de 7 páginas. **Importante:** Incluir gráficos del Loss Auxiliar demostrando que la relación max/min fue menor a 1.30. No pongan código en el PDF.
