# Análisis de Antecedentes para la Construcción del Router MoE

> Documento de referencia técnica para el diseño del router del sistema Mixture of Experts.
> Basado en los 6 papers proporcionados por el profesor como lectura obligatoria.

---

## 1. Paper: "An Image is Worth 16×16 Words" (Dosovitskiy et al., ICLR 2021)

### 1.1 ¿Qué es?

El paper fundacional del **Vision Transformer (ViT)**. Demuestra que un Transformer estándar (sin convoluciones) aplicado directamente a secuencias de parches de imagen alcanza o supera el estado del arte en clasificación cuando se preentrena a gran escala.

### 1.2 Arquitectura clave

| Componente | Detalle |
|------------|---------|
| Entrada | Imagen dividida en parches de `P×P` (ej. 16×16). Cada parche se aplana y se proyecta linealmente a dimensión `D`. |
| CLS token | Token aprendible prepuesto a la secuencia. Su estado en la salida (`z⁰_L`) es la representación global de la imagen. |
| Position embeddings | Embeddings 1D aprendibles sumados a los parches. No se benefician de versiones 2D más complejas. |
| Encoder | `L` bloques de `MSA → MLP` con LayerNorm pre-bloque y conexiones residuales post-bloque. |
| Salida | `y = LN(z⁰_L)` → cabeza de clasificación (MLP en pretraining, lineal en fine-tuning). |

Variantes del paper:

| Modelo | Capas | `D` (hidden) | MLP size | Heads | Params |
|--------|-------|-------------|----------|-------|--------|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

### 1.3 Observaciones clave para nuestro router

1. **El CLS token ES la representación global.** Este es el vector que usaremos como embedding para el ablation study. Sale del último bloque Transformer y condensa la información de toda la imagen.

2. **ViT carece de inductive bias visual.** No tiene localidad ni equivarianza traslacional nativos. Esto explica por qué funciona mal con pocos datos, pero su fortaleza es que aprende relaciones globales desde los datos. En nuestro caso, usamos un ViT **preentrenado** (ImageNet), así que ya superó esa limitación.

3. **Pretraining a escala es lo que importa.** ViT-Tiny preentrenado en ImageNet-21k tiene embeddings suficientemente ricos para distinguir modalidades médicas. No necesitamos modelos grandes: el paper muestra que modelos pequeños preentrenados a escala transfieren bien.

4. **Fine-tuning: se quita la cabeza y se pone una nueva.** Para nuestro router: quitamos la cabeza de clasificación (`num_classes=0` en timm) y usamos directamente `z⁰_L` como embedding. Exactamente lo que ya hacemos.

5. **Attention distance varía por capa.** Las capas tempranas tienen heads con atención local (similar a CNNs), las capas profundas integran información global. Esto justifica que el CLS del último bloque tenga información semántica completa de la imagen.

6. **Resolución flexible.** ViT puede procesar imágenes de distintas resoluciones (interpolando position embeddings). Esto es relevante para nuestro sistema donde el adapter convierte volúmenes 3D en slices 2D de 224×224.

### 1.4 Decisiones de diseño que valida este paper

- Usar `vit_tiny_patch16_224` preentrenado como backbone **congelado**.
- Extraer el CLS token como embedding único para el router.
- No necesitar entrenar el ViT completo: solo usarlo como extractor de features.

---

## 2. Paper: "CvT: Introducing Convolutions to Vision Transformers" (Wu et al., ICCV 2021)

### 2.1 ¿Qué es?

Propone una arquitectura híbrida que **inyecta operaciones convolucionales** dentro del Transformer para combinar lo mejor de CNNs (localidad, invarianza) con lo mejor de Transformers (atención dinámica, contexto global).

### 2.2 Arquitectura clave

CvT tiene tres innovaciones principales:

| Componente | Qué hace | Por qué importa |
|------------|----------|-----------------|
| **Convolutional Token Embedding** | Reemplaza el patch embedding lineal por una convolución con overlap y stride. | Captura contexto local desde el inicio y permite reducir tokens progresivamente (como downsampling en CNNs). |
| **Convolutional Projection** | Reemplaza la proyección lineal Q/K/V por convolución depth-wise separable. | Modela relaciones espaciales locales antes de la atención, reduciendo ambigüedad semántica. |
| **Multi-stage hierarchy** | 3 etapas con resolución decreciente y dimensión creciente. | Similar a pirámide de CNNs: features de bajo nivel → alto nivel. |

Variantes:

| Modelo | Params | FLOPs | ImageNet Top-1 |
|--------|--------|-------|----------------|
| CvT-13 | 20M | 4.5G | 81.6% |
| CvT-21 | 32M | 7.1G | 82.5% |
| CvT-W24 | 277M | 60.9G | 87.7% (con IN-22k) |

### 2.3 Observaciones clave para nuestro router

1. **Elimina la necesidad de position embeddings.** Las convoluciones inyectan información posicional de forma implícita. CvT sin position embeddings rinde **igual o mejor** que con ellos (81.6% vs 81.5%). Esto simplifica el diseño para entradas de resolución variable.

2. **Rendimiento superior con menos parámetros que ViT.** CvT-13 (20M params) supera a DeiT-B (86M) en accuracy con 63% menos parámetros. Para nuestro router, donde queremos un backbone liviano, un CvT-13 sería una alternativa competitiva a ViT-Tiny.

3. **Convolución depth-wise separable es eficiente.** Cada Convolutional Projection añade solo `s²C` parámetros extra (negligible). Costo computacional mínimo por la ganancia en modelado local.

4. **Stride en K/V reduce cómputo 4×.** Usando stride=2 en la convolución de Key y Value se reduce el costo de MHSA con solo 0.3% de pérdida en accuracy. Útil si queremos que el router sea rápido en inferencia.

5. **Transferencia efectiva.** CvT-W24 preentrenado en ImageNet-22k y transferido a downstream tasks supera a BiT-R152x4 (928M params) con solo 277M params.

### 2.4 Decisiones de diseño que valida este paper

- **Si buscamos máxima eficiencia en params:** considerar CvT como backbone del router en lugar de ViT puro. CvT-13 (20M) vs ViT-Tiny (5.7M): ViT-Tiny sigue siendo más liviano, pero CvT-13 produce embeddings potencialmente más ricos.
- **Si se quiere eliminar position embeddings:** CvT permite esto nativamente, simplificando el manejo de resoluciones variables (relevante para nuestro adapter 3D→2D).
- **Proyección convolucional:** si en algún momento entrenamos el router end-to-end, usar Convolutional Projection en Q/K/V podría mejorar la calidad del CLS sin costo significativo.

### 2.5 Limitación para nuestro proyecto

CvT no está disponible en `timm` con la misma facilidad que ViT-Tiny. Para el ablation study, donde el backbone es **congelado** y solo variamos la cabeza de routing, ViT preentrenado es la opción más práctica y reproducible. CvT sería relevante si quisiéramos **entrenar el backbone** desde cero o hacer fine-tuning profundo.

---

## 3. Paper: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (Liu et al., ICCV 2021)

### 3.1 ¿Qué es?

Un Vision Transformer **jerárquico** que reemplaza la self-attention global (cuadrática) por self-attention **local en ventanas** con un mecanismo de **shifted windows** para conectar ventanas vecinas. Logra complejidad lineal respecto al tamaño de imagen y se convierte en un backbone de propósito general (clasificación, detección, segmentación).

### 3.2 Arquitectura clave

| Componente | Detalle |
|------------|---------|
| **Patch partition** | Parches de 4×4 (no 16×16 como ViT), produciendo feature dimension = 48 (4×4×3) |
| **Patch merging** | Concatenación de 2×2 parches vecinos + proyección lineal entre stages → downsampling 2× por stage |
| **W-MSA** | Multi-Head Self-Attention dentro de ventanas locales no superpuestas de M×M (default M=7) |
| **SW-MSA** | Shifted Window MSA: ventanas desplazadas por (M/2, M/2) en bloques alternos para crear conexiones cross-window |
| **Relative position bias** | Bias `B ∈ ℝ^(M²×M²)` sumado a `QK^T/√d` en lugar de position embeddings absolutos |
| **4 stages** | Resoluciones: H/4, H/8, H/16, H/32 (idéntico a pirámide CNN) |

Variantes:

| Modelo | C | Layer numbers | Params | FLOPs | ImageNet Top-1 |
|--------|---|---------------|--------|-------|----------------|
| Swin-T | 96 | {2,2,6,2} | 29M | 4.5G | 81.3% |
| Swin-S | 96 | {2,2,18,2} | 50M | 8.7G | 83.0% |
| Swin-B | 128 | {2,2,18,2} | 88M | 15.4G | 83.5% / 86.4% (IN-22k) |
| Swin-L | 192 | {2,2,18,2} | 197M | 103.9G | 87.3% (IN-22k) |

### 3.3 Observaciones clave para nuestro router

1. **Complejidad lineal vs cuadrática.** `Ω(W-MSA) = 4hwC² + 2M²hwC` vs `Ω(MSA) = 4hwC² + 2(hw)²C`. Para imágenes médicas de alta resolución, esta diferencia es crítica. Si algún día procesamos CT/MRI a resoluciones mayores que 224×224, Swin es significativamente más escalable que ViT.

2. **Shifted windows > sliding windows.** El mecanismo de shifted windows tiene 4× más throughput que sliding windows naive (755 img/s vs 183 img/s en Swin-T), con accuracy equivalente (81.3% vs 81.4%). El cyclic shift implementation evita overhead.

3. **Relative position bias >> absolute position embedding.** El ablation (Table 4) muestra que relative position bias mejora +1.2% en ImageNet, +1.3 box AP en COCO, +2.3 mIoU en ADE20K respecto a no tener encoding posicional. Los embeddings absolutos son **peores** que relative bias en tareas densas.

4. **Global average pooling equivalente a CLS token.** Los autores encontraron que usar global average pooling del último stage produce accuracy idéntica a añadir un CLS token (Appendix A2.1). Esto sugiere que para nuestro router, si usamos Swin-T como backbone, el feature vector del GAP es tan bueno como el CLS.

5. **Feature maps jerárquicas.** Swin produce mapas de features a 4 resoluciones (como una FPN). Esto lo hace ideal para tareas densas pero también útil para clasificación. La riqueza multi-escala podría producir embeddings más discriminativos para routing que un ViT de resolución fija.

6. **Swin-T es comparable a ResNet-50 en cómputo.** 29M params, 4.5G FLOPs — es más pesado que ViT-Tiny (5.7M) pero produce features de mayor calidad. Si podemos tolerar el costo, Swin-T produce embeddings potencialmente superiores.

### 3.4 Decisiones de diseño que valida este paper

- **Swin-T como alternativa de backbone del router:** si ViT-Tiny no separa bien las modalidades médicas, Swin-T (disponible en `timm` como `swin_tiny_patch4_window7_224`) ofrece inductive bias jerárquico + local que puede distinguir mejor texturas 2D (radiografías, dermoscopia) vs estructuras 3D (CT slices).
- **Relative position bias:** si se entrena el backbone (no congelado), usar relative position bias es estrictamente superior a absolute position embedding.
- **Para imágenes grandes:** si necesitamos procesar a resoluciones > 224 (ej. 384 para dermatología), Swin escala linealmente mientras ViT escala cuadráticamente.

---

## 4. Paper: "Switch Transformers: Scaling to Trillion Parameter Models" (Fedus et al., JMLR 2022)

### 4.1 ¿Qué es?

El paper que simplifica los modelos Mixture-of-Experts para Transformers, introduciendo el **Switch Transformer**: una variante MoE que rutea cada token a **un solo experto** (top-1 en lugar de top-k), logrando escalabilidad hasta modelos de un trillón de parámetros con costo computacional constante por token.

### 4.2 Innovaciones clave

| Innovación | Detalle | Impacto |
|------------|---------|---------|
| **Switch routing (k=1)** | Cada token va a un solo experto en lugar de top-2 | Reduce cómputo del router, simplifica implementación, y **funciona igual o mejor** que top-2 |
| **Auxiliary load balancing loss** | `L_aux = α · N · Σ f_i · P_i` donde `f_i` = fracción de tokens al experto i, `P_i` = probabilidad media asignada al experto i | Garantiza distribución uniforme de tokens entre expertos. `α = 10⁻²` encontrado como óptimo |
| **Selective precision** | Router computado en float32 internamente, pero tensores dispatch/combine en bfloat16 | Estabiliza entrenamiento sin sacrificar velocidad. bfloat16 puro **diverge** |
| **Reduced initialization** | `σ = √(0.1/n)` en lugar de `σ = √(1/n)` (10× menor) | Reduce varianza y mejora estabilidad dramáticamente |
| **Expert dropout** | Dropout 0.1 en capas no-expert, 0.4 en capas expert durante fine-tuning | Mitiga overfitting del exceso de parámetros en tareas downstream pequeñas |
| **Capacity factor** | `expert_capacity = (tokens_per_batch / num_experts) × capacity_factor` | Buffer para desbalance. Factor 1.0 funciona bien con la aux loss |

### 4.3 Resultados de escala

| Modelo | Params | FLOPs/seq | Speedup vs Dense |
|--------|--------|-----------|-----------------|
| T5-Base | 0.2B | 124B | — |
| Switch-Base (128 experts) | 7B | 124B | **7× más rápido** al mismo quality |
| T5-Large | 0.7B | 425B | — |
| Switch-Large (128 experts) | 26B | 425B | 2.5× vs T5-Large |
| Switch-C (2048 experts) | 1571B | 890B | 4× vs T5-XXL |

### 4.4 Observaciones clave para nuestro router

1. **El router es una simple proyección lineal + softmax.** La ecuación fundamental es:

   ```
   h(x) = W_r · x          (logits)
   p_i(x) = softmax(h(x))_i  (probabilidad de cada experto)
   ```

   Donde `W_r ∈ ℝ^(d_model × N)` con N = número de expertos. **Esto es exactamente nuestra `LinearGatingHead`.**

2. **Top-1 es suficiente y preferible a top-2.** Contrario a la intuición de Shazeer et al. (2017) de que k>1 es necesario para gradientes no triviales, Switch demuestra que k=1 funciona mejor. Para nuestro router que asigna una imagen a un solo experto, esto es validación directa.

3. **La auxiliary loss es LA técnica de balanceo.** La fórmula exacta:
   - `f_i = (1/T) Σ 1{argmax p(x) = i}` (fracción de tokens asignados al experto i)
   - `P_i = (1/T) Σ p_i(x)` (probabilidad media del experto i)
   - `L_aux = α · N · Σ f_i · P_i`
   - Se minimiza bajo distribución uniforme. `α = 10⁻²` es el valor calibrado.

   **Esto es exactamente lo que pide la consigna** para el router Linear+Softmax.

4. **Selective precision es obligatorio.** El router debe computar en float32 internamente aunque el resto del modelo use FP16/bfloat16. La consigna pide FP16, así que debemos castear las operaciones del router a float32 y volver a FP16 después.

5. **Expert dropout para fine-tuning.** Con muchos parámetros y pocos datos downstream, usar dropout diferencial (alto en expertos, bajo en el resto) es la receta contra overfitting. Aplicable a nuestra fase de fine-tuning de expertos individuales.

6. **La destilación preserva ~30% de la ganancia.** Switch demuestra que un modelo sparse de 7B puede destilarse a un denso de 223M preservando 30% de la mejora. Relevante si en producción necesitamos un modelo más compacto.

7. **Input jitter para exploración.** De las estrategias de exploración probadas (argmax, sampling, input dropout, input jitter), **multiplicative jitter noise** en la representación de entrada al router dio el mejor resultado. Es una técnica simple: multiplicar los embeddings por ruido uniforme `U(1-ε, 1+ε)` antes del softmax.

### 4.5 Pseudocódigo del router (adaptado del paper)

```python
# Simplificado del Appendix F del paper
router_logits = einsum(token_embeddings, router_weights)  # [batch, n_experts]
if training:
    router_logits *= uniform(1-eps, 1+eps)  # jitter para exploración
router_logits = cast_to_float32(router_logits)  # selective precision
router_probs = softmax(router_logits)
expert_gate, expert_index = top_1(router_probs)
# Compute aux loss
f = mean(one_hot(expert_index))     # fracción de tokens por experto
P = mean(router_probs)              # probabilidad media por experto
aux_loss = alpha * n_experts * dot(f, P)
```

### 4.6 Decisiones de diseño que valida este paper

- **Router lineal + softmax** como mecanismo principal (confirmado como el diseño de referencia en la literatura MoE).
- **Auxiliary loss** con `α = 0.01` como técnica de balanceo de carga.
- **Selective precision** (float32 en router, FP16 en el resto) como requisito de estabilidad.
- **Top-1 routing** es suficiente para nuestro caso de 5 expertos.
- **Input jitter** como técnica de regularización/exploración en entrenamiento.

---

## 5. Paper: "Mixture of Experts for Image Classification: What's the Sweet Spot?" (Videau et al., TMLR 2025)

### 5.1 ¿Qué es?

Un estudio **sistemático y empírico** de cómo integrar capas MoE en arquitecturas de clasificación de imágenes (ConvNeXt y ViT), evaluando configuraciones, número de expertos, posición de capas, tipo de gate, y escalabilidad con datos.

### 5.2 Hallazgos principales

#### 5.2.1 ¿Cuándo ayuda MoE?

| Escala del modelo | Beneficio de MoE |
|-------------------|-----------------|
| Tiny / Small | **Mejora significativa** (+0.5% a +1% accuracy) |
| Medium | Mejora moderada |
| Base / Large (>100M params activos) | **Beneficio se desvanece** |

**Conclusión directa para nosotros:** Nuestro sistema tiene 5 expertos heterogéneos (no son copias del mismo bloque), y el router opera sobre modelos de tamaño moderado. MoE es útil precisamente en este régimen.

#### 5.2.2 Posición de las capas MoE

| Estrategia | ConvNeXt | ViT |
|------------|----------|-----|
| **Every-2** (cada 2 bloques) | Peor resultado | **Mejor** |
| **Last-2** (últimos 2 stages) | **Robusto** en todo | Segundo mejor |
| **Stage** (final de cada stage) | Aceptable | No probado |

**Regla heurística del paper:** **Last-2 es la opción más segura** cross-arquitectura. Para ViT, Every-2 es ligeramente mejor.

**Para nuestro router:** La cabeza de routing (Linear, GMM, NB, k-NN) opera **después** del backbone, no **dentro** de los bloques. Pero si en algún momento integramos MoE layers dentro del backbone del router, Last-2 es el punto de partida.

#### 5.2.3 Número de expertos

| Dataset | ConvNeXt óptimo | ViT óptimo |
|---------|----------------|-----------|
| ImageNet-1k | 4 expertos | 8 expertos |
| ImageNet-21k | Hasta 16 efectivos | Más expertos posibles |

**Observación:** 16+ expertos en datos limitados **empeora** accuracy. Más datos permiten más expertos.

**Para nosotros:** Tenemos 5 expertos (uno por dataset médico). Con ~150K muestras totales, 5 es un número razonable y está dentro del sweet spot (4–8).

#### 5.2.4 Tipo de gate/router

| Gate | Resultado |
|------|-----------|
| **Conv 1×1 (lineal)** | **Mejor en todos los casos** |
| Cosine similarity | Peor |
| L2 distance | Intermedio |

**Hallazgo fundamental:** Un router lineal simple (`W·z + b → softmax`) es **superior** a routers más complejos. La complejidad adicional en el routing **no aporta beneficio consistente**.

**Validación directa para la consigna:** La cabeza `LinearGatingHead` del mecanismo A es la que este paper confirma como la más efectiva. Los mecanismos estadísticos (GMM, NB, k-NN) del ablation son interesantes científicamente, pero este paper sugiere que el lineal debería ganar.

#### 5.2.5 Balanceo de carga

El paper usa la **Auxiliary Loss de Shazeer et al. (2017)** (la misma del Switch Transformer) con **Batch Prioritized Routing (BPR)**. Los resultados muestran que el balanceo funciona: la distribución de expertos es uniforme.

**Para nuestro proyecto:** Confirma que `L_aux = α · N · Σ f_i · P_i` es la técnica estándar y funcional para evitar expert collapse. El cociente `max(f_i)/min(f_i) < 1.30` de la consigna es alcanzable con esta loss.

#### 5.2.6 Especialización de expertos

**Hallazgo sorprendente:** Los expertos **no se especializan limpiamente** por clases semánticas. En capas profundas hay cierta correlación con clases (ej. animales vs objetos), pero es difusa. Los expertos cubren parches pequeños y no contiguos de la imagen.

**Para nuestro proyecto:** Esto es diferente de nuestro caso, donde la "especialización" es **por modalidad médica** (radiografía, dermoscopia, CT, etc.), no por clases dentro de un mismo dataset. Nuestros expertos son heterogéneos por diseño, lo que debería producir una partición más limpia que la observada en ImageNet.

#### 5.2.7 Robustez OOD

MoE con modelos pequeños que mejoran in-distribution **también mejoran** out-of-distribution. Pero para modelos grandes donde MoE no mejora ID, tampoco ayuda OOD. La robustez está atada al régimen donde MoE aporta.

---

## 6. Paper: "A Supervised Multi-Spectral Image Classification for Remote Sensing Data" (Zeki & Zaid, RACS 2015)

### 6.1 ¿Qué es?

Un paper de clasificación supervisada de imágenes multiespectrales de teledetección. Compara tres métodos clásicos: **Minimum Distance (MD)**, **Maximum Likelihood (ML)**, y **Probabilistic Neural Network (PNN)**, aplicados sobre datos Landsat-7 con PCA como extracción de features.

### 6.2 Métodos clave

| Método | Formulación | Tipo |
|--------|-------------|------|
| **Minimum Distance (MD)** | `D_i = \|\|x - z_i\|\|` = distancia euclidiana al centroide de clase `z_i`. Asigna x a la clase con `min D_i`. | Discriminativo, no paramétrico |
| **Maximum Likelihood (ML)** | `d_i(x) = ln P(W_i) - 0.5 ln\|C_i\| - 0.5 (x-z_i)' C_i⁻¹ (x-z_i)`. Usa la distribución gaussiana completa (media + covarianza). | Generativo, paramétrico |
| **Probabilistic Neural Network (PNN)** | Red neuronal con capa de patrones (kernel gaussiano por muestra de entrenamiento) y capa de salida (suma por clase). Decisión: `max O_q`. | Neural, no paramétrico |
| **PCA** | Extracción de features: reduce bandas espectrales manteniendo la máxima varianza. | Preprocesamiento |

### 6.3 Observaciones clave para nuestro router

1. **Minimum Distance ≡ k-NN con k=1 usando centroides.** Esto es una versión simplificada de nuestro mecanismo k-NN del ablation. Si calculamos el centroide de los embeddings CLS por dataset y asignamos la imagen al centroide más cercano, estamos haciendo Minimum Distance. Es la baseline más simple posible.

2. **Maximum Likelihood ≡ GMM con un componente por clase.** La formulación del paper es literalmente un GMM donde cada clase tiene su propia gaussiana (media + covarianza). Nuestro mecanismo GMM del ablation es una generalización directa: `sklearn.mixture.GaussianMixture` con un componente por experto.

3. **PNN es un kernel density estimator neural.** Cada muestra de entrenamiento se convierte en un kernel gaussiano. Para muchas muestras (100k+), esto es computacionalmente prohibitivo, pero el concepto fundamenta el Naive Bayes kernel: estimar la densidad por clase.

4. **PCA como reducción de dimensionalidad pre-clasificación.** El paper muestra que clasificar sobre features PCA es más rápido y más preciso que sobre los datos originales (6 bandas). **Esto valida directamente** la sugerencia de la consigna de aplicar PCA antes de k-NN/FAISS para reducir d=192 a d=32 y mitigar la maldición de dimensionalidad.

5. **Clasificación supervisada > no supervisada.** El paper confirma que cuando tienes etiquetas (nuestro caso: expert_id 0–4), la clasificación supervisada da más accuracy que clustering no supervisado. Esto valida nuestro enfoque de usar etiquetas de experto conocidas para entrenar/calibrar los mecanismos de routing.

### 6.4 Conexión directa con nuestro ablation study

| Método del paper | Nuestro mecanismo equivalente | Cómo se traduce |
|------------------|-------------------------------|-----------------|
| Minimum Distance | k-NN (k=1) sobre centroides | `faiss.IndexFlatL2` con búsqueda del vecino más cercano |
| Maximum Likelihood | GMM | `GaussianMixture(n_components=5, covariance_type='full')` |
| PNN | Naive Bayes (kernel) | `GaussianNB()` con asunción de independencia condicional |
| PCA + MD | k-NN con reducción PCA | `PCA(n_components=32)` → `faiss.IndexFlatL2` |

**El paper es un puente conceptual** entre los métodos clásicos de clasificación y nuestros mecanismos de routing. La diferencia es que nosotros operamos sobre embeddings CLS de 192 dimensiones en lugar de 6 bandas espectrales.

---

## 7. Síntesis cruzada: Lo que todos los papers validan

### 7.1 Tabla de validación contra la consigna

| Requisito de la consigna | ¿Lo respaldan los papers? | Referencia principal |
|--------------------------|--------------------------|---------------------|
| ViT como backbone del router | Sí | Dosovitskiy (ViT), Liu (Swin), Wu (CvT) — tres opciones válidas |
| CLS token como representación | Sí | ViT §3.1: `y = LN(z⁰_L)`. Swin: GAP equivalente |
| Linear + Softmax como router DL | **Confirmado como el mejor** | Switch Transformer §2.1, Videau §4.3.5 |
| Top-1 routing | Sí, **superior a top-2** | Switch Transformer: "simplification preserves quality" |
| Auxiliary loss para balanceo | Sí, fórmula exacta validada | Switch Transformer §2.2 Eq.4, Videau §3 |
| `α = 0.01` para aux loss | Sí, calibrado empíricamente | Switch Transformer: "sufficiently large to ensure load balancing" |
| Selective precision (FP32 en router) | **Obligatorio** para estabilidad | Switch Transformer §2.4 Table 2 |
| GMM como alternativa | Fundamento clásico sólido | Zeki: Maximum Likelihood = GMM monocomponente |
| Naive Bayes como alternativa | Fundamento clásico sólido | Zeki: PNN ≈ kernel density estimator |
| k-NN (FAISS) como alternativa | Válido con PCA previo | Zeki: Minimum Distance + PCA |
| 5 expertos heterogéneos | Dentro del sweet spot (4–8) | Videau §4.3.2 |
| Backbone congelado | Sí | ViT §4.1: few-shot sobre features congeladas |
| Posición de capas MoE (Last-2) | Sí, heurística robusta | Videau §4.3.1 |
| Swin-Tiny como alternativa | Sí, mejor inductive bias que ViT | Liu: Swin-T > DeiT-S en todo |
| Expert dropout en fine-tuning | Sí | Switch Transformer §2.4, Table 4 |

### 7.2 Observaciones para la Auxiliary Loss

- **La loss de balanceo funciona.** Confirmado por Switch Transformer y Videau et al.
- **`α = 10⁻²`** es el valor estándar. Switch Transformer barrió `10⁻¹` a `10⁻⁵`.
- Solo aplica al router **Linear + Softmax** (diferenciable). Los estadísticos no tienen gradiente.
- **Input jitter** (multiplicar embeddings por ruido `U(1-ε, 1+ε)`) mejora exploración.

### 7.3 Consideraciones de eficiencia

| Aspecto | Referencia | Impacto |
|---------|-----------|---------|
| FP16 / bfloat16 | Switch Transformer (obligatorio con selective precision en router) | Reduce VRAM ~50% |
| Backbone congelado | ViT: few-shot funciona bien | Extracción de CLS sin gradientes → rápido |
| Complejidad lineal | Swin: `O(M²hw)` vs ViT: `O((hw)²)` | Relevante si se procesan resoluciones > 224 |
| Stride en K/V | CvT: stride=2 reduce MHSA 4× | Solo si entrenamos backbone |
| Routers estadísticos en CPU | Operan sobre embeddings pre-extraídos | GMM, NB, k-NN no requieren GPU |
| Expert dropout | Switch: 0.4 en expertos, 0.1 en el resto | Anti-overfitting en fine-tuning |

---

## 8. Riesgos y puntos de atención

1. **El router lineal probablemente gane el ablation.** Tanto Switch Transformer como Videau et al. lo confirman. El valor del ablation está en **cuantificar la diferencia** y **discutir por qué**, no en que un método estadístico sorprenda.

2. **k-NN con d=192 puede sufrir curse of dimensionality.** Zeki et al. muestran que PCA pre-clasificación mejora resultados. Aplicar `PCA(n_components=32)` antes de FAISS es recomendable.

3. **El CLS de ViT-Tiny podría no separar bien 2D vs 3D.** El backbone está preentrenado en imágenes 2D. Los volúmenes 3D pasan por el adapter (slice central), pero la distribución de esos embeddings podría solaparse. Monitorear con t-SNE/UMAP de `Z_train`.

4. **Desbalance de muestras entre datasets.** NIH tiene ~112K imágenes vs Páncreas ~281 volúmenes. El split estratificado y posiblemente un sampling balanceado son esenciales.

5. **Selective precision es no negociable.** Switch Transformer demostró que bfloat16 puro **diverge** en el router. El cast a float32 en las operaciones del softmax del router es obligatorio.

6. **Expert dropout para fine-tuning de expertos.** Con VGG16-BN para Osteoarthritis u otros expertos pesados, usar dropout alto (0.3–0.4) en las capas FFN del experto y bajo (0.1) en el backbone compartido.

---

## 9. Propuesta de arquitectura del Router basada en los 6 papers

### 9.1 Diagrama conceptual

```
Imagen de entrada (cualquier modalidad: PNG, JPG, MHD, NIfTI, MHA)
    │
    ▼
┌─────────────────────────────┐
│  AdaptivePreprocessor        │  → Detecta 2D/3D, normaliza, resize
│  (ya implementado)           │  → Salida: tensor [C, H, W] o [1, D, H, W]
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  ViT_AdapterWrapper          │  → Convierte 3D → slice central 2D
│  (ya implementado)           │  → Salida: [3, 224, 224]
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  ImageNet Normalization      │  → mean=[0.485,0.456,0.406]
│                              │  → std=[0.229,0.224,0.225]
└─────────────────────────────┘
    │
    ▼
╔═════════════════════════════════════════════════════╗
║  BACKBONE DEL ROUTER (congelado, sin gradientes)    ║
║                                                     ║
║  Opción primaria:                                   ║
║    vit_tiny_patch16_224.augreg_in21k_ft_in1k        ║
║    → 5.7M params, d_model=192                       ║
║    → Fuente: ViT (Dosovitskiy), disponible en timm  ║
║                                                     ║
║  Alternativa si ViT-Tiny no discrimina bien:        ║
║    swin_tiny_patch4_window7_224.ms_in22k_ft_in1k    ║
║    → 29M params, d_model=768                        ║
║    → Fuente: Swin (Liu), jerárquico + local bias    ║
║    → Mayor costo pero mejor inductive bias          ║
║                                                     ║
║  Alternativa experimental (si se entrena backbone): ║
║    CvT-13 (Wu et al.)                               ║
║    → 20M params, sin position embeddings             ║
║    → No disponible fácilmente en timm                ║
║                                                     ║
║  Extracción:                                         ║
║    z = model.forward_features(x)                     ║
║    cls = z[:, 0, :]  # CLS token (ViT/Swin)         ║
║    # o cls = z.mean(dim=[2,3]) para Swin con GAP     ║
║                                                     ║
║  Selective precision (Switch Transformer):           ║
║    Backbone en FP16, router softmax en float32       ║
╚═════════════════════════════════════════════════════╝
    │
    │  z_cls ∈ ℝ^{d_model}  (d=192 para ViT-Tiny)
    │
    ▼
╔═════════════════════════════════════════════════════╗
║  CABEZA DE ROUTING (ablation study: 4 variantes)    ║
║                                                     ║
║  ┌─────────────────────────────────────────────┐    ║
║  │ A) Linear + Softmax (REFERENCIA)            │    ║
║  │    logits = W·z + b       [W ∈ ℝ^(d×5)]    │    ║
║  │    probs = softmax(logits)  → float32       │    ║
║  │    expert = argmax(probs)                   │    ║
║  │                                             │    ║
║  │    Training:                                │    ║
║  │      - Input jitter: z *= U(1-ε, 1+ε)      │    ║
║  │        (Switch Transformer Appendix C)      │    ║
║  │      - L_total = L_CE + α·L_aux             │    ║
║  │        α = 0.01 (Switch Transformer §2.2)   │    ║
║  │      - L_aux = N · Σ f_i · P_i              │    ║
║  │        (Fedus et al. Eq. 4)                 │    ║
║  │                                             │    ║
║  │    Fuente: Switch Transformer, Videau Tab.7 │    ║
║  └─────────────────────────────────────────────┘    ║
║                                                     ║
║  ┌─────────────────────────────────────────────┐    ║
║  │ B) GMM (Gaussian Mixture Model)             │    ║
║  │    sklearn.mixture.GaussianMixture(         │    ║
║  │      n_components=5,                        │    ║
║  │      covariance_type='full'                 │    ║
║  │    )                                        │    ║
║  │    .fit(Z_train, y_train_expert)            │    ║
║  │                                             │    ║
║  │    Predicción: expert = argmax P(x|class_k) │    ║
║  │                                             │    ║
║  │    Fundamento: Maximum Likelihood clásico    │    ║
║  │    (Zeki & Zaid: MLE con gaussiana completa)│    ║
║  └─────────────────────────────────────────────┘    ║
║                                                     ║
║  ┌─────────────────────────────────────────────┐    ║
║  │ C) Naive Bayes (Gaussiano)                  │    ║
║  │    sklearn.naive_bayes.GaussianNB()         │    ║
║  │    .fit(Z_train, y_train_expert)            │    ║
║  │                                             │    ║
║  │    Asunción: covarianza diagonal             │    ║
║  │    (simplificación del GMM)                 │    ║
║  │                                             │    ║
║  │    Fundamento: PNN simplificada (Zeki)      │    ║
║  └─────────────────────────────────────────────┘    ║
║                                                     ║
║  ┌─────────────────────────────────────────────┐    ║
║  │ D) k-NN con FAISS                           │    ║
║  │                                             │    ║
║  │    Preprocesamiento:                        │    ║
║  │      Z_reduced = PCA(n_components=32)       │    ║
║  │        .fit_transform(Z_train)              │    ║
║  │      (Zeki: PCA mejora MD)                  │    ║
║  │                                             │    ║
║  │    Índice:                                  │    ║
║  │      faiss.IndexFlatL2(32)                  │    ║
║  │      index.add(Z_reduced_train)             │    ║
║  │                                             │    ║
║  │    Predicción:                              │    ║
║  │      D, I = index.search(z_query, k=5)      │    ║
║  │      expert = majority_vote(y_train[I])     │    ║
║  │                                             │    ║
║  │    Fundamento: Minimum Distance (Zeki)      │    ║
║  │    + PCA para evitar curse of dim. (d=192)  │    ║
║  └─────────────────────────────────────────────┘    ║
╚═════════════════════════════════════════════════════╝
    │
    │  expert_id ∈ {0, 1, 2, 3, 4}
    │
    ▼
╔═════════════════════════════════════════════════════╗
║  EXPERTO SELECCIONADO                               ║
║                                                     ║
║  0 → NIH Chest X-ray Expert (14 clases, BCE)       ║
║  1 → ISIC 2019 Expert (8 clases, CE)               ║
║  2 → Osteoarthritis Expert (5 grados, CE)           ║
║  3 → LUNA16 Expert (nódulo/no-nódulo, BCE)          ║
║  4 → Pancreas Expert (tumor/normal, BCE)             ║
║                                                     ║
║  Cada experto aplica su preprocesamiento             ║
║  específico DESPUÉS del routing:                     ║
║    - CLAHE para Osteo (CvT: local context helps)    ║
║    - Augmentation agresiva para ISIC                 ║
║    - HU windowing para LUNA/Pancreas                 ║
║    - ImageNet norm para todos los 2D                 ║
║                                                     ║
║  Expert dropout en fine-tuning:                      ║
║    0.4 en capas FFN del experto                      ║
║    0.1 en backbone compartido                        ║
║    (Switch Transformer §2.4)                         ║
╚═════════════════════════════════════════════════════╝
```

### 9.2 Hiperparámetros recomendados

| Hiperparámetro | Valor | Fuente |
|----------------|-------|--------|
| Backbone | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` | ViT paper + consigna |
| d_model (dim CLS) | 192 | ViT-Tiny architecture |
| n_experts | 5 | Consigna (1 por dataset) |
| Routing: top-k | 1 | Switch Transformer (top-1 > top-2) |
| Aux loss α | 0.01 | Switch Transformer §2.2 |
| Aux loss formula | `L = α · N · Σ f_i · P_i` | Switch Transformer Eq. 4 |
| Input jitter ε | 0.01 | Switch Transformer Appendix C |
| Router precision | float32 (selective) | Switch Transformer §2.4 |
| Global precision | FP16 / bfloat16 | Consigna |
| PCA dims (para k-NN) | 32 | Zeki: PCA + MD mejora accuracy |
| k para k-NN | 5 | Heurística √N con N≈150k |
| Val split | 20% estratificado | Configuración actual |
| Balance metric | `max(f_i)/min(f_i) < 1.30` | Consigna |
| Expert dropout (fine-tuning) | 0.4 expert / 0.1 no-expert | Switch Transformer Table 4 |
| Optimizer (Linear router) | AdamW, lr=1e-3, wd=0.05 | CvT/Swin recipe |

### 9.3 Métricas del ablation study

| Métrica | Qué mide | Cómo calcular |
|---------|----------|---------------|
| **Routing Accuracy** | ¿Cuántas imágenes van al experto correcto? | `accuracy_score(y_true_expert, y_pred_expert)` sobre val set |
| **Routing F1 (macro)** | F1 balanceado entre los 5 expertos | `f1_score(y_true, y_pred, average='macro')` |
| **Load Balance Ratio** | Uniformidad de la distribución | `max(f_i) / min(f_i)`, target < 1.30 |
| **Latencia de inferencia** | Tiempo por imagen en el router | `time.perf_counter()` sobre 1000 muestras |
| **Params del router** | Costo en parámetros de la cabeza | Linear: `d×5 + 5 = 965`. Estadísticos: 0 params entrenables |

### 9.4 Resultado esperado del ablation

| Mecanismo | Routing Acc esperada | Balance | Latencia | Justificación |
|-----------|---------------------|---------|----------|---------------|
| **Linear + Softmax** | ~95-98% | < 1.30 (con aux loss) | ~0.1ms | Videau: "simple linear works best"; Switch: diseño de referencia |
| GMM | ~90-95% | N/A (no diferenciable) | ~0.5ms | Zeki: MLE funciona bien en features PCA. Depende de separabilidad |
| Naive Bayes | ~88-93% | N/A | ~0.05ms | Rápido pero covarianza diagonal puede perder correlaciones |
| k-NN (FAISS) | ~85-92% | N/A | ~0.2ms | Sensible a dimensionalidad. PCA a d=32 debería ayudar |

---

*Documento preparado como análisis técnico de antecedentes. Estos 6 papers constituyen la base teórica para justificar cada decisión de diseño del router en el reporte técnico final.*
