# Presentación — Proyecto MoE visión médica (~15 min)

Documento guion para diapositivas, alineado con `consigna.md`, la carpeta `notebooks/Expertos y doc/`, el código `notebooks/experts/arquitecturas_de_expertos.py` y el **ablation study** en curso (`06_Ablation_*.ipynb`, generados desde `scripts/build_ablation_notebooks.py`).

**Sustituir** los marcadores `[COMPLETAR: …]` con tus números reales antes de exponer.

---

## Resumen de tiempos (orientativo)

| Bloque | Minutos |
|--------|--------|
| Intro + pregunta científica | ~2 |
| Datos y preprocesado adaptativo | ~1,5 |
| Tabla de 5 expertos (este repo) | ~2 |
| Ablation study (router) | ~3,5 |
| MoE completo (L_task + L_aux) | ~1,5 |
| Resultados + balance | ~2 |
| Dashboard / demo | ~2 |
| Conclusiones | ~1 |

**Total ~15 min** + buffer para preguntas.

---

## Diapositiva 1 — Título

**Título:** Clasificación médica multimodal con Mixture of Experts (MoE)  
**Subtítulo:** Router Vision Transformer + ablation de mecanismos de enrutamiento  
**Pie:** [Nombre del equipo] · Incorporar Elementos de IA — Bloque Visión · 2026  

**Notas orales (30 s):** Presentar el objetivo del sistema: una sola entrada (imagen o volumen), sin metadatos de modalidad, y enrutamiento automático a expertos especializados.

---

## Diapositiva 2 — Motivación y restricción del enunciado

**Título:** ¿Por qué MoE en imágenes médicas?

**Bullets:**
- Un mismo sistema debe atender **varias modalidades** (radiografía, dermatoscopía, TC 3D, etc.) sin recibir etiquetas de tipo de estudio.
- **Entrada única:** tensores 2D o 3D; la modalidad se infiere solo del contenido (**penalización −20%** si se inyectan metadatos al router o al pipeline decisorio).
- **Pregunta científica (consigna):** ¿El **ViT como router** justifica su costo frente a métodos más ligeros que operan sobre el **mismo embedding**, o los baselines estadísticos/diferenciables son competitivos?

**Notas orales (1 min):** Enfatizar que la comparación justa es sobre **la misma representación** (CLS del ViT → proyección **z**), variando solo la **cabeza de routing**.

---

## Diapositiva 3 — Pipeline global (diagrama mental)

**Título:** Flujo del sistema

**Bullets (o figura):**
1. **Preprocesador adaptativo:** detecta rank del tensor → resize 224×224 (2D) o 64³ (3D), normalizaciones según modalidad.
2. **Backbone del router:** ViT-Tiny (`vit_tiny_patch16_224`) + `SwitchablePatchEmbed` → token **CLS** (dim 192).
3. **Proyector:** CLS → **z ∈ ℝ⁶⁴** (espacio común para cabezas de ablation).
4. **Gating:** probabilidades sobre **5 expertos** (uno por dominio/dataset de entrenamiento).
5. **Expertos:** redes **heterogéneas** (2D y 3D) con checkpoints entrenados por separado.
6. **Pérdida MoE:** **L = L_task + α · L_aux** (Switch Transformer), con **α** calibrado para balance de carga.

**Notas orales (1,5 min):** No leer el esquema línea a línea; apoyarse en **una sola figura** si la tienes del dashboard o del reporte.

---

## Diapositiva 4 — Cinco expertos (contenido de `Expertos y doc/` + `arquitecturas_de_expertos.py`)

**Título:** Expertos heterogéneos — datasets y arquitecturas

**Tabla (rellenar F1 si lo expones):**

| Exp. | Dataset | Notebook principal | Arquitectura | Clases | Notas |
|------|---------|----------------------|--------------|--------|--------|
| 1 | NIH Chest X-ray 14 (subset) | `NIH_ChestXray_Swin_Tiny_Training.ipynb` | **Swin-Tiny** (`swin_tiny_patch4_window7_224`) | 5 (multietiqueta) | ASL, F1-macro; split por paciente |
| 2 | ISIC 2019 | `ISIC2019_EfficientNetB3_Training_Final.ipynb` | **EfficientNet-B3** | 9 | Focal + class weights; sampler balanceado |
| 3 | Osteoarthritis rodilla | `Osteoarthritis_VGG16BN_Training.ipynb` | **VGG-16 BN** (entrada 1 canal) | 5 (KL) | CLAHE; fine-tuning en 2 fases |
| 4 | LUNA16 (TC pulmón 3D) | `LUNA16_R3D18_Training.ipynb` | **R3D-18** | 2 | Ventana HU, 64³, Focal; grad-CAM |
| 5 | Páncreas CT 3D | `Pancreas_R3D18_Training.ipynb` | **R3D-18** | 2 | Focal fuerte; gradient checkpointing |

**EDA / preprocesado previo:** `NIH_ChestXray_EDA_Preprocessing.ipynb`, `ISIC_2019_EDA_Preprocessing.ipynb`, `Osteoarthritis_eda_data_preprocecing.ipynb`, `Pancreas_eda_data_preprocecing.ipynb` (exploración y pipelines de datos).

**Checkpoints esperados (ejemplos en código):** `exp1_NIH_SwinTiny_best.pth`, `exp2_ISIC_EfficientNetB3_best.pth`, `exp3_Osteo_VGG16BN_best.pth`, `exp4_LUNA16_3D_best.pth`, `exp5_Pancreas_3D_best.pth`.

**Notas orales (2 min):** Justificar **heterogeneidad** (consigna): cada experto usa una familia de modelo distinta; los 3D comparten R3D-18 pero datasets y cabezas son distintos.

---

## Diapositiva 5 — Ablation study del router (lo que estás haciendo ahora)

**Título:** Ablation: mismos embeddings, distintas cabezas de decisión

**Diseño (notebooks `06_Ablation_NeuralGMM.ipynb`, `06_Ablation_LogGaussianNB.ipynb`, `06_Ablation_SoftKNN.ipynb`):**
- **Fase datos:** cache `unified_cls_tokens.npz` (CLS del ViT) y, tras warm-up, **`unified_z_64.npz`** (proyección **z** de dimensión 64) para inicializar y comparar.
- **Fase 1 — Warm-up maestro:** `VisionRouterAblation` con cabeza **`proxy_linear`** (proyector + softmax lineal sobre **z**); guarda backbone + proyector (`backbone_master_warmup.pth`, `projector_master.pth`).
- **Fase 2 — Cuatro mecanismos en competencia** (misma base ViT + proyector):
  1. **Baseline DL:** capa lineal + softmax sobre **z** (proxy / entrenamiento end-to-end del bloque router según celda).
  2. **NeuralGMM:** cabeza **GMM diferenciable** (`NeuralGMMHead`) — Gaussianas diagonales, priors **`log_pi`** aprendibles, **`σ` vía ELU** (estabilidad, inspirado en literatura MDN).
  3. **LogGaussianNB:** misma forma factorizada Gaussiana + Naive-Bayes-style sobre **z**.
  4. **SoftKNN:** prototipos / vecindad suave en **z** (`SoftKNNHead`).

**Métricas (alineadas con `notebooks/05_Router_Vit_Lineal_Solo.ipynb`):**  
En el 05, la validación rápida sobre CLS cacheado usa `eval_router_on_cls(router_head, Z_val_np, y_val_np)`. En los notebooks `06_Ablation_*.ipynb` (código generado en `scripts/build_ablation_notebooks.py`) la evaluación equivalente tras el proyector es `eval_router_on_z(...)`: **mismas fórmulas**, cambia solo que primero se aplica **CLS → z** con el `projector`.

| Clave (código) | Definición |
|----------------|------------|
| **`val_acc`** | Accuracy de routing en el split de validación: `pred = argmax(softmax(logits))` vs etiqueta de experto (dataset de origen). **Métrica principal** del ablation; en el 05 el objetivo guía es **> 0,80**. |
| **`val_ratio`** | Balance de cargas en **validación**: `max_i f_i / min_i f_i`, donde `f_i` es la fracción de predicciones duras al experto `i` (`_routing_ratio_from_preds`). Umbral consigna **≤ 1,30** para penalización del router ViT+Linear en el sistema final. |
| **`val_entropy`** | Media de la entropía de Shannon de las probabilidades de gating por muestra (`_entropy_mean` sobre `softmax(logits)`). Útil para comparar **confianza / dispersión** del gating entre métodos. |
| **`val_route_pct`** | Porcentaje de rutas al experto 0…4 en val (dict `E1:…%` …). Opcional en diapositiva si hay espacio. |
| **`val_cm`** | Matriz de confusión experto verdadero × experto predicho (para una figura en el reporte). |
| **Score compuesto (solo entrenamiento tipo 05 / `fit_router_with_eval`)** | `val_acc - 0.02 × max(0, val_ratio - 1.30)` — prioriza accuracy y castiga desbalance fuerte. Puedes reportar el **mejor score** por cabeza si entrenaste con el mismo criterio. |

**Durante entrenamiento** el 05 también registra por época: **`train_acc`** (`routing_acc`), **`train_aux`** (Switch Transformer `L_aux` en train), **`vram_mb`**. Úsalos para una fila “mejor época” solo del baseline lineal si contrastas con consigna; las cabezas estadísticas pueden no tener `aux` en el mismo sentido.

**Opcional consigna:** latencia por forward, VRAM pico, parámetros de la cabeza (columnas extra si las mediste).

**Notas orales (2 min):** Dejar claro que **lo único que cambia es la cabeza** sobre **z** (mismo proyector y mismos `Z_val`, `y_val`); la consigna valora **discusión honesta** aunque un baseline no sea el ganador.

---

## Diapositiva 6 — Resultados del ablation (tabla alineada al notebook 05)

**Título:** Comparación de mecanismos de routing (mismas métricas que `eval_router_on_cls` / `eval_router_on_z`)

**Tabla principal** — rellenar al **finalizar entrenamiento** de cada cabeza (mismo `y_val`, mismos embeddings de entrada por fila):

| Mecanismo | `val_acc` ↑ | `val_ratio` ↓ | `val_entropy` | Score* | Notas |
|-----------|-------------|-----------------|---------------|--------|--------|
| Warm-up / **proxy lineal** sobre **z** (`head_type='proxy_linear'`) | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | Baseline DL tras proyector; comparable al flujo del 05 sobre CLS pero en dim 64. |
| **NeuralGMM** (`NeuralGMMHead`) | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | |
| **LogGaussianNB** | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | |
| **SoftKNN** | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | |

\* **Score** = `val_acc - 0.02 × max(0, val_ratio - 1.30)` — opcional; solo tiene sentido si optimizaste o seleccionaste checkpoint con el mismo criterio que `fit_router_with_eval` en el 05.

**Tabla compacta opcional (una fila por método)** — si quieres mostrar el desglose de carga en validación:

| Mecanismo | E1 % | E2 % | E3 % | E4 % | E5 % |
|-----------|------|------|------|------|------|
| … | … | … | … | … | … |

*(Valores de `val_route_pct` en el código.)*

**Referencia numérica del 05 (ejemplo en salidas del notebook, no sustituir sin verificar tu corrida):** episodio con **`val_acc ≈ 0,952`**, **`val_ratio ≈ 1,131`**, **`val_entropy ≈ 0,515`** — sirve solo como orden de magnitud si tu tabla de ablation está en el **mismo split** y política de entrenamiento.

**Una frase de interpretación:** [COMPLETAR: ganador por `val_acc`; si dos métodos empatan, comparar `val_ratio` y `val_entropy`; si `val_ratio` > 1,30, mencionar riesgo de penalización solo para el router ViT+Linear en el MoE final.]

**Notas orales (1 min):** Si un método falló, **explicar** (pocas muestras por clase, varianzas nulas en NB/GMM, etc.) — la rúbrica premia el análisis sobre ocultar fallos.

---

## Diapositiva 7 — MoE integrado y balance de carga

**Título:** Entrenamiento del sistema completo

**Bullets:**
- **L_task:** agrega predicciones de expertos ponderadas por **gating** (flujo tipo profesor en notebooks de router).
- **L_aux (Switch Transformer):** penaliza desbalance **max(f_i)/min(f_i)**; umbral consigna **1.30** para el router **ViT+Linear** — por encima, **−40%** nota.
- **α ∈ [0.01, 0.1]** — [COMPLETAR valor usado].
- Técnicas HW: FP16, gradient accumulation, **gradient checkpointing** en expertos 3D.

**Notas orales (1,5 min):** Mostrar **cociente final** o gráfico de **f_i** si lo tienes en el dashboard.

---

## Diapositiva 8 — Resultados de clasificación por dataset

**Título:** F1 macro por dataset (expertos)

**Tabla:**

| Dataset | F1 macro (val/test) | [COMPLETAR] |
|---------|---------------------|-------------|
| NIH | | |
| ISIC | | |
| Osteoarthritis | | |
| LUNA16 | | |
| Páncreas | | |

**Umbral consigna (orientativo):** 2D > 0.72 (óptimo), > 0.65 aceptable; 3D > 0.65 / > 0.58.

**Notas orales (1 min):** Una matriz de confusión del **peor** experto si queda tiempo.

---

## Diapositiva 9 — Dashboard y demo

**Título:** Demostración interactiva

**Checklist consigna (voz en off):**
- Carga PNG / JPEG / NIfTI; preprocesado visible (antes/después).
- Inferencia: etiqueta, confianza, **ms**.
- **Attention heatmap** del router ViT.
- Panel **ablation** (tabla de los 4 métodos).
- **Load balance** (barras f_i).
- **OOD / entropía** del gating.

**Notas orales (2 min):** Demo corta: 1 imagen 2D + si es posible 1 volumen o captura 3D.

---

## Diapositiva 10 — Conclusiones y trabajo futuro

**Título:** Conclusiones

**Bullets:**
- **Respuesta a la pregunta científica:** [COMPLETAR en una o dos frases].
- **Limitaciones:** tamaño de datos 3D, imbalance, tiempo de entrenamiento, generalización OOD.
- **Reproducibilidad:** seeds, `requirements.txt`, rutas Drive documentadas.

**Notas orales (1 min):** Cierre con la **contribución del equipo**, no repetir el enunciado.

---

## Diapositiva opcional — Referencias mínimas (si no van solo en el reporte)

- Dosovitskiy et al., ViT; Fedus et al., Switch Transformers; Jacobs et al., Mixture of Experts; datasets NIH, ISIC, etc.

---

## Texto legal / entregables (slide pequeña o pie de tabla)

- Reporte técnico ≤ 7 páginas (ABET).  
- Repositorio Git con README y dependencias.  
- Figuras obligatorias del reporte: arquitectura, tabla ablation, curvas, balance, heatmaps.

---

*Generado para el proyecto en `proyecto 2`. Actualiza métricas y capturas antes de la presentación.*
