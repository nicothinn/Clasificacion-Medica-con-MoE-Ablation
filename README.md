# Clasificación médica con Mixture of Experts (MoE) — Router y ablation study

Proyecto académico: sistema MoE multimodal (cinco datasets médicos 2D/3D) con **Vision Router** entrenable y **estudio de ablación** comparando cuatro mecanismos de enrutamiento sobre los mismos embeddings CLS.

---

## Qué implementa el notebook principal

El archivo [`notebooks/03_Pipeline_Router_MoE.ipynb`](notebooks/03_Pipeline_Router_MoE.ipynb) concentra el **pipeline del router** alineado con el diagrama A→M del profesor (DataLoader mixto → patch embeddings homogéneos → ViT → CLS 1×192 → cabeza lineal 192→5 → gating + pérdida auxiliar de balanceo).

| Fase | Contenido |
|------|-----------|
| **Fase 0** | Instalación (`timm`, `monai`, etc.), montaje de Drive, rutas (`RAW_DIR`, `LOCAL_DEST`, `DATASET_ROOTS`, `WEIGHTS_DIR`). |
| **Arquitecturas embebidas** | Definición **en el propio notebook** de los 5 expertos (sin `.py` externo en Colab): NIH (**Swin-Tiny**, 5 clases), ISIC (EfficientNet-B3), Osteo (VGG16-BN), LUNA16 (**DCSwinBStyle3D**), Páncreas (R3D-18); `EXPERT_SPECS`, `build_expert`, `load_weights`, `load_all_experts_from_drive`, `print_expert_load_report`. |
| **Fase 1** | Extracción de ZIPs desde Drive a `/content/datasets/`. |
| **Fase 2** | `AdaptivePreprocessor`: carga 2D/3D, resize 224×224 / 64³, ventana HU en CT. |
| **Fase 3** | `scan_dataset_files` recursivo por extensiones médicas. |
| **Fase 4** | `MixedMedicalDataset`, `mixed_collate_fn`, `build_router_dataloader`; índices de **etiquetas de tarea** (NIH multilabel, ISIC, Osteo por carpetas KL, LUNA por CSV, Páncreas por `.npz`); celda opcional de **validación** de etiquetas. |
| **Fase 5** | `SwitchablePatchEmbed`: parches 16×16 (2D) / 8×8×8 (3D) → proyección a **dim 192**; padding de secuencia; CLS + posicionales. |
| **Fase 6** | `VisionRouter`: ViT-Tiny preentrenado (`timm`), 12 bloques de atención, `router_head` lineal **192 → 5**. |
| **Fase 7–8** | Prueba dummy; extracción y guardado de **CLS** (`Z_train`/`Z_val`) para ablation. |
| **Entrenamiento router** | `fit_router_with_eval`: `L_task` (CE al `dataset_id` o supervisión configurada), **`L_aux`** estilo Switch (balanceo), `alpha_aux`, **warm-up** (solo cabeza) + **partial unfreeze** de últimos bloques ViT, `CosineAnnealingLR`, métricas (accuracy de routing, ratio max/min de carga, entropía de gating, matriz de confusión, VRAM). |
| **DataLoader balanceado** | `WeightedRandomSampler` + `sample_cap` para no dominar NIH sobre volúmenes minoritarios. |
| **Ablation study** | Sobre embeddings guardados: **A** cabeza lineal sobre `Z`; **B** GMM; **C** Naive Bayes; **D** k-NN (FAISS); tabla comparativa. |
| **Evaluación KPI** | Routing accuracy, balance `max(f_i)/min(f_i) < 1.30`, pico de VRAM, OOD vía entropía del gating (si aplica). |
| **Expert on-demand** | `ExpertOnDemandManager`: expertos en CPU, carga LRU en GPU solo del experto activo (ahorro de VRAM en 2×12 GB). |

---

## Flujo conceptual (router)

1. Entrada: archivo 2D o 3D **sin metadato de modalidad** en la inferencia final (el entrenamiento del router usa `dataset_id` derivado del origen del archivo en el dataset mixto, según la consigna).
2. Preprocesado adaptativo → tensores nativos homogéneos para el siguiente paso.
3. Patch embedding conmutado 2D/3D → tokens **d = 192**.
4. ViT → token **CLS** (vector 192).
5. Capa lineal → **5 logits** (expertos); softmax → probabilidades de gating; **top-1** al experto.
6. Pérdida total: **`L = L_task + α · L_aux`** (balanceo de carga).

---

## Expertos (resumen)

| ID | Dataset | Arquitectura en el notebook | Notas |
|----|---------|-----------------------------|--------|
| 1 | NIH Chest X-ray | `SwinNIHClassifier` (`swin_tiny_patch4_window7_224`), **5 clases** | Entrenamiento de referencia: `NIH_ChestXray_Swin_Tiny_Training.ipynb`; checkpoint típico bajo `03_Weights/Experts_2D/`. |
| 2 | ISIC 2019 | EfficientNet-B3 | |
| 3 | Osteoartritis | VGG16-BN | |
| 4 | LUNA16 | DCSwinBStyle3D | Alineado con `LUNA16_Swin3D_Training.ipynb`. |
| 5 | Páncreas | R3D-18 | |

Los pesos **no** se versionan en Git (`.gitignore`); deben estar en Drive o ruta local según `WEIGHTS_DIR`.

---

## Requisitos

- Python 3.10+ recomendado.
- PyTorch, `torchvision`, `timm`, `monai`, `einops`, `scikit-learn`, `pandas`, opcional `faiss-cpu` para k-NN del ablation.
- GPU recomendada (entrenamiento del router y expertos 3D).

---

## Otros archivos del repositorio

- [`consigna.md`](consigna.md): KPIs del proyecto (p. ej. routing accuracy > 0.80, balance de carga, VRAM).
- [`notebooks/experts/`](notebooks/experts/): notebooks de entrenamiento por dataset y `arquitecturas_de_expertos.py` (referencia; el pipeline principal embebe las definiciones).
- [`dashboard/`](dashboard/): aplicación de demostración / inferencia (si aplica al entregable).
- [`guides/`](guides/): apuntes y papers de referencia.

---

## Clonar y uso rápido

```bash
git clone https://github.com/nicothinn/Clasificaci-n-M-dica-con-MoE-Ablation.git
cd Clasificaci-n-M-dica-con-MoE-Ablation
```

Abrir `notebooks/03_Pipeline_Router_MoE.ipynb` en Jupyter o **Google Colab**, montar Drive si usas rutas `/content/drive/...`, extraer datos con la función de la Fase 1 y ejecutar las celdas en orden.

---

## Autor / curso

Proyecto final — Bloque Visión (clasificación médica multimodal con MoE y ablation del router), según consigna del curso.
