# Clasificacion Medica Multimodal con Mixture of Experts y Ablation Study del Router

**Proyecto final** — Ingenieria de Datos e Inteligencia Artificial, Universidad Autonoma de Occidente.

**Autores:** Andres Alberto Enriquez (2222055), Nicolas Pena Irurita (2232049), Samuel Patino Lucumi.

---
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/b267890d-cb0e-4dcc-8d69-0e171fc182b1" />

## 1. Resumen del proyecto

Sistema de **Mixture of Experts (MoE)** para clasificacion de imagenes medicas multimodales (2D y 3D) sin metadatos clinicos. Un **Vision Transformer (ViT-Tiny)** compartido extrae embeddings CLS sobre los que un **router entrenable** decide cual de cinco expertos heterogeneos procesa cada imagen.

**Pregunta cientifica central:** *¿Justifica un Vision Transformer su costo computacional como router frente a metodos estadisticos clasicos (GMM, Naive Bayes, k-NN) que operan sobre los mismos embeddings?*

**Contribucion del equipo:**
1. Sistema MoE end-to-end con preprocesador adaptativo 2D/3D.
2. Ablation study formal de 4 mecanismos de routing (consigna).
3. Ablation extra: cabezas diferenciables (NeuralGMM, LogGaussianNB, SoftKNN) y Kolmogorov-Arnold (KAN).
4. Auxiliary loss de Switch Transformer calibrada para balance de carga.

---

## 2. Estructura del repositorio

```
proyecto-2/
|-- README.md                          <-- Este archivo
|-- requirements.txt                   <-- Dependencias (raiz, versiones exactas)
|-- consigna.md                        <-- Requisitos y KPIs del proyecto
|-- main.tex                           <-- Reporte tecnico (LaTeX IEEE)
|
|-- notebooks/
|   |-- 01_Exploracion_Estructura_Datasets.ipynb
|   |-- 02_Preprocesamiento_Multiple_Adaptive.ipynb
|   |-- 03_Pipeline_Router_MoE.ipynb           <-- Pipeline principal del MoE
|   |-- 04_Entrenamiento_MoE_Fases.ipynb
|   |-- 05_Router_Vit_Lineal_Solo.ipynb        <-- Router ViT + Linear (con feedback)
|   |-- 06_Ablation_Study_Statistical_Routers.ipynb  <-- Ablation obligatorio (4 metodos)
|   |-- 06_Ablation_NeuralGMM.ipynb            <-- Extra: GMM diferenciable
|   |-- 06_Ablation_LogGaussianNB.ipynb        <-- Extra: NB diferenciable
|   |-- 06_Ablation_SoftKNN.ipynb              <-- Extra: SoftKNN diferenciable
|   |-- 06_Router_ViT_KAN.ipynb                <-- Extra: cabeza KAN
|   |-- 06_GradCAM_Diagnostico_NIH_vs_Osteo.ipynb
|   |
|   `-- experts/                               <-- Entrenamiento de cada experto
|       |-- NIH_ChestXray_EDA_Preprocessing.ipynb
|       |-- NIH_ChestXray_Swin_Tiny_Training.ipynb
|       |-- ISIC_2019_EDA_Preprocessing.ipynb
|       |-- ISIC2019_EfficientNetB3_Training_Final.ipynb
|       |-- Osteoarthritis_eda_data_preprocecing.ipynb
|       |-- Osteoarthritis_VGG16BN_Training.ipynb
|       |-- LUNA16_EDA_Preprocessing.ipynb
|       |-- LUNA16_R3D18_Training.ipynb
|       |-- Pancreas_eda_data_preprocecing.ipynb
|       |-- Pancreas_R3D18_Training.ipynb
|       |-- GradCAM_Experts_ISIC.ipynb
|       `-- GradCAM_Pancreas_R3D18_Training.ipynb
|
|-- dashboard/
|   |-- app.py                     <-- Frontend Streamlit
|   |-- server.py                  <-- Backend FastAPI/Uvicorn
|   |-- moe_inference.py           <-- Motor de inferencia MoE
|   |-- preprocessing.py           <-- Preprocesado adaptativo
|   |-- heatmap_utils.py           <-- Attention heatmaps
|   |-- ood_utils.py               <-- OOD detection por entropia
|   |-- real_models.py             <-- Carga de modelos reales
|   |-- mock_models.py             <-- Modelos mock para demo
|   |-- test_inference.py
|   |-- test_client.py
|   |-- requirements.txt
|   `-- static/                    <-- HTML/CSS/JS del panel web
|
|-- data/                          <-- Datasets (NO versionados, ver .gitignore)
|-- weights/                       <-- Checkpoints (NO versionados)
|-- docs/                          <-- Documentacion interna
|-- guides/                        <-- Papers y apuntes de referencia
`-- scripts/                       <-- Utilidades auxiliares
```

> **Nota:** Los directorios `data/`, `weights/`, `docs/` y `guides/` estan excluidos del repositorio por `.gitignore`. Los pesos se almacenan en Google Drive bajo `PROYECTO_MOE_VISION/03_Weights/`.

---

## 3. Requisitos de hardware y entorno

### Cluster del curso (configuracion objetivo)
- **GPU:** 1x NVIDIA T4 / L4 (12-16 GB VRAM).
- **RAM:** 12+ GB.
- **Disco:** >= 50 GB libres para datasets descomprimidos.
- **Python:** 3.10+.
- **CUDA:** 11.8 o 12.1.

### Google Colab (alternativa)
- Runtime GPU (T4 gratuita o L4/A100 en Pro).
- Montar Google Drive para datasets y checkpoints.
- Los notebooks estan disenados para correr en Colab sin cambios.

---

## 4. Instalacion

### Opcion A: Cluster / maquina local

```bash
git clone https://github.com/nicothinn/Clasificaci-n-M-dica-con-MoE-Ablation.git
cd Clasificaci-n-M-dica-con-MoE-Ablation

# Crear entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Instalar dependencias (versiones exactas)
pip install -r requirements.txt
```

### Opcion B: Google Colab

Abrir el notebook deseado directamente en Colab. La Fase 0 de cada notebook instala las dependencias faltantes (`!pip install timm monai einops ...`). Montar Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 5. Ejecucion paso a paso (orden recomendado)

### Paso 1 — Entrenar expertos (si no se tienen checkpoints)

Ejecutar **en orden** los notebooks de `notebooks/experts/`:

| Orden | Notebook | Experto | Tiempo aprox. |
|------:|----------|---------|---------------|
| 1 | `NIH_ChestXray_Swin_Tiny_Training.ipynb` | Exp 1: Swin-Tiny (NIH, 5 cls) | ~3-4 h |
| 2 | `ISIC2019_EfficientNetB3_Training_Final.ipynb` | Exp 2: EfficientNet-B3 (ISIC, 8 cls) | ~1-2 h |
| 3 | `Osteoarthritis_VGG16BN_Training.ipynb` | Exp 3: VGG-16 BN (Osteo, 5 cls KL) | ~1 h |
| 4 | `LUNA16_R3D18_Training.ipynb` | Exp 4: R3D-18 (LUNA16, binario 3D) | ~4-6 h |
| 5 | `Pancreas_R3D18_Training.ipynb` | Exp 5: R3D-18 (Pancreas, binario 3D) | ~2-3 h |

Los checkpoints se guardan automaticamente en `03_Weights/` en Drive.

### Paso 2 — Pipeline MoE y extraccion de CLS tokens

```
notebooks/03_Pipeline_Router_MoE.ipynb
```

Ejecutar todas las celdas en orden:
- Fase 0: instalacion y montaje.
- Fase 1: descompresion de datasets (~1.5 h).
- Fase 2-3: preprocesado adaptativo.
- Fase 4: DataLoader mixto balanceado.
- Fase 5-6: SwitchablePatchEmbed + VisionRouter (ViT-Tiny).
- Fase 7-8: extraccion y guardado de CLS tokens (`Z_train`, `Z_val`).

### Paso 3 — Ablation study obligatorio (consigna)

```
notebooks/06_Ablation_Study_Statistical_Routers.ipynb
```

Compara 4 mecanismos sobre los mismos embeddings CLS:
- **A.** ViT + Linear (SGD, grid de lr y alpha).
- **B.** ViT + GMM (sklearn, EM).
- **C.** ViT + Naive Bayes (sklearn, MLE).
- **D.** ViT + k-NN (FAISS + PCA).

Produce la tabla comparativa con Routing Accuracy, Ratio y Score compuesto.

### Paso 4 — Router con feedback de expertos

```
notebooks/05_Router_Vit_Lineal_Solo.ipynb
```

Entrenamiento del router ViT + Linear con flujo profesor (`L_routing + L_task + alpha*L_aux`), expertos activos, dos fases (head_only + full finetune).

### Paso 5 — Ablation extra (diferenciables + KAN)

| Notebook | Cabeza |
|----------|--------|
| `06_Ablation_NeuralGMM.ipynb` | GMM diferenciable (warm-up maestro + GMM) |
| `06_Ablation_LogGaussianNB.ipynb` | Log-Gaussian Naive Bayes diferenciable |
| `06_Ablation_SoftKNN.ipynb` | SoftKNN (prototipos aprendibles) |
| `06_Router_ViT_KAN.ipynb` | FastKAN (Kolmogorov-Arnold) |

> Los notebooks `06_Ablation_NeuralGMM` genera el warm-up maestro (`backbone_master_warmup.pth`, `projector_master.pth`, `unified_z_64.npz`) que comparten SoftKNN y LogGaussianNB.

### Paso 6 — Evaluacion y artefactos

Los logs de cada notebook incluyen:
- `val_acc`, `val_ratio`, `val_entropy` por epoca.
- Matrices de confusion de routing.
- Checkpoints del mejor modelo por score compuesto.

Para Grad-CAM cruzado (diagnostico NIH vs Osteo):
```
notebooks/06_GradCAM_Diagnostico_NIH_vs_Osteo.ipynb
```

---

## 6. Dashboard

El dashboard cumple los items 15-22 de la consigna:

| Funcionalidad | Descripcion |
|---------------|-------------|
| Carga de imagen | PNG, JPEG, NIfTI. Deteccion automatica 2D/3D. |
| Preprocesado | Muestra dimensiones originales vs adaptadas. |
| Inferencia | Etiqueta predicha, confianza, latencia en ms. |
| Attention Heatmap | Mapa de calor del router ViT sobre la imagen. |
| Panel del experto | Nombre, arquitectura, dataset, gating score. |
| Panel ablation | Tabla comparativa de los 4 metodos de routing. |
| Load Balance | Grafica de barras con f_i acumulado y ratio. |
| OOD Detection | Alerta cuando la entropia del gating supera umbral. |

### Como correrlo

**Opcion 1 — Streamlit (frontend interactivo):**

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Se abre en `http://localhost:8501`.

**Opcion 2 — FastAPI (backend API REST):**

```bash
cd dashboard
uvicorn server:app --host 0.0.0.0 --port 8000
```

Endpoints disponibles en `http://localhost:8000/docs`.

**Opcion 3 — Colab:**

Ejecutar `app.py` en una celda de Colab con `!streamlit run app.py &` y usar el tunnel de Colab o `localtunnel` para acceder.

---

## 7. Reproducibilidad

### 7.1 Seeds fijas

Todos los notebooks fijan la semilla **SEED = 42** al inicio del entrenamiento:

```python
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Los notebooks de expertos ademas activan modo determinista de cuDNN:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Notebooks con seed verificada:**

| Notebook | Seed | cuDNN deterministic |
|----------|------|---------------------|
| `03_Pipeline_Router_MoE.ipynb` | 42 | No (rendimiento) |
| `05_Router_Vit_Lineal_Solo.ipynb` | 42 | No |
| `06_Ablation_SoftKNN.ipynb` | 42 | No |
| `06_Ablation_NeuralGMM.ipynb` | 42 | No |
| `06_Ablation_LogGaussianNB.ipynb` | 42 | No |
| `06_Router_ViT_KAN.ipynb` | 42 | No |
| `06_Ablation_Study_Statistical_Routers.ipynb` | 42 | N/A (sklearn) |
| `ISIC2019_EfficientNetB3_Training_Final.ipynb` | 42 | Si |
| `Pancreas_R3D18_Training.ipynb` | 42 | Si |
| `NIH_ChestXray_Swin_Tiny_Training.ipynb` | 42 | No |
| `LUNA16_R3D18_Training.ipynb` | 42 | No |
| `Osteoarthritis_VGG16BN_Training.ipynb` | 42 | No |

### 7.2 Rutas y checkpoints

Los pesos se almacenan en Google Drive:

```
PROYECTO_MOE_VISION/
`-- 03_Weights/
    |-- Experts_2D/
    |   |-- Expert1_NIH/         <-- Swin-Tiny
    |   |-- Expert2_ISIC/        <-- EfficientNet-B3
    |   |-- Expert3_Osteo/       <-- VGG-16 BN
    |   `-- Expert4_LUNA16/      <-- R3D-18
    |-- Experts_3D/
    |   `-- Expert5_Pancreas/    <-- R3D-18
    `-- Router/
        |-- router_professor_fase3_only_base.pth
        |-- router_professor_fase3_only_finetuned.pth
        |-- router_soft_knn.pth
        |-- router_neural_gmm.pth
        |-- router_log_gaussian_nb.pth
        |-- backbone_master_warmup.pth
        |-- projector_master.pth
        |-- unified_cls_tokens.npz
        `-- unified_z_64.npz
```

### 7.3 Variabilidad esperada

- Con `cudnn.deterministic=False` (router), variaciones de +/- 0.5% en `val_acc` entre runs.
- Con `cudnn.deterministic=True` (expertos ISIC/Pancreas), resultados identicos en misma GPU.
- Distintas GPUs (T4 vs A100) pueden producir diferencias menores por precision numerica.

---

## 8. Metricas y umbrales de la consigna

### Umbrales exigidos

| Metrica | Umbral | Descripcion |
|---------|--------|-------------|
| F1 Macro (expertos 2D) | > 0.72 | NIH, ISIC, Osteo |
| F1 Macro (expertos 3D) | > 0.65 | LUNA16, Pancreas |
| Routing Accuracy | > 0.80 | Mejor router del ablation |
| Load Balance ratio | < 1.30 | `max(f_i)/min(f_i)` en router ViT final |
| VRAM por GPU | < 11.5 GB | FP16 + grad checkpointing |

### Resultados obtenidos

| Exp | Dataset | F1-Macro | Umbral | Cumple |
|----:|---------|---------|--------|:------:|
| 1 | NIH ChestX-ray14 | 0.6063 | > 0.72 | No |
| 2 | ISIC 2019 | 0.7395 | > 0.72 | Si |
| 3 | Osteoartritis | 0.8176 | > 0.72 | Si |
| 4 | LUNA16 (3D) | 0.6620 | > 0.65 | Si |
| 5 | Pancreas (3D) | 0.6937 | > 0.65 | Si |

| Router (ablation obligatorio) | Routing Acc | Ratio | Score |
|-------------------------------|------------|-------|-------|
| ViT + k-NN (FAISS, PCA=96, k=3) | **0.9629** | 1.839 | **0.9521** |
| ViT + Linear (Softmax) | 0.8326 | 2.241 | 0.8138 |
| ViT + Naive Bayes | 0.8202 | 2.438 | 0.7975 |
| ViT + GMM (diag) | 0.8011 | 2.027 | 0.7866 |

| Router (extra, con feedback) | val_acc | ratio |
|------------------------------|---------|-------|
| ViT + SoftKNN (head_only) | **0.9483** | **1.13** |
| ViT + Linear diferenciable | 0.9472-0.9494 | 1.25-1.29 |
| ViT + NeuralGMM | 0.8169 (oscila) | 2.2-28.3 |
| ViT + LogGaussianNB | 0.7270 -> 0.1955 | > 10^8 (colapso) |

**Balance de carga MoE final:** ratio estabilizado en 1.21 (< 1.30).

---

## 9. Trazabilidad: consigna -> evidencia

| Requisito de consigna | Evidencia en el repositorio |
|----------------------|----------------------------|
| ViT como router | `notebooks/03_Pipeline_Router_MoE.ipynb` (VisionRouter, ViT-Tiny) |
| 3 routers estadisticos baseline | `notebooks/06_Ablation_Study_Statistical_Routers.ipynb` (GMM, NB, k-NN) |
| Ablation study formal (4 mecanismos) | Mismo notebook, tabla final con Acc/Ratio/Score |
| Preprocesador adaptativo 2D/3D | `notebooks/03_Pipeline_Router_MoE.ipynb` (AdaptivePreprocessor) |
| 5 expertos heterogeneos | `notebooks/experts/*_Training*.ipynb` (5 notebooks) |
| Auxiliary loss Switch Transformer | `notebooks/05_Router_Vit_Lineal_Solo.ipynb` (`_switch_aux_loss`) |
| Calibracion de alpha | Idem, alpha_aux = 0.01 (Linear), 0.10/0.04 (SoftKNN) |
| Load balance < 1.30 | Logs de `05_Router_Vit_Lineal_Solo.ipynb` (ratio 1.21-1.27) |
| F1 Macro por dataset | Logs de cada `*_Training*.ipynb` en `notebooks/experts/` |
| Gradient checkpointing 3D | `Pancreas_R3D18_Training.ipynb`, `LUNA16_R3D18_Training.ipynb` |
| Dashboard (items 15-22) | `dashboard/app.py` (Streamlit), `dashboard/server.py` (FastAPI) |
| OOD Detection (entropia gating) | `dashboard/ood_utils.py`, `dashboard/app.py` (panel OOD) |
| Attention Heatmap | `dashboard/heatmap_utils.py` |
| Seeds fijas | Funcion `set_seed(42)` en todos los notebooks |
| `requirements.txt` | `requirements.txt` (raiz) + `dashboard/requirements.txt` |
| Reporte tecnico (7 secciones, <= 7 pag.) | `main.tex` (IEEE, estructura de consigna) |
| Grad-CAM diagnostico NIH vs Osteo | `notebooks/06_GradCAM_Diagnostico_NIH_vs_Osteo.ipynb` |

---

## 10. Limitaciones y troubleshooting

### Tiempos de ejecucion

| Etapa | Tiempo aproximado |
|-------|-------------------|
| Descompresion de datasets | ~1.5 h |
| Normalizacion/preprocesado (CLAHE, HU, cache) | 1-3 h |
| Entrenamiento de cada experto | 1-6 h segun dataset |
| Extraccion de CLS tokens | ~30 min |
| Ablation estadistico (4 metodos, CPU) | ~15 min |
| Entrenamiento router con feedback (1 epoca) | 40-60 min |
| Dashboard (arranque) | ~30 s |

### Problemas comunes

| Problema | Solucion |
|----------|----------|
| `CUDA out of memory` en 3D | Reducir `batch_size` a 2-4; verificar que `gradient checkpointing` este activo |
| `FileNotFoundError` de pesos | Verificar que Drive este montado y la ruta `WEIGHTS_DIR` sea correcta |
| Validacion del router con val_acc muy baja | Regenerar cache CLS (`unified_cls_tokens.npz`) con el ViT actual |
| Confusion Exp 1 vs Exp 3 (NIH/Osteo) | Verificar que CLAHE este aplicado en preprocesado del router |
| NIH no alcanza F1 > 0.72 | Limitacion del dataset (ruido multilabel); no es del router |
| Colab se desconecta | Guardar checkpoints frecuentes en Drive; usar `SAVE_EVERY_N_EPOCHS` |

### Limitaciones de hardware

- Una sola GPU (12-16 GB) en entorno educativo.
- I/O desde Google Drive (mas lento que SSD local).
- Sin multi-GPU en la mayoria de runs.
- Consecuencia: menos exploracion de hiperparametros y menos seeds por configuracion.

---

## 11. Referencias principales

1. Fedus et al. (2022). *Switch Transformers.* JMLR.
2. Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words.* ICLR.
3. Liu et al. (2021). *Swin Transformer.* ICCV.
4. Tan & Le (2019). *EfficientNet.* ICML.
5. Snell et al. (2017). *Prototypical Networks.* NeurIPS.
6. Wang et al. (2017). *ChestX-ray8.* CVPR.
7. Combalia et al. (2019). *BCN20000 / ISIC 2019.*
8. Setio et al. (2017). *LUNA16.* Medical Image Analysis.
9. Roth et al. (2015). *DeepOrgan / Pancreas.* MICCAI.
10. Ridnik et al. (2021). *Asymmetric Loss.* ICCV.

---

*Proyecto final — Bloque Vision, clasificacion medica multimodal con MoE y ablation del router.*
