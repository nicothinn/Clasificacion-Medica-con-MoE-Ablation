# NIH Chest X-ray 14 — Guía del EDA alineado a Swin-Tiny

Documentación de [`NIH_ChestXray_EDA_Preprocessing.ipynb`](NIH_ChestXray_EDA_Preprocessing.ipynb) y su relación con [`NIH_ChestXray_Swin_Tiny_Training.ipynb`](NIH_ChestXray_Swin_Tiny_Training.ipynb).

---

## Tabla de contenidos

1. [Rol del notebook](#1-rol-del-notebook)
2. [Dataset y zip](#2-dataset-y-zip)
3. [Configuración compartida (5 clases)](#3-configuración-compartida-5-clases)
4. [EDA y decisiones de entrenamiento](#4-eda-y-decisiones-de-entrenamiento)
5. [Preprocesamiento y orden de operaciones](#5-preprocesamiento-y-orden-de-operaciones)
6. [MONAI e ImageNet](#6-monai-e-imagenet)
7. [Diagrama de flujo](#7-diagrama-de-flujo)

---

## 1. Rol del notebook

El EDA **no entrena** el modelo. Sirve para:

- Verificar ingesta desde Drive y cruce CSV ↔ archivos PNG.
- Cuantificar **desbalance** y **multietiqueta** en las mismas 5 clases que el experto.
- Justificar **GroupShuffleSplit por PatientID** (evitar fuga paciente entre train/val/test).
- Mostrar **antes/después** del preprocesamiento real (CLAHE, 224, MONAI) sobre pocas imágenes.

Versiones antiguas del EDA referían **DenseNet-121**, conversión **MHA** o pipelines con **UNet** de pulmón; eso **no** coincide con el entrenamiento Swin-Tiny documentado y fue retirado.

---

## 2. Dataset y zip

- Archivo típico: `NIH Chest X ray 14.zip` en `01_Data/Raw/` (Drive).
- Tras descomprimir aparecen CSV (`Data_Entry_2017.csv`) y carpetas de imágenes; a menudo hay **ZIPs internos** (`images_001.zip`, …) que deben extraerse (misma lógica que el training).

---

## 3. Configuración compartida (5 clases)

Lista alineada al `CONFIG['classes']` del entrenamiento:

| Clase         | Uso |
|---------------|-----|
| Mass          | Etiqueta binaria por imagen |
| Nodule        | Idem |
| Effusion      | Idem |
| Cardiomegaly  | Idem |
| Pneumothorax  | Idem |

**Filtro restrictivo:** se conservan solo filas donde `Finding Labels` contiene **al menos una** de estas cinco cadenas. Es el mismo criterio que reduce ~112k filas a ~27k muestras en el flujo de entrenamiento.

---

## 4. EDA y decisiones de entrenamiento

| Hallazgo en EDA | Decisión en `NIH_ChestXray_Swin_Tiny_Training` |
|-----------------|-----------------------------------------------|
| Frecuencias muy distintas entre patologías | **ASL** (Asymmetric Loss) y manejo de multietiqueta; opcional `WeightedRandomSampler` |
| Varios estudios por `PatientID` | **GroupShuffleSplit** por paciente |
| Contraste variable en radiografías | **CLAHE** adaptativo en LAB + **GaussianBlur** antes del resize |
| Compatibilidad con backbone `timm` / ImageNet | **NormalizeIntensityd** con media/std ImageNet **en el Dataset**, no en el `.npz` |
| Tamaño de entrada fijo | **224×224** tras CLAHE |

La co-ocurrencia **5×5** en el EDA ayuda a interpretar correlaciones entre patologías (multietiqueta real); no implica que el modelo prediga solo una clase.

---

## 5. Preprocesamiento y orden de operaciones

1. Leer PNG con OpenCV (`BGR`).
2. Convertir a **RGB**.
3. **CLAHE** en el canal L de **LAB** (parámetros adaptativos como en `materialize_nih_cache`).
4. **GaussianBlur** (5×5, `sigmaX=1`).
5. **Resize** a 224×224.
6. Dividir por 255 → tensor **CHW** en **[0, 1]**.
7. Guardar en `.npz` (en el notebook de training) **sin** restar media ImageNet.

---

## 6. MONAI e ImageNet

En entrenamiento, tras cargar el tensor [0,1] CHW:

\[
\hat{x}_{c} = \frac{x_{c} - \mu_{c}}{\sigma_{c}}, \quad \mu = [0.485, 0.456, 0.406],\; \sigma = [0.229, 0.224, 0.225]
\]

(con `NormalizeIntensityd` canal a canal, como en `CachedNIHDataset`).

`ScaleIntensityd` con `minv=0, maxv=1` mantiene el rango explícito antes de la normalización.

---

## 7. Diagrama de flujo

```mermaid
flowchart LR
  png[PNG_disk]
  bgr[cv2_imread_BGR]
  rgb[RGB]
  clahe[CLAHE_LAB_Gauss]
  r224[Resize_224]
  chw01[CHW_0_1]
  npz[npz_cache]
  aug[Augment_train_only]
  monai[MONAI_Scale_Normalize]
  swin[Swin_Tiny]

  png --> bgr --> rgb --> clahe --> r224 --> chw01 --> npz
  npz --> aug --> monai --> swin
```

---

*Última alineación con el pipeline descrito en `NIH_ChestXray_Swin_Tiny_Training.ipynb`.*
