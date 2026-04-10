# Estrategia de Preprocesamiento Adaptativo — Guia de Ingeniero Senior

## 1. Diagnostico: Por que la conversion masiva a MHA no es la estrategia correcta

### Por que Osteoarthritis fue rapido y NIH/ISIC es lento

| Dataset | Imagenes | Formato | Razon de la diferencia |
|---|---|---|---|
| Osteoarthritis | ~10,000 | JPG/PNG 224x224 | Imagenes pequeñas, ya a la resolucion final |
| NIH Chest | 112,120 | PNG 1024x1024 | **11x mas archivos, 20x mas pixels por imagen** |
| ISIC 2019 | 25,331 | JPG | Color RGB (3 canales), compresion JPEG a descomprimir |
| LUNA16 | 888 vol. | MHD+ZRAW | Volumenes 3D (~512x512x300 voxeles cada uno) |
| Pancreas | ~281 vol. | NIfTI .nii.gz | Compresion gzip adicional a descomprimir |

**Conclusion:** Convertir 112,120 PNGs de 1024x1024 (NIH) a MHA toma horas incluso con 60 GB de RAM. No es un problema de RAM sino de I/O de disco y tiempo de CPU.

---

## 2. Lo que pide el Profesor exactamente (desde la consigna)

La consigna tiene dos requisitos criticos que cambian TODA la estrategia:

> **"Sin metadatos — unica entrada: la imagen"**
> El sistema NO recibe texto, etiquetas de modalidad, nombres de archivo ni ningun metadato.
> La deteccion 2D/3D es automatica por la forma del tensor (rank=4 para 2D, rank=5 para 3D).
> Soluciones que pasen la modalidad como entrada adicional son penalizadas con -20%.

> **"Pipeline de preprocesado adaptativo que redimensione imagenes 2D y volumenes 3D automaticamente"**

**Implicacion directa:** El preprocesador NO necesita que todos los archivos esten en MHA.
Necesita ser capaz de **leer cualquier formato en tiempo de ejecucion** y convertirlo a tensor.

---

## 3. La Estrategia Correcta: AdaptivePreprocessor en lugar de conversion masiva

En lugar de hacer una conversion batch enorme de disco a disco (que tarda horas),
implementamos un `AdaptivePreprocessor` que es un transformador **on-the-fly**: lee el
archivo original en su formato nativo SOLO cuando el DataLoader lo necesita.

```
Filosofia: No conviertas los datos. Convierte la lectura.
```

### Diagrama de flujo correcto

```
[PNG / JPG / MHD / NIfTI en disco]
        |
        v
AdaptivePreprocessor.load(path)
  detect_format() → 'png' | 'jpg' | 'mhd' | 'nii.gz'
        |
        |── PNG/JPG  → PIL/SimpleITK → array numpy
        |── MHD      → SimpleITK (carga par .mhd + .zraw auto)
        |── NIfTI    → SimpleITK (lee .nii.gz nativo)
        v
[numpy array: 2D H×W o 3D D×H×W]
        |
        v
detect_rank() → rank==3 (2D) o rank==4 (3D)
        |
        |── 2D → resize(224, 224) → tensor [1, 224, 224]
        |── 3D → clip HU + normalize + resize(64, 64, 64) → tensor [1, 64, 64, 64]
        v
[Tensor listo para el backbone ViT del Router]
```

---

## 4. Paralelismo con 60 GB de RAM (para cuando SI necesites hacer conversion batch)

Si en algun momento necesitas una conversion batch (ej: para guardar en Drive pre-procesado),
usa `multiprocessing` de Python. Con 60 GB de RAM puedes lanzar **8-12 workers** en paralelo:

```python
from multiprocessing import Pool
import os

def convert_one(args):
    src_path, dest_path = args
    # tu logica de conversion aqui
    ...

# Lanzar con N workers = N CPUs que tenga Colab (tipicamente 2-4, pero hasta 8 en Pro)
with Pool(processes=os.cpu_count()) as pool:
    pool.map(convert_one, list_of_tuples)
```

**Por que Osteoarthritis fue mas rapido:**
El notebook de Osteoarthritis corre en Colab con GPU donde el disco local `/content/` es
SSD NVMe. Copiar de Drive a local primero y luego procesar en local es lo mas rapido posible.
Para NIH con 112K archivos simplemente hay mas trabajo. Con 8 workers en paralelo se pode
reducir el tiempo ~4-6x.

---

## 5. Estrategia de Almacenamiento (que guardar y donde)

### Recommendation: NO convertir todo a MHA en Drive

| Opcion | Ventajas | Desventajas | Recomendacion |
|---|---|---|---|
| Guardar todo en MHA en Drive | Lectura consistente | Ocupa el doble de espacio, horas de conversion | No recomendado |
| Guardar MHAs en ZIP en Drive | Menor espacio | Descomprimir en cada sesion de Colab | Solo si el espacio en Drive es muy limitado |
| Dejar formatos originales + AdaptivePreprocessor | Sin conversion, sin espacio extra, rapido | Implementar el preprocessor | **Recomendado** |

**Decision:** Dejar los ZIPs originales en Drive tal como estan.
El `AdaptivePreprocessor` se encarga de leer cualquier formato cuando el DataLoader
hace un `__getitem__`. Esta es exactamente la estrategia que usan los papers SOTA
en imagen medica (MONAI lo hace asi internamente).

---

## 6. Lo que tienes que desarrollar (lista priorizada)

### Prioridad 1: AdaptivePreprocessor (BLOQUE CRITICO)

Clase Python que el Router y los Expertos usan para cargar cualquier imagen.

```python
class AdaptivePreprocessor:
    def load(self, path: str) -> torch.Tensor:
        # detectar formato por extension
        # cargar con la libreria correcta
        # resize segun rank del tensor
        # retornar tensor normalizado
        ...
    
    def is_3d(self, tensor) -> bool:
        return tensor.ndim == 4  # [D, H, W] o [C, D, H, W]
```

### Prioridad 2: Dataset classes para cada fuente

Un `torch.utils.data.Dataset` por cada dataset que use el AdaptivePreprocessor:

```python
class NIHChestDataset(Dataset):
    # Lee PNG originales de disco con AdaptivePreprocessor
    # Etiqueta desde Data_Entry_2017.csv (multilabel)

class ISICDataset(Dataset):
    # Lee JPG originales con AdaptivePreprocessor
    # Etiqueta desde GroundTruth CSV (multiclass)

class LUNA16Dataset(Dataset):
    # Lee MHD+ZRAW con AdaptivePreprocessor
    # Etiqueta desde annotations.csv

class PancreasDataset(Dataset):
    # Lee NIfTI con AdaptivePreprocessor
    # Etiqueta binaria (cancer / no cancer)
```

### Prioridad 3: MixedMedicalDataLoader para el Router

DataLoader combinado proporcional que mezcla los 5 datasets para entrenar el router:

```python
class MixedMedicalDataLoader:
    # Mezcla proporcional: evita que NIH (112K imgs) domine sobre Pancreas (281 vol)
    # Etiqueta de EXPERTO (0-4) para training del router
    # No etiqueta de ENFERMEDAD
```

### Prioridad 4: Streamlit — Transformador en tiempo real

El dashboard en Streamlit necesita un transformador que:

1. Acepta PNG, JPEG, NIfTI como upload del usuario.
2. Llama a `AdaptivePreprocessor.load()` sobre el archivo subido.
3. Pasa el tensor al backbone ViT → router → experto correcto.
4. NO necesita que el archivo este en MHA.

```python
# En streamlit_app.py
uploaded = st.file_uploader("Subir imagen medica", type=["png","jpg","jpeg","nii","gz"])
if uploaded:
    with tempfile.NamedTemporaryFile(suffix=uploaded.name) as tmp:
        tmp.write(uploaded.read())
        tensor = preprocessor.load(tmp.name)
        prediction = model(tensor.unsqueeze(0))
```

---

## 7. Estructura de archivos recomendada para el proyecto

```
PROYECTO_MOE_VISION/
├── 01_Data/
│   └── Raw/                          ← ZIPs originales intactos (no tocar)
│       ├── NIH Chest X ray 14.zip
│       ├── ISIC 2019.zip
│       ├── Luna16 Lung Cancer Dataset.zip
│       ├── Pancreas Cancer.zip
│       └── Knee Osteoarthritis Classification.zip
├── 02_Data_Processed/                ← SOLO Osteoarthritis (ya procesado)
│   └── Experts_2D/                   ← MHAs de Osteoarthritis (CLAHE aplicado)
├── 03_Weights/
│   └── Experts_2D/                   ← .pth del VGG-16 BN (ya generado)
└── 04_Embeddings/                    ← CLS tokens extraidos del backbone ViT
    ├── NIH_cls_tokens.npy            ← Para el ablation study del Router
    ├── ISIC_cls_tokens.npy
    ├── LUNA16_cls_tokens.npy
    ├── Pancreas_cls_tokens.npy
    └── Osteo_cls_tokens.npy
```

---

## 8. Plan de Desarrollo Semana a Semana (alineado con la consigna)

### Semana S9 — Actual (Expertos individuales)

- [x] VGG-16 BN para Osteoarthritis (Experto 3) — COMPLETADO
- [ ] Implementar `AdaptivePreprocessor` (bloque critico)
- [ ] Implementar y entrenar Expertos 1 (NIH — ConvNeXt-Tiny) y 2 (ISIC — EfficientNet-B3)

### Semana S10 — Expertos 3D + Backbone ViT + Ablation

- [ ] Expertos 4 (LUNA16 — R3D-18 con grad checkpointing) y 5 (Pancreas)
- [ ] Backbone ViT-Tiny (timm) compartido → extraccion de CLS tokens → guardar en `04_Embeddings/`
- [ ] Ablation study: ViT+Linear vs GMM vs Naive Bayes vs k-NN (FAISS)

### Semana S11 — Sistema MoE + Calibracion

- [ ] Ensamble MoE con router ganador
- [ ] Calibrar alpha de L_aux (Switch Transformer)
- [ ] Verificar cociente max(f_i)/min(f_i) < 1.30

### Semana S12 — Dashboard Streamlit

- [ ] `AdaptivePreprocessor` integrado en el endpoint de carga
- [ ] Attention Heatmap del router ViT
- [ ] Panel de ablation study visual
- [ ] OOD Detection

---

## 9. Codigo base del AdaptivePreprocessor (para implementar en el notebook de Router)

```python
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

class AdaptivePreprocessor:
    SIZE_2D = (224, 224)
    SIZE_3D = (64, 64, 64)
    HU_MIN, HU_MAX = -1000, 400    # Rango general CT

    def load(self, path: str) -> torch.Tensor:
        path = str(path)
        ext = ''.join(Path(path).suffixes).lower()

        # Cargar segun formato
        if ext in ['.png', '.jpg', '.jpeg']:
            img = sitk.ReadImage(path)
        elif ext in ['.mhd', '.mha']:
            img = sitk.ReadImage(path)
        elif ext in ['.nii', '.nii.gz', '.gz']:
            img = sitk.ReadImage(path)
        else:
            raise ValueError(f'Formato no soportado: {ext}')

        arr = sitk.GetArrayFromImage(img).astype(np.float32)

        # Detectar 2D vs 3D por la forma del array
        if arr.ndim == 2:
            return self._process_2d(arr)
        elif arr.ndim == 3:
            # Puede ser 2D RGB (H, W, 3) o 3D volumetrico (D, H, W)
            if arr.shape[-1] in [1, 3, 4]:     # ultima dim es canal de color
                arr = arr.mean(axis=-1)         # convertir a gris
                return self._process_2d(arr)
            else:
                return self._process_3d(arr)   # volumen 3D
        else:
            raise ValueError(f'Shape no reconocida: {arr.shape}')

    def _process_2d(self, arr: np.ndarray) -> torch.Tensor:
        # Normalizar a [0, 1]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
        t = F.interpolate(t, size=self.SIZE_2D, mode='bilinear', align_corners=False)
        return t.squeeze(0)    # [1, 224, 224]

    def _process_3d(self, arr: np.ndarray) -> torch.Tensor:
        # Clipping HU + normalizacion
        arr = np.clip(arr, self.HU_MIN, self.HU_MAX)
        arr = (arr - self.HU_MIN) / (self.HU_MAX - self.HU_MIN)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # [1, 1, D, H, W]
        t = F.interpolate(t, size=self.SIZE_3D, mode='trilinear', align_corners=False)
        return t.squeeze(0)    # [1, 64, 64, 64]
```

---

## 10. El Flujo del Backbone ViT y el Router (Paso a Paso)

Una vez que la imagen pasa por el `AdaptivePreprocessor` y se convierte en un Tensor uniforme de PyTorch, entra al **Router**. El Router tiene dos partes fundamentales: el *Backbone* (extractor de caracteristicas) y la *Cabeza* (el tomador de decision).

**Conexión con el estado del arte (SIFA):** Tal como plantea el modelo SIFA (*Synergistic Image and Feature Adaptation*, Chen et al. 2019), el éxito en imágenes médicas de diferentes dominios (ej. CT vs MRI, o 2D vs 3D) se logra atacando el problema desde dos frentes simultáneos:
1. **Adaptación de Imagen (Image-level):** Lo que hace nuestro `AdaptivePreprocessor` proyectando todo a tensores estandarizados de [1, 224, 224] o [1, 64, 64, 64] y normalizando las ventanas HU.
2. **Adaptación de Features (Feature-level):** Lo que logrará nuestro ViT al extraer el `[CLS]` token congelado que debe funcionar como un "espacio o representación latente invariante" (domain-invariant feature) antes de que el modelo decida a qué experto enviarlo. Nuestro sistema MoE (Mixture of Experts) es inherentemente una red que aprende a manejar distribuciones heterogéneas.

### Paso 1: El Backbone ViT (Vision Transformer)

El tensor preprocesado (sea 2D de `224x224` o 3D proyectado a 2D) entra a un **Vision Transformer (ViT)** pre-entrenado.
- **Qué hace el ViT:** Corta la imagen en "parches" (patches) de 16x16 píxeles.
- **El poder del Self-Attention (Conexión con SASAN):** Tal como demostró el reciente paper SASAN (*Self-Attentive Spatial Adaptive Normalization*, Tomar et al. 2021), los mecanismos de auto-atención (self-attention) son superiores a las CNN puras en adaptación de dominio médico porque logran atender a diferentes estructuras anatómicas sin deformarlas, capturando la relación geométrica entre los tejidos sin importar la modalidad (ej. distinguir la aorta de la aurícula independientemente de si es CT o MRI).
- **El CLS Token (El Embedding):** El ViT añade un parche especial falso al principio llamado `[CLS]` (Classification Token). A medida que las capas de Atención procesan la imagen, este token `[CLS]` recolecta y resume TODA la información visual y espacial de las estructuras en un solo vector numérico (ej. de tamaño `[1, 192]`).
- **Estado en tu proyecto:** Este backbone ViT estará **congelado** (no se entrena). Su mecanismo de self-attention aprendido en ImageNet extraerá representaciones robustas generales.

### Paso 2: La Extraccion de Embeddings (Fase Offline)

En la Semana 10, no vas a entrenar redes enviando imagenes en tiempo real al principio. Lo que haras es:

1. Pasar las ~150,000 imagenes de tus 5 datasets por el ViT congelado.
2. Guardar a disco **solo el vector CLS de salida** de cada imagen (esto es extremadamente ligero, mb de peso). Has convertido 100GB de imagenes en unos pocos MB de vectores numericos ("embeddings").

### Paso 3: El Ablation Study del Router (Cabezas de Decision)

Ahora viene la competencia científica que pide el profesor. Sobre esos embeddings (CLS tokens) ya guardados, vas a entrenar 4 "cabezas" matemáticas diferentes para ver cual es mejor decidiendo a qué experto (1,2,3,4 o 5) pertenece el embedding:

1. **Cabeza Lineal (Deep Learning):** Capa densa `nn.Linear` conectada al CLS que aprende con la funcion softmax.
2. **GMM (Gaussian Mixture Model):** Distribuciones de probabilidad multivariadas para clusterizar a que enfermedad se parece mas el CLS.
3. **Naive Bayes:** Calculo de probabilidad condicional simple.
4. **k-NN (K-Nearest Neighbors con FAISS):** Busca en el espacio vectorial cuales son los otros 5 CLS historicos mas cercanos a este nuevo CLS y decide por "voto de la mayoría".

### Paso 4: Decision Final (Testing/Produccion)

La cabeza matemática que gane el Ablation Study se queda como tu **Router final**.
En la demostracion de Streamlit:
`Imagen User` -> `AdaptivePreprocessor` -> `Tensor` -> `ViT Congelado` -> `CLS Token` -> `Router Ganador` -> `Decide por ejemplo: "Experto 3 (Osteo)"` -> `Imagen fluye al Experto 3` -> `Predice Grado KL`.

---

## 11. Términos Clave (Keywords) para Buscar Papers

Para tu reporte técnico de 7 paginas y sustentacion, busca papers en Google Scholar, arXiv o IEEE usando estas frases y su traduccion rigurosa en ingles:

### Para la arquitectura y enrutamiento (El core de tu reporte)

* **Mezcla de Expertos (MoE):** *Mixture of Experts for Image Classification*
- **Equilibrio de Carga (Load Balancing):** *Expert Load Balancing in Switch Transformers* (Busca el mecanismo de *Auxiliary Loss*)
- **Estudio de Ablación del Router:** *Routing Mechanisms in Vision MoE* / *Gating Functions Ablation Study*
- **Colapso de Expertos (Cuando todos van al mismo):** *Expert Collapse mitigation in MoE models*
- **Modelos híbridos DL-Estadísticos:** *Non-parametric gating networks for Mixture of Experts* (Para justificar FAISS/GMM frente a Linear softmax).

### Para los datasets médicos (Para justificar los Preprocesamientos)

* **Adaptación de Dominio Sin Supervisión (UDA):** *Unsupervised domain adaptation for medical image segmentation / Synergistic Image and Feature Adaptation (SIFA)*. Usa este paper (Chen et al. 2019) para justificar que estandarizar imágenes de distintos dominios (CT vs MRI, 2D vs 3D) requiere tanto transformación de imagen (nuestro preprocesador) como invariancia de features (nuestro CLS Token).
* **Mecanismos de Auto-Atención para Preservar Geometría (UDA):** *Self-Attentive Spatial Adaptive Normalization for Cross-Modality Domain Adaptation (SASAN)*. Usa este paper (Tomar et al. 2021) para argumentar por qué un Vision Transformer basado en Self-Attention (que atiende espacialmente a los órganos) es el extractor de features ideal frente a los enfoques pasados.
* **Preprocesamiento Adaptativo 2D/3D:** *Domain Adaptation between 2D and 3D medical modalities* / *Cross-Modality Transfer Learning in Medical Imaging*
- **Problemas de Desbalanceo (NIH y Pancreas):** *Handling severe class imbalance with Focal Loss in Medical CT* / *Class-weighted formulation for multi-label classification*.

### Para el clasificador (Tu "cabeza" sobre el VGG o ViT)

* **Características del CLS:** *Representational power of ViT CLS token in transfer learning*
- **Ajuste fino (Fine-tuning):** *Few-shot Fine-tuning of Vision Transformers on Medical Datasets*.
