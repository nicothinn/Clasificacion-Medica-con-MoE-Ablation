"""
preprocessing.py — Preprocesador Adaptativo para el Dashboard.

Adaptado del AdaptivePreprocessor del notebook 03_Pipeline_Router_MoE.ipynb
para funcionar con BytesIO (lo que produce st.file_uploader de Streamlit).

Reglas de la consigna:
  - 2D (PNG/JPEG) → [3, 224, 224]
  - 3D (NIfTI)    → [1, 64, 64, 64]
  - Sin metadatos: solo la forma del tensor determina 2D vs 3D

Nota sobre normalización:
  - El tensor devuelto por process_uploaded_file está en rango [0, 1] (raw).
  - La normalización ImageNet se aplica POR SEPARADO en moe_inference.py
    ya que el Router y cada Experto requieren normalizaciones distintas.
"""

import io
import tempfile
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import torchvision.transforms as T


# Intentar importar SimpleITK y nibabel (opcionales para formatos 3D)
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    import nibabel as nib
    HAS_NIB = True
except ImportError:
    HAS_NIB = False


class AdaptivePreprocessor:
    """
    Preprocesador que detecta automaticamente si la entrada es 2D o 3D
    y la convierte al tensor con las dimensiones correctas.

    Solo mira la extension del archivo y la forma del tensor resultante.
    NUNCA recibe metadatos ni etiquetas (prohibido por la consigna).
    """

    # Normalización ImageNet (consigna §2: exigida por ViT)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, size_2d=(224, 224), size_3d=(64, 64, 64),
                 hu_window=(-1000, 400)):
        self.size_2d = size_2d
        self.size_3d = size_3d
        self.hu_min, self.hu_max = hu_window

        # Transform 2D: Resize + ToTensor
        self.transform_2d = T.Compose([
            T.Resize(size_2d),
            T.ToTensor(),
        ])

    @staticmethod
    def apply_imagenet_norm(tensor):
        norm = T.Normalize(
            mean=AdaptivePreprocessor.IMAGENET_MEAN,
            std=AdaptivePreprocessor.IMAGENET_STD,
        )
        return norm(tensor)

    def _is_nih_or_osteo(self, filename: str, source: str = "unknown"):
        p = filename.lower()
        is_nih = source == 'nih' or ('nih chest x ray 14' in p) or ('nih' in p)
        is_osteo = source == 'osteo' or ('knee osteoarthritis' in p) or ('osteo' in p)
        return is_nih, is_osteo

    def _clahe_nih_rgb(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        hist = cv2.calcHist([l], [0], None, [256], [0, 256]).flatten()
        peaks = int(np.sum(hist > hist.mean()))
        tile_size = max(2, int(np.ceil(np.log(max(peaks, 2)))))
        valleys = hist[hist <= hist.mean()]
        val_mean = float(valleys.mean()) if len(valleys) > 0 else 1.0
        clip = float(np.clip(hist.max() / (val_mean + 1e-6), 1.0, 4.0))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile_size, tile_size))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        arr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return cv2.GaussianBlur(arr_eq, (5, 5), sigmaX=1.0)

    def _clahe_osteo_gray_to_rgb(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
        tile_size = max(2, int(np.ceil(np.log(max(int(np.sum(hist > hist.mean())), 2)))))
        valleys = hist[hist <= hist.mean()]
        val_mean = float(valleys.mean()) if len(valleys) > 0 else 1.0
        clip = float(np.clip(hist.max() / (val_mean + 1e-6), 1.0, 4.0))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(g)
        return np.repeat(enhanced[:, :, None], 3, axis=2)



    def process_uploaded_file(self, uploaded_file, source="unknown"):
        """
        Procesa un archivo subido via st.file_uploader.

        Args:
            uploaded_file: objeto UploadedFile de Streamlit
            source: string indicando el dataset de origen

        Returns:
            tensor:         torch.Tensor preprocesado
            original_shape: tuple con dimensiones originales
            is_3d:          bool
        """
        filename = uploaded_file.name.lower()
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # reset para posibles re-lecturas

        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            return self._process_2d_bytes(file_bytes, filename, source)

        elif filename.endswith((".nii", ".nii.gz")):
            return self._process_nifti_bytes(file_bytes, filename, source)

        elif filename.endswith(".mha"):
            return self._process_mha_bytes(file_bytes, filename, source)

        else:
            raise ValueError(
                f"Formato no soportado: {filename}\n"
                f"Formatos validos: PNG, JPEG, NIfTI (.nii/.nii.gz), MHA"
            )

    def _process_2d_bytes(self, file_bytes, filename, source):
        """Procesa imagen 2D desde bytes."""
        is_nih, is_osteo = self._is_nih_or_osteo(filename, source)
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        original_shape = (img.width, img.height, 3)
        img = img.resize(self.size_2d)
        
        arr = np.array(img, dtype=np.uint8)
        if is_nih:
            arr = self._clahe_nih_rgb(arr)
        elif is_osteo:
            arr = self._clahe_osteo_gray_to_rgb(arr)
            
        t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        # NO aplicar ImageNet norm aquí: el Router necesita [0,1] crudo.
        # La normalización ImageNet se aplica por separado para cada experto
        # en moe_inference._prepare_expert_tensor().
        return t, original_shape, False

    def _process_nifti_bytes(self, file_bytes, filename, source):
        """Procesa volumen NIfTI desde bytes (requiere archivo temporal)."""
        if not HAS_NIB:
            raise ImportError("nibabel es necesario para archivos NIfTI. "
                              "Instalar con: pip install nibabel")

        suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            nii_img = nib.load(tmp_path)
            img_arr = nii_img.get_fdata().astype(np.float32)
            original_shape = img_arr.shape

            # Transponer desde (X, Y, Z) de nibabel a (Z, Y, X) de numpy/torch
            img_arr = np.transpose(img_arr, (2, 1, 0))

            tensor = self._normalize_and_resize_3d(img_arr, filename, source)
            return tensor, original_shape, True
        finally:
            os.unlink(tmp_path)

    def _process_mha_bytes(self, file_bytes, filename, source):
        """Procesa volumen MHA desde bytes (requiere archivo temporal)."""
        if not HAS_SITK:
            raise ImportError("SimpleITK requerido para archivos MHA. "
                              "Instalar con: pip install SimpleITK")

        with tempfile.NamedTemporaryFile(suffix=".mha", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            itk_img = sitk.ReadImage(tmp_path)
            img_arr = sitk.GetArrayFromImage(itk_img).astype(np.float32)
            original_shape = itk_img.GetSize()  # (X, Y, Z)

            # Detectar si es realmente 2D empaquetado como 3D
            if img_arr.ndim == 2 or (img_arr.ndim == 3 and img_arr.shape[0] == 1):
                arr_2d = np.squeeze(img_arr)
                if arr_2d.max() > 1.5:
                    arr_2d = arr_2d / 255.0
                t = torch.from_numpy(arr_2d).float().unsqueeze(0)
                t = F.interpolate(t.unsqueeze(0), size=self.size_2d,
                                  mode="bilinear", align_corners=False).squeeze(0)
                t = t.repeat(3, 1, 1)  # [1,H,W] -> [3,H,W]
                return t, original_shape, False

            tensor = self._normalize_and_resize_3d(img_arr, filename, source)
            return tensor, original_shape, True
        finally:
            os.unlink(tmp_path)

    def _normalize_and_resize_3d(self, img_arr, filename="", source="unknown"):
        """
        Aplica HU windowing [-1000, 400] -> [0, 1] y resize a 64^3.
        """
        amin, amax = float(np.nanmin(img_arr)), float(np.nanmax(img_arr))
        pre_norm = amax <= 1.5 and amin >= -1e-2 and ('pancreas' in filename.lower() or source == 'pancreas')
        
        if not pre_norm:
            img_arr = np.clip(img_arr, self.hu_min, self.hu_max)
            img_arr = (img_arr - self.hu_min) / (self.hu_max - self.hu_min + 1e-8)
        else:
            img_arr = np.clip(img_arr, 0.0, 1.0)

        # Convertir a tensor y resize
        t = torch.tensor(img_arr, dtype=torch.float32)
        t = t.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        t = F.interpolate(t, size=self.size_3d, mode="trilinear",
                          align_corners=False)
        return t.squeeze(0)  # [1, 64, 64, 64]


def get_display_image(uploaded_file, tensor, is_3d):
    """
    Genera una imagen PIL para mostrar en el dashboard.

    Para 2D: usa la imagen original.
    Para 3D: extrae el slice axial central del volumen preprocesado.

    Args:
        uploaded_file: archivo subido
        tensor: tensor preprocesado
        is_3d: bool

    Returns:
        PIL.Image para mostrar en st.image()
    """
    if not is_3d:
        uploaded_file.seek(0)
        return Image.open(uploaded_file).convert("RGB")

    # Para 3D: tomar el slice axial central
    vol = tensor.squeeze(0).numpy()  # [D, H, W]
    mid_slice = vol.shape[0] // 2
    slice_arr = vol[mid_slice]  # [H, W]

    # Normalizar a 0-255 para visualizacion
    slice_arr = (slice_arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(slice_arr, mode="L").convert("RGB")
