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

    @staticmethod
    def apply_conditional_clahe(img_pil: Image.Image) -> np.ndarray:
        """
        Detecta si la imagen es escala de grises y aplica CLAHE condicionalmente.
        Si es RGB (color real), no aplica CLAHE para no distorsionar colores.
        """
        arr_rgb = np.array(img_pil.convert("RGB"))
        
        # Deteccion de escala de grises:
        # Calculamos la varianza entre los canales R, G, B para cada pixel
        # Si la imagen es grayscale, R=G=B, por lo que la varianza es ~0.
        channel_var = np.var(arr_rgb, axis=2).mean()
        is_grayscale = channel_var < 5.0  # Umbral bajo para tolerar artefactos de compresion
        
        if is_grayscale:
            # Aplicar CLAHE a la imagen médica en escala de grises (Rayos X, Osteo, etc.)
            gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            return np.repeat(enhanced[:, :, None], 3, axis=2)
        else:
            # Si es imagen a color RGB (Dermatología), omitir CLAHE por completo
            return arr_rgb



    def process_uploaded_file(self, uploaded_files, source="unknown"):
        """
        Procesa archivos subidos (UploadFile) de forma segura en un entorno temporal.
        """
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
            
        import tempfile
        import os
        import shutil
        
        main_file = None
        main_path = None
        image_array = None
        original_shape = None
        is_mhd = False
        file_bytes = None
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            # 1. Guardar todos los archivos con sus nombres originales
            for f in uploaded_files:
                original_name = os.path.basename(f.filename)
                path = os.path.join(tmpdirname, original_name)
                
                with open(path, "wb") as buffer:
                    shutil.copyfileobj(f.file, buffer)
                f.file.seek(0)  # Resetear para la UI
                
                # 2. Identificar el archivo principal
                lower_name = original_name.lower()
                if lower_name.endswith((".png", ".jpg", ".jpeg", ".nii", ".nii.gz", ".mha", ".mhd")):
                    main_file = f
                    main_path = path

            if not main_file:
                raise ValueError("No se encontró archivo principal válido (.mhd, .nii, .png, etc.)")

            filename = os.path.basename(main_file.filename).lower()
            
            # 3 y 4. Lectura AÚN DENTRO del bloque
            if filename.endswith(".mhd"):
                if not HAS_SITK:
                    raise ImportError("SimpleITK requerido para archivos MHD.")
                
                # Inspección y validación del MHD
                raw_filename = None
                with open(main_path, "r", encoding="utf-8", errors="ignore") as f_mhd:
                    for line in f_mhd:
                        if line.startswith("ElementDataFile"):
                            raw_filename = line.split("=")[1].strip()
                            break
                
                if raw_filename:
                    expected_raw_path = os.path.join(tmpdirname, raw_filename)
                    if not os.path.exists(expected_raw_path):
                        # El nombre no coincide exactamente. Buscamos cualquier .raw o .zraw subido y lo renombramos.
                        found_raw = None
                        for f_name in os.listdir(tmpdirname):
                            if f_name.lower().endswith(('.raw', '.zraw')) and f_name != os.path.basename(main_path):
                                found_raw = os.path.join(tmpdirname, f_name)
                                break
                        
                        if found_raw:
                            os.rename(found_raw, expected_raw_path)
                            print(f"Raw renombrado automáticamente de '{os.path.basename(found_raw)}' a '{raw_filename}'")
                        else:
                            raise ValueError(f"El archivo MHD requiere el archivo RAW llamado '{raw_filename}', pero no se encontró en la subida.")
                
                is_mhd = True
                itk_image = sitk.ReadImage(main_path)
                image_array = sitk.GetArrayFromImage(itk_image).astype(np.float32)
                original_shape = itk_image.GetSize()
            else:
                with open(main_path, "rb") as bf:
                    file_bytes = bf.read()

        # AHORA SÍ, salimos del bloque with. La carpeta temporal se ha borrado,
        # pero image_array y file_bytes están cargados en memoria RAM.
        
        if is_mhd:
            if image_array.ndim == 2 or (image_array.ndim == 3 and image_array.shape[0] == 1):
                arr_2d = np.squeeze(image_array)
                if arr_2d.max() > 1.5:
                    arr_2d = arr_2d / 255.0
                t = torch.from_numpy(arr_2d).float().unsqueeze(0)
                t = F.interpolate(t.unsqueeze(0), size=self.size_2d,
                                  mode="bilinear", align_corners=False).squeeze(0)
                t = t.repeat(3, 1, 1)
                return t, original_shape, False, main_file

            tensor = self._normalize_and_resize_3d(image_array, filename, source)
            return tensor, original_shape, True, main_file
        else:
            if file_bytes is None:
                raise ValueError("No se pudo leer el contenido del archivo.")

            if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                t, s, is3d = self._process_2d_bytes(file_bytes, filename, source)
            elif filename.endswith((".nii", ".nii.gz")):
                t, s, is3d = self._process_nifti_bytes(file_bytes, filename, source)
            elif filename.endswith(".mha"):
                t, s, is3d = self._process_mha_bytes(file_bytes, filename, source)
            else:
                raise ValueError("Formato no soportado")
                
            return t, s, is3d, main_file

    def _process_2d_bytes(self, file_bytes, filename, source):
        """Procesa imagen 2D desde bytes."""
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        original_shape = (img.width, img.height, 3)
        img = img.resize(self.size_2d)
        
        # IMAGEN INFERENCIA: estrictamente sin CLAHE para evitar domain shift
        arr = np.array(img, dtype=np.uint8)
            
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

    def _process_mhd_path(self, mhd_path, filename, source):
        """Procesa volumen MHD leyendo desde un directorio temporal donde está el .raw."""
        if not HAS_SITK:
            raise ImportError("SimpleITK requerido para archivos MHD.")

        itk_img = sitk.ReadImage(mhd_path)
        img_arr = sitk.GetArrayFromImage(itk_img).astype(np.float32)
        original_shape = itk_img.GetSize()

        if img_arr.ndim == 2 or (img_arr.ndim == 3 and img_arr.shape[0] == 1):
            arr_2d = np.squeeze(img_arr)
            if arr_2d.max() > 1.5:
                arr_2d = arr_2d / 255.0
            t = torch.from_numpy(arr_2d).float().unsqueeze(0)
            t = F.interpolate(t.unsqueeze(0), size=self.size_2d,
                              mode="bilinear", align_corners=False).squeeze(0)
            t = t.repeat(3, 1, 1)
            return t, original_shape, False

        tensor = self._normalize_and_resize_3d(img_arr, filename, source)
        return tensor, original_shape, True

    def _normalize_and_resize_3d(self, img_arr, filename="", source="unknown"):
        """
        Aplica HU windowing [-1000, 400] -> [0, 1] y resize a 64^3.
        """
        amin, amax = float(np.nanmin(img_arr)), float(np.nanmax(img_arr))
        
        # Corrección dinámica para LUNA16 u otros datasets guardados como unsigned (Shift +1024)
        if amin >= 0 and amax > 1000:
            img_arr = img_arr - 1024
            amin, amax = float(np.nanmin(img_arr)), float(np.nanmax(img_arr))

        pre_norm = amax <= 1.5 and amin >= -1e-2 and ('pancreas' in filename.lower() or source == 'pancreas')
        
        if not pre_norm:
            # Ventana radiológica para Pulmón/Tórax (ej. LUNA16)
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
        # IMAGEN VISUAL: Le aplicamos CLAHE solo a la copia que va al dashboard
        uploaded_file.seek(0)
        img_pil = Image.open(uploaded_file).convert("RGB")
        arr_visual = AdaptivePreprocessor.apply_conditional_clahe(img_pil)
        return Image.fromarray(arr_visual).convert("RGB")

    # Para 3D: tomar el slice axial central
    vol = tensor.squeeze(0).numpy()  # [D, H, W]
    mid_slice = vol.shape[0] // 2
    slice_arr = vol[mid_slice]  # [H, W]

    # Normalizar a 0-255 para visualizacion
    slice_arr = (slice_arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(slice_arr, mode="L").convert("RGB")
