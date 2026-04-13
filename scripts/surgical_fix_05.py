import json
import os

nb_path = r'c:\Users\nicor\universidad\analitica\proyectos\proyecto 2\notebooks\05_Router_Profesor_Fase3_Solo.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update DATASET_ROOTS (LUNA path to ROI cache)
for cell in nb['cells']:
    if 'DATASET_ROOTS' in "".join(cell.get('source', [])):
        cell['source'] = [src.replace(
            '"LUNA16":   (os.path.join(LOCAL_DEST, "Luna16 Lung Cancer Dataset"), 3),',
            '"LUNA16":   (os.path.join(LOCAL_DEST, "Luna16_ROI_cache_v1"), 3),'
        ) for src in cell['source']]

# 2. Update AdaptivePreprocessor (Cell 5 - add .npz support and fix pre_norm)
for cell in nb['cells']:
    src = "".join(cell.get('source', []))
    if 'class AdaptivePreprocessor' in src:
        # Surgical replacement of specific lines
        lines = cell['source']
        for i, line in enumerate(lines):
            if "if ext.endswith(('.mhd', '.nii.gz', '.nii')):" in line:
                lines[i] = line.replace("('.mhd', '.nii.gz', '.nii')", "('.mhd', '.nii.gz', '.nii', '.npz')")
            if "pre_norm = amax <= 1.5 and amin >= -1e-2 and 'Pancreas' in str(path_hint)" in line:
                # Incluir 'Luna' y '.npz' en pre_norm
                lines[i] = "        pre_norm = (amax <= 1.5 and amin >= -1e-2) and ('Pancreas' in str(path_hint) or 'Luna' in str(path_hint) or str(path_hint).endswith('.npz'))\n"
            if "def _process_3d(self, path):" in line:
                # Inyectar logica de carga NPZ
                lines.insert(i+1, "        if path.endswith('.npz'):\n")
                lines.insert(i+2, "            d = np.load(path)\n")
                lines.insert(i+3, "            k = 'Z' if 'Z' in d else ('volume' if 'volume' in d else list(d.keys())[0])\n")
                lines.insert(i+4, "            return self._volume_array_to_tensor(d[k].astype(np.float32), path_hint=path)\n")
        cell['source'] = lines

# 3. Update resolve_task_label (Cell 7 - add .npz label extraction)
npz_label_helper = """
def _infer_label_from_npz(path):
    try:
        d = np.load(path)
        if 'y' in d: return int(d['y'])
        if 'label' in d: return int(d['label'])
    except: pass
    return None
"""

for cell in nb['cells']:
    src = "".join(cell.get('source', []))
    if 'def resolve_task_label' in src:
        # Inyectar helper y modificar lógica LUNA
        new_src = [npz_label_helper] + cell['source']
        for i, line in enumerate(new_src):
            if "if dataset_id == 3:  # LUNA16" in line:
                new_src[i+1] = "        if path.endswith('.npz'): return _infer_label_from_npz(path)\n"
        cell['source'] = new_src

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("05_Router_Profesor_Fase3_Solo.ipynb actualizado (quirúrgicamente).")
