import json
import os

nb_path = r'c:\Users\nicor\universidad\analitica\proyectos\proyecto 2\notebooks\05_Router_Profesor_Fase3_Solo.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = "".join(cell.get('source', []))
    if '_default_checkpoint_candidates' in src:
        # Remove MaxViT and old names to avoid accidental cross-loading
        cell['source'] = [src.replace(
            'os.path.join(weights_dir, "MaxViT_NIH_5cls.pth"),',
            ''
        ).replace(
            'os.path.join(weights_dir, "exp1_NIH_LungMaxViT_best.pth"),',
            ''
        )]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("05: Removidos checkpoints de MaxViT de la lista de candidatos de Exp1.")
