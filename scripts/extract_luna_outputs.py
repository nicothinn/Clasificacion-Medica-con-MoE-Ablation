import json

nb_path = r'c:\Users\nicor\universidad\analitica\proyectos\proyecto 2\notebooks\experts\LUNA16_Swin3D_Training.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

with open(r'c:\Users\nicor\universidad\analitica\proyectos\proyecto 2\scripts\luna_outputs.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        # Get all cell outputs (training logs)
        outputs = cell.get('outputs', [])
        for o in outputs:
            text = o.get('text', [])
            if text:
                joined = "".join(text)
                # Only relevant training output
                if any(kw in joined for kw in ['loss=', 'F1', 'f1', 'val_', 'best', 'Test', 'Epoch', 'acc=', 'AUC', 'confusion', 'recall', 'precision', 'nódulo', 'nodulo']):
                    out.write(f"\n--- CELL {i} OUTPUT ---\n")
                    out.write(joined[:4000])
                    out.write("\n")
