import json

nb_path = r'c:\Users\nicor\universidad\analitica\proyectos\proyecto 2\notebooks\experts\LUNA16_Swin3D_Training.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

out_path = r'c:\Users\nicor\universidad\analitica\proyectos\proyecto 2\scripts\luna_structure.txt'
with open(out_path, 'w', encoding='utf-8') as out:
    out.write(f"Total cells: {len(nb['cells'])}\n\n")
    for i, cell in enumerate(nb['cells']):
        ct = cell['cell_type']
        src = "".join(cell.get('source', []))
        # First 200 chars preview
        preview = src[:300].replace('\n', ' | ')
        out.write(f"--- CELL {i} ({ct}) ---\n")
        out.write(f"  Preview: {preview}\n")
        out.write(f"  Lines: {len(src.splitlines())}\n\n")
