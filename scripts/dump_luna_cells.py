import json

nb_path = r'c:\Users\nicor\universidad\analitica\proyectos\proyecto 2\notebooks\experts\LUNA16_Swin3D_Training.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

out_path = r'c:\Users\nicor\universidad\analitica\proyectos\proyecto 2\scripts\luna_cells_detail.txt'
with open(out_path, 'w', encoding='utf-8') as out:
    for i in [4, 5, 6, 7, 8, 9, 14]:
        src = "".join(nb['cells'][i].get('source', []))
        out.write(f"\n{'='*80}\n=== CELL {i} ({nb['cells'][i]['cell_type']}) ===\n{'='*80}\n")
        out.write(src)
        out.write("\n")
