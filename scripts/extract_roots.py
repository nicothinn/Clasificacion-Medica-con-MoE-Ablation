import json

nb_path = r'c:\Users\nicor\universidad\analitica\proyectos\proyecto 2\notebooks\05_Router_Profesor_Fase3_Solo.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

with open(r'c:\Users\nicor\universidad\analitica\proyectos\proyecto 2\scripts\roots_05.txt', 'w', encoding='utf-8') as f:
    for i, c in enumerate(nb['cells']):
        src = "".join(c.get('source', []))
        if 'DATASET_ROOTS' in src or 'RAW_DIR =' in src:
            f.write(f"--- CELL {i} ---\n")
            f.write(src)
            f.write("\n")
