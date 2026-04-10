import json

def get_expert1_model_name():
    with open(r"c:\Users\nicor\universidad\analitica\proyectos\proyecto 2\notebooks\experts\NIH_ChestXray_Swin_Tiny_Training.ipynb", encoding='utf-8') as f:
        nb = json.load(f)
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                src = "".join(cell.get('source', []))
                if "_model_name =" in src:
                    lines = [l for l in src.split('\n') if "_model_name =" in l]
                    print("Swin Training model name:", lines[0] if lines else "Not Found")

def dump_router_moe():
    with open(r"c:\Users\nicor\universidad\analitica\proyectos\proyecto 2\notebooks\03_Pipeline_Router_MoE.ipynb", encoding='utf-8') as f:
        nb = json.load(f)
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                src = "".join(cell.get('source', []))
                if "class LungMaxViT" in src or "class MaxViTClassifier" in src or "Expert 1" in src or "def load_experts" in src or "Expert_1" in src or "SwinClassifier" in src:
                    print(f"\n--- MoE Cell {i} ---")
                    print(src[:1000])

get_expert1_model_name()
dump_router_moe()
