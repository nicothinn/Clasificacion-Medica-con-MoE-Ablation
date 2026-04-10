"""
Script para restaurar el filtro restrictivo en NIH_ChestXray_LungMaxViT_Training.ipynb:
Solo mantiene las FILAS que tengan alguna de las 5 patologías objetivo.
"""
import json

def restore_filter(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    target_cell_found = False
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            if "# --- Filas con PNG en disco ---" in source and "Dataset total:" in source:
                # Restaurar el filtrado de filas
                new_source = [
                    "# --- Filas con PNG en disco ---\n",
                    "df_clean = df[df['FilePath'].notna()].copy().reset_index(drop=True)\n",
                    "\n",
                    "df_proc = df_clean\n",
                    "\n",
                    "# --- One-hot 5 clases ---\n",
                    "for cls in CONFIG['classes']:\n",
                    "    df_proc[cls] = df_proc['Finding Labels'].apply(lambda x: 1 if cls in str(x) else 0)\n",
                    "\n",
                    "# --- FILTRO: Solo muestras que tengan al menos 1 de las 5 clases target ---\n",
                    "mask = df_proc[CONFIG['classes']].sum(axis=1) > 0\n",
                    "n_all = len(df_proc)\n",
                    "df_proc = df_proc[mask].reset_index(drop=True)\n",
                    "\n",
                    "print(f\"Filtrado restrictivo: {n_all} -> {len(df_proc)} muestras con clases target\")\n",
                    "print(\"Distribucion de etiquetas (5 clases):\")\n",
                    "print(df_proc[CONFIG['classes']].sum().to_string())\n",
                    "\n",
                    "y_multi = df_proc[CONFIG['classes']].values\n",
                    "n = len(df_proc)\n",
                    "\n",
                    "from sklearn.model_selection import GroupShuffleSplit\n",
                    "gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=CONFIG['seed'])\n",
                    "train_idx, temp_idx = next(gss1.split(df_proc, groups=df_proc['PatientID']))\n",
                    "df_temp = df_proc.iloc[temp_idx].reset_index(drop=True)\n",
                    "gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=CONFIG['seed'])\n",
                    "val_idx_t, test_idx_t = next(gss2.split(df_temp, groups=df_temp['PatientID']))\n",
                    "\n",
                    "train_df = df_proc.iloc[train_idx].reset_index(drop=True)\n",
                    "val_df = df_temp.iloc[val_idx_t].reset_index(drop=True)\n",
                    "test_df = df_temp.iloc[test_idx_t].reset_index(drop=True)\n",
                    "\n",
                    "print(f\"Split por PatientID (SOLO CATEGORIAS TARGET): Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}\")\n"
                ]
                cell['source'] = new_source
                target_cell_found = True
                break

    if target_cell_found:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        return True
    return False

if __name__ == '__main__':
    path = r"c:\Users\nicor\universidad\analitica\proyectos\proyecto 2\notebooks\experts\NIH_ChestXray_LungMaxViT_Training.ipynb"
    if restore_filter(path):
        print("Filtro restaurado: El dataset ahora solo contiene las 5 categorías elegidas.")
    else:
        print("No se pudo aplicar la restauración.")
