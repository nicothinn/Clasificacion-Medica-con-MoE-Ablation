"""
fix_router_collapse.py
Parches para 05_Router_Profesor_Fase3_Solo.ipynb que corrigen el colapso de expertos
(ratio max/min -> millones en epoca 2).

Cambios:
  1. Perdida total: siempre incluye L_routing
  2. ALPHA_AUX: 0.03 -> 0.01 (Switch Transformer default)
  3. Filtro de homogeneidad: relajado
  4. Gradient clipping: max_norm=1.0
  5. Warmup: 2 epocas sin L_task

Uso:
  python scripts/fix_router_collapse.py
  python scripts/fix_router_collapse.py ruta/al/notebook.ipynb
"""

import json
import os
import sys


def _join(cell):
    return "".join(cell["source"])


def _split(text):
    lines = text.split("\n")
    result = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return result


def _patch(cell, old, new, label):
    src = _join(cell)
    if old not in src:
        print(f"  [SKIP] {label} -- patron no encontrado")
        return False
    n = src.count(old)
    src = src.replace(old, new)
    cell["source"] = _split(src)
    print(f"  [OK]   {label} ({n} ocurrencia(s))")
    return True


def fix1_combined_loss(cell):
    OLD = (
        "            if L_task is not None:\n"
        "                loss = L_task + alpha_aux * aux\n"
        "            else:\n"
        "                loss = L_routing + alpha_aux * aux\n"
    )
    NEW = (
        "            # Fix: L_routing siempre presente (Switch Transformer 3.1)\n"
        "            # Previene expert collapse cuando L_task domina\n"
        "            loss = L_routing + (L_task if L_task is not None else 0.0) + alpha_aux * aux\n"
    )
    return _patch(cell, OLD, NEW, "Fix 1: Perdida combinada L_routing + L_task + L_aux")


def fix3_relax_homogeneity(cell):
    # 24 spaces + 28 spaces (from the actual notebook)
    OLD = (
        "                        if int(expert_ids[i].item()) != eid_val:\n"
        "                            continue\n"
    )
    NEW = (
        "                        # Filtro de homogeneidad relajado: permite L_task cross-domain\n"
        "                        # (strict filtering starves L_task when router collapses)\n"
    )
    return _patch(cell, OLD, NEW, "Fix 3: Filtro de homogeneidad relajado")


def fix4_gradient_clipping(cell):
    OLD = (
        "        if use_amp:\n"
        "            scaler.scale(loss).backward()\n"
        "            scaler.step(optimizer)\n"
        "            scaler.update()\n"
        "        else:\n"
        "            loss.backward()\n"
        "            optimizer.step()\n"
    )
    NEW = (
        "        if use_amp:\n"
        "            scaler.scale(loss).backward()\n"
        "            scaler.unscale_(optimizer)\n"
        "            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n"
        "            scaler.step(optimizer)\n"
        "            scaler.update()\n"
        "        else:\n"
        "            loss.backward()\n"
        "            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n"
        "            optimizer.step()\n"
    )
    return _patch(cell, OLD, NEW, "Fix 4: Gradient clipping max_norm=1.0")


def fix5a_warmup_param(cell):
    OLD = (
        "    experts: dict | None = None,\n"
        "):\n"
    )
    NEW = (
        "    experts: dict | None = None,\n"
        "    warmup_epochs: int = 2,\n"
        "):\n"
    )
    return _patch(cell, OLD, NEW, "Fix 5a: Parametro warmup_epochs en firma")


def fix5b_warmup_logic(cell):
    OLD = (
        "        train_m = train_router_one_epoch(\n"
        "            model, dataloader, optimizer, device,\n"
        "            alpha_aux=alpha_aux,\n"
        "            label_smoothing=label_smoothing,\n"
        "            scaler=scaler if use_cuda else None,\n"
        "            max_batches=max_batches_per_epoch,\n"
        "            experts=experts,\n"
        "        )\n"
    )
    NEW = (
        "        # Warmup: primeras N epocas solo L_routing + L_aux (sin expert feedback)\n"
        "        # Estabiliza el routing antes de introducir L_task (V-MoE, 4.2)\n"
        "        use_experts = experts if ep > warmup_epochs else None\n"
        "        if ep == warmup_epochs + 1 and experts is not None:\n"
        "            print(f'  -> Warmup completo. Activando feedback de expertos desde epoca {ep}.')\n"
        "        train_m = train_router_one_epoch(\n"
        "            model, dataloader, optimizer, device,\n"
        "            alpha_aux=alpha_aux,\n"
        "            label_smoothing=label_smoothing,\n"
        "            scaler=scaler if use_cuda else None,\n"
        "            max_batches=max_batches_per_epoch,\n"
        "            experts=use_experts,\n"
        "        )\n"
    )
    return _patch(cell, OLD, NEW, "Fix 5b: Warmup logic en loop de epocas")


def fix5c_warmup_docstring(cell):
    OLD = (
        "    experts: dict {expert_key: nn.Module} — si se proporciona, activa el flujo del\n"
        "      profesor (forward del experto elegido + L_task). El dataloader debe haber sido\n"
        "      construido con include_task_label=True para que los batches incluyan etiquetas de\n"
        "      tarea. Sin experts, el entrenamiento usa solo L_routing + L_aux (flujo base).\n"
    )
    NEW = (
        "    experts: dict {expert_key: nn.Module} — si se proporciona, activa el flujo del\n"
        "      profesor (forward del experto elegido + L_task). El dataloader debe haber sido\n"
        "      construido con include_task_label=True para que los batches incluyan etiquetas de\n"
        "      tarea. Sin experts, el entrenamiento usa solo L_routing + L_aux (flujo base).\n"
        "\n"
        "    warmup_epochs: int -- epocas iniciales SIN L_task (solo L_routing + L_aux).\n"
        "      Estabiliza el gating antes de que los expert_logits x g_scores dominen.\n"
    )
    return _patch(cell, OLD, NEW, "Fix 5c: Docstring warmup")


def fix2_alpha_aux(cell):
    OLD = "ALPHA_AUX = 0.03\n"
    NEW = "ALPHA_AUX = 0.01  # Switch Transformer default (era 0.03)\n"
    return _patch(cell, OLD, NEW, "Fix 2: ALPHA_AUX 0.03 -> 0.01")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(
        script_dir, "..", "notebooks", "05_Router_Profesor_Fase3_Solo.ipynb"
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    path = os.path.abspath(path)

    # If a previous broken run already wrote a partial file, restore from backup
    backup = path + ".bak"
    if os.path.exists(backup):
        print(f"Restaurando desde backup: {backup}")
        with open(backup, "r", encoding="utf-8") as f:
            nb = json.load(f)
    else:
        print(f"Leyendo: {path}")
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        # Create backup
        with open(backup, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False)
        print(f"Backup creado: {backup}")

    applied = 0
    total = 0

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = _join(cell)

        # Cell with train_router_one_epoch + fit_router_with_eval
        if "def train_router_one_epoch(" in src:
            print(f"\n{'='*60}")
            print(f"Celda {i} -- train_router_one_epoch + fit_router_with_eval")
            print(f"{'='*60}")

            total += 1; applied += fix1_combined_loss(cell)
            total += 1; applied += fix3_relax_homogeneity(cell)
            total += 1; applied += fix4_gradient_clipping(cell)
            total += 1; applied += fix5a_warmup_param(cell)
            total += 1; applied += fix5b_warmup_logic(cell)
            total += 1; applied += fix5c_warmup_docstring(cell)

        # Cell with training hyperparameters
        if "ALPHA_AUX = 0.03" in src:
            print(f"\n{'='*60}")
            print(f"Celda {i} -- Hiperparametros de entrenamiento")
            print(f"{'='*60}")

            total += 1; applied += fix2_alpha_aux(cell)

    print(f"\n{'='*60}")
    print(f"RESUMEN: {applied}/{total} parches aplicados")
    print(f"{'='*60}")

    if applied == 0:
        print("No se aplicaron parches.")
        return

    print(f"\nEscribiendo notebook parcheado: {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"\nListo. Cambios aplicados:")
    print(f"   1. Perdida: loss = L_routing + L_task + a*L_aux (siempre combinada)")
    print(f"   2. a_aux: 0.03 -> 0.01 (paper default)")
    print(f"   3. Filtro homogeneidad: relajado (L_task ya no queda vacio)")
    print(f"   4. Gradient clipping: max_norm=1.0 (AMP + no-AMP)")
    print(f"   5. Warmup: 2 epocas sin L_task antes de activar expert feedback")
    print(f"\n   Restaurar original: copiar .bak sobre .ipynb")


if __name__ == "__main__":
    main()
