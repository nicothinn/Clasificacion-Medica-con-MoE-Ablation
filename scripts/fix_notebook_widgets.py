"""
Fix 'metadata.widgets' in Jupyter notebooks so GitHub can render them.

GitHub/nbformat requires each entry under metadata.widgets to have a 'state' key.
This script wraps the existing widget data inside {'state': ...} when 'state' is
missing, or removes metadata.widgets entirely if it's empty/broken.
"""

import json
import sys
from pathlib import Path


def fix_notebook(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)

    meta = nb.get("metadata", {})
    widgets = meta.get("widgets")
    if widgets is None:
        return False

    changed = False

    for mime, value in list(widgets.items()):
        if not isinstance(value, dict):
            continue
        if "state" not in value:
            widgets[mime] = {"state": value, "version_major": 2, "version_minor": 0}
            changed = True

    if not widgets:
        meta.pop("widgets", None)
        changed = True

    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")

    return changed


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    notebooks = sorted(root.rglob("*.ipynb"))
    fixed = 0
    for nb_path in notebooks:
        if ".ipynb_checkpoints" in str(nb_path):
            continue
        if fix_notebook(nb_path):
            print(f"  FIXED: {nb_path}")
            fixed += 1
    print(f"\n{fixed}/{len(notebooks)} notebooks corregidos.")


if __name__ == "__main__":
    main()
