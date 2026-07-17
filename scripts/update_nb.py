import json

with open("notebooks/Mamba3Yolo_Kaggle_Runbook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scripts/validate_core.py", "r", encoding="utf-8") as f:
    validate_code = "%%writefile scripts/validate_core.py\n" + f.read()

def to_lines(text):
    lines = text.split('\n')
    return [l + '\n' for l in lines[:-1]] + [lines[-1]] if lines else []

nb['cells'][10]['source'] = to_lines(validate_code)

with open("notebooks/Mamba3Yolo_Kaggle_Runbook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
