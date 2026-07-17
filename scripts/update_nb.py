import json

with open("notebooks/Mamba3Yolo_Kaggle_Runbook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("src/blocks/mamba3_ref.py", "r", encoding="utf-8") as f:
    ref_code = "%%writefile src/blocks/mamba3_ref.py\n" + f.read()
    
with open("src/blocks/mamba3_odss.py", "r", encoding="utf-8") as f:
    odss_code = "%%writefile src/blocks/mamba3_odss.py\n" + f.read()

def to_lines(text):
    lines = text.split('\n')
    return [l + '\n' for l in lines[:-1]] + [lines[-1]] if lines else []

nb['cells'][6]['source'] = to_lines(ref_code)
nb['cells'][7]['source'] = to_lines(odss_code)

with open("notebooks/Mamba3Yolo_Kaggle_Runbook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
