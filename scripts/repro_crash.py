import traceback
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ultra_mamba3 import register, make_yaml
from ultralytics import YOLO

try:
    register()
    y = YOLO(make_yaml('s'))
    y.train(data='coco8.yaml', epochs=1, imgsz=256, batch=2, device='cpu', workers=0, plots=False, verbose=False, exist_ok=True, name='coco8_smoke')
except Exception as e:
    traceback.print_exc()
