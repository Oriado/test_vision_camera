import numpy as np
from typing import List, Dict, Any
from src.interfaces import IDetector
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class YoloDetector(IDetector):
    def __init__(self):
        self.model = None

    def load_model(self, config: Dict[str, Any]) -> None:
        if YOLO is None:
            print("Warning: ultralytics not installed. YoloDetector will fail.")
            return
        
        model_path = config.get('model_path', 'yolov8n.pt')
        self.confidence_thresh = config.get('confidence', 0.5)
        print(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None:
            return []

        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_thresh:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                label = self.model.names[cls]
                
                detections.append({
                    'bbox': (x1, y1, w, h),
                    'label': label,
                    'score': conf
                })
        
        return detections
