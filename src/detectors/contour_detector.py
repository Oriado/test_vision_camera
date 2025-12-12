import cv2
import numpy as np
from typing import List, Dict, Any
from src.interfaces import IDetector

class ContourDetector(IDetector):
    def __init__(self):
        self.config = {}

    def load_model(self, config: Dict[str, Any]) -> None:
        self.config = config

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        thresh_min = self.config.get('threshold_min', 100)
        thresh_max = self.config.get('threshold_max', 255)
        
        _, thresh = cv2.threshold(blur, thresh_min, thresh_max, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        min_area = self.config.get('min_area', 500)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append({
                    'bbox': (x, y, w, h),
                    'label': 'object',
                    'score': 1.0, # Synthetic score
                    'contour': cnt # Store contour for visualization if needed
                })
        
        # Sort by area (optional, user picked max, here we return all valid)
        detections.sort(key=lambda d: d['bbox'][2] * d['bbox'][3], reverse=True)
        
        return detections
