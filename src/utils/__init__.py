import cv2
import numpy as np
from typing import List, Dict, Tuple, Any

def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    for det in detections:
        x, y, w, h = det['bbox']
        label = det.get('label', 'obj')
        score = det.get('score', 0.0)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def draw_tracking_and_distance(frame: np.ndarray, tracked_objects: Dict[int, Tuple[int, int, int, int]], distances: Dict[int, float]) -> np.ndarray:
    for obj_id, bbox in tracked_objects.items():
        x, y, w, h = bbox
        
        # Draw centroid
        cX = int(x + w/2)
        cY = int(y + h/2)
        cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
        
        text = f"ID: {obj_id}"
        if obj_id in distances:
            dist = distances[obj_id]
            if dist != -1.0:
                text += f" Dist: {dist:.1f}"
                
        cv2.putText(frame, text, (x, y+h+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw box if not already drawn by detector (though usually we redraw)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)
        
    return frame
