import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
from src.interfaces import ITracker

class OpenCVTracker(ITracker):
    def __init__(self):
        self.trackers = {} # id -> tracker instance
        self.bboxes = {} # id -> bbox
        self.tracker_type = "csrt"
        self.next_obj_id = 0

    def load_model(self, config: Dict[str, Any]) -> None:
        self.tracker_type = config.get('tracker_type', 'csrt').lower()

    def _create_tracker(self):
        # Try legacy namespace first (OpenCV 4.5+ with contrib) or standard
        try:
            if self.tracker_type == 'csrt':
                if hasattr(cv2, 'TrackerCSRT_create'): return cv2.TrackerCSRT_create()
                if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'): return cv2.legacy.TrackerCSRT_create()
            
            elif self.tracker_type == 'kcf':
                if hasattr(cv2, 'TrackerKCF_create'): return cv2.TrackerKCF_create()
                if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'): return cv2.legacy.TrackerKCF_create()

            elif self.tracker_type == 'mil':
                if hasattr(cv2, 'TrackerMIL_create'): return cv2.TrackerMIL_create()
                if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMIL_create'): return cv2.legacy.TrackerMIL_create()
        except Exception as e:
            print(f"Tracker creation error: {e}")

        # Fallback or Error
        print("Error: Tracking algorithms not found. Please run: pip install opencv-contrib-python")
        return None # Will likely crash later, but message is clear

    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> Dict[int, Tuple[int, int, int, int]]:
        # This is a naive implementation where we re-init if detections are present
        # In a real heavy pipeline, we would only detect every N frames.
        
        # If we have detections, we prioritizing re-initializing (or matching).
        # For this prototype, if detections are provided, we clear and re-init to sync.
        # This effectively makes it just a "Smoother" if detection is skipped, 
        # but if main loop calls detect every frame, this is redundant.
        
        # However, to demonstrate "tracking", we assume main loop might NOT pass detections every time.
        
        current_bboxes = {}

        if detections:
            # Re-initialize everything based on fresh detections
            # Note: This kills ID persistence if not matched. 
            # A real hybrid requires matching detections to existing trackers.
            # Here keeping it simple: Fresh Detections -> Reset.
            self.trackers = {}
            self.bboxes = {}
            # Reset ID counter or keep incrementing? 
            # If we reset, we lose history. 
            # Let's simple create new trackers for all detections.
            for det in detections:
                tracker = self._create_tracker()
                if tracker is None:
                    continue # Skip if tracker creation failed
                bbox = det['bbox']
                # Tracker init takes (x, y, w, h)
                tracker.init(frame, bbox)
                self.trackers[self.next_obj_id] = tracker
                self.bboxes[self.next_obj_id] = bbox
                self.next_obj_id += 1
            
            return self.bboxes
        else:
            # Update existing trackers
            ids_to_remove = []
            for obj_id, tracker in self.trackers.items():
                success, bbox = tracker.update(frame)
                if success:
                    # bbox is float tuple
                    self.bboxes[obj_id] = tuple(map(int, bbox))
                else:
                    ids_to_remove.append(obj_id)
            
            for obj_id in ids_to_remove:
                del self.trackers[obj_id]
                del self.bboxes[obj_id]
                
            return self.bboxes
