from typing import Tuple, Dict, Any
from src.interfaces import IDistanceEstimator, FrameData

class ReferenceDistance(IDistanceEstimator):
    def __init__(self):
        self.known_width = 15.0 # cm
        self.focal_length = 500.0 # pixels
        
    def load_configuration(self, config: Dict[str, Any]):
        self.known_width = config.get('known_width_cm', 15.0)
        self.focal_length = config.get('focal_length', 500.0)

    def estimate(self, frame_data: FrameData, object_id: int, bbox: Tuple[int, int, int, int]) -> float:
        # Distance = (KnownWidth * FocalLength) / PerceivedWidth
        _, _, w, _ = bbox
        if w == 0:
            return -1.0
        
        distance = (self.known_width * self.focal_length) / w
        return distance
