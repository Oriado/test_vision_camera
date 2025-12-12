from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional
import numpy as np

class FrameData:
    """Data class to hold frame information passed through the pipeline."""
    def __init__(self, frame: np.ndarray, timestamp: float):
        self.original_frame = frame
        self.timestamp = timestamp
        self.detections: List[Dict[str, Any]] = [] # x, y, w, h, class_id, conf
        self.tracked_objects: Dict[int, Any] = {} # id -> (centroid, bbox)
        self.distances: Dict[int, float] = {} # id -> distance

class IDetector(ABC):
    """Interface for Object Detection modules."""
    
    @abstractmethod
    def load_model(self, config: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns a list of detections. 
        Each detection is a dict: {'bbox': (x, y, w, h), 'label': str, 'score': float}
        """
        pass

class ITracker(ABC):
    """Interface for Object Tracking modules."""

    @abstractmethod
    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Updates the tracker with new frame and detections.
        Returns a dict of {object_id: (x, y, w, h)}.
        """
        pass

class IDistanceEstimator(ABC):
    """Interface for Distance Estimation modules."""

    @abstractmethod
    def estimate(self, frame_data: FrameData, object_id: int, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculates distance to the object in specific units (e.g., cm or meters).
        Returns -1.0 if distance cannot be calculated.
        """
        pass
