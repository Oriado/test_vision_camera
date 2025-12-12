import sys
import os

# Allow running this script directly for testing
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from typing import Tuple, Dict, Any
import numpy as np
import cv2
from src.interfaces import IDistanceEstimator, FrameData
try:
    from transformers import pipeline
    from PIL import Image
except ImportError:
    pipeline = None

class DepthEstimator(IDistanceEstimator):
    def __init__(self):
        self.pipe = None
        self.last_depth_map = None
        self.last_timestamp = -1

    def load_configuration(self, config: Dict[str, Any]):
        if pipeline is None:
            print("Transformers/PIL not installed. DepthEstimator disabled.")
            return
            
        model_id = config.get('model', "LiheYoung/depth-anything-small-hf")
        print(f"Loading Depth Anything model: {model_id}...")
        try:
            # device=0 for GPU, -1 for CPU
            self.pipe = pipeline(task="depth-estimation", model=model_id, device=-1) 
        except Exception as e:
            print(f"Failed to load Depth model: {e}")

    def estimate(self, frame_data: FrameData, object_id: int, bbox: Tuple[int, int, int, int]) -> float:
        if self.pipe is None:
            return -1.0

        # Optimization: Only compute depth map once per frame
        if self.last_timestamp != frame_data.timestamp:
            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(frame_data.original_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Inference
            depth_output = self.pipe(pil_image)
            self.last_depth_map = np.array(depth_output["depth"])
            self.last_timestamp = frame_data.timestamp

        # Get average depth in bbox
        x, y, w, h = bbox
        # Ensure bounds
        H, W = self.last_depth_map.shape
        x = max(0, x)
        y = max(0, y)
        w = min(W - x, w)
        h = min(H - y, h)

        if w <= 0 or h <= 0:
            return -1.0

        roi = self.last_depth_map[y:y+h, x:x+w]
        avg_depth = np.mean(roi)
        
        # Note: Depth Anything returns relative depth (inverse depth mostly or metric depending on model)
        # Without calibration, this number is just "closeness".
        # Creating a synthetic mapping to CM just for demo.
        # usually 0 = far, 255 = close or vice versa depending on output.
        # Assuming metric-ish output or just returning raw value.
        return float(avg_depth)
