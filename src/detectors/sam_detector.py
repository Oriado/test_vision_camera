import numpy as np
from typing import List, Dict, Any
import os
import torch
import sys
import cv2

# Ensure sam2 is in path if not installed to site-packages, though pip install -e . should handle it.
# We might need to import hydra or omegaconf if build_sam2 uses it internally, but sam2 handles it.

from src.interfaces import IDetector

# Flexible import block
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    SAM2_AVAILABLE = True
except ImportError:
    print("Warning: SAM 2 not installed or import error.")
    SAM2_AVAILABLE = False
    SAM2AutomaticMaskGenerator = None
    build_sam2 = None

class SamDetector(IDetector):
    def __init__(self):
        self.mask_generator = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
             # Optimization for CPU: use float32 or bfloat16 if supported, but usually float32 is safest
             pass

    def load_model(self, config: Dict[str, Any]) -> None:
        if not SAM2_AVAILABLE:
            print("SAM 2 libraries missing.")
            return

        # Checkpoint path
        default_ckpt = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../sam2_repo/checkpoints/sam2.1_hiera_small.pt"))
        checkpoint = config.get('checkpoint', default_ckpt)
        
        # Determine config file based on model type or explicit path
        # config for 'small' model is sam2.1_hiera_s.yaml
        
        # We assume the user has the sam2 repo cloned at d:/hakaton/sam2_repo
        # The config file is at sam2_repo/sam2/configs/sam2.1/sam2.1_hiera_s.yaml
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../sam2_repo"))
        model_cfg = config.get('model_config', "sam2.1_hiera_s.yaml")
        
        # build_sam2 expects the config NAME relative to sam2 package config root 
        # OR absolute path. The sam2 logic is a bit complex with hydra.
        # However, passing the full path to the yaml file usually works if it's external.
        # But commonly we pass the config name e.g. "sam2.1_hiera_s.yaml" and sam2 finds it if installed.
        # Since we installed via pip -e, the package structure might rely on hydra search paths.
        
        # Let's try passing the relative path inside the configs folder if we can.
        # Actually, let's verify if we can just pass the path.
        if not os.path.exists(checkpoint):
             print(f"Error: SAM 2 Checkpoint not found at {checkpoint}")
             return
             
        print(f"Loading SAM 2 from {checkpoint} with config {model_cfg} on {self.device}...")
        
        try:
             # mode="eval" is usually default
             # apply_postprocessing=False if we want raw masks
             # Since we are using AutomaticGen, we build the model first
             sam_model = build_sam2(model_cfg, checkpoint, device=self.device, apply_postprocessing=False)
             
             self.mask_generator = SAM2AutomaticMaskGenerator(sam_model)
        except Exception as e:
             print(f"Failed to load SAM 2: {e}")
             import traceback
             traceback.print_exc()

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.mask_generator is None:
            return []

        # SAM 2 Automatic Generator
        # input: RGB image (HWC) uint8
        # output: list of dicts
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        # On CPU, autocast with float32 is not supported/needed (warnings). Use nullcontext or valid dtype.
        if self.device == "cuda":
            autocast_context = torch.autocast(self.device, dtype=torch.bfloat16)
        else:
            # CPU: Default to float32 (no autocast) to avoid warnings and "wrong type" errors
            # bfloat16 on CPU is possible but requires support. Safest is raw float32.
            from contextlib import nullcontext
            autocast_context = nullcontext()

        # Print trace to show it's working (SAM is slow)
        # print("Running SAM 2 inference...", end="\r") 
        
        with torch.inference_mode(), autocast_context:
             masks = self.mask_generator.generate(frame_rgb)
        
        detections = []
        for mask_data in masks:
            # mask_data keys usually: segmentation, area, bbox, predicted_iou, stability_score...
            x, y, w, h = mask_data['bbox']
            score = mask_data.get('predicted_iou', 0.5)
            
            detections.append({
                'bbox': (int(x), int(y), int(w), int(h)),
                'label': 'object',
                'score': float(score),
                'mask': mask_data['segmentation'] 
            })
            
        return detections
