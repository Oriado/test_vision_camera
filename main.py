import cv2
import yaml
import time
import sys
import importlib
from src.interfaces import FrameData

# Dynamic import helpers
def get_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def broadcast_config(instance, full_config, key):
    # Pass a slice of config to the instance
    if hasattr(instance, 'load_model'):
        # Pass generic modules config or specific key config
        specific_settings = full_config.get(f"{key}_settings", {}).get(full_config['modules'][key], {})
        instance.load_model(specific_settings)
    if hasattr(instance, 'load_configuration'):
        specific_settings = full_config.get(f"{key}_settings", {}).get(full_config['modules'][key], {})
        instance.load_configuration(specific_settings)

def main():
    print("Initializing Modular CV System...")
    config = load_config()
    
    # Factory Logic
    detectors_map = {
        'contour': ('src.detectors.contour_detector', 'ContourDetector'),
        'yolo': ('src.detectors.yolo_detector', 'YoloDetector'),
        'sam': ('src.detectors.sam_detector', 'SamDetector')
    }
    trackers_map = {
        'centroid': ('src.trackers.centroid_tracker', 'CentroidTracker'),
        'opencv': ('src.trackers.opencv_tracker', 'OpenCVTracker')
    }
    distance_map = {
        'reference': ('src.distance.reference_distance', 'ReferenceDistance'),
        'depth_anything': ('src.distance.depth_anything', 'DepthEstimator')
    }
    
    # Instantiate modules
    det_name = config['modules']['detector']
    mod_path, cls_name = detectors_map.get(det_name, detectors_map['contour'])
    detector = get_class(mod_path, cls_name)()
    broadcast_config(detector, config, 'detector')
    
    trk_name = config['modules']['tracker']
    mod_path, cls_name = trackers_map.get(trk_name, trackers_map['centroid'])
    tracker = get_class(mod_path, cls_name)()
    broadcast_config(tracker, config, 'tracker')
    
    dst_name = config['modules']['distance']
    mod_path, cls_name = distance_map.get(dst_name, distance_map['reference'])
    distance_estimator = get_class(mod_path, cls_name)()
    broadcast_config(distance_estimator, config, 'distance')
    
    # Camera Init
    cap = cv2.VideoCapture(config['system']['camera_index'])
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    from src.utils import draw_detections, draw_tracking_and_distance

    print(f"Modules Loaded: Detector={det_name}, Tracker={trk_name}, Distance={dst_name}")
    print("Press 'q' to quit.")

    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.time()
        timestamp = current_time
        
        # Create FrameData
        frame_data = FrameData(frame, timestamp)
        
        # 1. Detect
        # In a faster pipeline, we might skip detection every N frames
        detections = detector.detect(frame)
        frame_data.detections = detections
        
        # 2. Track
        # For CentroidTracker, we pass detections. For OpenCVTracker, we typically pass nothing unless re-init.
        # Our interfaces are generic: update(frame, detections)
        tracked_objects = tracker.update(frame, detections)
        frame_data.tracked_objects = tracked_objects
        
        # 3. Distance Estimation
        distances = {}
        for obj_id, bbox in tracked_objects.items():
            dist = distance_estimator.estimate(frame_data, obj_id, bbox)
            distances[obj_id] = dist
        frame_data.distances = distances
        
        # Visualization
        # draw_detections(frame, detections) # Optional: if we only want to see tracked objs
        draw_tracking_and_distance(frame, tracked_objects, distances)
        
        if config['system']['show_fps']:
            fps = 1 / (current_time - prev_time) if current_time > prev_time else 0
            prev_time = current_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Modular CV System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
