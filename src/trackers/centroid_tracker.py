from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist
from typing import List, Dict, Tuple, Any
from src.interfaces import ITracker

class CentroidTracker(ITracker):
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.bboxes = OrderedDict() # Store bbox for each ID

    def load_model(self, config: Dict[str, Any]) -> None:
        self.max_disappeared = config.get('max_disappeared', 50)

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bboxes[object_id]

    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> Dict[int, Tuple[int, int, int, int]]:
        # Extract centroids from detections
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        input_bboxes = []
        for i, d in enumerate(detections):
            x, y, w, h = d['bbox']
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            input_centroids[i] = (cX, cY)
            input_bboxes.append((x, y, w, h))

        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.bboxes

        if len(self.objects) == 0:
            for i in range(0, len(detections)):
                self.register(input_centroids[i], input_bboxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                # Objects disappeared
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # New objects appeared
                for col in unused_cols:
                    self.register(input_centroids[col], input_bboxes[col])

        return self.bboxes
