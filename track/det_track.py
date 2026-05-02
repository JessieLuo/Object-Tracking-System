from track.utils.fps import FpsMeter, draw_fps
from streaming.rtsp_pub import RtspWriter
from streaming.subscribe import FrameSource
from ultralytics import YOLO

import numpy as np
import cv2


class YOLOTracker:
    def __init__(self, model_path="", conf=0.3, classes=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.classes = classes

    def infer(self, frame: np.ndarray):
        results = self.model.track(
            frame,
            conf=self.conf,
            classes=self.classes,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return np.empty((0, 7), dtype=np.float32), frame

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy()

        if r.boxes.id is None:
            ids = -np.ones(len(boxes))
        else:
            ids = r.boxes.id.cpu().numpy()

        # (N,7)
        dets = np.column_stack((boxes, scores, clses, ids)).astype(np.float32)

        vis = r.plot()
        # vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        return dets, vis
