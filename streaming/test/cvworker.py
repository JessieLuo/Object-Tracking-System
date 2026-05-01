import cv2 as cv
import numpy as np


class CvWorker:
    """Pure frame processor"""

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return frame

        h, w = frame.shape[:2]

        box_w = int(w * 0.3)
        box_h = int(h * 0.3)

        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # center point
        cx, cy = w // 2, h // 2
        cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        return frame
