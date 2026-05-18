"""https://github.com/ANGJustinl/gcxl_2025/blob/6f595e8f868145d0cd744595a9001056d18262a5/Vision/NCNN_Yolo.py
verified that ultralytics can run ncnn yolo directly"""
import numpy as np
from ultralytics import YOLO


class UltralyticsYoloDetector:
    def __init__(self,
                 model_path: str,
                 conf: float = 0.3,
                 classes: list[str] | None = None
                 ):
        self.model = YOLO(model_path, task="detect")
        self.conf = conf
        self.classes = classes

    def _load_input(self, img):
        if isinstance(img, np.ndarray):
            return img

        if isinstance(img, str):
            # Ultralytics accepts file path directly, so just return the path str
            return img

        raise TypeError(f"Unsupported input type: {type(img)}")

    def inference(self, img: str | np.ndarray) -> np.ndarray:
        """img can be either:
        - np.ndarray (BGR)
        - str (image file path); only used for offline/local debugging

        return: (N, 6)
        [x1, y1, x2, y2, score, class_id]
        """
        inp = self._load_input(img)

        results = self.model.predict(
            inp, conf=self.conf,
            classes=self.classes,
            verbose=False)

        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        boxes = r.boxes.xyxy.cpu().numpy()  # (N, 4)
        scores = r.boxes.conf.cpu().numpy()  # (N,)
        class_ids = r.boxes.cls.cpu().numpy()  # (N,)

        return np.column_stack((boxes, scores, class_ids)).astype(np.float32)
