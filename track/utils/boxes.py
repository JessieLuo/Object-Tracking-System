import numpy as np


def xyxy2xywh(bboxes: np.ndarray) -> np.ndarray:
    """center-based format
    [x1, y1, x2, y2] -> [cx, cy, w, h]."""
    out = bboxes.copy().astype(np.float32)
    out[:, 2] = out[:, 2] - out[:, 0]
    out[:, 3] = out[:, 3] - out[:, 1]
    out[:, 0] = out[:, 0] + out[:, 2] / 2
    out[:, 1] = out[:, 1] + out[:, 3] / 2
    return out


def xyxy2ltwh(bboxes: np.ndarray) -> np.ndarray:
    """left-top-based format
    [x1, y1, x2, y2] -> [x1, y1, w, h]."""
    out = bboxes.copy().astype(np.float32)
    out[:, 2] = out[:, 2] - out[:, 0]
    out[:, 3] = out[:, 3] - out[:, 1]
    return out


def nms(
        boxes: np.ndarray,
        scores: np.ndarray, iou_thr: float
) -> np.ndarray:
    """
    NMS for float bbox (xyxy), aligned with modern YOLO pipeline.

    Args:
        boxes:  (N, 4) float32 [x1, y1, x2, y2]
        scores: (N,)
        iou_thr: float

    Returns:
        keep indices: (K,) int32 ndarray
    """
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)
