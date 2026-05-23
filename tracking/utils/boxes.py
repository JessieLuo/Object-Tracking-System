import numpy as np


def bbox_iou(
        box1,
        box2,
) -> float:
    """
    IoU for TLBR boxes.

    box:
        [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)

    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(0.0, bx2 - bx1) * max(
        0.0,
        by2 - by1,
    )

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return float(inter / union)


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


def tlbr_to_xyah(box):
    """
    Convert a bounding box from TLBR format to XYAH format.
    tlbr:
        [x1, y1, x2, y2]
        where:
            (x1, y1) = top-left corner
            (x2, y2) = bottom-right corner
    XYAH:
        [cx, cy, a, h]
        where:
            cx = box center x-coordinate
            cy = box center y-coordinate
            a  = aspect ratio (width / height)
            h  = box height

    This representation is commonly used in Kalman-filter-based tracking
    because center position and box scale evolve more smoothly over time
    than raw corner coordinates.
    """
    x1, y1, x2, y2 = box

    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    cx = x1 + w / 2
    cy = y1 + h / 2
    a = w / h  # box size ratio

    return np.array([cx, cy, a, h], dtype=np.float32)


def xyah_to_tlbr(x):
    cx, cy, a, h = x[:4]

    w = a * h

    return np.array(
        [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
        ],
        dtype=np.float32,
    )


def nms_dets(
        dets: np.ndarray,
        iou_thr: float = 0.5,
) -> np.ndarray:
    """
    Apply NMS to detection array.

    dets:
        (N, 6) = [x1, y1, x2, y2, score, cls]
    """
    if dets is None or len(dets) == 0:
        return np.empty((0, 6), dtype=np.float32)

    dets = np.asarray(dets, dtype=np.float32)

    boxes = dets[:, :4]
    scores = dets[:, 4]

    keep = nms(boxes, scores, iou_thr=iou_thr)

    return dets[keep]
