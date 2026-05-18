import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def match_by_cost(cost, thresh):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    rows, cols = linear_sum_assignment(cost)

    matches = []
    unmatched_rows = set(range(cost.shape[0]))
    unmatched_cols = set(range(cost.shape[1]))

    for r, c in zip(rows, cols):
        if cost[r, c] > thresh:
            continue

        matches.append((r, c))
        unmatched_rows.discard(r)
        unmatched_cols.discard(c)

    return matches, list(unmatched_rows), list(unmatched_cols)


def match_by_cost_dynamic_thresh(cost, row_thresh_fn):
    """
    row_thresh_fn(row_index) -> max allowed cost for this row
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    rows, cols = linear_sum_assignment(cost)

    matches = []
    unmatched_rows = set(range(cost.shape[0]))
    unmatched_cols = set(range(cost.shape[1]))

    for r, c in zip(rows, cols):
        thresh = row_thresh_fn(r)

        if cost[r, c] > thresh:
            continue

        matches.append((r, c))
        unmatched_rows.discard(r)
        unmatched_cols.discard(c)

    return matches, list(unmatched_rows), list(unmatched_cols)


def build_iou_cost(tracks, detections):
    """Matching object by bboxes IoU value"""
    cost = np.ones((len(tracks), len(detections)), dtype=np.float32)

    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            cost[i, j] = 1.0 - iou(trk.box, det[:4])

    return cost


def build_appearance_cost(tracks, det_feats, appearance_store):
    """Build appearance cost by track_id -> appearance memory."""
    cost = np.ones((len(tracks), len(det_feats)), dtype=np.float32)

    for i, trk in enumerate(tracks):
        for j, feat in enumerate(det_feats):
            sim = appearance_store.similarity(trk.id, feat)

            if sim < 0:
                cost[i, j] = 1.0
            else:
                cost[i, j] = 1.0 - sim

    return cost
