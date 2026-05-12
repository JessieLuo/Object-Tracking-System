import cv2
import numpy as np

from collections import deque
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanBox
from .matching import iou, match_by_cost
from .reid import OSNetReID


class Track:
    def __init__(
        self,
        box,
        score,
        cls_id,
        track_id,
        feat=None,
        bank_size=20,
        ema_alpha=0.9,
    ):
        self.id = track_id
        self.score = float(score)
        self.cls_id = float(cls_id)

        self.kf = KalmanBox(box)
        self.box = np.asarray(box, dtype=np.float32)

        self.age = 1
        self.hits = 1
        self.lost = 0
        self.state = "active"

        self.features = deque(maxlen=bank_size)
        self.smooth_feat = None
        self.ema_alpha = ema_alpha

        if feat is not None:
            self.update_feature(feat)

    def update_feature(self, feat):
        if feat is None:
            return

        feat = feat.astype(np.float32)

        norm = np.linalg.norm(feat)
        if norm < 1e-6:
            return

        feat = feat / norm

        self.features.append(feat)

        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = (
                self.ema_alpha * self.smooth_feat
                + (1.0 - self.ema_alpha) * feat
            )
            self.smooth_feat = self.smooth_feat / max(
                np.linalg.norm(self.smooth_feat),
                1e-6,
            )

    def predict(self):
        self.box = self.kf.predict()
        self.age += 1
        self.lost += 1

        return self.box

    def update(self, box, score, cls_id, feat=None):
        self.box = self.kf.update(box)
        self.score = float(score)
        self.cls_id = float(cls_id)

        self.hits += 1
        self.lost = 0
        self.state = "active"

        self.update_feature(feat)

    def appearance_sim(self, feat):
        if feat is None:
            return -1.0

        feat = feat.astype(np.float32)
        feat = feat / max(np.linalg.norm(feat), 1e-6)

        smooth_sim = -1.0
        bank_sim = -1.0

        if self.smooth_feat is not None:
            smooth_sim = float(np.dot(self.smooth_feat, feat))

        if len(self.features) > 0:
            bank_sim = max(float(np.dot(f, feat)) for f in self.features)

        if smooth_sim < 0 and bank_sim < 0:
            return -1.0

        if smooth_sim < 0:
            return bank_sim

        if bank_sim < 0:
            return smooth_sim

        return 0.7 * smooth_sim + 0.3 * bank_sim


class ReIDTracker:
    def __init__(
        self,
        iou_thresh=0.3,
        reid_thresh=0.55,
        max_lost=30,
        min_hits=1,
        reid_interval=5,
        bank_size=20,
        ema_alpha=0.9, # ??
        reid_min_h=80,
        reid_weights="weights/osnet_x0_25_market.pth"
    ):
        self.active_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []

        self.next_id = 0
        self.frame_id = 0

        self.iou_thresh = iou_thresh
        self.reid_thresh = reid_thresh
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.reid_interval = reid_interval
        self.bank_size = bank_size
        self.ema_alpha = ema_alpha # ??

        self.reid = OSNetReID(
            weights_path=reid_weights,
            min_h=reid_min_h,
        )

    def reid_threshold_by_lost(self, lost):
        """dynamic threshold based on lost frames"""
        if lost <= 10:
            return self.reid_thresh

        if lost <= 30:
            return self.reid_thresh + 0.03

        if lost <= 60:
            return self.reid_thresh + 0.06

        return self.reid_thresh + 0.10

    def update(self, frame, detections):
        """Trackers core logic"""
        self.frame_id += 1

        detections = np.asarray(detections, dtype=np.float32)

        for trk in self.active_tracks:
            trk.predict()

        for trk in self.lost_tracks:
            trk.predict()

        if len(detections) == 0:
            for trk in self.active_tracks:
                trk.state = "lost"
                self.lost_tracks.append(trk)

            self.active_tracks = []
            self._clear_dead()

            tracks = self.export_tracks()
            vis = self.plot(frame, tracks)

            return tracks, vis

        new_active = []

        # =====================================================
        # 1. active tracks: IoU association
        # =====================================================

        active = self.active_tracks

        cost_iou = np.ones((len(active), len(detections)), dtype=np.float32)

        for i, trk in enumerate(active):
            for j, det in enumerate(detections):
                cost_iou[i, j] = 1.0 - iou(trk.box, det[:4])

        matches, unmatched_active, unmatched_dets = match_by_cost(
            cost_iou,
            thresh=1.0 - self.iou_thresh,
        )

        for ti, di in matches:
            trk = active[ti]
            det = detections[di]

            feat = None

            if self.frame_id % self.reid_interval == 0:
                feat = self.reid.extract(frame, det[:4])

            trk.update(det[:4], det[4], det[5], feat)
            new_active.append(trk)

        for ti in unmatched_active:
            trk = active[ti]
            trk.state = "lost"
            self.lost_tracks.append(trk)

        # =====================================================
        # 2. lost tracks: ReID association
        # =====================================================

        remain_dets = [detections[i] for i in unmatched_dets]

        if len(self.lost_tracks) > 0 and len(remain_dets) > 0:
            det_feats = []

            for det in remain_dets:
                feat = self.reid.extract(frame, det[:4])
                det_feats.append(feat)

            cost_reid = np.ones(
                (len(self.lost_tracks), len(remain_dets)),
                dtype=np.float32,
            )

            for i, trk in enumerate(self.lost_tracks):
                for j, feat in enumerate(det_feats):
                    sim = trk.appearance_sim(feat)
                    cost_reid[i, j] = 1.0 - sim

            # ========= enhancement start ==============
            rows, cols = linear_sum_assignment(cost_reid)

            reid_matches = []
            used_lost = set()
            used_remain = set()

            for li, rdi in zip(rows, cols):
                trk = self.lost_tracks[li]

                sim = 1.0 - cost_reid[li, rdi]
                thresh = self.reid_threshold_by_lost(trk.lost)

                if sim < thresh:
                    continue

                reid_matches.append((li, rdi))
                used_lost.add(li)
                used_remain.add(rdi)

            unmatched_lost = [
                i for i in range(len(self.lost_tracks))
                if i not in used_lost
            ]

            unmatched_remain = [
                i for i in range(len(remain_dets))
                if i not in used_remain
            ]
            # ========= enhancement end ==============

            recovered = []

            for li, rdi in reid_matches:
                trk = self.lost_tracks[li]
                det = remain_dets[rdi]
                feat = det_feats[rdi]

                trk.update(det[:4], det[4], det[5], feat)
                recovered.append(trk)

            new_active.extend(recovered)

            self.lost_tracks = [self.lost_tracks[i] for i in unmatched_lost]
            remain_dets = [remain_dets[i] for i in unmatched_remain]

        # =====================================================
        # 3. unmatched detections: new tracks
        # =====================================================

        for det in remain_dets:
            feat = self.reid.extract(frame, det[:4])

            trk = Track(
                box=det[:4],
                score=det[4],
                cls_id=det[5],
                track_id=self.next_id,
                feat=feat,
                bank_size=self.bank_size,
                ema_alpha=self.ema_alpha, # ?? 
            )

            self.next_id += 1
            new_active.append(trk)

        self.active_tracks = new_active

        self._clear_dead()

        tracks = self.export_tracks()
        vis = self.plot(frame, tracks)

        return tracks, vis

    def _clear_dead(self):
        keep_lost = []

        for trk in self.lost_tracks:
            if trk.lost <= self.max_lost:
                keep_lost.append(trk)
            else:
                trk.state = "removed"
                self.removed_tracks.append(trk)

        self.lost_tracks = keep_lost

    def export_tracks(self):
        outputs = []

        for trk in self.active_tracks:
            if trk.hits < self.min_hits:
                continue

            x1, y1, x2, y2 = trk.box

            outputs.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                    trk.id,
                    trk.score,
                    trk.cls_id,
                ]
            )

        if len(outputs) == 0:
            return np.empty((0, 7), dtype=np.float32)

        return np.asarray(outputs, dtype=np.float32)

    def plot(self, frame, tracks):
        vis = frame.copy()

        for trk in tracks:
            x1, y1, x2, y2, tid, score, cls_id = trk

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            tid = int(tid)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                vis,
                f"ID {tid}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return vis
