import cv2
import numpy as np

from collections import deque

from .kalman import KalmanFilterXYAH
from .matching import iou, match_by_cost, match_by_cost_dynamic_thresh
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

        self.box = np.asarray(box, dtype=np.float32)
        self.kf = KalmanFilterXYAH(box)

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
        reid_thresh=0.50,
        max_lost=60,
        min_hits=2,
        reid_interval=15,
        bank_size=20,
        ema_alpha=0.9,
        reid_min_h=80,
        reid_model_name="osnet_x0_25",
        reid_weights="weights/osnet_x0_25_msmt17.pt",
        high_thresh=0.4,
        low_thresh=0.1,
        low_iou_thresh=0.2,
        max_reid_per_frame=2,
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
        self.ema_alpha = ema_alpha

        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.low_iou_thresh = low_iou_thresh

        self.max_reid_per_frame = max_reid_per_frame
        self.reid_count = 0

        self.reid = OSNetReID(
            model_name=reid_model_name,
            weights_path=reid_weights,
            min_h=reid_min_h,
        )

    def extract_reid(self, frame, box):
        """restrict reid count"""
        if self.reid_count >= self.max_reid_per_frame:
            return None

        feat = self.reid.extract(frame, box) # ! Heavy Calculation
        self.reid_count += 1

        return feat

    def reid_threshold_by_lost(self, lost):
        if lost <= 10:
            return self.reid_thresh

        if lost <= 30:
            return self.reid_thresh + 0.03

        if lost <= 60:
            return self.reid_thresh + 0.06

        return self.reid_thresh + 0.10

    def update(self, frame, detections):
        self.frame_id += 1
        self.reid_count = 0

        detections = np.asarray(detections, dtype=np.float32)

        if detections.size == 0:
            detections = np.empty((0, 6), dtype=np.float32)

        if detections.ndim == 1:
            detections = detections.reshape(-1, 6)

        if len(detections) > 0:
            high_dets = detections[detections[:, 4] >= self.high_thresh]

            low_dets = detections[
                (detections[:, 4] >= self.low_thresh)
                & (detections[:, 4] < self.high_thresh)
            ]
        else:
            high_dets = np.empty((0, 6), dtype=np.float32)
            low_dets = np.empty((0, 6), dtype=np.float32)

        # =====================================================
        # 1. predict
        # =====================================================
        for trk in self.active_tracks:
            trk.predict()

        for trk in self.lost_tracks:
            trk.predict()

        # =====================================================
        # 2. no detections
        #
        # 这是真正的 no object，不输出预测框。
        # active -> lost，但不画 lost。
        # =====================================================
        if len(detections) == 0:
            for trk in self.active_tracks:
                trk.state = "lost"
                self.lost_tracks.append(trk)

            self.active_tracks = []
            self._clear_dead()

            tracks = np.empty((0, 7), dtype=np.float32)
            vis = self.plot(frame, tracks)

            return tracks, vis

        new_active = []
        active = self.active_tracks

        # =====================================================
        # 3. active tracks <-> high detections by IoU
        # =====================================================
        cost_iou = np.ones((len(active), len(high_dets)), dtype=np.float32)

        for i, trk in enumerate(active):
            for j, det in enumerate(high_dets):
                cost_iou[i, j] = 1.0 - iou(trk.box, det[:4])

        matches, unmatched_active, unmatched_high = match_by_cost(
            cost_iou,
            thresh=1.0 - self.iou_thresh,
        )

        for ti, di in matches:
            trk = active[ti]
            det = high_dets[di]

            feat = None

            if self.frame_id % self.reid_interval == 0:
                feat = self.extract_reid(frame, det[:4])

            trk.update(det[:4], det[4], det[5], feat)
            new_active.append(trk)

        # =====================================================
        # 4. unmatched active <-> low detections by IoU rescue
        #
        # 低分框只允许维持已有 active track：
        # - 不建新 track
        # - 不 ReID 恢复 lost track
        # - 不更新 feature bank
        # =====================================================
        still_unmatched_active = unmatched_active

        if len(still_unmatched_active) > 0 and len(low_dets) > 0:
            low_active = [active[i] for i in still_unmatched_active]

            cost_low = np.ones(
                (len(low_active), len(low_dets)),
                dtype=np.float32,
            )

            for i, trk in enumerate(low_active):
                for j, det in enumerate(low_dets):
                    cost_low[i, j] = 1.0 - iou(trk.box, det[:4])

            low_matches, low_unmatched_active, _ = match_by_cost(
                cost_low,
                thresh=1.0 - self.low_iou_thresh,
            )

            for local_ti, di in low_matches:
                trk = low_active[local_ti]
                det = low_dets[di]

                trk.update(det[:4], det[4], det[5], feat=None)
                new_active.append(trk)

            still_unmatched_active = [
                still_unmatched_active[i]
                for i in low_unmatched_active
            ]

        # unmatched active -> lost
        for ti in still_unmatched_active:
            trk = active[ti]
            trk.state = "lost"
            self.lost_tracks.append(trk)

        # 只有 high det 能进入 ReID 恢复和新建 track。
        remain_dets = [high_dets[i] for i in unmatched_high]

        # =====================================================
        # 5. lost tracks <-> remaining high detections by ReID
        #
        # 不再使用硬 IoU gate。
        # 遮挡后 Kalman 预测位置很容易错，硬 gate 会直接毁掉 ReID。
        # =====================================================
        if len(self.lost_tracks) > 0 and len(remain_dets) > 0:
            det_feats = []

            for det in remain_dets:
                feat = self.extract_reid(frame, det[:4])
                det_feats.append(feat)

            cost_reid = np.ones(
                (len(self.lost_tracks), len(remain_dets)),
                dtype=np.float32,
            )

            for i, trk in enumerate(self.lost_tracks):
                for j, feat in enumerate(det_feats):
                    sim = trk.appearance_sim(feat)

                    if sim < 0:
                        cost_reid[i, j] = 1.0
                        continue

                    cost_reid[i, j] = 1.0 - sim

            reid_matches, unmatched_lost, unmatched_remain = (
                match_by_cost_dynamic_thresh(
                    cost_reid,
                    row_thresh_fn=lambda li: 1.0
                    - self.reid_threshold_by_lost(
                        self.lost_tracks[li].lost
                    ),
                )
            )

            recovered = []

            for li, rdi in reid_matches:
                trk = self.lost_tracks[li]
                det = remain_dets[rdi]
                feat = det_feats[rdi]

                trk.update(det[:4], det[4], det[5], feat)
                recovered.append(trk)

            new_active.extend(recovered)

            self.lost_tracks = [
                self.lost_tracks[i]
                for i in unmatched_lost
            ]

            remain_dets = [
                remain_dets[i]
                for i in unmatched_remain
            ]

        # =====================================================
        # 6. unmatched high detections -> new tracks
        #
        # 注意：
        # 这里不再偷偷尝试恢复旧 ID。
        # ReID 恢复只能发生在 step 5。
        # =====================================================
        for det in remain_dets:
            feat = self.extract_reid(frame, det[:4])

            trk = Track(
                box=det[:4],
                score=det[4],
                cls_id=det[5],
                track_id=self.next_id,
                feat=feat,
                bank_size=self.bank_size,
                ema_alpha=self.ema_alpha,
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
            if trk.state != "active":
                continue

            if trk.lost != 0:
                continue

            # 新 track 必须连续命中 min_hits 次才显示。
            # 这用于压制单帧误检造成的幽灵框。
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
