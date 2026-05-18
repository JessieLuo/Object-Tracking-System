import cv2
import numpy as np

from tracking.track_state import Track
from tracking.appearance.memory import AppearanceStore


class Tracker:
    def __init__(
            self,
            motion_factory,
            appearance_model,
            association,
            max_lost=60,
            min_hits=2,
            appearance_interval=15,
            bank_size=30,
            ema_alpha=0.9,
            high_thresh=0.4,
            low_thresh=0.1,
            max_appearance_per_frame=2,
    ):
        self.motion_factory = motion_factory
        self.appearance_model = appearance_model
        self.association = association

        self.appearance_store = None
        if self.appearance_model is not None:
            self.appearance_store = AppearanceStore(
                bank_size=bank_size,
                ema_alpha=ema_alpha,
            )

        self.max_lost = max_lost
        self.min_hits = min_hits
        self.appearance_interval = appearance_interval
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_appearance_per_frame = max_appearance_per_frame

        self.active_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []

        self.next_id = 0
        self.frame_id = 0

    def split_detections(self, dets):
        if dets is None or len(dets) == 0:
            empty = np.empty((0, 6), dtype=np.float32)
            return empty, empty

        dets = np.asarray(dets, dtype=np.float32)

        high_dets = dets[dets[:, 4] >= self.high_thresh]

        low_dets = dets[
            (dets[:, 4] >= self.low_thresh)
            & (dets[:, 4] < self.high_thresh)
        ]

        return high_dets, low_dets

    def predict_tracks(self):
        for track in self.active_tracks:
            track.predict()

        for track in self.lost_tracks:
            track.predict()

    def extract_feature(self, frame, box):
        if self.appearance_model is None:
            return None

        return self.appearance_model.extract(frame, box)

    def build_high_det_features(self, frame, high_dets):
        if self.appearance_model is None:
            return None

        det_feats = [None for _ in range(len(high_dets))]

        if self.frame_id % self.appearance_interval != 0:
            return det_feats

        extracted = 0

        for i, det in enumerate(high_dets):
            if extracted >= self.max_appearance_per_frame:
                break

            feat = self.extract_feature(frame, det[:4])
            det_feats[i] = feat

            if feat is not None:
                extracted += 1

        return det_feats

    def create_track(self, det, feat=None):
        box = det[:4]
        score = det[4]
        cls_id = det[5] if len(det) > 5 else 0

        motion = self.motion_factory(box)

        track = Track(
            box=box,
            score=score,
            cls_id=cls_id,
            track_id=self.next_id,
            motion=motion,
        )

        self.next_id += 1

        if self.appearance_store is not None and feat is not None:
            self.appearance_store.update(track.id, feat)

        return track

    def update_active_matches(self, result, high_dets, high_det_feats):
        for track_i, det_i in result.active_matches:
            track = self.active_tracks[track_i]
            det = high_dets[det_i]

            track.update(
                box=det[:4],
                score=det[4],
                cls_id=det[5] if len(det) > 5 else 0,
            )

            if (
                    self.appearance_store is not None
                    and high_det_feats is not None
                    and high_det_feats[det_i] is not None
            ):
                self.appearance_store.update(
                    track.id,
                    high_det_feats[det_i],
                )

    def update_low_matches(self, result, low_dets):
        for track_i, det_i in result.low_matches:
            track = self.active_tracks[track_i]
            det = low_dets[det_i]

            track.update(
                box=det[:4],
                score=det[4],
                cls_id=det[5] if len(det) > 5 else 0,
            )

    def recover_lost_tracks(self, result, high_dets, high_det_feats):
        recovered_lost_ids = set()

        for lost_i, det_i in result.recovered_matches:
            track = self.lost_tracks[lost_i]
            det = high_dets[det_i]

            track.update(
                box=det[:4],
                score=det[4],
                cls_id=det[5] if len(det) > 5 else 0,
            )

            if (
                    self.appearance_store is not None
                    and high_det_feats is not None
                    and high_det_feats[det_i] is not None
            ):
                self.appearance_store.update(
                    track.id,
                    high_det_feats[det_i],
                )

            self.active_tracks.append(track)
            recovered_lost_ids.add(lost_i)

        self.lost_tracks = [
            track
            for i, track in enumerate(self.lost_tracks)
            if i not in recovered_lost_ids
        ]

    def move_unmatched_active_to_lost(self, unmatched_active_ids):
        unmatched_set = set(unmatched_active_ids)

        moved_to_lost = []

        for i, track in enumerate(self.active_tracks):
            if i in unmatched_set:
                track.mark_lost()
                moved_to_lost.append(track)

        self.active_tracks = [
            track
            for i, track in enumerate(self.active_tracks)
            if i not in unmatched_set
        ]

        self.lost_tracks.extend(moved_to_lost)

    def create_new_tracks(self, high_dets, high_det_feats, unmatched_high_ids):
        for det_i in unmatched_high_ids:
            feat = None

            if high_det_feats is not None:
                feat = high_det_feats[det_i]

            track = self.create_track(high_dets[det_i], feat=feat)
            self.active_tracks.append(track)

    def prune_lost_tracks(self):
        kept_lost = []

        for track in self.lost_tracks:
            if track.lost > self.max_lost:
                track.mark_removed()
                self.removed_tracks.append(track)

                if self.appearance_store is not None:
                    self.appearance_store.remove(track.id)
            else:
                kept_lost.append(track)

        self.lost_tracks = kept_lost

    def visible_tracks(self):
        output = []

        for track in self.active_tracks:
            if track.hits < self.min_hits:
                continue

            x1, y1, x2, y2 = track.box

            output.append([
                x1,
                y1,
                x2,
                y2,
                track.id,
                track.score,
                track.cls_id,
            ])

        if len(output) == 0:
            return np.empty((0, 7), dtype=np.float32)

        return np.asarray(output, dtype=np.float32)

    def draw_tracks(self, frame):
        vis = frame.copy()

        for track in self.active_tracks:
            if track.hits < self.min_hits:
                continue

            x1, y1, x2, y2 = map(int, track.box)

            cv2.rectangle(
                vis,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                vis,
                f"id={track.id}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return vis

    def update(self, frame, dets):
        self.frame_id += 1

        high_dets, low_dets = self.split_detections(dets)

        self.predict_tracks()

        high_det_feats = self.build_high_det_features(frame, high_dets)

        result = self.association.associate(
            active_tracks=self.active_tracks,
            lost_tracks=self.lost_tracks,
            high_dets=high_dets,
            low_dets=low_dets,
            high_det_feats=high_det_feats,
            appearance_store=self.appearance_store,
        )

        self.update_active_matches(result, high_dets, high_det_feats)
        self.update_low_matches(result, low_dets)
        self.recover_lost_tracks(result, high_dets, high_det_feats)
        self.move_unmatched_active_to_lost(result.unmatched_active_ids)
        self.create_new_tracks(
            high_dets,
            high_det_feats,
            result.unmatched_high_ids,
        )
        self.prune_lost_tracks()

        tracks = self.visible_tracks()
        vis = self.draw_tracks(frame)

        return tracks, vis