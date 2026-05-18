from dataclasses import dataclass

from tracking.association.matching import (
    build_iou_cost,
    build_appearance_cost,
    match_by_cost,
    match_by_cost_dynamic_thresh,
)


@dataclass
class AssociationResult:
    active_matches: list
    low_matches: list
    recovered_matches: list
    unmatched_active_ids: list
    unmatched_lost_ids: list
    unmatched_high_ids: list


class MotionAppearanceAssociation:
    """Kalman-predicted IoU association + appearance recovery."""

    def __init__(
            self,
            iou_thresh=0.3,
            low_iou_thresh=0.2,
            appearance_thresh=0.50,
            use_low_score_rescue=True,
            lost_association="appearance",
    ):
        self.iou_thresh = iou_thresh
        self.low_iou_thresh = low_iou_thresh
        self.appearance_thresh = appearance_thresh
        self.use_low_score_rescue = use_low_score_rescue
        self.lost_association = lost_association

    def threshold_by_lost(self, lost):
        if lost <= 10:
            return self.appearance_thresh

        if lost <= 30:
            return self.appearance_thresh + 0.03

        if lost <= 60:
            return self.appearance_thresh + 0.06

        return self.appearance_thresh + 0.10

    def associate(
            self,
            active_tracks,
            lost_tracks,
            high_dets,
            low_dets,
            high_det_feats,
            appearance_store,
    ):
        # 1. active tracks have already been motion-predicted.
        # Match predicted boxes with high-score detections by IoU.
        cost_motion_iou = build_iou_cost(active_tracks, high_dets)

        active_matches, unmatched_active, unmatched_high = match_by_cost(
            cost_motion_iou,
            thresh=1.0 - self.iou_thresh,
        )

        # 2. unmatched active tracks <-> low-score detections by IoU rescue.
        low_matches_global = []
        still_unmatched_active = unmatched_active

        if (
                self.use_low_score_rescue
                and len(still_unmatched_active) > 0
                and len(low_dets) > 0
        ):
            low_active = [active_tracks[i] for i in still_unmatched_active]
            cost_low_iou = build_iou_cost(low_active, low_dets)

            low_matches, low_unmatched_active, _ = match_by_cost(
                cost_low_iou,
                thresh=1.0 - self.low_iou_thresh,
            )

            for local_ti, low_di in low_matches:
                global_ti = still_unmatched_active[local_ti]
                low_matches_global.append((global_ti, low_di))

            still_unmatched_active = [
                still_unmatched_active[i]
                for i in low_unmatched_active
            ]

        # 3. lost tracks <-> remaining high detections by appearance.
        recovered_matches = []
        unmatched_lost = list(range(len(lost_tracks)))

        can_recover_by_appearance = (
                self.lost_association == "appearance"
                and appearance_store is not None
                and high_det_feats is not None
                and len(lost_tracks) > 0
                and len(unmatched_high) > 0
        )

        if can_recover_by_appearance:
            remain_feats = [high_det_feats[i] for i in unmatched_high]

            cost_appearance = build_appearance_cost(
                lost_tracks,
                remain_feats,
                appearance_store,
            )

            recovered_local, unmatched_lost, unmatched_remain = (
                match_by_cost_dynamic_thresh(
                    cost_appearance,
                    row_thresh_fn=lambda li: 1.0 - self.threshold_by_lost(
                        lost_tracks[li].lost
                    ),
                )
            )

            for lost_i, remain_i in recovered_local:
                high_i = unmatched_high[remain_i]
                recovered_matches.append((lost_i, high_i))

            unmatched_high = [
                unmatched_high[i]
                for i in unmatched_remain
            ]

        return AssociationResult(
            active_matches=active_matches,
            low_matches=low_matches_global,
            recovered_matches=recovered_matches,
            unmatched_active_ids=still_unmatched_active,
            unmatched_lost_ids=unmatched_lost,
            unmatched_high_ids=unmatched_high,
        )
