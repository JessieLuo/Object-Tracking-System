# tracking/association/associate.py

from tracking.interfaces import (
    MotionModule,
    AppearanceModule,
)

from tracking.association.assign import hungarian_assign

from tracking.association.costs import (
    build_iou_cost,
    build_motion_cost,
)

from tracking.association.gating import gate_cost


class MotionAppearanceAssociation:
    def __init__(
            self,
            active_iou_thresh=0.7,
            lost_iou_thresh=0.9,
            motion_thresh=9.4877,
    ):
        self.active_iou_thresh = active_iou_thresh
        self.lost_iou_thresh = lost_iou_thresh
        self.motion_thresh = motion_thresh

    def associate(
            self,
            tracks,
            observations,
            motion: MotionModule | None = None,
            appearance: AppearanceModule | None = None,
    ):
        if len(tracks) == 0 or len(observations) == 0:
            return (
                [],
                list(range(len(tracks))),
                list(range(len(observations))),
            )

        iou_cost = build_iou_cost(
            tracks,
            observations,
        )

        motion_cost = build_motion_cost(
            tracks,
            observations,
            motion,
        )

        cost = gate_cost(
            iou_cost,
            motion_cost > self.motion_thresh,
        )

        # -------------------------------------------------
        # adaptive threshold
        # -------------------------------------------------

        matches = []

        unmatched_tracks = list(range(len(tracks)))
        unmatched_obs = list(range(len(observations)))

        for track_i, track in enumerate(tracks):

            if track.state.name == "LOST":
                max_cost = self.lost_iou_thresh
            else:
                max_cost = self.active_iou_thresh

            local_matches, remain_t, remain_o = (
                hungarian_assign(
                    cost[track_i:track_i + 1],
                    max_cost=max_cost,
                )
            )

            if len(local_matches) > 0:

                _, obs_i = local_matches[0]

                if obs_i in unmatched_obs:

                    matches.append((
                        track_i,
                        obs_i,
                    ))

                    if track_i in unmatched_tracks:
                        unmatched_tracks.remove(track_i)

                    if obs_i in unmatched_obs:
                        unmatched_obs.remove(obs_i)

        return (
            matches,
            unmatched_tracks,
            unmatched_obs,
        )