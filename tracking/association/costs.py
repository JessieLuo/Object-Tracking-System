import numpy as np

from tracking.interfaces import (
    MotionModule,
    AppearanceModule,
)

from tracking.utils.boxes import bbox_iou


def build_iou_cost(
        tracks,
        observations,
):
    cost = np.ones(
        (len(tracks), len(observations)),
        dtype=np.float32,
    )

    for i, track in enumerate(tracks):
        if track.last_observation is None:
            continue

        track_box = track.last_observation.box

        for j, obs in enumerate(observations):
            cost[i, j] = 1.0 - bbox_iou(
                track_box,
                obs.box,
            )

    return cost


def build_motion_cost(
        tracks,
        observations,
        motion: MotionModule | None,
):
    if motion is None:
        return np.zeros(
            (len(tracks), len(observations)),
            dtype=np.float32,
        )

    boxes = np.asarray(
        [obs.box for obs in observations],
        dtype=np.float32,
    )

    cost = np.zeros(
        (len(tracks), len(observations)),
        dtype=np.float32,
    )

    for i, track in enumerate(tracks):
        cost[i] = motion.gating_distance(
            track,
            boxes,
        )

    return cost


def build_appearance_cost(
        tracks,
        observations,
        appearance: AppearanceModule | None,
):
    cost = np.ones(
        (len(tracks), len(observations)),
        dtype=np.float32,
    )

    if appearance is None:
        return cost

    for i, track in enumerate(tracks):
        for j, obs in enumerate(observations):
            sim = appearance.similarity(
                track,
                obs.feat,
            )

            if sim < 0:
                cost[i, j] = 1.0
            else:
                cost[i, j] = 1.0 - sim

    return cost