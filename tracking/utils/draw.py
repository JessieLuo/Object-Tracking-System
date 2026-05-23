# tracking/visualization/draw.py

import cv2


def draw_tracks(frame, tracks):
    vis = frame.copy()

    for track in tracks:
        obs = track.last_observation

        if obs is None:
            continue

        x1, y1, x2, y2 = obs.box.astype(int)

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            vis,
            f"id:{track.id}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return vis