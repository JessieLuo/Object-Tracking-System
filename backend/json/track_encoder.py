# import time


def encode_tracks(
        camera_id,
        frame_id,
        frame_timestamp,
        tracks,
):

    payload_tracks = []

    for track in tracks:
        obs = track.last_observation

        if obs is None:
            continue

        payload_tracks.append({
            "track_id": int(track.id),
            "bbox": obs.box.astype(float).tolist(),
            "score": float(obs.score),
            # "state": track.state.name,
        })

    payload = {
        "camera_id": camera_id,
        "frame_id": int(frame_id),
        "timestamp": frame_timestamp, # the time before frame processing
        # "timestamp": time.time(), # the time after frame processing
        "tracks": payload_tracks,
    }

    return payload