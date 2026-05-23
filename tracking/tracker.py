# tracking/tracker.py

from tracking.track import Track, TrackState
from tracking.interfaces import MotionModule, AppearanceModule
from tracking.utils.boxes import bbox_iou


class Tracker:
    def __init__(
            self,
            association,
            motion: MotionModule | None = None,
            appearance: AppearanceModule | None = None,
            max_missed=60,
            min_hits=2,
            soft_lost=10,
    ):
        self.association = association

        self.motion = motion
        self.appearance = appearance

        self.max_missed = max_missed
        self.min_hits = min_hits
        self.soft_lost = soft_lost

        self.tracks = []

        self.next_id = 0
        self.frame_id = 0

    def create_track(self, observation):

        track = Track(
            track_id=self.next_id,
            observation=observation,
        )

        # tentative starts with one hit
        track.hits = 1

        self.next_id += 1

        return track

    def remove_dead_tracks(self):

        alive_tracks = []

        for track in self.tracks:

            if track.missed > self.max_missed:

                track.mark_removed()

                if self.motion is not None:
                    self.motion.remove(track)

                if self.appearance is not None:
                    self.appearance.remove(track)

            else:
                alive_tracks.append(track)

        self.tracks = alive_tracks

    def update(
            self,
            frame,
            observations,
    ):
        self.frame_id += 1

        # -------------------------------------------------
        # appearance frame context
        # -------------------------------------------------

        if self.appearance is not None:

            self.appearance.begin_frame(
                frame,
                self.frame_id,
            )

        # -------------------------------------------------
        # timestep advance
        # -------------------------------------------------

        for track in self.tracks:
            track.step()

        # -------------------------------------------------
        # motion predict
        # -------------------------------------------------

        if self.motion is not None:
            self.motion.predict(self.tracks)

        # -------------------------------------------------
        # association
        # -------------------------------------------------

        matches, unmatched_tracks, unmatched_obs = (
            self.association.associate(
                tracks=self.tracks,
                observations=observations,
                motion=self.motion,
                appearance=self.appearance,
            )
        )

        # -------------------------------------------------
        # matched tracks
        # -------------------------------------------------

        for track_i, obs_i in matches:

            track = self.tracks[track_i]

            observation = observations[obs_i]

            track.update(observation)

            if self.motion is not None:
                self.motion.update(track, observation)

            if self.appearance is not None:
                self.appearance.update(track, observation)

        # -------------------------------------------------
        # unmatched tracks
        # -------------------------------------------------

        for track_i in unmatched_tracks:

            track = self.tracks[track_i]

            if track.state == TrackState.REMOVED:
                continue

            if track.missed > self.soft_lost:
                track.mark_lost()

        # -------------------------------------------------
        # duplicate suppression
        # -------------------------------------------------

        filtered_unmatched_obs = []

        for obs_i in unmatched_obs:

            observation = observations[obs_i]

            duplicate = False

            for track in self.tracks:

                if track.state == TrackState.REMOVED:
                    continue

                if track.last_observation is None:
                    continue

                iou = bbox_iou(
                    track.last_observation.box,
                    observation.box,
                )

                # existing track already explains this detection
                if iou > 0.5:

                    duplicate = True

                    # keep old track alive
                    track.missed = min(
                        track.missed,
                        self.soft_lost,
                    )

                    break

            if not duplicate:
                filtered_unmatched_obs.append(obs_i)

        # -------------------------------------------------
        # create new tracks
        # -------------------------------------------------

        for obs_i in filtered_unmatched_obs:

            observation = observations[obs_i]

            track = self.create_track(observation)

            self.tracks.append(track)

            if self.motion is not None:
                self.motion.initiate(track, observation)

            if self.appearance is not None:
                self.appearance.initiate(track, observation)

        # -------------------------------------------------
        # cleanup
        # -------------------------------------------------

        self.remove_dead_tracks()

        return self.active_tracks()

    def active_tracks(self):

        outputs = []

        for track in self.tracks:

            if track.state != TrackState.ACTIVE:
                continue

            if not track.just_updated:
                continue

            if track.hits < self.min_hits:
                continue

            outputs.append(track)

        return outputs