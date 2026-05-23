from tracking.appearance.memory import AppearanceStore


class ScheduledAppearance:
    def __init__(
            self,
            reid,
            memory: AppearanceStore,
            interval=5,
            max_per_frame=4,
    ):
        self.reid = reid
        self.memory = memory

        self.interval = interval
        self.max_per_frame = max_per_frame

        self.frame = None
        self.frame_id = 0

        self.used_this_frame = 0

    def reset_budget(self):
        self.used_this_frame = 0

    def begin_frame(
            self,
            frame,
            frame_id: int,
    ):
        self.frame = frame
        self.frame_id = frame_id

        self.reset_budget()

    def should_extract(self):
        return (
                self.frame is not None
                and
                self.frame_id % self.interval == 0
                and
                self.used_this_frame < self.max_per_frame
        )

    def extract_scheduled(
            self,
            observation,
    ):
        if observation.feat is not None:
            return

        if not self.should_extract():
            return

        feat = self.reid.extract(
            self.frame,
            observation.box,
        )

        if feat is None:
            return

        observation.feat = feat

        self.used_this_frame += 1

    def extract_force(
            self,
            observation,
    ):
        if observation.feat is not None:
            return

        if self.frame is None:
            return

        feat = self.reid.extract(
            self.frame,
            observation.box,
        )

        if feat is None:
            return

        observation.feat = feat

    def extract_candidates(
            self,
            observations,
            limit=None,
    ):
        if self.frame is None:
            return

        if limit is None:
            limit = self.max_per_frame

        extracted = 0

        for obs in observations:

            if extracted >= limit:
                break

            if self.used_this_frame >= self.max_per_frame:
                break

            if obs.feat is not None:
                continue

            feat = self.reid.extract(
                self.frame,
                obs.box,
            )

            if feat is None:
                continue

            obs.feat = feat

            extracted += 1
            self.used_this_frame += 1

    def initiate(
            self,
            track,
            observation,
    ):
        self.extract_force(observation)

        self.memory.initiate(
            track,
            observation,
        )

    def update(
            self,
            track,
            observation,
    ):
        self.extract_scheduled(observation)

        self.memory.update(
            track,
            observation,
        )

    def similarity(
            self,
            track,
            feat,
    ):
        return self.memory.similarity(
            track,
            feat,
        )

    def remove(
            self,
            track,
    ):
        self.memory.remove(track)