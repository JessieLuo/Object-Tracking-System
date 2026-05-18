import numpy as np


class Track:
    def __init__(
            self,
            box,
            score,
            cls_id,
            track_id,
            motion,
    ):
        self.id = int(track_id)
        self.score = float(score)
        self.cls_id = int(cls_id)

        self.box = np.asarray(box, dtype=np.float32)
        self.motion = motion

        self.age = 1
        self.hits = 1
        self.lost = 0
        self.state = "active"

    def predict(self):
        self.box = self.motion.predict()
        self.age += 1
        self.lost += 1
        return self.box

    def update(self, box, score, cls_id):
        self.box = self.motion.update(box)
        self.score = float(score)
        self.cls_id = int(cls_id)

        self.hits += 1
        self.lost = 0
        self.state = "active"

    def mark_lost(self):
        self.state = "lost"

    def mark_removed(self):
        self.state = "removed"