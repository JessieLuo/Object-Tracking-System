# tracking/track.py

from enum import Enum, auto
from typing import Optional

from tracking.observation import Observation


class TrackState(Enum):
    TENTATIVE = auto()
    ACTIVE = auto()
    LOST = auto()
    REMOVED = auto()


class Track:
    def __init__(
            self,
            track_id: int,
            observation: Optional[Observation] = None,
    ):
        self.id = int(track_id)

        self.state = TrackState.TENTATIVE

        self.age = 0
        self.hits = 0
        self.missed = 0

        self.just_updated = False

        self.last_observation: Optional[Observation] = observation

    def step(self):
        self.age += 1
        self.missed += 1

        self.just_updated = False

    def update(
            self,
            observation: Observation,
    ):
        self.last_observation = observation

        self.hits += 1
        self.missed = 0

        self.just_updated = True

        if self.state == TrackState.TENTATIVE:

            if self.hits >= 3:
                self.state = TrackState.ACTIVE

        elif self.state != TrackState.REMOVED:

            self.state = TrackState.ACTIVE

    def mark_lost(self):

        if self.state != TrackState.REMOVED:
            self.state = TrackState.LOST

    def mark_removed(self):
        self.state = TrackState.REMOVED