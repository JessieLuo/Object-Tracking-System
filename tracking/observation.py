# tracking/observation.py

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Observation:
    box: np.ndarray
    score: float
    feat: Optional[np.ndarray] = None