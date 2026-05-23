import numpy as np


INF_COST = 1e5


def gate_cost(
        cost_matrix: np.ndarray,
        gate_mask: np.ndarray,
) -> np.ndarray:
    gated = cost_matrix.copy()
    gated[gate_mask] = INF_COST
    return gated