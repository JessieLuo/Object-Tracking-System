from scipy.optimize import linear_sum_assignment


INF_COST = 1e5


def hungarian_assign(
        cost_matrix,
        max_cost,
):
    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    rows, cols = linear_sum_assignment(cost_matrix)

    matches = []

    unmatched_tracks = set(range(cost_matrix.shape[0]))
    unmatched_observations = set(range(cost_matrix.shape[1]))

    for r, c in zip(rows, cols):
        cost = cost_matrix[r, c]

        if cost >= INF_COST:
            continue

        if cost > max_cost:
            continue

        matches.append((r, c))

        unmatched_tracks.discard(r)
        unmatched_observations.discard(c)

    return (
        matches,
        list(unmatched_tracks),
        list(unmatched_observations),
    )