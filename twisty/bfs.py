"""Brute-force BFS ground truth: exact optimal distance-from-solved for shallow states.

Single-source BFS from `solved_state()`. Feasible up to about max_depth=6 (~7.6M states);
depth 7+ (100M+ states, per published HTM state-count data) is out of scope for this
single-source approach — a bidirectional meet-in-the-middle search would be needed there.
"""

from __future__ import annotations

import numpy as np

from twisty.cube3x3 import Cube3x3

_MOVE_FACES = [name[0] for name in Cube3x3.MOVE_NAMES]


def _state_key(state: dict[str, np.ndarray]) -> bytes:
    packed = np.concatenate(
        [state["corner_perm"], state["corner_ori"], state["edge_perm"], state["edge_ori"]]
    ).astype(np.int8)
    return packed.tobytes()


def bfs_distances(cube: Cube3x3, max_depth: int = 5) -> dict[bytes, int]:
    """Map every state reachable within `max_depth` moves of solved to its exact distance.

    Prunes moves on the same face as the previous move: since the generator set already
    includes the quarter, prime, and double turn for every face, turning a face again right
    after turning it is always redundant (equivalent to a single move already in the set, or
    identity).
    """
    solved = cube.solved_state()
    root_key = _state_key(solved)
    distances = {root_key: 0}
    frontier = [(solved, -1)]

    for depth in range(1, max_depth + 1):
        next_frontier = []
        for state, last_move in frontier:
            last_face = _MOVE_FACES[last_move] if last_move >= 0 else None
            for move_idx, face in enumerate(_MOVE_FACES):
                if face == last_face:
                    continue
                new_state = cube.apply_move(state, move_idx)
                key = _state_key(new_state)
                if key in distances:
                    continue
                distances[key] = depth
                next_frontier.append((new_state, move_idx))
        frontier = next_frontier

    return distances


def distance_of(distances: dict[bytes, int], state: dict[str, np.ndarray]) -> int | None:
    return distances.get(_state_key(state))
