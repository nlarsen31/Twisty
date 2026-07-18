import numpy as np
import pytest

from twisty.bfs import bfs_distances, distance_of
from twisty.cube3x3 import Cube3x3


@pytest.fixture
def cube():
    return Cube3x3()


@pytest.fixture(scope="module")
def distances():
    return bfs_distances(Cube3x3(), max_depth=3)


def test_solved_state_has_distance_zero(cube, distances):
    assert distance_of(distances, cube.solved_state()) == 0


def test_all_single_moves_have_distance_one(cube, distances):
    solved = cube.solved_state()
    for move_idx in range(len(cube.MOVE_NAMES)):
        state = cube.apply_move(solved, move_idx)
        assert distance_of(distances, state) == 1


def test_distance_counts_match_known_htm_data(distances):
    # Published HTM state counts by distance from solved (cube20.org / Rokicki et al.).
    expected_counts = [1, 18, 243, 3240]
    counts = [0] * len(expected_counts)
    for dist in distances.values():
        counts[dist] += 1
    assert counts == expected_counts


def test_scrambled_state_distance_never_exceeds_scramble_length(cube, distances):
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(0, 4))  # keep within the depth-3 table
        state, _ = cube.scramble(n, rng=rng)
        d = distance_of(distances, state)
        assert d is not None
        assert d <= n


def test_same_face_pruning_does_not_lose_states(cube, distances):
    solved = cube.solved_state()

    for alg in [["R", "U"], ["U", "R", "U'"]]:
        state = solved
        for name in alg:
            state = cube.apply_move(state, cube.MOVE_NAMES.index(name))
        d = distance_of(distances, state)
        assert d is not None
        assert d <= len(alg)
