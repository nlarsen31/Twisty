from collections import Counter

import numpy as np
import pytest

from twisty.cube3x3 import Cube3x3
from twisty.visualize import FACE_COLORS, state_to_facelets


@pytest.fixture
def cube():
    return Cube3x3()


def _assert_valid_facelets(state):
    grid = state_to_facelets(state)
    all_colors = []
    for face in FACE_COLORS:
        assert (grid[face] != None).all()  # noqa: E711 (object array, need elementwise)
        all_colors.extend(grid[face].flatten().tolist())
    counts = Counter(all_colors)
    assert set(counts) == set(FACE_COLORS.values())
    assert all(v == 9 for v in counts.values())
    return grid


def test_solved_state_is_six_solid_faces(cube):
    grid = _assert_valid_facelets(cube.solved_state())
    for face, color in FACE_COLORS.items():
        assert (grid[face] == color).all()


@pytest.mark.parametrize("move_idx", range(12))
def test_every_single_move_produces_valid_facelets(cube, move_idx):
    state = cube.apply_move(cube.solved_state(), move_idx)
    _assert_valid_facelets(state)


def test_u_move_leaves_uniform_rows_on_side_faces(cube):
    """A single quarter turn relocates whole solid-colored strips of
    stickers; it must never scatter 3 different colors into one row."""
    state = cube.apply_move(cube.solved_state(), cube.MOVE_NAMES.index("U"))
    grid = state_to_facelets(state)
    for face in ("L", "F", "R", "B"):
        top_row = grid[face][0]
        assert len(set(top_row.tolist())) == 1, f"{face} top row not uniform: {top_row}"
        for row in (1, 2):
            assert (grid[face][row] == FACE_COLORS[face]).all()


def test_scrambled_states_produce_valid_facelets(cube):
    rng = np.random.default_rng(0)
    for _ in range(100):
        state, _ = cube.scramble(int(rng.integers(1, 30)), rng=rng)
        _assert_valid_facelets(state)
