import numpy as np
import pytest

from twisty.cube3x3 import Cube3x3


@pytest.fixture
def cube():
    return Cube3x3()


@pytest.fixture
def solved(cube):
    return cube.solved_state()


def states_equal(s1, s2):
    return all(np.array_equal(s1[k], s2[k]) for k in s1)


def test_solved_state_has_zero_orientation(cube, solved):
    assert np.array_equal(solved["corner_perm"], np.arange(8))
    assert np.array_equal(solved["edge_perm"], np.arange(12))
    assert (solved["corner_ori"] == 0).all()
    assert (solved["edge_ori"] == 0).all()


@pytest.mark.parametrize("face", ["U", "D", "L", "R", "F", "B"])
def test_quarter_turn_applied_four_times_is_identity(cube, solved, face):
    move_idx = cube.MOVE_NAMES.index(face)
    state = solved
    for _ in range(4):
        state = cube.apply_move(state, move_idx)
    assert states_equal(state, solved)


@pytest.mark.parametrize("face", ["U", "D", "L", "R", "F", "B"])
def test_double_move_applied_twice_is_identity(cube, solved, face):
    move_idx = cube.MOVE_NAMES.index(f"{face}2")
    state = solved
    for _ in range(2):
        state = cube.apply_move(state, move_idx)
    assert states_equal(state, solved)


@pytest.mark.parametrize("face", ["U", "D", "L", "R", "F", "B"])
def test_double_move_equals_two_quarter_turns(cube, solved, face):
    quarter = cube.MOVE_NAMES.index(face)
    double = cube.MOVE_NAMES.index(f"{face}2")
    via_quarters = cube.apply_move(cube.apply_move(solved, quarter), quarter)
    via_double = cube.apply_move(solved, double)
    assert states_equal(via_quarters, via_double)


@pytest.mark.parametrize("face", ["U", "D", "L", "R", "F", "B"])
def test_move_and_prime_are_inverses(cube, solved, face):
    quarter = cube.MOVE_NAMES.index(face)
    prime = cube.MOVE_NAMES.index(f"{face}'")
    state = cube.apply_move(solved, quarter)
    state = cube.apply_move(state, prime)
    assert states_equal(state, solved)


def test_u_move_only_affects_top_layer(cube, solved):
    state = cube.apply_move(solved, cube.MOVE_NAMES.index("U"))
    moved_corners = {i for i in range(8) if state["corner_perm"][i] != i}
    moved_edges = {i for i in range(12) if state["edge_perm"][i] != i}
    assert moved_corners == {0, 1, 2, 3}
    assert moved_edges == {0, 1, 2, 3}


def test_sexy_move_has_order_six(cube, solved):
    seq = [cube.MOVE_NAMES.index(m) for m in ["R", "U", "R'", "U'"]]
    state = solved
    for _ in range(6):
        for m in seq:
            state = cube.apply_move(state, m)
    assert states_equal(state, solved)


def test_scramble_preserves_invariants(cube):
    rng = np.random.default_rng(0)
    for _ in range(200):
        state, seq = cube.scramble(int(rng.integers(1, 30)), rng=rng)
        assert len(seq) > 0
        assert state["corner_ori"].sum() % 3 == 0
        assert state["edge_ori"].sum() % 2 == 0
        assert sorted(state["corner_perm"].tolist()) == list(range(8))
        assert sorted(state["edge_perm"].tolist()) == list(range(12))


def test_apply_move_batch_matches_apply_move(cube, solved):
    rng = np.random.default_rng(1)
    states = [cube.scramble(int(rng.integers(1, 20)), rng=rng)[0] for _ in range(16)]
    batch = {
        "corner_perm": np.stack([s["corner_perm"] for s in states]),
        "corner_ori": np.stack([s["corner_ori"] for s in states]),
        "edge_perm": np.stack([s["edge_perm"] for s in states]),
        "edge_ori": np.stack([s["edge_ori"] for s in states]),
    }
    move_idx = 3
    batch_result = cube.apply_move_batch(batch, move_idx)
    for i, s in enumerate(states):
        expected = cube.apply_move(s, move_idx)
        for k in expected:
            assert np.array_equal(batch_result[k][i], expected[k])


def test_encode_shape_and_one_hot(cube, solved):
    enc = cube.encode(solved)
    assert enc.shape == (cube.input_dim(),)
    assert enc.sum() == 20  # one one-hot activation per piece (8 corners + 12 edges)


def test_scramble_is_reproducible_with_same_rng_seed(cube):
    s1, seq1 = cube.scramble(20, rng=np.random.default_rng(7))
    s2, seq2 = cube.scramble(20, rng=np.random.default_rng(7))
    assert seq1 == seq2
    assert states_equal(s1, s2)
