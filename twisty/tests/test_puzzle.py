import pytest

from twisty.puzzle import Move, TwistyPuzzle


def test_twisty_puzzle_is_abstract():
    with pytest.raises(TypeError):
        TwistyPuzzle(num_pieces=1, orientations_per_piece=1, move_generators=[])


def test_move_is_a_simple_named_value():
    m = Move("U")
    assert m.name == "U"
    assert m == Move("U")
    assert m != Move("U'")
