from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Move:
    """A named generator move. Concrete puzzles map these to permutation/orientation tables."""

    name: str


class TwistyPuzzle(ABC):
    """Abstract base class all specific puzzles implement."""

    def __init__(
        self,
        num_pieces: int,
        orientations_per_piece: int,
        move_generators: list[Move],
    ):
        self.num_pieces = num_pieces
        self.orientations_per_piece = orientations_per_piece
        self.move_generators = move_generators

    @abstractmethod
    def solved_state(self) -> Any:
        """Return identity state (permutation + orientation arrays)."""
        raise NotImplementedError

    @abstractmethod
    def apply_move(self, state: Any, move_idx: int) -> Any:
        """Return new state after applying move_idx. Must support batched states."""
        raise NotImplementedError

    @abstractmethod
    def apply_move_batch(self, states: Any, move_idx: int) -> Any:
        """Vectorized version for batch data generation. Critical for speed."""
        raise NotImplementedError

    @abstractmethod
    def scramble(
        self, n_moves: int, rng: np.random.Generator | None = None
    ) -> tuple[Any, list[int]]:
        """Apply n random moves from solved, return final state + move sequence."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, state: Any) -> np.ndarray:
        """Convert state to network input tensor (one-hot per piece position+orientation)."""
        raise NotImplementedError

    def num_actions(self) -> int:
        return len(self.move_generators)

    @abstractmethod
    def input_dim(self) -> int:
        """Size of encoded state vector — puzzle-specific."""
        raise NotImplementedError
