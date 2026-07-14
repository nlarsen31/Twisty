from __future__ import annotations

import numpy as np

from twisty.puzzle import Move, TwistyPuzzle

# Reference slot positions in cube coordinates: x=right(+1)/left(-1),
# y=up(+1)/down(-1), z=front(+1)/back(-1). Order is arbitrary but fixed.
CORNER_POS = [
    (1, 1, 1),  # 0 URF
    (-1, 1, 1),  # 1 UFL
    (-1, 1, -1),  # 2 ULB
    (1, 1, -1),  # 3 UBR
    (1, -1, 1),  # 4 DFR
    (-1, -1, 1),  # 5 DLF
    (-1, -1, -1),  # 6 DLB
    (1, -1, -1),  # 7 DRB
]

EDGE_POS = [
    (1, 1, 0),  # 0 UR
    (0, 1, 1),  # 1 UF
    (-1, 1, 0),  # 2 UL
    (0, 1, -1),  # 3 UB
    (1, -1, 0),  # 4 DR
    (0, -1, 1),  # 5 DF
    (-1, -1, 0),  # 6 DL
    (0, -1, -1),  # 7 DB
    (1, 0, 1),  # 8 FR
    (-1, 0, 1),  # 9 FL
    (-1, 0, -1),  # 10 BL
    (1, 0, -1),  # 11 BR
]

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}

# (axis, layer coordinate value, base direction) for each face; the primed
# move is the same axis/layer with direction negated. Directions are chosen
# so each move is a clockwise turn viewed from outside that face (standard
# notation): our rotation matrices follow the usual convention where d=+1 is
# counterclockwise when viewed from the positive-axis side, so the layer at
# the positive coordinate needs d=-1 for clockwise-from-outside, and the
# layer at the negative coordinate needs d=+1.
_FACE_DEFS = {
    "U": ("y", 1, -1),
    "U'": ("y", 1, 1),
    "D": ("y", -1, 1),
    "D'": ("y", -1, -1),
    "L": ("x", -1, 1),
    "L'": ("x", -1, -1),
    "R": ("x", 1, -1),
    "R'": ("x", 1, 1),
    "F": ("z", 1, -1),
    "F'": ("z", 1, 1),
    "B": ("z", -1, 1),
    "B'": ("z", -1, -1),
}


def _axis_matrix(axis: str, direction: int) -> np.ndarray:
    """90-degree right-handed rotation about `axis` (signed permutation matrix)."""
    d = direction
    if axis == "x":
        return np.array([[1, 0, 0], [0, 0, -d], [0, d, 0]])
    if axis == "y":
        return np.array([[0, 0, d], [0, 1, 0], [-d, 0, 0]])
    return np.array([[0, -d, 0], [d, 0, 0], [0, 0, 1]])  # z


def _edge_primary_axis(pos: tuple[int, int, int]) -> str:
    """Reference facelet axis for edge orientation: U/D if the edge touches
    the U/D layer, otherwise F/B (this matches the standard "good/bad edge"
    convention used in cube theory)."""
    return "y" if pos[1] != 0 else "z"


def _build_corner_table(axis: str, layer_val: int, direction: int) -> tuple[np.ndarray, np.ndarray]:
    axis_i = _AXIS_IDX[axis]
    R = _axis_matrix(axis, direction)
    perm = np.arange(8)
    ori_delta = np.zeros(8, dtype=int)
    # Corners alternate chirality (checkerboard-like) depending on the sign
    # parity of their position, so the axis->orientation mapping must flip
    # accordingly, or the sum-of-orientations-mod-3 invariant breaks.
    axis_to_ori_pos = {1: 0, 2: 1, 0: 2}  # sticker ends up along y->0, z->1, x->2
    axis_to_ori_neg = {1: 0, 0: 1, 2: 2}
    for old_idx, pos in enumerate(CORNER_POS):
        if pos[axis_i] != layer_val:
            continue
        new_pos = tuple(int(v) for v in R @ np.array(pos))
        new_idx = CORNER_POS.index(new_pos)
        perm[new_idx] = old_idx
        result = R @ np.array([0, 1, 0])  # where the U/D (primary) sticker goes
        axis_out = int(np.nonzero(result)[0][0])
        parity = new_pos[0] * new_pos[1] * new_pos[2]
        axis_to_ori = axis_to_ori_pos if parity == 1 else axis_to_ori_neg
        ori_delta[new_idx] = axis_to_ori[axis_out]
    return perm, ori_delta


def _build_edge_table(axis: str, layer_val: int, direction: int) -> tuple[np.ndarray, np.ndarray]:
    axis_i = _AXIS_IDX[axis]
    R = _axis_matrix(axis, direction)
    perm = np.arange(12)
    ori_delta = np.zeros(12, dtype=int)
    axis_name = {0: "x", 1: "y", 2: "z"}
    for old_idx, pos in enumerate(EDGE_POS):
        if pos[axis_i] != layer_val:
            continue
        new_pos = tuple(int(v) for v in R @ np.array(pos))
        new_idx = EDGE_POS.index(new_pos)
        perm[new_idx] = old_idx
        primary_old = _edge_primary_axis(pos)
        primary_vec = np.array([0, 1, 0]) if primary_old == "y" else np.array([0, 0, 1])
        result = R @ primary_vec
        axis_out = axis_name[int(np.nonzero(result)[0][0])]
        primary_new = _edge_primary_axis(new_pos)
        ori_delta[new_idx] = 0 if axis_out == primary_new else 1
    return perm, ori_delta


def _compose_table(perm: np.ndarray, ori: np.ndarray, mod: int) -> tuple[np.ndarray, np.ndarray]:
    """Compose a move table with itself: the table for applying it twice
    (used to derive 180-degree double moves like U2 from the U quarter turn)."""
    return perm[perm], (ori[perm] + ori) % mod


_BASE_MOVE_NAMES = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"]


class Cube3x3(TwistyPuzzle):
    """3x3x3 Rubik's Cube, represented as 8 corner + 12 edge cubies (not stickers)."""

    NUM_CORNERS = 8
    NUM_EDGES = 12
    # Quarter turns plus double (180-degree) moves, so move counts here match
    # the Half-Turn Metric (HTM) used for e.g. God's number = 20.
    MOVE_NAMES = [
        "U", "U'", "U2",
        "D", "D'", "D2",
        "L", "L'", "L2",
        "R", "R'", "R2",
        "F", "F'", "F2",
        "B", "B'", "B2",
    ]

    def __init__(self):
        moves = [Move(name) for name in self.MOVE_NAMES]
        super().__init__(
            num_pieces=self.NUM_CORNERS + self.NUM_EDGES,
            orientations_per_piece=3,
            move_generators=moves,
        )
        base = {}
        for name in _BASE_MOVE_NAMES:
            axis, layer_val, direction = _FACE_DEFS[name]
            cp, co = _build_corner_table(axis, layer_val, direction)
            ep, eo = _build_edge_table(axis, layer_val, direction)
            base[name] = (cp, co, ep, eo)

        corner_perms, corner_oris, edge_perms, edge_oris = [], [], [], []
        for name in self.MOVE_NAMES:
            if name.endswith("2"):
                cp, co, ep, eo = base[name[:-1]]
                cp, co = _compose_table(cp, co, 3)
                ep, eo = _compose_table(ep, eo, 2)
            else:
                cp, co, ep, eo = base[name]
            corner_perms.append(cp)
            corner_oris.append(co)
            edge_perms.append(ep)
            edge_oris.append(eo)
        self._corner_perm_table = np.array(corner_perms)  # (18, 8)
        self._corner_ori_table = np.array(corner_oris)  # (18, 8)
        self._edge_perm_table = np.array(edge_perms)  # (18, 12)
        self._edge_ori_table = np.array(edge_oris)  # (18, 12)

    def solved_state(self) -> dict[str, np.ndarray]:
        return {
            "corner_perm": np.arange(8),
            "corner_ori": np.zeros(8, dtype=int),
            "edge_perm": np.arange(12),
            "edge_ori": np.zeros(12, dtype=int),
        }

    def apply_move(self, state: dict[str, np.ndarray], move_idx: int) -> dict[str, np.ndarray]:
        m_cp = self._corner_perm_table[move_idx]
        m_co = self._corner_ori_table[move_idx]
        m_ep = self._edge_perm_table[move_idx]
        m_eo = self._edge_ori_table[move_idx]
        return {
            "corner_perm": state["corner_perm"][m_cp],
            "corner_ori": (state["corner_ori"][m_cp] + m_co) % 3,
            "edge_perm": state["edge_perm"][m_ep],
            "edge_ori": (state["edge_ori"][m_ep] + m_eo) % 2,
        }

    def apply_move_batch(self, states: dict[str, np.ndarray], move_idx: int) -> dict[str, np.ndarray]:
        m_cp = self._corner_perm_table[move_idx]
        m_co = self._corner_ori_table[move_idx]
        m_ep = self._edge_perm_table[move_idx]
        m_eo = self._edge_ori_table[move_idx]
        return {
            "corner_perm": states["corner_perm"][:, m_cp],
            "corner_ori": (states["corner_ori"][:, m_cp] + m_co) % 3,
            "edge_perm": states["edge_perm"][:, m_ep],
            "edge_ori": (states["edge_ori"][:, m_ep] + m_eo) % 2,
        }

    def scramble(
        self, n_moves: int, rng: np.random.Generator | None = None
    ) -> tuple[dict[str, np.ndarray], list[int]]:
        rng = rng if rng is not None else np.random.default_rng()
        state = self.solved_state()
        move_seq = []
        for _ in range(n_moves):
            m = int(rng.integers(0, self.num_actions()))
            state = self.apply_move(state, m)
            move_seq.append(m)
        return state, move_seq

    def encode(self, state: dict[str, np.ndarray]) -> np.ndarray:
        corner_cat = state["corner_perm"] * 3 + state["corner_ori"]  # (8,) in [0,24)
        edge_cat = state["edge_perm"] * 2 + state["edge_ori"]  # (12,) in [0,24)
        corner_oh = np.eye(24, dtype=np.float32)[corner_cat].reshape(-1)
        edge_oh = np.eye(24, dtype=np.float32)[edge_cat].reshape(-1)
        return np.concatenate([corner_oh, edge_oh])

    def input_dim(self) -> int:
        return self.NUM_CORNERS * 24 + self.NUM_EDGES * 24
