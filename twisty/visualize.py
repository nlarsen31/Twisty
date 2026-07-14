from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from twisty.cube3x3 import CORNER_POS, EDGE_POS

FACE_COLORS = {
    "U": "#ffffff",
    "D": "#ffd500",
    "F": "#00a651",
    "B": "#0051ba",
    "R": "#c41e3a",
    "L": "#ff5800",
}

# (normal_axis, normal_sign) -> face name. axis 0=x, 1=y, 2=z.
_AXIS_SIGN_TO_FACE = {
    (1, 1): "U",
    (1, -1): "D",
    (2, 1): "F",
    (2, -1): "B",
    (0, 1): "R",
    (0, -1): "L",
}

# Each face's (row, col) grid coordinates as a function of (x, y, z).
_FACE_ROW_COL = {
    "U": (lambda x, y, z: z + 1, lambda x, y, z: x + 1),
    "D": (lambda x, y, z: 1 - z, lambda x, y, z: x + 1),
    "F": (lambda x, y, z: 1 - y, lambda x, y, z: x + 1),
    "B": (lambda x, y, z: 1 - y, lambda x, y, z: 1 - x),
    "R": (lambda x, y, z: 1 - y, lambda x, y, z: 1 - z),
    "L": (lambda x, y, z: 1 - y, lambda x, y, z: z + 1),
}

# Net layout: (col_offset, row_offset) of each face's bottom-left corner, in
# units of one sticker, for the unfolded cross layout.
_NET_OFFSET = {
    "U": (3, 6),
    "L": (0, 3),
    "F": (3, 3),
    "R": (6, 3),
    "B": (9, 3),
    "D": (3, 0),
}

# axis_to_ori maps used when building corner move tables (twisty/cube3x3.py),
# inverted here to recover which axis/sign the U/D sticker maps to given ori.
_INV_AXIS_TO_ORI_POS = {0: 1, 1: 2, 2: 0}
_INV_AXIS_TO_ORI_NEG = {0: 1, 1: 0, 2: 2}


def _corner_rotation(
    old_pos: tuple[int, int, int],
    new_pos: tuple[int, int, int],
    axis_out: int,
    sign_out: int,
) -> np.ndarray:
    """Reconstruct the 3x3 rotation matrix that moved a corner from old_pos to
    new_pos, given where its primary (U/D) sticker ends up. A proper rotation
    is fully determined by the images of any 2 linearly independent vectors
    plus det=+1, so this also pins down where the other two stickers land."""
    u1, u2 = np.array([0, 1, 0]), np.array(old_pos)
    v1 = np.zeros(3, dtype=int)
    v1[axis_out] = sign_out
    v2 = np.array(new_pos)
    u3, v3 = np.cross(u1, u2), np.cross(v1, v2)
    U = np.stack([u1, u2, u3], axis=1)
    V = np.stack([v1, v2, v3], axis=1)
    return np.round(V @ np.linalg.inv(U)).astype(int)


def _color_for(axis: int, sign: int) -> str:
    return FACE_COLORS[_AXIS_SIGN_TO_FACE[(axis, sign)]]


def _face_row_col(pos: tuple[int, int, int], axis: int) -> tuple[str, int, int]:
    sign = pos[axis]
    face = _AXIS_SIGN_TO_FACE[(axis, sign)]
    row_fn, col_fn = _FACE_ROW_COL[face]
    return face, row_fn(*pos), col_fn(*pos)


def state_to_facelets(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert cubie state to a dict of 3x3 color grids, one per face."""
    grid = {face: np.empty((3, 3), dtype=object) for face in FACE_COLORS}
    for face, color in FACE_COLORS.items():
        grid[face][1, 1] = color

    slot_parity = [p[0] * p[1] * p[2] for p in CORNER_POS]
    for slot_idx in range(8):
        pos = CORNER_POS[slot_idx]
        piece_id = int(state["corner_perm"][slot_idx])
        ori = int(state["corner_ori"][slot_idx])
        orig_pos = CORNER_POS[piece_id]

        inv_map = _INV_AXIS_TO_ORI_POS if slot_parity[slot_idx] == 1 else _INV_AXIS_TO_ORI_NEG
        axis_out = inv_map[ori]
        # The helper vector tracked during table-building is the unsigned
        # (0,1,0), not the corner's actual y-sticker direction, so recovering
        # the true rotation needs an orig_pos[1] correction factor here.
        sign_out = orig_pos[1] * pos[axis_out]
        R = _corner_rotation(orig_pos, pos, axis_out, sign_out)

        for axis in range(3):
            color = _color_for(axis, orig_pos[axis])
            vec = np.zeros(3, dtype=int)
            vec[axis] = orig_pos[axis]
            result_axis = int(np.nonzero(R @ vec)[0][0])
            face, row, col = _face_row_col(pos, result_axis)
            grid[face][row, col] = color

    for slot_idx in range(12):
        pos = EDGE_POS[slot_idx]
        piece_id = int(state["edge_perm"][slot_idx])
        ori = int(state["edge_ori"][slot_idx])
        orig_pos = EDGE_POS[piece_id]

        primary_axis_orig = 1 if orig_pos[1] != 0 else 2
        secondary_axis_orig = next(
            a for a in range(3) if orig_pos[a] != 0 and a != primary_axis_orig
        )
        primary_color = _color_for(primary_axis_orig, orig_pos[primary_axis_orig])
        secondary_color = _color_for(secondary_axis_orig, orig_pos[secondary_axis_orig])

        primary_axis_new = 1 if pos[1] != 0 else 2
        secondary_axis_new = next(
            a for a in range(3) if pos[a] != 0 and a != primary_axis_new
        )
        if ori == 0:
            primary_target, secondary_target = primary_axis_new, secondary_axis_new
        else:
            primary_target, secondary_target = secondary_axis_new, primary_axis_new

        face, row, col = _face_row_col(pos, primary_target)
        grid[face][row, col] = primary_color
        face, row, col = _face_row_col(pos, secondary_target)
        grid[face][row, col] = secondary_color

    return grid


def plot_net(state: dict[str, np.ndarray], ax: plt.Axes | None = None) -> plt.Axes:
    """Render the cube state as an unfolded 2D net of the 6 faces."""
    grid = state_to_facelets(state)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))

    for face, (col_off, row_off) in _NET_OFFSET.items():
        for r in range(3):
            for c in range(3):
                color = grid[face][r, c]
                x0 = col_off + c
                y0 = row_off + (2 - r)
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=1.5,
                    )
                )

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
