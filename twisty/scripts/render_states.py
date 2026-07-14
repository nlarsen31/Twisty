"""Render a handful of cube states (solved, one move, scrambles) to a PNG grid."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twisty.cube3x3 import Cube3x3
from twisty.visualize import plot_net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="cube_states.png", help="output PNG path")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for scrambles")
    args = parser.parse_args()

    cube = Cube3x3()
    solved = cube.solved_state()
    rng = np.random.default_rng(args.seed)

    corner_alg = [cube.MOVE_NAMES.index(m) for m in [
        "U", "R", "U'", "L'", "U", "R'", "U'", "L",
        ]]
    edge_alg = [cube.MOVE_NAMES.index(m) for m in [
        "F", "F", "U'", "R'", "L", "F", "F", "R", "L'", "U'", "F", "F",
        ]]
    after_corner_alg = solved
    for m in corner_alg:
        after_corner_alg = cube.apply_move(after_corner_alg, m)

    after_edge_alg = solved
    for m in edge_alg:
        after_edge_alg = cube.apply_move(after_edge_alg, m)

    face_panels = [
        (f"after {face}", cube.apply_move(solved, cube.MOVE_NAMES.index(face)))
        for face in ["U", "D", "L", "R", "F", "B",
            "U'", "D'", "L'", "R'", "F'", "B'",
            "U2", "D2", "L2", "R2", "F2", "B2"]
    ]

    panels = [
        ("solved", solved),
        *face_panels,
        ("scrambled(5)", cube.scramble(5, rng=rng)[0]),
        ("scrambled(15)", cube.scramble(15, rng=rng)[0]),
        ("scrambled(25)", cube.scramble(25, rng=rng)[0]),
        ("After Corner Alg", after_corner_alg),
        ("After Edge Alg", after_edge_alg),
    ]

    ncols = 3
    nrows = -(-len(panels) // ncols)  # ceil div
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes = axes.flatten()
    for ax, (title, state) in zip(axes, panels):
        plot_net(state, ax=ax)
        ax.set_title(title)
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.output, dpi=110)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
