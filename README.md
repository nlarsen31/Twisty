# Twisty Puzzle Solver — Design Doc

## Goal

Build a Python/PyTorch pipeline that trains a neural network to estimate distance-from-solved
for twisty puzzles (starting with 3x3x3 Rubik's Cube), using an autodidactic/self-play style
training loop, with the puzzle representation generalized so new puzzles (2x2x2, pyraminx,
megaminx, skewb, NxNxN, etc.) can be added without touching the training or search code.

## Implementation Status

What's actually built so far (data model + visualization; network/search/eval are still future
work per the build order below):

- `twisty/puzzle.py` — the `TwistyPuzzle` abstract base class and `Move` dataclass, matching
  the skeleton below.
- `twisty/cube3x3.py` — `Cube3x3`: cubie-based state (corner/edge perm + orientation arrays),
  geometrically-derived move tables (not hand-hardcoded — built from 90° rotation matrices
  and verified against known cube invariants), `apply_move`/`apply_move_batch`, `scramble`,
  `encode`/`input_dim`.
  - Generator set is 18 moves, not 12: the 12 quarter turns (`U, U', D, D', ...`) plus the 6
    double/180° moves (`U2, D2, ...`), so move counts here match the Half-Turn Metric (HTM)
    that "God's number = 20" is quoted under.
  - Move directions follow standard notation (clockwise viewed from outside each face).
- `twisty/visualize.py` — `plot_net`/`state_to_facelets`: renders a cube state as an unfolded
  2D net (6 colored faces) via matplotlib. Reconstructing which sticker shows on which face
  from the compressed (perm, orientation) state turned out to be non-trivial for corners (a
  piece's orientation value alone isn't enough — the axis mapping for relocated corners has
  to be reconstructed geometrically); see git history / commit messages for the details.
- `twisty/scripts/render_states.py` + `scripts/visualize.sh` — renders a grid of panels (solved,
  one turn of each of the 12 quarter/prime moves and 6 double moves, a couple of scrambles, and
  a named algorithm) to a PNG for quick visual sanity-checking. Run via `./scripts/visualize.sh
  [output.png] [seed]`.
- `twisty/tests/` — pytest suite (order-4 for quarter turns, order-2 for doubles, move/prime and
  double/quarter-turn-pair consistency, orientation invariants under random scrambling, `U` only
  touching the top layer, `apply_move_batch` matching `apply_move`, facelet-completeness and
  uniform-row checks on the rendered net). Run via `source .venv/bin/activate && python -m
  pytest twisty/tests`.
- Virtualenv: `.venv` (managed with `uv`), with `numpy`, `matplotlib`, `pytest` installed.

## Core Abstraction

Every twisty puzzle reduces to:

- A fixed set of **pieces** (cubies, edges, tips, etc.)
- A set of **generator moves**, each a permutation (possibly with orientation changes) over pieces
- A **solved state** (identity permutation/orientation)
- Current state = some sequence of generators applied to the solved state

```python
class TwistyPuzzle:
    """Abstract base class all specific puzzles implement."""

    def __init__(self, num_pieces, orientations_per_piece, move_generators):
        self.num_pieces = num_pieces
        self.orientations_per_piece = orientations_per_piece
        self.move_generators = move_generators  # list of Move objects

    def solved_state(self):
        """Return identity state (permutation + orientation arrays)."""
        raise NotImplementedError

    def apply_move(self, state, move_idx):
        """Return new state after applying move_idx. Must support batched states."""
        raise NotImplementedError

    def apply_move_batch(self, states, move_idx):
        """Vectorized version for batch data generation. Critical for speed."""
        raise NotImplementedError

    def scramble(self, n_moves, rng=None):
        """Apply n random moves from solved, return final state + move sequence."""
        raise NotImplementedError

    def encode(self, state):
        """Convert state to network input tensor (one-hot per piece position+orientation)."""
        raise NotImplementedError

    def num_actions(self):
        return len(self.move_generators)

    def input_dim(self):
        """Size of encoded state vector — puzzle-specific."""
        raise NotImplementedError
```

### Variance points between puzzles (design around these)

| Aspect | 3x3x3 | Pyraminx | NxNxN (general) |
|---|---|---|---|
| Piece count | 20 (8 corner + 12 edge) | 4 corners + 6 edges + tips | grows with N |
| Orientations/piece | corners:3, edges:2 | corners:3, edges:2, tips: trivial | varies |
| # generators (actions) | 12 (face turns) | 4-8 depending on tip inclusion | many more (layer turns) |
| Symmetry group size | 48 (24 rotations × mirror) | 12 (rotations only, no reliable mirror) | puzzle-specific |
| God's number (known optimal max distance) | 20 (HTM) | known, smaller | mostly unknown for large N |

Because God's number / max distance is unknown for most non-cube puzzles, the 3x3x3 case is
your main calibration tool — small-depth scrambles (1-8 moves) can be brute-force verified as
ground truth, which won't be available at scale for other puzzles.

## 3x3x3 Implementation Notes

- Represent state as **cubies, not stickers**: 8 corner pieces + 12 edge pieces, each with
  a permutation index and an orientation value. This is smaller than the 54-sticker encoding
  and structurally forbids illegal states (parity-violating configurations).
- Corner orientation: 0, 1, or 2 (mod 3 twist). Edge orientation: 0 or 1 (flipped or not).
- Encode via one-hot: for each of the 20 pieces, one-hot over (position × orientation)
  combinations, concatenated into a flat vector. This is `input_dim()` for `Cube3x3`.
- 18 face turns (12 quarter turns U, U', D, D', L, L', R, R', F, F', B, B' plus 6 double moves
  U2, D2, L2, R2, F2, B2) as the generator set. Each is a fixed permutation + orientation delta
  applied to the corner/edge arrays — implement as precomputed index arrays so moves are pure
  NumPy indexing (fast, vectorizable across a batch of states at once). Double moves are just
  the quarter-turn table composed with itself, not separately derived.

## Data Generation (Autodidactic Iteration)

You can't get true optimal distances at scale — the state space is too large
(~4.3×10^19 for 3x3x3) for brute-force labeling. Instead:

1. **Bootstrap labels**: generate scrambles of random depth `k` (1 to ~20-30 moves) from
   solved. `k` is a *noisy upper bound* on true distance (move sequences can cancel, so true
   distance ≤ k, often <).
2. **Refine via self-play**: for each training state `s`, compute
   `label(s) = 1 + min over all neighbor moves m of current_network(apply_move(s, m))`,
   with the solved state always pinned to label 0. Retrain the network on these refined
   labels. Repeat — this bootstrapping is what makes it "autodidactic" (no human solve data
   needed).
3. **Sanity check**: periodically evaluate against brute-force-verified optimal distances
   for shallow scrambles (depth 1-8, computable exactly via BFS) to catch calibration drift.

```python
def generate_batch(puzzle, N, max_scramble=20, rng=None):
    states, depths = [], []
    for _ in range(N):
        k = rng.integers(1, max_scramble + 1)
        state, _ = puzzle.scramble(k, rng=rng)
        states.append(state)
        depths.append(k)  # noisy label, refine later via self-play pass
    return states, depths
```

## Network

- Simple MLP is sufficient: input (one-hot state) → dense(1000) → dense(1000) → dense(1),
  optionally with a couple residual blocks. Capacity is not the bottleneck — the training
  loop and label quality are.
- Loss: MSE or Huber on the refined cost-to-go labels.
- Output is a **heuristic**, not a ground-truth distance — never trust it directly for
  solving; always wrap it in a search.

## Search / Solving

The trained network alone will not give exact optimal solutions. Use it as a heuristic inside:
- **Weighted A\*** (DeepCubeA-style): `f(n) = g(n) + w * h(n)`, where `h` is the network output.
  Weight `w` trades optimality for speed.
- or **Batch MCTS**: use the network as both value and rough policy prior, batch-evaluate many
  leaf nodes per network call.

Both consume the generic `TwistyPuzzle.apply_move` interface, so they work unmodified once you
add new puzzle subclasses.

## Suggested Build Order

1. ✅ `TwistyPuzzle` abstract base class + `Move` representation.
2. ✅ `Cube3x3` subclass: state arrays, move table generation, `encode()`, `scramble()`. Also
   added a matplotlib net visualizer and a pytest suite (not originally scoped as a numbered
   step, but done alongside the data model per user request).
3. Brute-force BFS solver for depth ≤ 8 (ground truth for calibration + unit tests). *(not
   started)*
4. Data generation pipeline (batched, vectorized). *(not started — `apply_move_batch` exists,
   but no batched scramble-generation pipeline yet)*
5. MLP model + training loop (first pass: raw scramble-depth labels).
6. Autodidactic refinement loop (self-play relabeling).
7. Weighted A* search wrapper using the trained network as heuristic.
8. Evaluate: solve rate, average solution length vs. optimal (on the depth ≤ 8 test set),
   average search time.
9. Only after 3x3x3 works end-to-end: add a second `TwistyPuzzle` subclass (e.g. 2x2x2 or
   pyraminx) to validate the abstraction actually holds up, before investing in more puzzles.

## Open Questions to Resolve Early

- Exact one-hot encoding scheme for corners/edges (position×orientation indexing) — get this
  right first, since `encode()`/`input_dim()` ripple through everything.
- How much symmetry augmentation to apply during training (48-fold for cube) — helps sample
  efficiency but adds complexity to the data pipeline.
- Whether to precompute move tables as static index arrays (recommended) vs. computing
  permutations on the fly.