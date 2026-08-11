"""Rules engine for 3-player tic-tac-toe on a 3x3 board.

Board representation
---------------------
A board is a length-9 tuple of ints. Index i is cell (row, col) = (i // 3, i % 3).
0 means empty; 1, 2, 3 mean occupied by that player. Boards are immutable tuples
so they can be used directly as dict keys (needed for the transposition-table
demo in the tree-reuse scenario).

Turn order is fixed and cyclic: player 1, then 2, then 3, then 1, ... The player
to move is therefore a pure function of how many cells are filled -- it is never
stored separately, so there is no way for board and turn to fall out of sync.

Reward convention
------------------
A finished game returns a reward vector {1: r1, 2: r2, 3: r3} with r1+r2+r3 == 1
always: winner gets 1 and the other two get 0, a draw splits 1/3 each. This
constant-sum property is what the Max^N backpropagation in mcts_core.py relies
on, and it is also what lets the classic 2-player "flip the sign" trick be
derived as a special case (shown explicitly in the zero-sum notebook scenario)
rather than assumed.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Tuple

Board = Tuple[int, ...]

PLAYERS: Tuple[int, int, int] = (1, 2, 3)
EMPTY = 0

# Presentation constants live here (not in the viz module) because every module
# that talks about a player -- engine, mcts_core, viz -- needs the same symbol
# and color for that player, and engine.py is the one module all the others
# already depend on.
PLAYER_SYMBOLS: Dict[int, str] = {1: "X", 2: "O", 3: "\u0394"}  # X, O, Δ
PLAYER_NAMES: Dict[int, str] = {1: "Player 1 (X)", 2: "Player 2 (O)", 3: "Player 3 (\u0394)"}
PLAYER_COLORS: Dict[int, str] = {1: "#3B6FE0", 2: "#E0662E", 3: "#2E9E5B"}  # blue, orange, green

LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
)

_CELL_TYPE: Dict[int, str] = {
    4: "center",
    0: "corner", 2: "corner", 6: "corner", 8: "corner",
    1: "edge", 3: "edge", 5: "edge", 7: "edge",
}


def cell_type(cell: int) -> str:
    """Return 'center', 'corner', or 'edge' for a board index, for readable node labels."""
    return _CELL_TYPE[cell]


def initial_board() -> Board:
    return (EMPTY,) * 9


def num_moves_played(board: Board) -> int:
    return sum(1 for c in board if c != EMPTY)


def player_to_move(board: Board) -> Optional[int]:
    """Whose turn it is, derived from move count. Returns None if the game is over."""
    if is_terminal(board):
        return None
    return PLAYERS[num_moves_played(board) % 3]


def legal_moves(board: Board) -> List[int]:
    return [i for i, v in enumerate(board) if v == EMPTY]


def apply_move(board: Board, cell: int, player: int) -> Board:
    if board[cell] != EMPTY:
        raise ValueError(f"cell {cell} is already occupied by player {board[cell]}")
    b = list(board)
    b[cell] = player
    return tuple(b)


def check_winner(board: Board) -> int:
    """Return the winning player (1/2/3) or 0 if nobody has three in a row yet."""
    for a, b, c in LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def is_full(board: Board) -> bool:
    return all(c != EMPTY for c in board)


def is_terminal(board: Board) -> bool:
    return check_winner(board) != 0 or is_full(board)


def terminal_reward(board: Board) -> Dict[int, float]:
    """Constant-sum reward vector for a finished game. Sums to 1.0 always."""
    if not is_terminal(board):
        raise ValueError("terminal_reward called on a non-terminal board")
    w = check_winner(board)
    if w != 0:
        return {p: (1.0 if p == w else 0.0) for p in PLAYERS}
    return {p: 1.0 / 3.0 for p in PLAYERS}


def render_ascii(board: Board) -> str:
    rows = []
    for r in range(3):
        cells = [PLAYER_SYMBOLS.get(board[r * 3 + c], ".") for c in range(3)]
        rows.append(" | ".join(cells))
    return "\n---------\n".join(rows)


# ---------------------------------------------------------------------------
# Spatial symmetry (dihedral group of order 8), used only for the tree-reuse /
# transposition demo. This reduces the 3x3 grid, not the player labels -- a
# board is only folded onto another board if one is a literal rotation or
# reflection of the other with the same player at every corresponding cell.
# ---------------------------------------------------------------------------

def _build_symmetries() -> List[Tuple[int, ...]]:
    coords = [(r, c) for r in range(3) for c in range(3)]
    index_of = {rc: i for i, rc in enumerate(coords)}

    def rot90(rc):
        r, c = rc
        return (c, 2 - r)

    def reflect(rc):
        r, c = rc
        return (r, 2 - c)

    perms = []
    cur = coords
    for _ in range(4):
        perms.append(tuple(index_of[rc] for rc in cur))
        reflected = [reflect(rc) for rc in cur]
        perms.append(tuple(index_of[rc] for rc in reflected))
        cur = [rot90(rc) for rc in cur]
    return perms


SYMMETRIES: Tuple[Tuple[int, ...], ...] = tuple(_build_symmetries())
assert len(SYMMETRIES) == 8


def canonical_board(board: Board) -> Board:
    """Smallest board (by tuple order) among all 8 rotations/reflections of `board`."""
    variants = (tuple(board[perm[i]] for i in range(9)) for perm in SYMMETRIES)
    return min(variants)


if __name__ == "__main__":
    # Minimal self-check, run with: python3 ttt3_engine.py
    b = initial_board()
    assert player_to_move(b) == 1
    b = apply_move(b, 4, 1)
    assert player_to_move(b) == 2
    assert cell_type(4) == "center"
    b = apply_move(b, 0, 2)
    b = apply_move(b, 1, 3)
    assert not is_terminal(b)
    win_board = (1, 1, 1, 0, 2, 0, 3, 0, 2)
    assert check_winner(win_board) == 1
    assert terminal_reward(win_board) == {1: 1.0, 2: 0.0, 3: 0.0}
    draw_needs_check = sum(terminal_reward((1, 2, 3, 2, 3, 1, 3, 1, 2)).values())
    assert abs(draw_needs_check - 1.0) < 1e-9
    assert len(SYMMETRIES) == 8
    print("ttt3_engine self-check passed")
    print(render_ascii(win_board))
