"""Pure rules for two 2-player games on a 3x3 board, plus action_group/action_label helpers."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

Board = Tuple[int, ...]
ClassicAction = int
NumericalAction = Tuple[int, int]
Action = Union[ClassicAction, NumericalAction]

PLAYERS: Tuple[int, int] = (1, 2)

CENTER: int = 4
CORNERS: Tuple[int, ...] = (0, 2, 6, 8)
EDGES: Tuple[int, ...] = (1, 3, 5, 7)

_CELL_TYPE: Dict[int, str] = {CENTER: "center"}
_CELL_TYPE.update({c: "corner" for c in CORNERS})
_CELL_TYPE.update({c: "edge" for c in EDGES})

LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)

PLAYER_COLORS: Dict[int, str] = {1: "#3B6FE0", 2: "#E0662E"}


def cell_type(cell: int) -> str:
    """Return 'center', 'corner', or 'edge' for a board index."""
    return _CELL_TYPE[cell]


def action_cell(action: Action) -> int:
    """Extract the board cell from either a classic or numerical action."""
    return action if isinstance(action, int) else action[0]


def action_group(action: Action) -> str:
    """Return the cell-type band ('center'/'corner'/'edge') an action falls in."""
    return cell_type(action_cell(action))


def action_label(game: "Game", action: Optional[Action], mover: Optional[int]) -> str:
    """Short node label, e.g. classic 'X4', numerical '5@4', or 'root'."""
    if action is None:
        return "root"
    if game.name == "classic":
        return f"{game.SYMBOLS[mover]}{action}"
    cell, number = action
    return f"{number}@{cell}"


def _num_moves_played(board: Board) -> int:
    return sum(1 for c in board if c != 0)


def zero_sum_reward(winner: int) -> Dict[int, float]:
    """Constant-sum reward for a finished 2-player game. Sums to 1.0 always."""
    if winner == 0:
        return {1: 0.5, 2: 0.5}
    return {p: (1.0 if p == winner else 0.0) for p in PLAYERS}


class Game:
    """Shared scaffolding both games specialize: move counting, mover parity."""

    name: str

    def initial_board(self) -> Board:
        return (0,) * 9

    def next_mover(self, board: Board) -> int:
        """Whose turn it is, ignoring whether the game has already ended."""
        return PLAYERS[_num_moves_played(board) % 2]

    def player_to_move(self, board: Board) -> Optional[int]:
        """Whose turn it is, or None if the game is over."""
        if self.is_terminal(board):
            return None
        return self.next_mover(board)

    def is_full(self, board: Board) -> bool:
        return all(c != 0 for c in board)

    def is_terminal(self, board: Board) -> bool:
        return self.check_winner(board) != 0 or self.is_full(board)

    def reward(self, board: Board) -> Dict[int, float]:
        return zero_sum_reward(self.check_winner(board))

    def check_winner(self, board: Board) -> int:
        raise NotImplementedError

    def legal_moves(self, board: Board) -> List[Action]:
        raise NotImplementedError

    def apply_move(self, board: Board, action: Action, player: int) -> Board:
        raise NotImplementedError

    def cell_display(self, board: Board, cell: int) -> str:
        raise NotImplementedError

    def mover_label(self, player: int) -> str:
        raise NotImplementedError


class ClassicGame(Game):
    """Classic tic-tac-toe: first to three in a row, zero-sum."""

    name = "classic"
    SYMBOLS: Dict[int, str] = {1: "X", 2: "O"}
    NAMES: Dict[int, str] = {1: "Player 1 (X)", 2: "Player 2 (O)"}

    def legal_moves(self, board: Board) -> List[ClassicAction]:
        return [i for i, v in enumerate(board) if v == 0]

    def apply_move(self, board: Board, action: ClassicAction, player: int) -> Board:
        if board[action] != 0:
            raise ValueError(f"cell {action} already occupied")
        b = list(board)
        b[action] = player
        return tuple(b)

    def check_winner(self, board: Board) -> int:
        for a, b, c in LINES:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a]
        return 0

    def cell_display(self, board: Board, cell: int) -> str:
        v = board[cell]
        return self.SYMBOLS[v] if v != 0 else ""

    def mover_label(self, player: int) -> str:
        return self.NAMES[player]


class NumericalGame(Game):
    """Numerical (Graham) tic-tac-toe: first line to sum to exactly 15 wins."""

    name = "numerical"
    PLAYER_NUMBERS: Dict[int, Tuple[int, ...]] = {1: (1, 3, 5, 7, 9), 2: (2, 4, 6, 8)}
    NAMES: Dict[int, str] = {1: "Player 1 (odds)", 2: "Player 2 (evens)"}

    def remaining_numbers(self, board: Board, player: int) -> Tuple[int, ...]:
        used = set(board)
        return tuple(n for n in self.PLAYER_NUMBERS[player] if n not in used)

    def legal_moves(self, board: Board) -> List[NumericalAction]:
        player = self.player_to_move(board)
        if player is None:
            return []
        cells = [i for i, v in enumerate(board) if v == 0]
        numbers = self.remaining_numbers(board, player)
        return [(cell, number) for cell in cells for number in numbers]

    def apply_move(self, board: Board, action: NumericalAction, player: int) -> Board:
        cell, number = action
        if board[cell] != 0:
            raise ValueError(f"cell {cell} already occupied")
        b = list(board)
        b[cell] = number
        return tuple(b)

    def check_winner(self, board: Board) -> int:
        completed = any(
            board[a] != 0 and board[b] != 0 and board[c] != 0 and board[a] + board[b] + board[c] == 15
            for a, b, c in LINES
        )
        if not completed:
            return 0
        moves_played = _num_moves_played(board)
        if moves_played == 0:
            return 0
        return PLAYERS[(moves_played - 1) % 2]

    def cell_display(self, board: Board, cell: int) -> str:
        v = board[cell]
        return str(v) if v != 0 else ""

    def mover_label(self, player: int) -> str:
        return self.NAMES[player]


CLASSIC = ClassicGame()
NUMERICAL = NumericalGame()


def render_ascii(game: Game, board: Board) -> str:
    rows = []
    for r in range(3):
        cells = [game.cell_display(board, r * 3 + c) or "." for c in range(3)]
        rows.append(" | ".join(f"{c:>2}" if game.name == "numerical" else c for c in cells))
    return "\n---------------\n".join(rows)


def self_check() -> None:
    """Assertions covering both games' rules; prints a short confirmation."""
    b = CLASSIC.initial_board()
    assert CLASSIC.player_to_move(b) == 1
    b = CLASSIC.apply_move(b, 4, 1)
    assert CLASSIC.player_to_move(b) == 2
    assert cell_type(4) == "center"
    assert action_group(4) == "center"
    win_board = (1, 1, 1, 0, 2, 0, 2, 0, 0)
    assert CLASSIC.check_winner(win_board) == 1
    assert CLASSIC.reward(win_board) == {1: 1.0, 2: 0.0}
    draw_board = (1, 2, 1, 1, 2, 2, 2, 1, 1)
    assert CLASSIC.is_terminal(draw_board) and CLASSIC.check_winner(draw_board) == 0
    assert abs(sum(CLASSIC.reward(draw_board).values()) - 1.0) < 1e-9

    n = NUMERICAL.initial_board()
    assert NUMERICAL.player_to_move(n) == 1
    assert NUMERICAL.remaining_numbers(n, 1) == (1, 3, 5, 7, 9)
    n = NUMERICAL.apply_move(n, (0, 9), 1)
    assert NUMERICAL.player_to_move(n) == 2
    assert NUMERICAL.remaining_numbers(n, 1) == (1, 3, 5, 7)
    assert NUMERICAL.check_winner(n) == 0
    n = NUMERICAL.apply_move(n, (1, 4), 2)
    n = NUMERICAL.apply_move(n, (5, 5), 1)
    assert NUMERICAL.check_winner(n) == 0
    n = NUMERICAL.apply_move(n, (2, 2), 2)
    assert NUMERICAL.check_winner(n) == 2
    assert NUMERICAL.reward(n) == {1: 0.0, 2: 1.0}
    assert action_label(NUMERICAL, (2, 1), 1) == "1@2"
    assert action_label(CLASSIC, 4, 1) == "X4"

    forced_start = NUMERICAL.initial_board()
    forced_start = NUMERICAL.apply_move(forced_start, (4, 5), 1)
    assert NUMERICAL.check_winner(forced_start) == 0
    assert len(NUMERICAL.legal_moves(forced_start)) == 8 * 4
    print("engine self-check passed")
    print(render_ascii(CLASSIC, win_board))
    print(render_ascii(NUMERICAL, n))


if __name__ == "__main__":
    self_check()
