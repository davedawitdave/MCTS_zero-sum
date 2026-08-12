"""Board + tree rendering, the dashboard, and human-vs-MCTS play (colors/bands: see README)."""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

import engine as eng
import mcts as mc

COLOR_NORMAL = "#8891A6"
COLOR_PATH = "#2E9E5B"
COLOR_NEW = "#E0A62E"
COLOR_ROOT = "#3B6FE0"
COLOR_UNTRIED = "#D8DCE6"
COLOR_TEXT = "#1A1D24"
COLOR_BEST = "#D4AF00"

BAND_ORDER = ("center", "corner", "edge")
BAND_TITLES = {"center": "CENTER", "corner": "CORNERS", "edge": "EDGES"}


def _compact(x: float) -> str:
    s = f"{x:.2f}"
    return s[1:] if s.startswith("0.") else (s[0] + s[2:] if s.startswith("-0.") else s)


def _cell_color(game: eng.Game, value: int) -> str:
    if game.name == "classic":
        return eng.PLAYER_COLORS[value]
    return eng.PLAYER_COLORS[1 if value % 2 == 1 else 2]


def draw_board(ax, game: eng.Game, board: eng.Board, highlight_cell: Optional[int] = None,
               title: Optional[str] = None) -> None:
    ax.clear()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.axis("off")

    for i in (1, 2):
        ax.add_line(Line2D([i, i], [0, 3], color=COLOR_TEXT, linewidth=2))
        ax.add_line(Line2D([0, 3], [i, i], color=COLOR_TEXT, linewidth=2))

    for cell in range(9):
        row, col = divmod(cell, 3)
        cx, cy = col + 0.5, 2 - row + 0.5
        if highlight_cell == cell:
            ax.add_patch(plt.Rectangle((col, 2 - row), 1, 1, facecolor="#FFF3C4", zorder=0))
        text = game.cell_display(board, cell)
        if text:
            ax.text(cx, cy, text, fontsize=26 if game.name == "numerical" else 30,
                     fontweight="bold", ha="center", va="center",
                     color=_cell_color(game, board[cell]), zorder=2)

    mover = game.player_to_move(board)
    if title:
        ax.set_title(title, fontsize=11, color=COLOR_TEXT, loc="left")
    if mover is not None:
        ax.text(1.5, -0.28, f"to move: {game.mover_label(mover)}", fontsize=11,
                 ha="center", color=eng.PLAYER_COLORS[mover], fontweight="bold")
    else:
        w = game.check_winner(board)
        msg = f"{game.mover_label(w)} wins" if w else "draw"
        ax.text(1.5, -0.28, msg, fontsize=11, ha="center", color=COLOR_TEXT, fontweight="bold")


def compute_tree_layout(nodes: Dict[int, mc.MCTSNode], root_id: int, x_spacing: float = 1.9
                         ) -> Tuple[Dict[int, Tuple[float, int]], Dict[str, Tuple[float, int]]]:
    depth: Dict[int, int] = {}

    def assign_depth(nid: int, d: int) -> None:
        depth[nid] = d
        for cid in nodes[nid].children.values():
            assign_depth(cid, d + 1)

    assign_depth(root_id, 0)

    x: Dict[int, float] = {}
    counter = [0]

    def assign_x(nid: int) -> float:
        node = nodes[nid]
        if not node.children:
            x[nid] = float(counter[0]) * x_spacing
            counter[0] += 1
            return x[nid]
        child_xs = [assign_x(cid) for cid in node.children.values()]
        x[nid] = sum(child_xs) / len(child_xs)
        return x[nid]

    assign_x(root_id)

    band_labels: Dict[str, Tuple[float, int]] = {}
    root = nodes[root_id]
    if root.children:
        groups: Dict[str, List[int]] = {"center": [], "corner": [], "edge": []}
        for a, cid in root.children.items():
            groups[eng.action_group(a)].append(cid)

        def apply_shift(nid: int, shift: float) -> None:
            x[nid] += shift
            for cid in nodes[nid].children.values():
                apply_shift(cid, shift)

        cursor = 0.0
        gap = x_spacing * 0.9
        for band in BAND_ORDER:
            ids = groups[band]
            if not ids:
                continue
            ids.sort(key=lambda cid: x[cid])
            local_min = min(x[cid] for cid in ids)
            shift = cursor - local_min
            for cid in ids:
                apply_shift(cid, shift)
            span_min = min(x[cid] for cid in ids)
            span_max = max(x[cid] for cid in ids)
            band_labels[band] = ((span_min + span_max) / 2.0, len(ids))
            cursor = span_max + x_spacing + gap
        x[root_id] = sum(x[cid] for cid in root.children.values()) / len(root.children)

    layout = {nid: (x[nid], depth[nid]) for nid in nodes}
    return layout, band_labels


def tree_figsize(nodes: Dict[int, mc.MCTSNode], root_id: int) -> Tuple[float, float]:
    """Figure size proportional to the tree's own data extent, so aspect='equal' leaves no dead space."""
    layout, _ = compute_tree_layout(nodes, root_id)
    xs = [p[0] for p in layout.values()]
    ys = [p[1] for p in layout.values()]
    x_span = (max(xs) - min(xs)) + 1.8
    y_span = (max(ys) - min(ys)) + 1.5
    scale = 0.62
    raw_width, raw_height = x_span * scale, y_span * scale
    width_cap_scale = 15.0 / raw_width if raw_width > 15.0 else 1.0
    width, height = raw_width * width_cap_scale, raw_height * width_cap_scale
    if height < 3.0:
        height = 3.0
        width = min(15.0, width * (3.0 / max(raw_height * width_cap_scale, 0.1)))
    width = max(width, 5.0)
    return width, height


def draw_tree(ax, game: eng.Game, nodes: Dict[int, mc.MCTSNode], root_id: int,
              path: Sequence[int] = (), expanded_id: Optional[int] = None,
              show_prior: bool = True, title: Optional[str] = None,
              show_untried_stub: bool = True, show_best_ring: bool = True) -> None:
    ax.clear()
    ax.axis("off")
    ax.set_aspect("equal")
    layout, band_labels = compute_tree_layout(nodes, root_id)
    path_set = set(path)

    root = nodes[root_id]
    best_child_id = None
    if show_best_ring and root.children:
        best_child_id = max(root.children.values(), key=lambda cid: nodes[cid].N)
        if nodes[best_child_id].N == 0:
            best_child_id = None

    if show_untried_stub:
        for nid, node in nodes.items():
            if game.is_terminal(node.board):
                continue
            untried = node.untried_actions()
            if not untried:
                continue
            px, py = layout[nid]
            n_stub = min(len(untried), 4)
            for k in range(n_stub):
                offset = (k - (n_stub - 1) / 2) * 0.30
                sx, sy = px + offset, py + 0.65
                ax.add_line(Line2D([px, sx], [py, sy], color=COLOR_UNTRIED, linewidth=1, zorder=1))
                ax.add_patch(Circle((sx, sy), 0.09, facecolor="white",
                                     edgecolor=COLOR_UNTRIED, linewidth=1.3, zorder=2))
            if len(untried) > n_stub:
                ax.text(px, py + 1.0, f"+{len(untried) - n_stub}", fontsize=6.5,
                         ha="center", color=COLOR_UNTRIED)

    for nid, node in nodes.items():
        if node.parent is None:
            continue
        px, py = layout[node.parent]
        cx, cy = layout[nid]
        on_path = nid in path_set and node.parent in path_set
        ax.add_line(Line2D([px, cx], [py, cy],
                            color=COLOR_PATH if on_path else "#B9BFCC",
                            linewidth=2.4 if on_path else 1.2, zorder=1))

    for nid, node in nodes.items():
        cx, cy = layout[nid]
        if nid == expanded_id:
            color = COLOR_NEW
        elif nid == root_id:
            color = COLOR_ROOT
        elif nid in path_set:
            color = COLOR_PATH
        else:
            color = COLOR_NORMAL
        radius = 0.22 if nid == root_id else 0.18
        if nid == best_child_id:
            ax.add_patch(Circle((cx, cy), radius + 0.07, facecolor="none",
                                 edgecolor=COLOR_BEST, linewidth=2.2, zorder=2.5))
        ax.add_patch(Circle((cx, cy), radius, facecolor=color, edgecolor="white",
                             linewidth=1.5, zorder=3))

        q = node.Q()
        stat = f"N{node.N} Q{_compact(q)}"
        if show_prior and node.parent is not None:
            stat += f" P{_compact(node.prior)}"
        ax.text(cx, cy - radius - 0.09, node.label(game), fontsize=7.6, ha="center", va="top",
                color=COLOR_TEXT, fontweight="bold", zorder=4)
        ax.text(cx, cy - radius - 0.27, stat, fontsize=6.3, ha="center", va="top",
                color="#4A5268", zorder=4)

    for band, (bx, count) in band_labels.items():
        by = 1 - 0.62
        ax.text(bx, by, f"{BAND_TITLES[band]} ({count})", fontsize=7.2, ha="center",
                color="#5B6478", fontweight="bold", style="italic")

    xs = [p[0] for p in layout.values()]
    ys = [p[1] for p in layout.values()]
    ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
    ax.set_ylim(max(ys) + 0.75, -0.75)
    if title:
        ax.set_title(title, fontsize=11, color=COLOR_TEXT, loc="left")


class TreeBoardDashboard:
    """Tree (left) + board (right), Prev / Next / Reset above."""

    def __init__(self, game: eng.Game, frames: List[mc.Frame], root_board: eng.Board, title: str = ""):
        if not frames:
            raise ValueError("no frames to display -- did MCTS.run(record=True) get called?")
        self.game = game
        self.frames = frames
        self.root_board = root_board
        self.title = title

    def display(self):
        import ipywidgets as W
        from IPython.display import display, clear_output

        tree_out = W.Output()
        board_out = W.Output()
        info_out = W.Output()
        prev_btn = W.Button(description="\u25c0 Prev")
        next_btn = W.Button(description="Next \u25b6")
        reset_btn = W.Button(description="\u21ba Reset")
        iter_label = W.Label()
        header = W.HBox([prev_btn, next_btn, reset_btn, iter_label])

        state = {"idx": 0}

        def redraw():
            idx = state["idx"]
            frame = self.frames[idx]
            nodes, root_id = frame.nodes_snapshot, frame.root_id
            root = nodes[root_id]

            with tree_out:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(6.6, 5.6))
                draw_tree(ax, self.game, nodes, root_id, path=frame.path,
                          expanded_id=frame.expanded_id,
                          title=f"{self.title} \u2014 iteration {frame.iteration}, {frame.phase}")
                plt.show()
                plt.close(fig)

            with board_out:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(3.6, 4.2))
                highlight = None
                if len(frame.path) > 1:
                    first_child = nodes[frame.path[1]]
                    highlight = eng.action_cell(first_child.action)
                draw_board(ax, self.game, root.board, highlight_cell=highlight,
                           title="position under analysis")
                plt.show()
                plt.close(fig)

            with info_out:
                clear_output(wait=True)
                path_labels = " \u2192 ".join(nodes[n].label(self.game) for n in frame.path)
                msg = f"**{frame.phase.upper()}**  |  path: {path_labels}"
                if frame.reward is not None:
                    r = {p: round(v, 2) for p, v in frame.reward.items()}
                    msg += f"  |  value backed up: {r}"
                if frame.note:
                    msg += f"  |  {frame.note}"
                print(msg)

            iter_label.value = f"frame {idx + 1} / {len(self.frames)}"

        def on_prev(_):
            state["idx"] = max(0, state["idx"] - 1)
            redraw()

        def on_next(_):
            state["idx"] = min(len(self.frames) - 1, state["idx"] + 1)
            redraw()

        def on_reset(_):
            state["idx"] = 0
            redraw()

        prev_btn.on_click(on_prev)
        next_btn.on_click(on_next)
        reset_btn.on_click(on_reset)

        display(header)
        display(W.HBox([tree_out, board_out]))
        display(info_out)
        redraw()


def dashboard_frames(mcts_obj: "mc.MCTS", n_stepped_iters: int = 6) -> List[mc.Frame]:
    """First `n_stepped_iters` iterations' frames, plus one synthetic final-tree frame."""
    steps = [f for f in mcts_obj.frames if f.iteration <= n_stepped_iters]
    final = mc.Frame(
        iteration=mcts_obj.frames[-1].iteration if mcts_obj.frames else 0,
        phase="final tree", nodes_snapshot=copy.deepcopy(mcts_obj.nodes),
        root_id=mcts_obj.root_id, path=(), expanded_id=None, reward=None,
        note="search complete")
    return steps + [final]


def render_final_tree(game: eng.Game, mcts_obj: "mc.MCTS", ax, title: str = "") -> None:
    draw_tree(ax, game, mcts_obj.nodes, mcts_obj.root_id, title=title)


def render_side_by_side(game: eng.Game, panels: List[Tuple[str, "mc.MCTS"]], figsize=None):
    """Panel widths proportional to each tree's own size, so panels don't distort each other."""
    sizes = [tree_figsize(m.nodes, m.root_id) for _, m in panels]
    widths = [w for w, _ in sizes]
    height = max(h for _, h in sizes)
    total_width = sum(widths)
    if figsize is None:
        scale = min(1.0, 15.0 / total_width) if total_width > 0 else 1.0
        figsize = (total_width * scale, max(height * scale, 3.0))
    fig, axes = plt.subplots(1, len(panels), figsize=figsize,
                              gridspec_kw={"width_ratios": widths})
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, m) in zip(axes, panels):
        render_final_tree(game, m, ax, title=title)
    plt.tight_layout()
    return fig


def plot_parameter_sweep(values: List[float], metric: List[float], xlabel: str, ylabel: str,
                          title: str = "", mark_transition: bool = True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(values, metric, marker="o", color=COLOR_ROOT, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(alpha=0.25)
    if mark_transition and len(metric) > 2:
        diffs = [abs(metric[i + 1] - metric[i]) for i in range(len(metric) - 1)]
        idx = max(range(len(diffs)), key=lambda i: diffs[i])
        xt = (values[idx] + values[idx + 1]) / 2
        ax.axvline(xt, color=COLOR_NEW, linestyle="--", linewidth=1.5)
        ax.annotate("largest change here", xy=(xt, metric[idx]), xytext=(12, -14),
                    textcoords="offset points", color=COLOR_NEW, fontsize=9)
    return ax


def _actions_for_player(game: eng.Game, board: eng.Board, player: int) -> List:
    """Legal actions for `player`, regardless of whose turn `board` says it is."""
    if game.name == "classic":
        return [i for i, v in enumerate(board) if v == 0]
    cells = [i for i, v in enumerate(board) if v == 0]
    numbers = game.remaining_numbers(board, player)
    return [(cell, number) for cell in cells for number in numbers]


def _immediate_threats(game: eng.Game, board: eng.Board, player: int) -> List:
    """Actions that would win for `player` right now, one ply deep."""
    return [a for a in _actions_for_player(game, board, player)
            if game.check_winner(game.apply_move(board, a, player)) == player]


def build_human_vs_mcts(game: eng.Game, mcts_factory, human_player: int = 1,
                         computer_iterations: int = 10):
    """mcts_factory() -> a fresh mc.MCTS for the computer's move search."""
    import ipywidgets as W
    from IPython.display import display, clear_output

    state = {"board": game.initial_board(), "pending_cell": None}
    board_out = W.Output()
    status_out = W.Output()
    threat_out = W.Output()
    grid = W.GridspecLayout(3, 3, width="220px", height="220px")
    picker_out = W.Output()

    def cell_text(cell: int) -> str:
        v = state["board"][cell]
        return game.cell_display(state["board"], cell) or str(cell)

    def redraw_board(highlight=None):
        with board_out:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(3.4, 3.9))
            draw_board(ax, game, state["board"], highlight_cell=highlight)
            plt.show()
            plt.close(fig)
        for cell in range(9):
            btn = grid[cell // 3, cell % 3]
            btn.description = cell_text(cell)
            btn.disabled = state["board"][cell] != 0

    def update_threat_status(mover: int) -> None:
        with threat_out:
            clear_output(wait=True)
            if game.is_terminal(state["board"]):
                return
            threats = _immediate_threats(game, state["board"], mover)
            if not threats:
                print("No immediate threat; computer follows positional preference.")
                return
            cells = sorted(set(eng.action_cell(t) for t in threats))
            if len(cells) == 1:
                print(f"You created an immediate threat at cell {cells[0]} \u2014 computer must block.")
            else:
                cell_list = ", ".join(str(c) for c in cells)
                print(f"You created a fork \u2014 threats at cells {cell_list} \u2014 "
                      f"computer can only block one.")

    def finish_move(action):
        player = game.player_to_move(state["board"])
        state["board"] = game.apply_move(state["board"], action, player)
        state["pending_cell"] = None
        with picker_out:
            clear_output(wait=True)
        redraw_board(highlight=eng.action_cell(action))
        update_threat_status(player)
        maybe_computer_move()

    def maybe_computer_move():
        with status_out:
            clear_output(wait=True)
            if game.is_terminal(state["board"]):
                w = game.check_winner(state["board"])
                print(f"{game.mover_label(w)} wins!" if w else "Draw.")
                return
            mover = game.player_to_move(state["board"])
            print(f"{game.mover_label(mover)} to move")
        if game.player_to_move(state["board"]) not in (None, human_player):
            m = mcts_factory()
            m.new_tree(state["board"])
            m.run(computer_iterations, record=False)
            action = m.best_action("visits")
            state["board"] = game.apply_move(state["board"], action, m.root().player)
            redraw_board(highlight=eng.action_cell(action))
            with status_out:
                clear_output(wait=True)
                if game.is_terminal(state["board"]):
                    w = game.check_winner(state["board"])
                    print(f"{game.mover_label(w)} wins!" if w else "Draw.")
                else:
                    print(f"{game.mover_label(game.player_to_move(state['board']))} to move")

    def on_cell_click(cell):
        def handler(_):
            if game.is_terminal(state["board"]) or state["board"][cell] != 0:
                return
            if game.player_to_move(state["board"]) != human_player:
                return
            if game.name == "classic":
                finish_move(cell)
                return
            state["pending_cell"] = cell
            numbers = game.remaining_numbers(state["board"], human_player)
            with picker_out:
                clear_output(wait=True)
                display(W.Label(f"place which number in cell {cell}?"))
                num_buttons = [W.Button(description=str(n), layout=W.Layout(width="42px"))
                               for n in numbers]

                def make_num_handler(n):
                    def h(_):
                        finish_move((state["pending_cell"], n))
                    return h

                for b, n in zip(num_buttons, numbers):
                    b.on_click(make_num_handler(n))
                display(W.HBox(num_buttons))
        return handler

    for cell in range(9):
        btn = W.Button(description="", layout=W.Layout(width="70px", height="70px"))
        btn.on_click(on_cell_click(cell))
        grid[cell // 3, cell % 3] = btn

    reset_btn = W.Button(description="\u21ba New game")

    def on_reset(_):
        state["board"] = game.initial_board()
        state["pending_cell"] = None
        with picker_out:
            clear_output(wait=True)
        with threat_out:
            clear_output(wait=True)
        redraw_board()
        maybe_computer_move()

    reset_btn.on_click(on_reset)

    display(W.VBox([W.HBox([board_out, grid]), picker_out, threat_out, status_out, reset_btn]))
    redraw_board()
    maybe_computer_move()
