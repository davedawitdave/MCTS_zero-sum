"""Rendering and the interactive dashboard.

Every scenario notebook cell builds an MCTS, calls .run(), and hands the
resulting frames to one shared TreeBoardDashboard class. No scenario draws
its own board or tree from scratch -- if the rendering needs to change, it
changes here once, for every scenario at once.

Color convention used throughout:
  - gray-blue   : an ordinary, already-visited node
  - green       : a node on the CURRENT iteration's path -- "look here now"
  - amber       : the node just created this iteration (expansion)
  - light gray  : an untried action, drawn as a faint stub so the branching
                   factor at a node is visible even before it is explored
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.lines import Line2D

import scripts.ttt3_engine as eng
import scripts.mcts_core as mc

COLOR_NORMAL = "#8891A6"
COLOR_PATH = "#2E9E5B"
COLOR_NEW = "#E0A62E"
COLOR_ROOT = "#3B6FE0"
COLOR_UNTRIED = "#D8DCE6"
COLOR_TEXT = "#1A1D24"


def _compact(x: float) -> str:
    s = f"{x:.2f}"
    return s[1:] if s.startswith("0.") else (s[0] + s[2:] if s.startswith("-0.") else s)


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

def draw_board(ax, board: eng.Board, highlight_cell: Optional[int] = None,
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
        player = board[cell]
        if player != eng.EMPTY:
            ax.text(cx, cy, eng.PLAYER_SYMBOLS[player], fontsize=30, fontweight="bold",
                     ha="center", va="center", color=eng.PLAYER_COLORS[player], zorder=2)

    mover = eng.player_to_move(board)
    if title:
        ax.set_title(title, fontsize=11, color=COLOR_TEXT, loc="left")
    if mover is not None:
        ax.text(1.5, -0.28, f"to move: {eng.PLAYER_NAMES[mover]}", fontsize=11,
                 ha="center", color=eng.PLAYER_COLORS[mover], fontweight="bold")
    else:
        w = eng.check_winner(board)
        msg = f"{eng.PLAYER_NAMES[w]} wins" if w else "draw"
        ax.text(1.5, -0.28, msg, fontsize=11, ha="center", color=COLOR_TEXT, fontweight="bold")


# ---------------------------------------------------------------------------
# Tree layout: post-order x assignment (leaves get sequential slots, internal
# nodes sit above the mean of their children), depth-based y. Bounded to
# roughly N+1 nodes by the lazy-expansion design in mcts_core, so this stays
# readable without needing graphviz.
# ---------------------------------------------------------------------------

def compute_tree_layout(nodes: Dict[int, mc.MCTSNode], root_id: int, x_spacing: float = 1.9
                         ) -> Dict[int, Tuple[float, int]]:
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
    return {nid: (x[nid], depth[nid]) for nid in nodes}


def tree_figsize(nodes: Dict[int, mc.MCTSNode], root_id: int) -> Tuple[float, float]:
    """Size the figure to the tree's own footprint so a 9-wide root row and a
    narrow 2-wide deep branch don't share the same cramped default size."""
    layout = compute_tree_layout(nodes, root_id)
    n_leaves = sum(1 for nid, node in nodes.items() if not node.children)
    max_depth = max(d for _, d in layout.values())
    width = min(15.0, max(6.5, n_leaves * 1.35))
    height = min(9.0, max(4.5, (max_depth + 1) * 1.9))
    return width, height


def draw_tree(ax, nodes: Dict[int, mc.MCTSNode], root_id: int, mover_for_scores: int,
              path: Sequence[int] = (), expanded_id: Optional[int] = None,
              show_prior: bool = True, title: Optional[str] = None,
              show_untried_stub: bool = True) -> None:
    """mover_for_scores: which player's Q is printed on each node (usually the
    root player, so every node's number answers 'how good for the player
    deciding at the root'). Untried actions at expanded nodes are drawn as
    faint unlabeled stubs so the branching factor is visible before it's
    explored."""
    ax.clear()
    ax.axis("off")
    ax.set_aspect("equal")
    layout = compute_tree_layout(nodes, root_id)
    path_set = set(path)

    # untried-action stubs, drawn first so real nodes sit on top
    if show_untried_stub:
        for nid, node in nodes.items():
            if eng.is_terminal(node.board):
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

    # edges
    for nid, node in nodes.items():
        if node.parent is None:
            continue
        px, py = layout[node.parent]
        cx, cy = layout[nid]
        on_path = nid in path_set and node.parent in path_set
        ax.add_line(Line2D([px, cx], [py, cy],
                            color=COLOR_PATH if on_path else "#B9BFCC",
                            linewidth=2.4 if on_path else 1.2, zorder=1))

    # nodes
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
        ax.add_patch(Circle((cx, cy), radius, facecolor=color, edgecolor="white",
                             linewidth=1.5, zorder=3))

        q = node.Q(mover_for_scores)
        stat = f"N{node.N} Q{_compact(q)}"
        if show_prior and node.parent is not None:
            stat += f" P{_compact(node.prior)}"
        ax.text(cx, cy - radius - 0.09, node.label(), fontsize=7.6, ha="center", va="top",
                color=COLOR_TEXT, fontweight="bold", zorder=4)
        ax.text(cx, cy - radius - 0.27, stat, fontsize=6.3, ha="center", va="top",
                color="#4A5268", zorder=4)

    xs = [p[0] for p in layout.values()]
    ys = [p[1] for p in layout.values()]
    ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
    ax.set_ylim(max(ys) + 0.75, -0.55)  # inverted: root at top
    if title:
        ax.set_title(title, fontsize=11, color=COLOR_TEXT, loc="left")


# ---------------------------------------------------------------------------
# The dashboard: tree (left) + board (right), next/back above, one frame
# per MCTS sub-phase (selection / expansion / evaluation / backprop).
# ---------------------------------------------------------------------------

class TreeBoardDashboard:
    """Tree (left) + board (right), next/back above. Reads each frame's own
    `nodes_snapshot` directly -- the tree shown at any step is exactly what
    mcts_core recorded at that moment, not a reconstruction."""

    def __init__(self, frames: List[mc.Frame], root_board: eng.Board, title: str = ""):
        if not frames:
            raise ValueError("no frames to display -- did MCTS.run(record=True) get called?")
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
        iter_label = W.Label()
        header = W.HBox([prev_btn, next_btn, iter_label])

        state = {"idx": 0}

        def redraw():
            idx = state["idx"]
            frame = self.frames[idx]
            nodes, root_id = frame.nodes_snapshot, frame.root_id
            root = nodes[root_id]

            with tree_out:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(6.4, 5.4))
                draw_tree(ax, nodes, root_id, mover_for_scores=root.player,
                          path=frame.path, expanded_id=frame.expanded_id,
                          title=f"{self.title} \u2014 iteration {frame.iteration}, {frame.phase}")
                plt.show()
                plt.close(fig)

            with board_out:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(3.6, 4.2))
                highlight = None
                if len(frame.path) > 1:
                    first_child = nodes[frame.path[1]]
                    highlight = first_child.action
                draw_board(ax, root.board, highlight_cell=highlight,
                           title="position under analysis")
                plt.show()
                plt.close(fig)

            with info_out:
                clear_output(wait=True)
                path_labels = " \u2192 ".join(nodes[n].label() for n in frame.path)
                msg = f"**{frame.phase.upper()}**  |  path: {path_labels}"
                if frame.reward is not None:
                    r = {eng.PLAYER_SYMBOLS[p]: round(v, 2) for p, v in frame.reward.items()}
                    msg += f"  |  reward backed up: {r}"
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

        prev_btn.on_click(on_prev)
        next_btn.on_click(on_next)

        display(header)
        display(W.HBox([tree_out, board_out]))
        display(info_out)
        redraw()


# ---------------------------------------------------------------------------
# Static helpers for the comparison / sweep scenarios (no stepping needed).
# ---------------------------------------------------------------------------

def render_final_tree(mcts: "mc.MCTS", ax, title: str = "") -> None:
    root = mcts.root()
    draw_tree(ax, mcts.nodes, mcts.root_id, mover_for_scores=root.player, title=title)


def render_side_by_side(panels: List[Tuple[str, "mc.MCTS"]], figsize=(15, 5)):
    """panels: list of (title, mcts) pairs, drawn as final trees left to right."""
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, m) in zip(axes, panels):
        render_final_tree(m, ax, title=title)
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
