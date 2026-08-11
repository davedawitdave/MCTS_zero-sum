"""MCTS for N-player constant-sum games, specialised to 3-player tic-tac-toe.

Design note: why Max^N instead of negamax
-------------------------------------------
Classic 2-player MCTS backpropagates a single scalar and flips its sign at
every level (v_parent = 1 - v_child), because with exactly two players and a
constant-sum reward, player B's outcome is *exactly determined* by player A's
outcome (r_B = 1 - r_A). With three players that determinism is gone: knowing
player 2 lost tells you nothing about how player 1's and player 3's rewards
split the remaining 1.0. So every node stores a full reward *vector*
{1: w1, 2: w2, 3: w3}, backprop adds the same vector to every node on the
path (no flip), and a node scores its children using the vector component
that belongs to *that node's own mover*. This is the textbook Max^n
algorithm (Luckhart & Irani 1986; Sturtevant 2000); the 2-player negamax
trick is the special case N=2. See `demonstrate_broken_two_player_flip` at
the bottom for what goes wrong if the 2-player flip is used anyway.

Lazy expansion, one new node per iteration
---------------------------------------------
A node tracks which of its legal actions already have a child and which are
still "untried". Selection always considers *every legal action* at a node,
not just the ones with an existing child: an untried action is scored using
its prior with N=0 (no node needs to exist yet to compute that). Only once
an untried action is chosen does a single new node get created for it. This
means one MCTS iteration creates at most one new tree node, so an N-iteration
search has at most N+1 nodes total -- small, bounded, and exactly what makes
the tree readable when it is drawn iteration by iteration.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import scripts.ttt3_engine as eng
from scripts.ttt3_engine import Board, PLAYERS

PriorFn = Callable[[Board, int], Dict[int, float]]
EvalFn = Callable[[Board, int], Dict[int, float]]
# score_fn(parent, child_or_None, prior_of_this_action, mover) -> float
ScoreFn = Callable[[Optional["MCTSNode"], Optional["MCTSNode"], float, int], float]


@dataclass
class MCTSNode:
    board: Board
    parent: Optional[int]
    action: Optional[int]          # cell index taken from parent to reach here; None for root
    player: Optional[int]          # player to move AT this node; None if terminal
    prior: float = 0.0             # P(action | parent) that led to this node
    children: Dict[int, int] = field(default_factory=dict)  # action -> child node id
    action_priors: Optional[Dict[int, float]] = None        # cached prior_fn(board, player)
    N: int = 0
    W: Dict[int, float] = field(default_factory=lambda: {p: 0.0 for p in PLAYERS})

    def Q(self, player: int) -> float:
        return self.W[player] / self.N if self.N > 0 else 0.0

    def snapshot(self) -> "MCTSNode":
        """Independent copy for a Frame's record -- mutating the live tree
        after this call must not change what the snapshot shows."""
        return MCTSNode(
            board=self.board, parent=self.parent, action=self.action, player=self.player,
            prior=self.prior, children=dict(self.children),
            action_priors=dict(self.action_priors) if self.action_priors is not None else None,
            N=self.N, W=dict(self.W),
        )

    def untried_actions(self) -> List[int]:
        return [a for a in eng.legal_moves(self.board) if a not in self.children]

    def label(self) -> str:
        if self.action is None:
            return "root"
        # Whoever placed the mark that created this node was the (k-1)-th
        # mover, where k is how many cells are filled on this node's board --
        # true whether or not the resulting board happens to be terminal.
        k = eng.num_moves_played(self.board)
        mover_who_moved = PLAYERS[(k - 1) % 3]
        sym = eng.PLAYER_SYMBOLS[mover_who_moved]
        return f"{sym}{self.action}"


# ---------------------------------------------------------------------------
# Selection score functions. `child` may be None (action never tried yet);
# `prior` is always available regardless, from the parent's cached action_priors.
# ---------------------------------------------------------------------------

def uct_score(parent: MCTSNode, child: Optional[MCTSNode], prior: float, mover: int,
              c: float = 1.4) -> float:
    if child is None or child.N == 0:
        return math.inf
    return child.Q(mover) + c * math.sqrt(math.log(max(parent.N, 1)) / child.N)


def puct_score(parent: MCTSNode, child: Optional[MCTSNode], prior: float, mover: int,
               c: float = 1.4, fpu: float = 0.0) -> float:
    n = child.N if child is not None else 0
    q = child.Q(mover) if (child is not None and n > 0) else fpu
    bonus = c * prior * math.sqrt(parent.N) / (1 + n)
    return q + bonus


def make_uct_selector(c: float = 1.4) -> ScoreFn:
    return lambda parent, child, prior, mover: uct_score(parent, child, prior, mover, c=c)


def make_puct_selector(c: float = 1.4, fpu: float = 0.0) -> ScoreFn:
    return lambda parent, child, prior, mover: puct_score(parent, child, prior, mover, c=c, fpu=fpu)


# ---------------------------------------------------------------------------
# Prior and evaluation functions.
# ---------------------------------------------------------------------------

def uniform_prior(board: Board, player: int) -> Dict[int, float]:
    moves = eng.legal_moves(board)
    return {a: 1.0 / len(moves) for a in moves}


def random_rollout_eval(board: Board, player: int, rng: random.Random) -> Dict[int, float]:
    """Play uniformly at random to a terminal state and return the true outcome."""
    b = board
    while not eng.is_terminal(b):
        a = rng.choice(eng.legal_moves(b))
        p = eng.player_to_move(b)
        b = eng.apply_move(b, a, p)
    return eng.terminal_reward(b)


def make_random_rollout_eval(rng: random.Random) -> EvalFn:
    return lambda board, player: random_rollout_eval(board, player, rng)


def make_constant_prior(fixed: Dict[int, float]) -> PriorFn:
    """For hand-authored demo priors: fixed[cell] = probability, renormalised
    over whatever is still legal so it stays a valid distribution at any node."""
    def fn(board: Board, player: int) -> Dict[int, float]:
        moves = eng.legal_moves(board)
        raw = {a: fixed.get(a, 0.0) for a in moves}
        total = sum(raw.values())
        if total <= 0:
            # None of the hand-set cells are legal here (e.g. this node is
            # deeper than the demo's priors were written for) -- fall back to
            # uniform rather than returning a degenerate all-zero prior.
            return {a: 1.0 / len(moves) for a in moves}
        return {a: v / total for a, v in raw.items()}
    return fn


def make_constant_value(fixed: Dict[int, float]) -> EvalFn:
    def fn(board: Board, player: int) -> Dict[int, float]:
        return dict(fixed)
    return fn


# ---------------------------------------------------------------------------
# The search itself.
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """One sub-step of one iteration, for the next/back dashboard.

    `nodes_snapshot` and `root_id` are a genuine independent copy of the tree
    at the moment this frame was recorded -- not something reconstructed
    later by replaying the search. Replaying would be wrong whenever eval_fn
    is stochastic (e.g. a random rollout): a replay draws fresh numbers from
    the shared RNG and can diverge from what actually happened, so the
    dashboard must render exactly what was captured here, nothing else.
    """
    iteration: int
    phase: str                 # "selection" | "expansion" | "evaluation" | "backprop"
    path: List[int]            # node ids visited this iteration, root to leaf
    nodes_snapshot: Dict[int, "MCTSNode"] = field(default_factory=dict)
    root_id: int = 0
    expanded_id: Optional[int] = None
    reward: Optional[Dict[int, float]] = None
    note: str = ""


class MCTS:
    """Max^N tree search over 3-player tic-tac-toe.

    Parameters
    ----------
    score_fn: decides which action to take at a node (uct_score / puct_score,
        or a closure from make_uct_selector / make_puct_selector).
    prior_fn: P(action | board, player), cached once per node on first visit.
        Defaults to uniform (pure MCTS has no notion of a prior).
    eval_fn: reward-vector estimate for a freshly created leaf. Pass a
        rollout closure for classic MCTS, or a constant/learned value function
        for PUCT / AlphaZero-style search.
    seed: every stochastic entry point in this project takes an explicit seed
        so a run is reproducible from its parameters alone.
    """

    def __init__(
        self,
        score_fn: ScoreFn,
        eval_fn: EvalFn,
        prior_fn: PriorFn = uniform_prior,
        seed: int = 0,
    ):
        self.score_fn = score_fn
        self.eval_fn = eval_fn
        self.prior_fn = prior_fn
        self.rng = random.Random(seed)
        self.nodes: Dict[int, MCTSNode] = {}
        self._next_id = 0
        self.root_id: Optional[int] = None

    # -- tree construction ---------------------------------------------------

    def _new_node(self, board: Board, parent: Optional[int], action: Optional[int],
                  prior: float = 0.0) -> int:
        node_id = self._next_id
        self._next_id += 1
        self.nodes[node_id] = MCTSNode(
            board=board, parent=parent, action=action,
            player=eng.player_to_move(board), prior=prior,
        )
        return node_id

    def new_tree(self, board: Board) -> int:
        self.nodes = {}
        self._next_id = 0
        self.root_id = self._new_node(board, parent=None, action=None, prior=1.0)
        return self.root_id

    def reroot(self, action: int) -> int:
        """Tree reuse: commit to `action` at the current root, discard every
        other branch, and make that child the new root. Visit counts and
        values already accumulated under it are kept, not recomputed."""
        old_root = self.nodes[self.root_id]
        if action not in old_root.children:
            raise ValueError(f"action {action} was never expanded at the current root")
        new_root_id = old_root.children[action]
        keep = self._reachable_ids(new_root_id)
        self.nodes = {nid: self.nodes[nid] for nid in keep}
        self.nodes[new_root_id].parent = None
        self.nodes[new_root_id].action = None
        self.root_id = new_root_id
        return new_root_id

    def _reachable_ids(self, start: int) -> List[int]:
        seen = []
        stack = [start]
        while stack:
            nid = stack.pop()
            seen.append(nid)
            stack.extend(self.nodes[nid].children.values())
        return seen

    # -- one MCTS iteration ---------------------------------------------------

    def _select(self, root_id: int) -> Tuple[List[int], bool, Optional[int]]:
        """Walk down existing children; stop at a terminal board, or at the
        first node with an untried action (returning which action to expand)."""
        path = [root_id]
        node = self.nodes[root_id]
        while True:
            if eng.is_terminal(node.board):
                return path, True, None
            if node.action_priors is None:
                node.action_priors = self.prior_fn(node.board, node.player)
            untried = node.untried_actions()
            if untried:
                best_a, best_s = None, -math.inf
                for a in untried:
                    s = self.score_fn(node, None, node.action_priors.get(a, 0.0), node.player)
                    if s > best_s:
                        best_s, best_a = s, a
                return path, False, best_a
            best_a, best_s = None, -math.inf
            for a, cid in node.children.items():
                child = self.nodes[cid]
                s = self.score_fn(node, child, child.prior, node.player)
                if s > best_s:
                    best_s, best_a = s, a
            next_id = node.children[best_a]
            path.append(next_id)
            node = self.nodes[next_id]

    def _snapshot_nodes(self) -> Dict[int, "MCTSNode"]:
        return {nid: node.snapshot() for nid, node in self.nodes.items()}

    def _backprop(self, path: List[int], reward: Dict[int, float]) -> None:
        for nid in path:
            node = self.nodes[nid]
            node.N += 1
            for p in PLAYERS:
                node.W[p] += reward[p]

    def run(self, n_iterations: int, record: bool = True) -> List[Frame]:
        frames: List[Frame] = []

        def snap(phase: str, path: List[int], **kw) -> None:
            if record:
                frames.append(Frame(
                    iteration=it, phase=phase, path=list(path),
                    nodes_snapshot=self._snapshot_nodes(), root_id=self.root_id, **kw,
                ))

        for it in range(1, n_iterations + 1):
            path, reached_terminal, expand_action = self._select(self.root_id)
            snap("selection", path)

            if reached_terminal:
                leaf = self.nodes[path[-1]]
                reward = eng.terminal_reward(leaf.board)
                snap("evaluation", path, reward=reward,
                     note="reached a finished game, nothing to expand")
            else:
                parent = self.nodes[path[-1]]
                child_board = eng.apply_move(parent.board, expand_action, parent.player)
                prior = parent.action_priors.get(expand_action, 0.0)
                child_id = self._new_node(child_board, parent=path[-1], action=expand_action,
                                           prior=prior)
                parent.children[expand_action] = child_id
                path.append(child_id)
                snap("expansion", path, expanded_id=child_id)
                leaf = self.nodes[child_id]
                if eng.is_terminal(leaf.board):
                    # the move that created this leaf ended the game -- use
                    # the exact outcome, never eval_fn (leaf.player is None
                    # here, there is no "player to move" on a finished board)
                    reward = eng.terminal_reward(leaf.board)
                    snap("evaluation", path, reward=reward,
                         note="the expanded move ended the game, exact outcome used")
                else:
                    reward = self.eval_fn(leaf.board, leaf.player)
                    snap("evaluation", path, reward=reward)

            self._backprop(path, reward)
            snap("backprop", path, reward=reward)

        for node in self.nodes.values():
            if node.N > 0:
                assert abs(sum(node.W.values()) - node.N) < 1e-6, (
                    f"constant-sum invariant broken at node with N={node.N}, "
                    f"sum(W)={sum(node.W.values())}"
                )
        return frames

    # -- convenience -----------------------------------------------------------

    def root(self) -> MCTSNode:
        return self.nodes[self.root_id]

    def best_action(self, by: str = "visits") -> int:
        root = self.root()
        if by == "visits":
            return max(root.children.items(), key=lambda kv: self.nodes[kv[1]].N)[0]
        if by == "value":
            mover = root.player
            return max(root.children.items(), key=lambda kv: self.nodes[kv[1]].Q(mover))[0]
        raise ValueError(f"unknown selection rule {by!r}")

    def visit_distribution(self) -> Dict[int, int]:
        root = self.root()
        return {a: self.nodes[cid].N for a, cid in root.children.items()}


# ---------------------------------------------------------------------------
# Deliberately-wrong contrast function for the zero-sum scenario: this is what
# happens if you keep the 2-player "flip the sign every level" rule and just
# apply it at a 3-player node anyway. It is never used by the MCTS class above
# -- it exists only so the notebook can show, with real numbers, why it is
# wrong, rather than merely asserting that it is.
# ---------------------------------------------------------------------------

def demonstrate_broken_two_player_flip() -> str:
    """Hand-built 3-node example: a player-3 node with two children whose true
    (Max^N) values disagree with what the naive 1-v flip would compute."""
    child_A = {1: 0.10, 2: 0.60, 3: 0.30}   # bad for player 1, great for player 2
    child_B = {1: 0.50, 2: 0.10, 3: 0.40}   # good for player 1, bad for player 2

    correct_pick = "A" if child_A[3] > child_B[3] else "B"
    correct_values = f"Q3(A)={child_A[3]:.2f}, Q3(B)={child_B[3]:.2f}"

    naive_v_A = 1 - child_A[1]   # arbitrarily treats player 1 as "the" opponent
    naive_v_B = 1 - child_B[1]
    naive_pick = "A" if naive_v_A > naive_v_B else "B"

    lines = [
        f"Child A true values: {child_A}  (player 3's own share: {child_A[3]:.2f})",
        f"Child B true values: {child_B}  (player 3's own share: {child_B[3]:.2f})",
        f"Correct Max^N choice for player 3: {correct_pick}  [{correct_values}]",
        f"Naive 1-v flip (using player 1 as 'the' opponent): "
        f"v(A)={naive_v_A:.2f}, v(B)={naive_v_B:.2f} -> picks {naive_pick}",
        f"Mismatch: {'YES' if correct_pick != naive_pick else 'no'} "
        f"-- the naive flip is undefined here because there are two opponents, "
        f"not one, and player 3's own outcome does not move in lockstep with "
        f"either of theirs.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    rng = random.Random(42)
    mcts = MCTS(
        score_fn=make_uct_selector(c=1.4),
        eval_fn=make_random_rollout_eval(rng),
        prior_fn=uniform_prior,
        seed=42,
    )
    mcts.new_tree(eng.initial_board())
    frames = mcts.run(40, record=True)
    root = mcts.root()
    assert root.N == 40
    assert len(mcts.nodes) <= 41, f"expected at most 41 nodes for 40 iterations, got {len(mcts.nodes)}"
    print(f"nodes created: {len(mcts.nodes)}, frames recorded: {len(frames)}")
    print("root visit distribution:", {eng.cell_type(a): mcts.nodes[cid].N for a, cid in root.children.items()})
    best = mcts.best_action("visits")
    print("most-visited root action:", eng.cell_type(best), best)

    n_before = len(mcts.nodes)
    mcts.reroot(best)
    n_after = len(mcts.nodes)
    assert n_after < n_before
    assert mcts.root().N > 0
    print(f"reroot: nodes {n_before} -> {n_after}, new root N={mcts.root().N}")

    print()
    print(demonstrate_broken_two_player_flip())
    print("mcts_core self-check passed")
