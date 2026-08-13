"""2-player MCTS: lazy expansion, UCT/PUCT selection, scalar negamax backprop (v -> 1-v per level toward the root)."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import engine as eng

PriorFn = Callable[[eng.Game, eng.Board, int], Dict[eng.Action, float]]
EvalFn = Callable[[eng.Game, eng.Board, int], float]
ScoreFn = Callable[["MCTSNode", Optional["MCTSNode"], float], float]


@dataclass
class MCTSNode:
    id: int
    board: eng.Board
    player: int
    parent: Optional[int]
    action: Optional[eng.Action]
    moved_by: Optional[int] = None
    prior: float = 0.0
    N: int = 0
    W: float = 0.0
    children: Dict[eng.Action, int] = field(default_factory=dict)
    legal_actions: List[eng.Action] = field(default_factory=list)
    priors: Dict[eng.Action, float] = field(default_factory=dict)

    def Q(self) -> float:
        """Value from this node's own mover's perspective."""
        return self.W / self.N if self.N else 0.0

    def untried_actions(self) -> List[eng.Action]:
        return [a for a in self.legal_actions if a not in self.children]

    def label(self, game: eng.Game) -> str:
        return eng.action_label(game, self.action, self.moved_by)


@dataclass
class Frame:
    iteration: int
    phase: str
    nodes_snapshot: Dict[int, MCTSNode]
    root_id: int
    path: Tuple[int, ...] = ()
    expanded_id: Optional[int] = None
    reward: Optional[Dict[int, float]] = None
    note: str = ""


def uniform_prior(game: eng.Game, board: eng.Board, player: int) -> Dict[eng.Action, float]:
    """Every legal action equally likely."""
    legal = game.legal_moves(board)
    if not legal:
        return {}
    p = 1.0 / len(legal)
    return {a: p for a in legal}


def make_constant_prior(cell_weights: Dict[int, float]) -> PriorFn:
    """Fixed per-cell weights (e.g. center > corner > edge), renormalized over legal actions."""
    def fn(game: eng.Game, board: eng.Board, player: int) -> Dict[eng.Action, float]:
        legal = game.legal_moves(board)
        if not legal:
            return {}
        raw = {a: cell_weights.get(eng.action_cell(a), 0.0) for a in legal}
        total = sum(raw.values())
        if total <= 0:
            p = 1.0 / len(legal)
            return {a: p for a in legal}
        return {a: v / total for a, v in raw.items()}
    return fn


def make_random_rollout_eval(rng) -> EvalFn:
    """Play uniformly random moves to a finished game, return the reward for `player`."""
    def fn(game: eng.Game, board: eng.Board, player: int) -> float:
        b = board
        while not game.is_terminal(b):
            p = game.player_to_move(b)
            a = rng.choice(game.legal_moves(b))
            b = game.apply_move(b, a, p)
        return game.reward(b)[player]
    return fn


_DEFAULT_GROUP_RANK: Dict[str, int] = {"center": 0, "corner": 1, "edge": 2}


def make_heuristic_rollout_eval(rng, group_rank: Optional[Dict[str, int]] = None) -> EvalFn:
    """Rollout guided by a light heuristic instead of pure randomness: at each ply, take an
    immediate win if one exists, else avoid a move that hands the opponent an immediate
    win next turn, else prefer center > corner > edge among what's left. A uniformly
    random playout on a 3x3 board is noisy enough that it under-counts how bad "hand the
    opponent an open line" really is (see mcts_games.ipynb's rollout-bottleneck note); this
    cuts that noise the same way a real player would, without hardcoding either game --
    it only uses the generic Game interface (check_winner/legal_moves/apply_move) plus
    action_group's center/corner/edge banding, so it works for Classic and Numerical alike.
    """
    ranks = group_rank or _DEFAULT_GROUP_RANK

    def choose(game: eng.Game, board: eng.Board, mover: int) -> eng.Action:
        legal = game.legal_moves(board)
        for a in legal:
            if game.check_winner(game.apply_move(board, a, mover)) == mover:
                return a
        opponent = 1 if mover == 2 else 2

        def hands_opponent_a_win(a: eng.Action) -> bool:
            after = game.apply_move(board, a, mover)
            if game.is_terminal(after):
                return False
            return any(game.check_winner(game.apply_move(after, b, opponent)) == opponent
                       for b in game.legal_moves(after))

        safe = [a for a in legal if not hands_opponent_a_win(a)]
        pool = safe if safe else legal
        best_rank = min(ranks[eng.action_group(a)] for a in pool)
        preferred = [a for a in pool if ranks[eng.action_group(a)] == best_rank]
        return rng.choice(preferred)

    def fn(game: eng.Game, board: eng.Board, player: int) -> float:
        b = board
        while not game.is_terminal(b):
            mover = game.player_to_move(b)
            a = choose(game, b, mover)
            b = game.apply_move(b, a, mover)
        return game.reward(b)[player]
    return fn


def make_constant_value(values: Dict[int, float]) -> EvalFn:
    """A stand-in for a trained value function: a fixed value per player."""
    def fn(game: eng.Game, board: eng.Board, player: int) -> float:
        return values.get(player, 0.5)
    return fn


def make_uct_selector(c: float = 1.4) -> ScoreFn:
    """Classic UCT. Untried actions score +inf so every child is tried once first.

    Node.Q() is "value from this node's own mover's perspective" (see MCTSNode.Q). The
    parent is choosing on behalf of *its own* mover, who is the opposite player from the
    child's mover (players strictly alternate), so the Q(s,a) the UCT formula wants is
    1 - child.Q(), not child.Q() directly -- same negamax flip _backprop already applies
    once per level on the way up. The formula itself (Q + c*sqrt(ln N(s)/N(s,a))) is
    unchanged; this only fixes which value gets plugged in for Q(s,a).
    """
    def score(parent: MCTSNode, child: Optional[MCTSNode], prior: float) -> float:
        if child is None or child.N == 0:
            return math.inf
        q_for_parent = 1.0 - child.Q()
        return q_for_parent + c * math.sqrt(math.log(parent.N + 1) / child.N)
    return score


def make_puct_selector(c: float = 1.4, fpu: float = 0.0) -> ScoreFn:
    """AlphaZero-style PUCT: exploration weighted by the prior, no singularity at N=0.

    Same perspective fix as make_uct_selector: Q(s,a) is 1 - child.Q(), converting the
    child's own-mover value into the parent's-mover value before comparing. `fpu` is
    already a direct assumed value for the parent's mover on an untried action, so it is
    used as-is (no flip). The formula itself is unchanged.
    """
    def score(parent: MCTSNode, child: Optional[MCTSNode], prior: float) -> float:
        n_child = child.N if child is not None else 0
        q_for_parent = (1.0 - child.Q()) if (child is not None and child.N > 0) else fpu
        return q_for_parent + c * prior * math.sqrt(parent.N + 1) / (1 + n_child)
    return score


class MCTS:
    """One search tree over one `engine.Game`."""

    def __init__(self, game: eng.Game, score_fn: ScoreFn, eval_fn: EvalFn, prior_fn: PriorFn,
                 seed: int = 0):
        self.game = game
        self.score_fn = score_fn
        self.eval_fn = eval_fn
        self.prior_fn = prior_fn
        import random
        self.rng = random.Random(seed)
        self.nodes: Dict[int, MCTSNode] = {}
        self.root_id: Optional[int] = None
        self._next_id = 0
        self.frames: List[Frame] = []

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _make_node(self, board: eng.Board, parent_id: Optional[int],
                    action: Optional[eng.Action]) -> MCTSNode:
        player = self.game.next_mover(board)
        moved_by = self.nodes[parent_id].player if parent_id is not None else None
        node = MCTSNode(id=self._new_id(), board=board, player=player, parent=parent_id,
                         action=action, moved_by=moved_by)
        if parent_id is not None:
            node.prior = self.nodes[parent_id].priors.get(action, 0.0)
        if not self.game.is_terminal(board):
            node.legal_actions = self.game.legal_moves(board)
            node.priors = self.prior_fn(self.game, board, player)
        self.nodes[node.id] = node
        return node

    def new_tree(self, board: eng.Board) -> MCTSNode:
        self.nodes = {}
        self._next_id = 0
        self.frames = []
        root = self._make_node(board, None, None)
        self.root_id = root.id
        return root

    def root(self) -> MCTSNode:
        return self.nodes[self.root_id]

    def _select(self, root_id: int) -> Tuple[List[int], Optional[eng.Action]]:
        path = [root_id]
        nid = root_id
        while True:
            node = self.nodes[nid]
            if self.game.is_terminal(node.board):
                return path, None
            best_a, best_score = None, -math.inf
            for a in node.legal_actions:
                child = self.nodes[node.children[a]] if a in node.children else None
                s = self.score_fn(node, child, node.priors.get(a, 0.0))
                if s > best_score:
                    best_score, best_a = s, a
            if best_a not in node.children:
                return path, best_a
            nid = node.children[best_a]
            path.append(nid)

    def _expand(self, path: List[int], action: Optional[eng.Action]) -> Tuple[int, Optional[int]]:
        if action is None:
            return path[-1], None
        parent = self.nodes[path[-1]]
        child_board = self.game.apply_move(parent.board, action, parent.player)
        child = self._make_node(child_board, parent.id, action)
        parent.children[action] = child.id
        path.append(child.id)
        return child.id, child.id

    def _evaluate(self, node_id: int) -> float:
        node = self.nodes[node_id]
        if self.game.is_terminal(node.board):
            return self.game.reward(node.board)[node.player]
        return self.eval_fn(self.game, node.board, node.player)

    def _backprop(self, path: List[int], value: float) -> None:
        val = value
        for nid in reversed(path):
            node = self.nodes[nid]
            node.N += 1
            node.W += val
            val = 1.0 - val

    def _snapshot(self, iteration: int, phase: str, path: List[int],
                   expanded_id: Optional[int] = None, reward: Optional[Dict[int, float]] = None,
                   note: str = "") -> Frame:
        return Frame(iteration=iteration, phase=phase, nodes_snapshot=copy.deepcopy(self.nodes),
                     root_id=self.root_id, path=tuple(path), expanded_id=expanded_id,
                     reward=reward, note=note)

    def run(self, n_iterations: int, record: bool = False) -> None:
        for it in range(1, n_iterations + 1):
            path, expand_action = self._select(self.root_id)
            if record:
                self.frames.append(self._snapshot(it, "selection", path))
            leaf_id, expanded_id = self._expand(path, expand_action)
            if record:
                self.frames.append(self._snapshot(it, "expansion", path, expanded_id=expanded_id))
            value = self._evaluate(leaf_id)
            if record:
                mover = self.nodes[leaf_id].player
                self.frames.append(self._snapshot(
                    it, "evaluation", path, expanded_id=expanded_id,
                    note=f"leaf value for player {mover}: {value:.2f}"))
            self._backprop(path, value)
            if record:
                reward_disp: Dict[int, float] = {}
                val = value
                for nid in reversed(path):
                    reward_disp[self.nodes[nid].player] = round(val, 3)
                    val = 1.0 - val
                self.frames.append(self._snapshot(
                    it, "backprop", path, expanded_id=expanded_id, reward=reward_disp))

    def best_action(self, criterion: str = "visits") -> Optional[eng.Action]:
        root = self.root()
        items = [(a, self.nodes[cid]) for a, cid in root.children.items()]
        if not items:
            return None
        if criterion == "visits":
            return max(items, key=lambda kv: (kv[1].N, kv[1].Q()))[0]
        return max(items, key=lambda kv: kv[1].Q())[0]

    def visit_distribution(self) -> Dict[eng.Action, int]:
        root = self.root()
        return {a: self.nodes[cid].N for a, cid in root.children.items()}

    def reroot(self, action: eng.Action) -> int:
        """Discard everything outside the subtree reached by `action`; return visits reused."""
        root = self.root()
        if action in root.children:
            new_root_id = root.children[action]
            keep = set()
            stack = [new_root_id]
            while stack:
                nid = stack.pop()
                if nid in keep:
                    continue
                keep.add(nid)
                stack.extend(self.nodes[nid].children.values())
            reused_visits = self.nodes[new_root_id].N
            self.nodes = {nid: self.nodes[nid] for nid in keep}
            self.nodes[new_root_id].parent = None
            self.nodes[new_root_id].action = None
            self.nodes[new_root_id].moved_by = None
            self.root_id = new_root_id
        else:
            child_board = self.game.apply_move(root.board, action, root.player)
            reused_visits = 0
            self._next_id = 0
            new_root = self._make_node(child_board, None, None)
            self.nodes = {new_root.id: new_root}
            self.root_id = new_root.id
        self.frames = []
        return reused_visits


def self_check() -> None:
    """Search self-check: node counts and the visit-conservation invariant, both games."""
    import random

    m = MCTS(eng.CLASSIC, make_uct_selector(1.4), make_random_rollout_eval(random.Random(0)),
              uniform_prior, seed=0)
    m.new_tree(eng.CLASSIC.initial_board())
    m.run(30, record=True)
    assert len(m.nodes) <= 31

    def check_visit_invariant(mcts_obj):
        """Each non-terminal node's visits equal its children's visits, plus one for its own creation."""
        for node in mcts_obj.nodes.values():
            if node.N > 0 and not mcts_obj.game.is_terminal(node.board):
                child_sum = sum(mcts_obj.nodes[cid].N for cid in node.children.values())
                own_birth = 0 if node.id == 0 else 1
                assert child_sum == node.N - own_birth, (node.id, child_sum, node.N, own_birth)

    check_visit_invariant(m)
    best = m.best_action("visits")
    assert best is not None

    m2 = MCTS(eng.NUMERICAL, make_puct_selector(1.4),
              make_constant_value({1: 0.6, 2: 0.4}),
              make_constant_prior({4: 0.5, 0: 0.125, 2: 0.125, 6: 0.125, 8: 0.125}), seed=0)
    m2.new_tree(eng.NUMERICAL.initial_board())
    m2.run(20, record=False)
    assert len(m2.nodes) <= 21
    check_visit_invariant(m2)
    a = m2.best_action("visits")
    assert a is not None and a in m2.root().children

    kept = m2.reroot(a)
    assert m2.root().parent is None
    m2.run(10, record=False)
    assert len(m2.nodes) <= 21 + 10
    check_visit_invariant(m2)

    print("mcts self-check passed:", len(m.nodes), "classic nodes,", kept, "numerical visits reused")


if __name__ == "__main__":
    self_check()
