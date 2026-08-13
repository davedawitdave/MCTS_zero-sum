"""Scenario presets, opponent policies, and a tiny self-play policy/value network."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import engine as eng
import mcts as mc

@dataclass
class ScenarioConfig:
    name: str
    description: str
    game: eng.Game
    mode: str = "uct"
    c: float = 1.4
    fpu: float = 0.0
    n_iterations: int = 10
    seed: int = 0
    cell_weights: Optional[Dict[int, float]] = None
    values: Optional[Dict[int, float]] = None


SCENARIO_PURE_UCT = ScenarioConfig(
    name="Pure MCTS (UCT)",
    game=eng.CLASSIC,
    description=(
        "No priors, no learned value. Every freshly created leaf is evaluated "
        "by playing a uniformly random game out to a finished result."
    ),
    mode="uct", c=1.4, n_iterations=10, seed=7,
)

# Center > corner > edge, edges kept strictly positive (never 0) so they stay reachable.
BAND_WEIGHTS: Dict[int, float] = {
    4: 0.40,
    0: 0.12, 2: 0.12, 6: 0.12, 8: 0.12,
    1: 0.03, 3: 0.03, 5: 0.03, 7: 0.03,
}

SCENARIO_PUCT_PRIOR = ScenarioConfig(
    name="PUCT with hand-set priors",
    game=eng.CLASSIC,
    description=(
        "Center is believed strongest, corners next, edges a distant but "
        "nonzero third -- a stand-in for what a trained policy network "
        "would output -- and leaves are evaluated directly by a stand-in "
        "value function instead of a rollout."
    ),
    mode="puct", c=1.4, n_iterations=10, seed=1,
    cell_weights=BAND_WEIGHTS,
    values={1: 0.62, 2: 0.38},
)


def build_mcts(cfg: ScenarioConfig) -> mc.MCTS:
    if cfg.mode == "uct":
        score_fn = mc.make_uct_selector(c=cfg.c)
        prior_fn = mc.uniform_prior
        eval_fn = mc.make_random_rollout_eval(random.Random(cfg.seed))
    elif cfg.mode == "puct":
        score_fn = mc.make_puct_selector(c=cfg.c, fpu=cfg.fpu)
        prior_fn = mc.make_constant_prior(cfg.cell_weights or {})
        eval_fn = mc.make_constant_value(cfg.values or {p: 0.5 for p in eng.PLAYERS})
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")
    return mc.MCTS(cfg.game, score_fn, eval_fn, prior_fn, seed=cfg.seed)


@dataclass
class PlayConfig:
    """One knob-panel for the interactive demos: search budget, selection, priors, tree
    reuse, who moves first, and how leaves get evaluated. Feeds build_mcts_from_playconfig
    and viz.build_human_vs_mcts -- nothing here changes the UCT/PUCT formulas themselves."""

    # Search budget
    n_iterations: int = 500                       # default per move -- see notebook note on budgets
    use_graded_budget: bool = False               # if True: first move 30, later moves 20-28
    graded_schedule: Tuple[int, ...] = (30, 28, 24, 20)

    # Selection
    mode: str = "uct"                             # "uct" | "puct"
    c: float = 1.414                               # sqrt(2) default
    fpu: float = 0.0                               # for PUCT only

    # Priors / branching bias
    prior_style: str = "band"                      # "uniform" | "band"
    # "band"    = center > corner > edge (BAND_WEIGHTS above) -- the codebase default.
    # "uniform" = flat over legal moves.
    # A "lines" style (bias toward cells that complete/block a winning line) is a natural
    # next step but isn't wired up here -- see the notebook markdown.

    # Tree reuse
    tree_reuse: bool = True                        # reroot() between moves instead of rebuilding

    # Who starts (human-vs-MCTS widget)
    computer_starts: bool = True

    # Leaf evaluation
    eval_style: str = "heuristic_rollout"           # "heuristic_rollout" | "rollout" | "constant" | "net"
    # "heuristic_rollout" plays win>block>center/corner/edge during the rollout instead of
    # pure randomness -- far less noisy on a 3x3 board (see mcts.make_heuristic_rollout_eval).
    # "rollout" is the pure-random version used by the pedagogical baseline-story scenarios.
    values: Optional[Dict[int, float]] = None        # used when eval_style == "constant"
    seed: int = 0

    def iterations_for_move(self, move_number: int) -> int:
        """Iteration budget for the move_number-th (0-indexed) search this config drives."""
        if not self.use_graded_budget:
            return self.n_iterations
        schedule = self.graded_schedule
        return schedule[move_number] if move_number < len(schedule) else schedule[-1]


def _prior_fn_for_playconfig(cfg: PlayConfig) -> mc.PriorFn:
    if cfg.prior_style == "band":
        return mc.make_constant_prior(BAND_WEIGHTS)
    if cfg.prior_style == "uniform":
        return mc.uniform_prior
    raise NotImplementedError(
        f"prior_style={cfg.prior_style!r} is a future idea, not wired up yet -- "
        "use 'band' or 'uniform' (see the notebook markdown)."
    )


def _eval_fn_for_playconfig(cfg: PlayConfig, game: eng.Game, net: Optional["TinyNet"],
                             rng: random.Random) -> mc.EvalFn:
    if cfg.eval_style == "heuristic_rollout":
        return mc.make_heuristic_rollout_eval(rng)
    if cfg.eval_style == "rollout":
        return mc.make_random_rollout_eval(rng)
    if cfg.eval_style == "constant":
        return mc.make_constant_value(cfg.values or {p: 0.5 for p in eng.PLAYERS})
    if cfg.eval_style == "net":
        active_net = net if net is not None else TinyNet(game=game, seed=cfg.seed)
        return net_as_eval_fn(active_net)
    raise ValueError(f"unknown eval_style {cfg.eval_style!r}")


def build_mcts_from_playconfig(cfg: PlayConfig, game: eng.Game,
                                net: Optional["TinyNet"] = None,
                                seed: Optional[int] = None) -> mc.MCTS:
    """Build one mc.MCTS wired from a PlayConfig: mode/c/fpu pick the existing selector,
    prior_style picks the prior, eval_style picks the leaf evaluator. No new search logic --
    this only assembles the pieces mcts.py already provides."""
    use_seed = cfg.seed if seed is None else seed
    rng = random.Random(use_seed)

    if cfg.mode == "uct":
        score_fn = mc.make_uct_selector(c=cfg.c)
    elif cfg.mode == "puct":
        score_fn = mc.make_puct_selector(c=cfg.c, fpu=cfg.fpu)
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    prior_fn = _prior_fn_for_playconfig(cfg)
    eval_fn = _eval_fn_for_playconfig(cfg, game, net, rng)
    return mc.MCTS(game, score_fn, eval_fn, prior_fn, seed=use_seed)


_GROUP_RANK: Dict[str, int] = {"center": 0, "corner": 1, "edge": 2}


def random_policy(game: eng.Game, board: eng.Board, player: int, rng: random.Random):
    return rng.choice(game.legal_moves(board))


def heuristic_policy(game: eng.Game, board: eng.Board, player: int, rng: random.Random):
    """Win now; else avoid handing the opponent a win; else prefer center > corner > edge among safe moves."""
    moves = game.legal_moves(board)
    for a in moves:
        if game.check_winner(game.apply_move(board, a, player)) == player:
            return a
    opponent = 1 if player == 2 else 2

    def hands_opponent_a_win(a) -> bool:
        after = game.apply_move(board, a, player)
        if game.is_terminal(after):
            return False
        return any(game.check_winner(game.apply_move(after, b, opponent)) == opponent
                   for b in game.legal_moves(after))

    safe = [a for a in moves if not hands_opponent_a_win(a)]
    pool = safe if safe else moves
    best_rank = min(_GROUP_RANK[eng.action_group(a)] for a in pool)
    preferred = [a for a in pool if _GROUP_RANK[eng.action_group(a)] == best_rank]
    return rng.choice(preferred)


def make_mcts_policy(game: eng.Game, n_iterations: int, c: float = 1.4):
    def policy(g: eng.Game, board: eng.Board, player: int, rng: random.Random):
        m = mc.MCTS(g, mc.make_uct_selector(c=c), mc.make_random_rollout_eval(rng),
                     mc.uniform_prior, seed=rng.randint(0, 2**31 - 1))
        m.new_tree(board)
        m.run(n_iterations, record=False)
        return m.best_action("visits")
    return policy


def play_full_game(game: eng.Game, policies: Dict[int, Callable], seed: int = 0
                    ) -> Tuple[int, List]:
    """policies[player] -> policy(game, board, player, rng). Returns (winner, history)."""
    rng = random.Random(seed)
    board = game.initial_board()
    history: List = []
    while not game.is_terminal(board):
        p = game.player_to_move(board)
        a = policies[p](game, board, p, rng)
        board = game.apply_move(board, a, p)
        history.append(a)
    return game.check_winner(board), history


def round_robin_eval(game: eng.Game, policies: Dict[int, Callable], n_games: int, seed: int = 0
                      ) -> Dict[int, int]:
    """Play n_games with a fixed player->policy assignment; win counts per player, 0=draws."""
    wins = {0: 0, 1: 0, 2: 0}
    for g in range(n_games):
        w, _ = play_full_game(game, policies, seed=seed * 10_000 + g)
        wins[w] += 1
    return wins


def policy_dim(game: eng.Game) -> int:
    return 9 if game.name == "classic" else 81


def action_index(game: eng.Game, action) -> int:
    if game.name == "classic":
        return action
    cell, number = action
    return cell * 9 + (number - 1)


def action_from_index(game: eng.Game, idx: int):
    if game.name == "classic":
        return idx
    cell, number = divmod(idx, 9)
    return (cell, number + 1)


def featurize(game: eng.Game, board: eng.Board, player: int) -> np.ndarray:
    """20-dim ownership features per cell (mine/theirs) plus whose turn it is."""
    x = np.zeros(20, dtype=np.float64)
    for cell, v in enumerate(board):
        if v == 0:
            continue
        owner = v if game.name == "classic" else (1 if v % 2 == 1 else 2)
        x[cell * 2 + (owner - 1)] = 1.0
    x[18 + (player - 1)] = 1.0
    return x


@dataclass
class TinyNet:
    """One hidden layer, policy + value heads over 2 players; see the gradient check at the bottom of this file."""
    game: eng.Game
    hidden_dim: int = 24
    lr: float = 0.15
    seed: int = 0
    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    Wp: np.ndarray = field(init=False)
    bp: np.ndarray = field(init=False)
    Wv: np.ndarray = field(init=False)
    bv: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        pdim = policy_dim(self.game)
        scale_in = (1.0 / 20) ** 0.5
        scale_h = (1.0 / self.hidden_dim) ** 0.5
        self.W1 = rng.normal(0, scale_in, size=(20, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.Wp = rng.normal(0, scale_h, size=(self.hidden_dim, pdim))
        self.bp = np.zeros(pdim)
        self.Wv = rng.normal(0, scale_h, size=(self.hidden_dim, 2))
        self.bv = np.zeros(2)

    @staticmethod
    def _softmax(logits: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        z = logits.copy()
        if mask is not None:
            z = np.where(mask, z, -1e9)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def forward(self, board: eng.Board, player: int) -> dict:
        x = featurize(self.game, board, player)
        pre_h = x @ self.W1 + self.b1
        h = np.maximum(pre_h, 0.0)
        policy_logits = h @ self.Wp + self.bp
        value_logits = h @ self.Wv + self.bv
        legal = self.game.legal_moves(board)
        idxs = [action_index(self.game, a) for a in legal]
        mask = np.zeros(policy_dim(self.game), dtype=bool)
        mask[idxs] = True
        policy = self._softmax(policy_logits, mask=mask)
        value = self._softmax(value_logits)
        return dict(x=x, pre_h=pre_h, h=h, policy=policy, value=value, mask=mask,
                    legal=legal, idxs=idxs)

    def predict(self, board: eng.Board, player: int) -> Tuple[Dict, Dict[int, float]]:
        out = self.forward(board, player)
        prior = {a: float(out["policy"][i]) for a, i in zip(out["legal"], out["idxs"])}
        value = {p: float(out["value"][p - 1]) for p in eng.PLAYERS}
        return prior, value

    def backward(self, cache: dict, pi_target: np.ndarray, z_target: np.ndarray) -> dict:
        dpolicy_logits = cache["policy"] - pi_target
        dvalue_logits = cache["value"] - z_target
        dh = dpolicy_logits @ self.Wp.T + dvalue_logits @ self.Wv.T
        dpre_h = dh * (cache["pre_h"] > 0)
        return dict(
            dWp=np.outer(cache["h"], dpolicy_logits), dbp=dpolicy_logits,
            dWv=np.outer(cache["h"], dvalue_logits), dbv=dvalue_logits,
            dW1=np.outer(cache["x"], dpre_h), db1=dpre_h,
        )

    def step(self, grads: dict) -> None:
        self.W1 -= self.lr * grads["dW1"]
        self.b1 -= self.lr * grads["db1"]
        self.Wp -= self.lr * grads["dWp"]
        self.bp -= self.lr * grads["dbp"]
        self.Wv -= self.lr * grads["dWv"]
        self.bv -= self.lr * grads["dbv"]

    def loss(self, cache: dict, pi_target: np.ndarray, z_target: np.ndarray) -> float:
        eps = 1e-9
        policy_ce = -np.sum(pi_target * np.log(cache["policy"] + eps))
        value_ce = -np.sum(z_target * np.log(cache["value"] + eps))
        return float(policy_ce + value_ce)


def net_as_prior_fn(net: TinyNet) -> mc.PriorFn:
    return lambda game, board, player: net.predict(board, player)[0]


def net_as_eval_fn(net: TinyNet) -> mc.EvalFn:
    return lambda game, board, player: net.predict(board, player)[1][player]


def self_play_training_loop(game: eng.Game, n_games: int, mcts_iterations: int,
                             net: Optional[TinyNet] = None, c: float = 1.4,
                             dirichlet_alpha: float = 0.5, dirichlet_eps: float = 0.25,
                             seed: int = 0) -> Tuple[TinyNet, List[float]]:
    """One self-play game per call: PUCT-guided moves distilled into policy/value targets, one gradient step per game."""
    if net is None:
        net = TinyNet(game=game, seed=seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    losses: List[float] = []
    pdim = policy_dim(game)

    for g in range(n_games):
        board = game.initial_board()
        trace: List[Tuple[eng.Board, int, np.ndarray]] = []
        while not game.is_terminal(board):
            player = game.player_to_move(board)
            base_prior_fn = net_as_prior_fn(net)

            def noisy_prior_fn(gm, b, p, _base=base_prior_fn, _root_board=board):
                base = _base(gm, b, p)
                if b != _root_board:
                    return base
                legal = list(base.keys())
                noise = np_rng.dirichlet([dirichlet_alpha] * len(legal))
                return {a: (1 - dirichlet_eps) * base[a] + dirichlet_eps * n
                        for a, n in zip(legal, noise)}

            m = mc.MCTS(game, mc.make_puct_selector(c=c), net_as_eval_fn(net),
                        noisy_prior_fn, seed=rng.randint(0, 2**31 - 1))
            m.new_tree(board)
            m.run(mcts_iterations, record=False)
            visits = m.visit_distribution()
            total = sum(visits.values()) or 1
            pi = np.zeros(pdim)
            for a, n in visits.items():
                pi[action_index(game, a)] = n / total
            trace.append((board, player, pi))
            action = max(visits.items(), key=lambda kv: kv[1])[0]
            board = game.apply_move(board, action, player)

        outcome = game.reward(board)
        z_full = np.array([outcome[p] for p in eng.PLAYERS])
        game_losses = []
        for b, player, pi in trace:
            cache = net.forward(b, player)
            game_losses.append(net.loss(cache, pi, z_full))
            grads = net.backward(cache, pi, z_full)
            net.step(grads)
        losses.append(float(np.mean(game_losses)) if game_losses else 0.0)

    return net, losses


def self_check() -> None:
    """Opponent-ladder sanity check plus a finite-difference gradient check, both games."""
    rng = random.Random(0)
    w, hist = play_full_game(eng.CLASSIC, {1: random_policy, 2: heuristic_policy}, seed=1)
    assert w in (0, 1, 2)
    wins = round_robin_eval(eng.CLASSIC, {1: random_policy, 2: heuristic_policy}, n_games=6, seed=0)
    assert sum(wins.values()) == 6

    for game in (eng.CLASSIC, eng.NUMERICAL):
        net = TinyNet(game=game, seed=0)
        board = game.initial_board()
        player = game.player_to_move(board)
        cache = net.forward(board, player)
        legal = game.legal_moves(board)
        pi = np.zeros(policy_dim(game))
        for a in legal[: max(1, len(legal) // 2)]:
            pi[action_index(game, a)] = 1.0 / max(1, len(legal) // 2)
        z = np.array([0.7, 0.3])
        grads = net.backward(cache, pi, z)

        eps = 1e-5
        rng_np = np.random.default_rng(1)
        for name, param, grad in (("W1", net.W1, grads["dW1"]), ("Wp", net.Wp, grads["dWp"]),
                                   ("Wv", net.Wv, grads["dWv"])):
            idxs = [tuple(rng_np.integers(0, s) for s in param.shape) for _ in range(4)]
            for idx in idxs:
                orig = param[idx]
                param[idx] = orig + eps
                loss_plus = net.loss(net.forward(board, player), pi, z)
                param[idx] = orig - eps
                loss_minus = net.loss(net.forward(board, player), pi, z)
                param[idx] = orig
                numeric = (loss_plus - loss_minus) / (2 * eps)
                analytic = grad[idx]
                assert abs(numeric - analytic) < 1e-3, (game.name, name, idx, numeric, analytic)

    cfg = PlayConfig(n_iterations=6, mode="uct", prior_style="uniform", eval_style="rollout", seed=2)
    m = build_mcts_from_playconfig(cfg, eng.CLASSIC)
    m.new_tree(eng.CLASSIC.initial_board())
    m.run(cfg.n_iterations, record=False)
    assert m.best_action("visits") is not None

    cfg2 = PlayConfig(n_iterations=6, mode="puct", prior_style="band", eval_style="constant",
                       values={1: 0.6, 2: 0.4}, seed=2)
    m2 = build_mcts_from_playconfig(cfg2, eng.NUMERICAL)
    m2.new_tree(eng.NUMERICAL.initial_board())
    m2.run(cfg2.n_iterations, record=False)
    assert m2.best_action("visits") is not None

    graded = PlayConfig(use_graded_budget=True, graded_schedule=(30, 28, 24, 20))
    assert [graded.iterations_for_move(i) for i in range(6)] == [30, 28, 24, 20, 20, 20]

    print("scenarios self-check passed (opponent ladder + gradient check + PlayConfig wiring, both games)")


if __name__ == "__main__":
    self_check()
