"""Scenario presets, opponent policies, and a tiny trainable policy/value net.

Why a hand-rolled linear-algebra network instead of a framework
--------------------------------------------------------------------
3-player tic-tac-toe's state space is small enough that a one-hidden-layer
network trained with plain numpy is enough to demonstrate the *mechanism*
AlphaZero uses -- self-play, MCTS as a policy-improvement operator, and
distillation of the improved visit distribution back into the network --
without adding a deep-learning framework dependency to a notebook that is
meant to run anywhere. The network is smaller than AlphaZero's; the loop
around it (self-play -> MCTS-improved targets -> gradient step -> repeat)
is the same loop, not a simplified stand-in for it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import scripts.ttt3_engine as eng
import scripts.mcts_core as mc
from scripts.ttt3_engine import Board, PLAYERS

# ---------------------------------------------------------------------------
# Baseline scenario configs (scenarios 1 and 2 from the write-up, now run to
# real depth instead of stopping at the root).
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    name: str
    description: str
    mode: str = "uct"          # "uct" | "puct"
    c: float = 1.4
    fpu: float = 0.0
    n_iterations: int = 24
    seed: int = 0
    priors: Optional[Dict[int, float]] = None
    values: Optional[Dict[int, float]] = None


SCENARIO_1_PURE_MCTS = ScenarioConfig(
    name="Pure MCTS (UCT)",
    description=(
        "No priors, no learned value. Every freshly created leaf is evaluated "
        "by playing a uniformly random game out to a finished result."
    ),
    mode="uct", c=1.4, n_iterations=26, seed=7,
)

SCENARIO_2_PUCT = ScenarioConfig(
    name="PUCT with hand-set priors",
    description=(
        "Center is believed strongest, corners next, edges weakest -- a "
        "stand-in for what a trained policy network would output -- and "
        "leaves are evaluated directly by a stand-in value function instead "
        "of a rollout."
    ),
    mode="puct", c=1.4, n_iterations=26, seed=1,
    priors={4: 0.50, 0: 0.125, 2: 0.125, 6: 0.125, 8: 0.125},
    values={1: 0.62, 2: 0.20, 3: 0.18},
)


def build_mcts(cfg: ScenarioConfig) -> mc.MCTS:
    if cfg.mode == "uct":
        score_fn = mc.make_uct_selector(c=cfg.c)
        prior_fn = mc.uniform_prior
        eval_fn = mc.make_random_rollout_eval(random.Random(cfg.seed))
    elif cfg.mode == "puct":
        score_fn = mc.make_puct_selector(c=cfg.c, fpu=cfg.fpu)
        prior_fn = mc.make_constant_prior(cfg.priors or {})
        eval_fn = mc.make_constant_value(cfg.values or {p: 1 / 3 for p in PLAYERS})
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")
    return mc.MCTS(score_fn=score_fn, eval_fn=eval_fn, prior_fn=prior_fn, seed=cfg.seed)


# ---------------------------------------------------------------------------
# Opponent policies, weakest to strongest, for the opponent-strength scenario.
# Each has the same signature: policy(board, player, rng) -> action.
# ---------------------------------------------------------------------------

def random_policy(board: Board, player: int, rng: random.Random) -> int:
    return rng.choice(eng.legal_moves(board))


def heuristic_policy(board: Board, player: int, rng: random.Random) -> int:
    """Win now if possible; else avoid moves that hand the next player an
    immediate win; else random among what's left."""
    moves = eng.legal_moves(board)
    for a in moves:
        if eng.check_winner(eng.apply_move(board, a, player)) == player:
            return a
    next_player = PLAYERS[(PLAYERS.index(player) + 1) % 3]

    def hands_opponent_a_win(a: int) -> bool:
        after = eng.apply_move(board, a, player)
        return any(eng.check_winner(eng.apply_move(after, b, next_player)) == next_player
                   for b in eng.legal_moves(after))

    safe = [a for a in moves if not hands_opponent_a_win(a)]
    return rng.choice(safe) if safe else rng.choice(moves)


def make_mcts_policy(n_iterations: int, c: float = 1.4) -> Callable[[Board, int, random.Random], int]:
    def policy(board: Board, player: int, rng: random.Random) -> int:
        m = mc.MCTS(score_fn=mc.make_uct_selector(c=c),
                     eval_fn=mc.make_random_rollout_eval(rng),
                     prior_fn=mc.uniform_prior, seed=rng.randint(0, 2**31 - 1))
        m.new_tree(board)
        m.run(n_iterations, record=False)
        return m.best_action("visits")
    return policy


def play_full_game(policies: Dict[int, Callable[[Board, int, random.Random], int]],
                    seed: int = 0) -> Tuple[int, List[int]]:
    """policies[player] -> a policy(board, player, rng) function for each of 1,2,3.
    Returns (winner, move_history); winner is 0 for a draw."""
    rng = random.Random(seed)
    board = eng.initial_board()
    history: List[int] = []
    while not eng.is_terminal(board):
        p = eng.player_to_move(board)
        a = policies[p](board, p, rng)
        board = eng.apply_move(board, a, p)
        history.append(a)
    return eng.check_winner(board), history


def round_robin_eval(policies: Dict[int, Callable], n_games: int, seed: int = 0
                      ) -> Dict[int, int]:
    """Play n_games full games with a fixed player->policy assignment, return
    win counts per player (key 0 = draws)."""
    wins = {0: 0, 1: 0, 2: 0, 3: 0}
    for g in range(n_games):
        w, _ = play_full_game(policies, seed=seed * 10_000 + g)
        wins[w] += 1
    return wins


# ---------------------------------------------------------------------------
# Tiny policy/value network for the AlphaZero-lite self-play scenario.
# ---------------------------------------------------------------------------

def featurize(board: Board, player: int) -> np.ndarray:
    """30 features: for each of 9 cells, a one-hot over {player1, player2,
    player3} mark presence (27 dims), plus a one-hot of whose turn it is (3
    dims). Empty cells contribute all zeros for their 3 slots."""
    x = np.zeros(30, dtype=np.float64)
    for cell, occ in enumerate(board):
        if occ != eng.EMPTY:
            x[cell * 3 + (occ - 1)] = 1.0
    x[27 + (player - 1)] = 1.0
    return x


@dataclass
class TinyNet:
    """One hidden layer, two heads (policy logits over 9 cells, value logits
    over 3 players). Manually differentiated -- see gradient check in
    ttt3p_selfcheck at the bottom of this file before this is trusted."""
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
        scale_in = (1.0 / 30) ** 0.5
        scale_h = (1.0 / self.hidden_dim) ** 0.5
        self.W1 = rng.normal(0, scale_in, size=(30, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.Wp = rng.normal(0, scale_h, size=(self.hidden_dim, 9))
        self.bp = np.zeros(9)
        self.Wv = rng.normal(0, scale_h, size=(self.hidden_dim, 3))
        self.bv = np.zeros(3)

    @staticmethod
    def _softmax(logits: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        z = logits.copy()
        if mask is not None:
            z = np.where(mask, z, -1e9)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def forward(self, board: Board, player: int) -> dict:
        x = featurize(board, player)
        pre_h = x @ self.W1 + self.b1
        h = np.maximum(pre_h, 0.0)  # ReLU
        policy_logits = h @ self.Wp + self.bp
        value_logits = h @ self.Wv + self.bv
        legal = eng.legal_moves(board)
        mask = np.zeros(9, dtype=bool)
        mask[legal] = True
        policy = self._softmax(policy_logits, mask=mask)
        value = self._softmax(value_logits)
        return dict(x=x, pre_h=pre_h, h=h, policy=policy, value=value, mask=mask)

    def predict(self, board: Board, player: int) -> Tuple[Dict[int, float], Dict[int, float]]:
        out = self.forward(board, player)
        prior = {a: float(out["policy"][a]) for a in eng.legal_moves(board)}
        value = {p: float(out["value"][p - 1]) for p in PLAYERS}
        return prior, value

    def backward(self, cache: dict, pi_target: np.ndarray, z_target: np.ndarray) -> dict:
        """Cross-entropy on both heads. dL/dlogits = pred - target is the
        standard softmax+cross-entropy gradient; masking illegal actions to
        -inf before softmax makes their probability and gradient exactly 0."""
        dpolicy_logits = cache["policy"] - pi_target
        dvalue_logits = cache["value"] - z_target
        dh = dpolicy_logits @ self.Wp.T + dvalue_logits @ self.Wv.T
        dpre_h = dh * (cache["pre_h"] > 0)
        grads = dict(
            dWp=np.outer(cache["h"], dpolicy_logits), dbp=dpolicy_logits,
            dWv=np.outer(cache["h"], dvalue_logits), dbv=dvalue_logits,
            dW1=np.outer(cache["x"], dpre_h), db1=dpre_h,
        )
        return grads

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
    return lambda board, player: net.predict(board, player)[0]


def net_as_eval_fn(net: TinyNet) -> mc.EvalFn:
    return lambda board, player: net.predict(board, player)[1]


def self_play_training_loop(n_games: int, mcts_iterations: int, net: Optional[TinyNet] = None,
                             c: float = 1.4, dirichlet_alpha: float = 0.5, dirichlet_eps: float = 0.25,
                             seed: int = 0) -> Tuple[TinyNet, List[float]]:
    """One game of self-play = one sequence of MCTS-then-distill steps. At
    every move: run MCTS guided by the current net (with Dirichlet noise
    mixed into the root prior so self-play does not collapse onto one line
    too early), take the visit distribution pi as the policy target, store
    (features-context, pi, board, player), then once the game ends every
    stored position gets the actual outcome as its value target and one
    gradient step is taken -- this is the AlphaZero loop, scaled down."""
    if net is None:
        net = TinyNet(seed=seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    losses: List[float] = []

    for g in range(n_games):
        board = eng.initial_board()
        trace: List[Tuple[Board, int, np.ndarray]] = []
        while not eng.is_terminal(board):
            player = eng.player_to_move(board)
            prior_fn = net_as_prior_fn(net)

            def noisy_prior_fn(b, p, _base=prior_fn, _root_board=board):
                base = _base(b, p)
                if b != _root_board:
                    return base
                legal = list(base.keys())
                noise = np_rng.dirichlet([dirichlet_alpha] * len(legal))
                return {a: (1 - dirichlet_eps) * base[a] + dirichlet_eps * n
                        for a, n in zip(legal, noise)}

            m = mc.MCTS(score_fn=mc.make_puct_selector(c=c),
                        eval_fn=net_as_eval_fn(net), prior_fn=noisy_prior_fn,
                        seed=rng.randint(0, 2**31 - 1))
            m.new_tree(board)
            m.run(mcts_iterations, record=False)
            visits = m.visit_distribution()
            total = sum(visits.values()) or 1
            pi = np.zeros(9)
            for a, n in visits.items():
                pi[a] = n / total
            trace.append((board, player, pi))
            action = max(visits.items(), key=lambda kv: kv[1])[0]
            board = eng.apply_move(board, action, player)

        outcome = eng.terminal_reward(board)
        z_full = np.array([outcome[p] for p in PLAYERS])
        game_losses = []
        for b, player, pi in trace:
            cache = net.forward(b, player)
            game_losses.append(net.loss(cache, pi, z_full))
            grads = net.backward(cache, pi, z_full)
            net.step(grads)
        losses.append(float(np.mean(game_losses)) if game_losses else 0.0)

    return net, losses
