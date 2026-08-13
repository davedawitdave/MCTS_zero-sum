# MCTS, UCT, and PUCT: two 2-player games, one search

A presentation-ready notebook where one Monte Carlo Tree Search implementation — lazy
expansion, UCT/PUCT selection, negamax backprop — explains itself on two different games,
an interactive config panel, and then plays against you live.

## The two games

| | Classic tic-tac-toe | Numerical (Graham) tic-tac-toe |
|---|---|---|
| Marks | X vs O | player 1 = odds {1,3,5,7,9}, player 2 = evens {2,4,6,8} |
| Win condition | three in a row | any line sums to exactly 15 |
| Notable | draw with perfect play on both sides | first player has a forced win with perfect play |

Both are 2-player and zero-sum, so a single scalar `(N, W)` per node and a
flip-the-sign backup at every step toward the root is all that's needed — no
Max^N reward vector required.

## How to play

1. Install dependencies and open the notebook (see [Running it](#running-it) below), then
   run every cell top to bottom at least once.
2. Scroll to **"Configure the live search"**. This panel controls both play boards below
   it through one shared `PlayConfig`:
   - `n_iterations` — how much search budget the computer gets per move.
   - `mode` — `uct` (visit-count-only exploration) or `puct` (prior-weighted, AlphaZero-style).
   - `c` — the exploration constant in both formulas below.
   - `tree_reuse` — reroot onto the move actually played instead of restarting the search
     from scratch every turn.
   - `computer_starts` — who moves first.
3. Re-run the **"Play Classic against MCTS"** / **"Play Numerical against MCTS"** cells to
   build (or rebuild) the boards with the panel's current settings.
4. Click an empty square to move (Numerical then asks which of your remaining numbers to
   place there). The line under the board reports an immediate threat — `"immediate threat
   at cell X — computer must block"` or `"no immediate threat"` — before the computer
   replies, so you can check its tactics for yourself.
5. **New game** resets the board and, if `computer_starts` is on, lets the computer take
   the opening move again.

See [Known-good budgets](#known-good-budgets) below for how much search each game actually
needs before the computer stops losing.

## Monte Carlo Tree Search

Monte Carlo Tree Search (MCTS) is a heuristic search algorithm well suited to problems
whose search space is too large to enumerate exhaustively (a full 3x3 tic-tac-toe tree is
small enough for exhaustive minimax, but the same exhaustive recursion on a game like
chess is not — see the notebook's introduction for that comparison).

MCTS builds an asymmetric search tree one simulation at a time, in four phases:

1. **Selection** — from the root, repeatedly pick the highest-scoring action until reaching
   a node with an untried action or a finished game.
2. **Expansion** — create the one new node for that untried action (lazy expansion: the
   tree never grows past what's actually been explored).
3. **Evaluation** — score the new leaf: a rollout to a finished game (either uniformly
   random, or heuristic-guided — see below), or a learned/stand-in value.
4. **Backpropagation** — carry that value back up to the root, flipping it (v -> 1 - v)
   at every step, since the two players alternate and rewards are constant-sum (win = 1,
   draw = 0.5, loss = 0 for whoever's turn it is at that node).

Repeating this thousands of times concentrates the tree's visits on the actions that keep
looking best, so the search spends most of its budget on the lines worth spending it on
instead of exploring everything equally.

### Exploration-exploitation tradeoff: UCT

Classic MCTS picks the action at each node maximizing the Upper Confidence bound applied
to Trees:

```
UCT(s, a) = Q(s, a) + c * sqrt( ln(N(s)) / N(s, a) )
```

- `Q(s, a)` — mean value of the simulations that took action `a` from state `s`, from the
  perspective of the player choosing at `s` (win rate, with draws counted as 0.5).
- `N(s)` — total number of times state `s` has been visited.
- `N(s, a)` — number of times action `a` has been taken from `s`.
- `c` — the exploration constant (`c = 1.414 ~ sqrt(2)` by default here).

The first term favors actions that have performed well so far (exploitation); the second
grows for actions that have been tried relatively rarely (exploration). An action with
`N(s, a) = 0` scores `+inf`, so UCT always tries every legal move at a node at least once
before it can start exploiting any of them — fine for tic-tac-toe's small branching
factor, wasteful in games with many more legal moves per position.

### Prior-guided search: PUCT

The `puct` mode replaces UCT's visit-count-only exploration term with one weighted by a
prior `P(s, a)` over actions (AlphaZero's selection rule):

```
PUCT(s, a) = Q(s, a) + c * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

- `P(s, a)` — the prior probability assigned to action `a` before any of it has been
  searched. This project's `"band"` prior weights center > corner > edge; `"uniform"`
  spreads it flat over legal moves.
- The `1 + N(s, a)` denominator stays finite at `N(s, a) = 0` (unlike UCT's `ln(...)/0`),
  so an untried action gets a bonus scaled by its prior instead of an automatic infinite
  one — the prior, not raw visit-count arithmetic, decides which untried branches get
  attention first.

### Negamax bookkeeping

Every node stores `Q(s, a)` from the perspective of *its own* mover. Since the two players
strictly alternate, the parent's mover is always the opposite player from the child's, so
reading a child's value back out for the parent's selection formula requires the flip
`1 - Q(child)`, matching the same `v -> 1 - v` flip backpropagation applies once per level.
Getting this flip right on both the write side (backprop) and the read side (selection) is
what keeps a single scalar `(N, W)` per node — rather than a separate value per player —
consistent all the way to the root.

## Known-good budgets

Measured by playing many games against the one-ply heuristic opponent, alternating who
moves first:

- **Classic**, `mode="uct"`: draws become the normal outcome (no losses) from roughly
  **~200 iterations** on, using the heuristic-guided rollout below.
- **Classic**, `mode="puct"` with band priors: the same reliability from roughly
  **~50-100 iterations** — PUCT needs less budget because the prior already points it at
  reasonable moves before any search has happened.
- **Numerical**: a much larger branching factor (45 legal moves at the opening vs.
  Classic's 9) means it needs substantially more search before the first player's (odds)
  forced-win advantage holds reliably — around **~5,000 iterations** in testing, with
  noticeably fewer losses even at a few hundred.

The rollout used to evaluate a fresh leaf matters as much as the budget: a uniformly
random playout is noisy enough on a board this small that it under-counts how bad "leave
an open line for the opponent" really is. `PlayConfig`'s default `eval_style` is
`"heuristic_rollout"` — win now, else avoid handing the opponent an immediate win, else
prefer center > corner > edge — which is what gets the budgets above down into double and
triple digits instead of the thousands a uniformly random rollout needs for the same
reliability. The pedagogical "pure UCT" scenario earlier in the notebook still uses a
plain random rollout on purpose, to show what that noise looks like.

## How this was built

1. **Rules first** (`engine.py`) — board, legal moves, win detection for both games behind
   one shared interface, so nothing downstream needs to know which game it's looking at.
2. **Search next** (`mcts.py`) — lazy expansion, UCT and PUCT selection, scalar negamax
   backprop, frame recording so any iteration can be replayed step by step, and both a
   uniformly random and a heuristic-guided rollout evaluator.
3. **Then rendering** (`viz.py`) — board and tree drawing, a two-pass layout that clusters
   the root's children into CENTER/CORNERS/EDGES bands, the Prev/Next/Reset dashboard, and
   the human-vs-MCTS play widgets.
4. **Scenarios on top** (`scenarios.py`) — search configs (including the interactive
   `PlayConfig`), an opponent ladder, and a tiny numpy self-play network, all built only
   from the two modules above.
5. **The notebook** (`mcts_games.ipynb`) — rules -> engine self-checks -> why not
   exhaustive search -> the Monte Carlo method -> pure UCT vs. PUCT with priors -> the
   exploration constant -> tree reuse -> an opponent ladder -> self-play -> a configurable
   search panel -> you vs. MCTS on both games.
6. **Corrections, verified rather than assumed:**
   - A genuine selection bug was found and fixed: `Node.Q()` is defined as the value from
     a node's *own* mover's perspective, but both selectors were plugging `child.Q()`
     straight into the UCT/PUCT formulas instead of `1 - child.Q()` for the *parent's*
     perspective — the formulas themselves were always correct, only which value got
     plugged into `Q(s, a)` was wrong. This alone took pure UCT from losing essentially
     every game to a one-ply heuristic to drawing every one of them at ~1,000 iterations.
   - A heuristic-guided rollout was added and made the default for interactive play,
     closing the remaining gap between "usually draws" and "never loses" at much lower,
     still-interactive iteration counts (see [Known-good budgets](#known-good-budgets)).
   - `TreeBoardDashboard` used to make three separate top-level `display()` calls per
     dashboard (header, tree+board, info line), rendering as three stacked output blocks;
     consolidated into one `VBox` and one `display()` call per dashboard.
   - The human-vs-MCTS widget used to draw the board once at setup and then immediately
     redraw it again if the computer moved first; restructured so any pending computer
     move is computed before the single visible redraw.
   - center/corner/edge priors and the heuristic opponent were re-weighted so edges get a
     small positive prior instead of zero; a one-ply threat scan was added so the
     human-play widget reacts to the player's move before the computer responds; the
     dashboard's frame-stepping was stress-tested by walking Prev/Next/Reset across every
     recorded frame in both directions, not just the first frame a static execution
     happens to touch.
7. **Cleanup pass** — every docstring cut to one precise line, every stray comment removed,
   so the scripts read as rules, search, and rendering, and nothing else.

Every step was checked, not just written: each script carries a `self_check()` exercised
on both games, and the full notebook is executed end to end after every change.

## Structure

```
mcts_games.ipynb          the notebook: rules through you-vs-MCTS, ~20 code cells + markdown
README.md                 this file
scripts/
  engine.py                 rules for both games, action_group/action_label helpers
  mcts.py                    the search: MCTSNode, negamax backprop, UCT/PUCT, lazy
                             expansion, frame recording, random + heuristic rollouts
  viz.py                     board + tree rendering, TreeBoardDashboard, human-vs-MCTS
                             play, sweep plots
  scenarios.py               scenario configs, PlayConfig, opponent-ladder policies, the
                             tiny self-play network
```

The notebook imports from `scripts/`, so keep the four `.py` files in a `scripts/`
subfolder next to `mcts_games.ipynb`. The `.ipynb` is edited and re-executed directly —
there's no separate build script generating it.

## Running it

```
pip install numpy matplotlib ipywidgets nbformat jupyter
jupyter notebook mcts_games.ipynb
```

Run cells top to bottom. The notebook ships with outputs already populated from a verified
execution, so it's readable before you run anything — the dashboard and play widgets just
need a live kernel re-run to respond to clicks.

## Self-checks

```
python3 scripts/engine.py
python3 scripts/mcts.py
python3 scripts/scenarios.py
```

Each prints a short confirmation: engine rules on both games, a visit-conservation
invariant after every search (plus `PlayConfig` wiring through both selectors and all
three leaf-evaluation styles), and a finite-difference gradient check on the self-play
network's hand-derived backward pass. These check structure, not playing strength — the
Q-perspective bug above passed every one of them, which is exactly why it's worth playing
real games against a real opponent before trusting a search implementation.
