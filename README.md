# MCTS, UCT, and PUCT: two 2-player games, one search

A presentation-ready notebook where one Monte Carlo Tree Search implementation — lazy
expansion, UCT/PUCT selection, negamax backprop — explains itself on two different games,
then plays against the audience live.

## The two games

| | Classic tic-tac-toe | Numerical (Graham) tic-tac-toe |
|---|---|---|
| Marks | X vs O | player 1 = odds {1,3,5,7,9}, player 2 = evens {2,4,6,8} |
| Win condition | three in a row | any line sums to exactly 15 |
| Notable | the familiar case | first player has a forced win with perfect play |

Both are 2-player and zero-sum, so a single scalar `(N, W)` per node and a
flip-the-sign backup at every step toward the root is all that's needed — no
Max^N reward vector required.

## How this was built

1. **Rules first** (`engine.py`) — board, legal moves, win detection for both games behind
   one shared interface, so nothing downstream needs to know which game it's looking at.
2. **Search next** (`mcts.py`) — lazy expansion, UCT and PUCT selection, scalar negamax
   backprop, and frame recording so any iteration can be replayed step by step.
3. **Then rendering** (`viz.py`) — board and tree drawing, a two-pass layout that clusters
   the root's children into CENTER/CORNERS/EDGES bands, the Prev/Next/Reset dashboard, and
   the human-vs-MCTS play widgets.
4. **Scenarios on top** (`scenarios.py`) — search configs, an opponent ladder, and a tiny
   numpy self-play network, all built only from the three modules above.
5. **The notebook assembled last** (`build_notebook.py`) — an 11-part story: rules → engine
   self-checks → pure UCT vs. PUCT with priors → the exploration constant → tree reuse → an
   opponent ladder → self-play → you vs. MCTS on both games.
6. **Corrections, verified rather than assumed** — center/corner/edge priors and the
   heuristic opponent were re-weighted so edges get a small positive prior instead of zero;
   a one-ply threat scan was added so the human-play widget reacts to the player's move
   before the computer responds; every markdown section was tightened to open with one
   claim; and the dashboard's frame-stepping was stress-tested by walking Prev/Next/Reset
   across *every* recorded frame in both directions, not just the first frame a static
   execution happens to touch.
7. **Cleanup pass** — every docstring cut to one precise line, every stray comment removed,
   so the scripts read as rules, search, and rendering, and nothing else.

Every step was checked, not just written: each script carries a `self_check()` exercised
on both games, and the full notebook is executed end to end after every change.

## Structure

```
mcts_games.ipynb          the notebook: 11-part story, ~15 code cells + markdown
build_notebook.py         regenerates the notebook from scratch (source of truth for cell content)
scripts/
  engine.py                 rules for both games, action_group/action_label helpers
  mcts.py                    the search: MCTSNode, negamax backprop, UCT/PUCT, lazy expansion, frame recording
  viz.py                     board + tree rendering, TreeBoardDashboard, human-vs-MCTS play, sweep plots
  scenarios.py               scenario configs, opponent-ladder policies, the tiny self-play network
```

## Running it

```
pip install numpy matplotlib ipywidgets nbformat jupyter
jupyter notebook mcts_games.ipynb
```

Run cells top to bottom. The notebook ships with outputs already populated from a verified
execution, so it's readable before you run anything — the dashboard and play widgets just
need a live kernel re-run to respond to clicks.

## Regenerating the notebook

The `.ipynb` is a build artifact, not something to hand-edit:

```
python3 build_notebook.py
```

## Self-checks

```
python3 scripts/engine.py
python3 scripts/mcts.py
python3 scripts/scenarios.py
```

Each prints a short confirmation: engine rules on both games, a visit-conservation
invariant after every search, and a finite-difference gradient check on the self-play
network's hand-derived backward pass.
