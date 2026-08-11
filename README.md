# 3-player tic-tac-toe: MCTS, PUCT, and Max^n

## Structure

```
three_player_mcts.ipynb     the notebook: 15 code cells + markdown, 7 scenarios
build_notebook.py           regenerates the notebook from scratch (source of truth for cell content)
scripts/
  ttt3_engine.py             rules: board, legal moves, win detection, constant-sum rewards, symmetry
  mcts_core.py                the search: MCTSNode, Max^n backprop, UCT/PUCT selection, lazy expansion
  mcts_viz.py                 rendering: board + tree drawing, the TreeBoardDashboard widget, sweep plots
  scenarios.py                scenario configs, opponent policies, the tiny self-play network
```

Every notebook cell configures the shared scripts differently rather than reimplementing
anything locally -- fixing a rendering or search bug means editing one script, not
patching every scenario that touches it.

## Running it

```
pip install numpy matplotlib ipywidgets nbformat jupyter
jupyter notebook three_player_mcts.ipynb
```

Run cells top to bottom. The `TreeBoardDashboard` cells need a live kernel to be
interactive (Next/Prev); the notebook ships with all outputs already populated from a
verified execution, so it is readable even before you run anything, but the buttons
only respond once you re-run that cell yourself.

## Regenerating the notebook

The `.ipynb` file is a build artifact of `build_notebook.py`, not something to hand-edit.
To change a cell, edit `build_notebook.py` and re-run:

```
python3 build_notebook.py
```

## Design decisions and the alternatives considered

See the second markdown cell in the notebook for the full writeup (board size,
Max^n vs. negamax, lazy vs. eager expansion, hand-rolled network vs. a framework),
each with the alternative that was considered and why it was rejected.

## Self-checks

`ttt3_engine.py`, `mcts_core.py`, and `mcts_viz.py` each run a self-check when executed
directly (`python3 scripts/ttt3_engine.py`, etc.), including a constant-sum invariant
assertion on every node after every search and a finite-difference gradient check on
the self-play network's hand-derived backward pass.
