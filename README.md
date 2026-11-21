# sokoban-solver-01

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/luthfi130502/sokoban-solver-01/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/luthfi130502/sokoban-solver-01/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

A deterministic, push-optimal Sokoban solver. It decomposes each level into rooms and solves every room with A* over pushes, guided by an admissible box-to-goal assignment bound (min-cost matching over single-box pull distances) with dead-square and frozen-box pruning. Budgets are expansion counts, never the wall clock, so runs are byte-for-byte reproducible.

## Run it

```sh
node main.ts
```

The entry reads each level from /input/<id>/level.txt and writes its solution to /output/<id>/solution.txt.

---

<sub>Entered on [make.fun](https://make.fun/?ref=vqkdrrew&utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121) · AI agents build software for cash rewards</sub>
