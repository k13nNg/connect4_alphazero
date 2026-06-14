# Connect4 AlphaZero

A Python implementation of the AlphaZero algorithm applied to Connect Four.

The goal is to train an agent that learns to play Connect Four at a strong level purely through self-play — no hardcoded strategies or human game data, just a neural network and a search algorithm improving each other over time.

## How it works

AlphaZero combines two components:

- **Monte Carlo Tree Search (MCTS)** — guides move selection by simulating future game states
- **Neural network** — evaluates board positions and suggests move probabilities, replacing the random rollouts used in traditional MCTS

The training loop runs in three stages per iteration:
1. **Self-play** — the current network plays games against itself, guided by MCTS, generating training data
2. **Training** — the network is updated on the self-play data to better predict winning moves and outcomes
3. **Evaluation** — the updated network is pitted against the previous best; it replaces the best model only if it wins convincingly

Over many iterations, the network and search progressively improve together.