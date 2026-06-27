import numpy as np
import torch
from game_engine.game import Connect4, GameState
from neural_net.network import AlphaZeroNetwork
from mcts.mcts import MCTS
from config import Config

def play_one_game(network: AlphaZeroNetwork, mcts: MCTS, config: Config) -> list:
    '''
    Play one full game of Connect4 using MCTS.
    
    Returns a list of (state, mcts_policy, outcome) tuples,
    one for each move in the game.
    '''
    state = Connect4.initial_state()
    history = []    # list of (state, mcts_policy) — outcome added later

    while True:
        # run MCTS to get visit counts
        visit_counts = mcts.search(state)

        # convert visit counts to policy (normalize to sum to 1)
        mcts_policy = visit_counts / visit_counts.sum()

        # pick a move
        # use temperature: sample proportionally early, greedy later
        if len(history) < config.TEMPERATURE_THRESHOLD:
            col = int(np.random.choice(7, p=mcts_policy))   # EXPLORE early (sample)
        else:
            col = int(np.argmax(mcts_policy))               # EXPLOIT late (greedy)

        # store (state, mcts_policy) — outcome unknown until game ends
        history.append((state, mcts_policy))

        # apply the move
        last_player = state.current_player
        state = Connect4.make_move(state, col)

        # check if game is over
        done, value = Connect4.is_terminal(state, last_player)
        if done:
            break

    # assign outcomes to every position in history
    return assign_outcomes(history, value, last_player)

def assign_outcomes(history, value, last_player) -> list:
    '''
    Go through history and assign the correct outcome to each position.
    
    value    = outcome for last_player (+1 win, -1 loss, 0 draw)
    history  = list of (state, mcts_policy)
    
    Returns list of (state, mcts_policy, outcome)
    '''
    result = []

    for state, mcts_policy in reversed(history):
        result.append((state, mcts_policy, value))
        value = -value

    return result
