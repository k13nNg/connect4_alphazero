from game_engine.game import Connect4
from neural_net.network import AlphaZeroNetwork
from mcts.mcts import MCTS
from config import Config

import numpy as np
import torch

def evaluate(new_network: AlphaZeroNetwork, 
             old_network: AlphaZeroNetwork, 
             config: Config,
             device: str) -> bool:
    '''
    Play NUM_EVAL_GAMES games between new and old network.
    Return True if new network wins more than WIN_THRESHOLD of games.
    '''
    new_wins = 0
    old_wins = 0
    draws = 0

    new_mcts = MCTS(Connect4, new_network, config, device)
    old_mcts = MCTS(Connect4, old_network, config, device)

    for i in range(config.NUM_EVAL_GAMES):
        # alternate who plays first to remove first-mover advantage
        if i % 2 == 0:
            result = play_game(new_mcts, old_mcts, config)
        else:
            result = play_game(old_mcts, new_mcts, config)
            result = -result   # flip result since new_network played second

        if result == 1:
            new_wins += 1
        elif result == -1:
            old_wins += 1
        else:
            draws += 1

    win_rate = new_wins / config.NUM_EVAL_GAMES
    print(f"New: {new_wins} | Old: {old_wins} | Draws: {draws} | Win rate: {win_rate:.2%}")
    return win_rate > config.WIN_THRESHOLD

def play_game(p1_mcts: MCTS, p2_mcts: MCTS, config: Config) -> float:
    '''
    Play one game between two MCTS players.
    Returns +1 if p1 wins, -1 if p2 wins, 0 if draw.
    '''
    state = Connect4.initial_state()

    while True:
        # p1 moves
        visits = p1_mcts.search(state)
        col = int(np.argmax(visits))
        last_player = state.current_player
        state = Connect4.make_move(state, col)

        done, value = Connect4.is_terminal(state, last_player)
        if done:
            return value   # +1 p1 won, -1 p1 lost, 0 draw

        # p2 moves
        visits = p2_mcts.search(state)
        col = int(np.argmax(visits))
        last_player = state.current_player
        state = Connect4.make_move(state, col)

        done, value = Connect4.is_terminal(state, last_player)
        if done:
            return -value  # flip: p2 won means p1 lost
