from collections import deque
from game_engine.game import Connect4

import numpy as np
import random
import torch

class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)  # automatically discards oldest when full

    def add_game(self, game_data: list):
        '''
        Add all (state, mcts_policy, outcome) tuples from one game.
        '''
        for i in game_data:
            self.buffer.append(i)

    def sample(self, batch_size: int):
        '''
        Sample a random minibatch.
        Returns (states, policies, outcomes) as numpy arrays.
        '''
        batch = random.sample(self.buffer, batch_size)
        states, policies, outcomes = zip(*batch)

        states = torch.stack([Connect4.encode(s) for s in states])
        policies = torch.tensor(np.array(policies), dtype = torch.float32)
        outcomes = torch.tensor(np.array(outcomes), dtype = torch.float32)

        return states, policies, outcomes

    def __len__(self):
        return len(self.buffer)
