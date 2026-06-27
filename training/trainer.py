import torch
import torch.nn.functional as F
from neural_net.network import AlphaZeroNetwork
from training.replay_buffer import ReplayBuffer
from config import Config

class Trainer:
    def __init__(self, network: AlphaZeroNetwork, config: Config):
        self.network = network
        self.config = config
        self.optimizer = torch.optim.Adam(  network.parameters(),
                                            lr = config.LEARNING_RATE, 
                                            weight_decay=config.WEIGHT_DECAY)   # Adam optimizer, use config.LEARNING_RATE and config.WEIGHT_DECAY

    def train(self, replay_buffer: ReplayBuffer):
        '''
        Sample minibatches from replay buffer and update network weights.
        Run for NUM_EPOCHS epochs.
        '''
        self.network.train()   # set network to training mode

        for epoch in range(self.config.NUM_EPOCHS):
            # 1. sample a minibatch
            states, policies, outcomes = replay_buffer.sample(self.config.BATCH_SIZE)

            # 2. forward pass
            pred_policies, pred_values = self.network.forward(states)

            # 3. compute losses
            policy_loss = F.cross_entropy(pred_policies, policies)   # CrossEntropy
            value_loss  = F.mse_loss(pred_values.squeeze(), outcomes)   # MSE

            total_loss = policy_loss + value_loss

            # 4. backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return total_loss.item()
