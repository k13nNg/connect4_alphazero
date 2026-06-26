from .res_block import ResBlock

import torch
import torch.nn as nn

class AlphaZeroNetwork(nn.Module):
    '''
    A class representing AlphaGo Zero's neural network, structured as described in AlphaGo Zero paper
    '''

    # Define some constants
    INPUT_DIM = 2
    OUTPUT_DIM = 256

    def __init__(self, num_res_blocks=5, num_channels=256):
        super().__init__()

        '''
        --- The convolutional block ---

        The convolutional block applies the following modules:
            1. A conlution of 256 filters of kernel size 3x3 with stride 1
            2. Batch normalization
            3. A rectifier nonlinearity
        '''
        conv_block = nn.Sequential(
            nn.Conv2d(self.INPUT_DIM, self.OUTPUT_DIM, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(self.OUTPUT_DIM),
            nn.ReLU()
        )

        '''
        --- The residual tower ---

        The residual tower consists of a single convolution block, followed by either 19 or 39 residual blocks
        '''
        # Stack num_res_blocks ResBlocks
        self.res_tower = nn.Sequential(
            conv_block,
            *[ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        '''
        --- The policy head ---
        
        The policy head applies teh following modules:
            1. A convolution of 2 filters of kernel size 1x1 with stride 1
            2. Batch normalization
            3. A rectifier nonlinearity
            4. A fully connected linear layer that outputs a vector of size 7 (you can only choose 1 of the 7 columns to place your chip)
        '''
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1, stride = 1, bias = False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(), # Flatten the tensor so that we can feed it into the linear layer
            nn.Linear(2 * 6 * 7, 7)
        )

        '''
        --- The value head ---

        The value head applies the following modules:
            1. A convolution of 1 filter of kernel size with stride 1
            2. Batch normalization
            3. A rectifier nonlinearity
            4. A fully connected linear layer to a hidden layer of size 256
            5. A rectifier nonlinearity
            6. A fully connected linear layer to a scalar
            7. A tanh nonlinearity outputting a sclar in the range [-1,1]
        '''

        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 6 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        '''
        Do a forward pass of x throughout the network
        '''
        # x shape: (batch, 2, 6, 7)

        # 1. Residual tower
        x = self.res_tower(x)

        # 2. Policy head → return raw logits, no softmax
        logits = self.policy_head(x)

        # 4. Value head → return scalar in [-1, 1]
        output = self.value_head(x)

        # return policy, value
        return (logits, output)
