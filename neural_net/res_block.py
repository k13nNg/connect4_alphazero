import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    '''
    A class representing the residual block in AlphaGo Zero's neural network, structured as described in AlphaGo Zero paper

    "
    Each residual block applies the following modules sequentially to its input:
        1. A convolution of 256 filters of kernal size 3x3 with stride 1
        2. Batch normalization
        3. A rectifier nonlinearity (ReLU)
        4. A convolution of 256 filters of kernel size 3x3 with stride 1
        5. Batch normalization
        6. A skip connection that adds the input to the block
        7. A rectifier nonlinearity (ReLU)
    "
    '''

    def __init__(self, in_planes=256, planes=None, stride=1):
        super().__init__()
        planes = planes or in_planes

        self.in_planes = in_planes
        self.stride = stride
        self.planes = planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.skip_connection = nn.Sequential()
    
    def forward(self, x: torch.Tensor):
        '''
        Pass the input through the series of operations specified above
        '''
        # 1. A convolution of 256 filters of kernal size 3x3 with stride 1
        out = self.conv1(x)

        # 2. Batch normalization
        out = self.bn1(out)

        # 3. A rectifier nonlinearity (ReLU)
        out = F.relu(out)

        # 4. A convolution of 256 filters of kernel size 3x3 with stride 1
        out = self.conv2(out)

        # 5. Batch normalization
        out = self.bn2(out)

        # 6. A skip connection that adds the input to the block
        out += self.skip_connection(x)

        # 7. A rectifier nonlinearity (ReLU)
        out = F.relu(out)

        return out
