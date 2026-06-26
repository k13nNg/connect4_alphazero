from network import AlphaZeroNetwork
import torch 

net = AlphaZeroNetwork()
dummy = torch.zeros(1, 2, 6, 7)
policy, value = net(dummy)
assert policy.shape == (1, 7)
assert value.shape == (1, 1)
print("Network OK!")