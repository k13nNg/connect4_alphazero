from .mcts_node import MCTSNode
from game_engine.game import GameState
from neural_net.network import AlphaZeroNetwork

import numpy as np
import torch

class MCTS:
    def __init__(self, game:GameState, network: AlphaZeroNetwork, config, device):
        self.game = game
        self.network = network
        self.config = config
        self.device = device

    def search(self, root_state:GameState):
        '''
        Run NUM_SIMULATIONS simulations from root_state.
        Return visit count distribution over all 7 columns.
        '''
        # switch the network into evaluation mode so we don't mess with the gradients
        self.network.eval()          

        with torch.no_grad():
            root = MCTSNode(root_state)
            self._expand(root)
            self._add_dirichlet_noise(root)

            for _ in range(self.config.NUM_SIMULATIONS):
                node = self._select(root)
                value = self._evaluate(node)
                self._backup(node, value)

        # return visit counts for all 7 columns
        visits = np.zeros(7)
        for col, child in root.children.items():
            visits[col] = child.N
        return visits

    def _select(self, root: MCTSNode) -> MCTSNode:
        '''
        Walk down the tree from root, always picking the child 
        with the highest UCB score, until we reach a leaf.
        '''
        node = root

        while node.is_leaf() == False:
            max_ucb_child = None
            max_ucb_score = -float('inf')

            # iterate through the node's children, pick the one with the highest UCB score
            for c in node.children.values():
                if c.ucb_score(self.config.C_PUCT) > max_ucb_score:
                    max_ucb_child = c
                    max_ucb_score = c.ucb_score(self.config.C_PUCT)

            node = max_ucb_child

        return node

    def _expand(self, node: MCTSNode):
        '''
        Ask the network for (policy, value).
        Mask illegal moves, renormalize policy.
        Create child nodes with priors from policy.
        '''
        state = node.state

        # encode the game state
        encoded_state = self.game.encode(state)
        # add the batch dimension before feeding the encoded state into the network
        input_tensor = encoded_state.unsqueeze(0).float().to(self.device)

        logits, value = self.network(input_tensor)

        # remove the batch dimension from logits
        logits = logits.squeeze(0)

        # make illegal moves
        legal_moves = self.game.get_legal_moves(state)
        mask = torch.full((7,), -float('inf'), device = self.device)
        
        # change the value of legal columns to zero
        for i in legal_moves:
            mask[i] = 0.0
        
        # apply mask to logits
        logits += mask

        # apply softmax to logits to get move probabilities
        move_probs = torch.softmax(logits, dim = 0)

        # create children nodes
        for col in legal_moves:
            child_state = self.game.make_move(state, col)
            node.children[col] = MCTSNode(
                state=child_state,
                parent=node,
                action_taken=col,
                prior=move_probs[col].item()
            )
        
        # return the predicted game value
        return value.item()

    def _evaluate(self, node):
        '''
        If node is terminal, return the terminal value.
        Otherwise expand the node and return the network's value.
        '''
        
        # get the last player of game state, because is_terminal() is checking if the last player won the game
        last_player = -node.state.current_player 

        is_done, value = self.game.is_terminal(node.state, last_player)

        # if the last player won the game, return the value dirently
        if(is_done):
            return value
        
        # else, expand from the current node and return the network's predicted value
        else:
            return self._expand(node)

    def _backup(self, node, value):
        '''
        Walk from node back to root.
        At each step: N += 1, W += value, then negate value.
        '''
        while node != None:
            node.N += 1
            node.W += value
            # flip value when tracing backup the parent, because we alternate players at each level
            value *= -1
            node = node.parent

    def _add_dirichlet_noise(self, root):
        '''
        Add Dirichlet noise to root node priors for exploration.
        '''
        
        # get the list of all children (MCTSNode type) of the root node
        children = list(root.children.values())
        # generate the noise
        noise = np.random.dirichlet([self.config.DIRICHLET_ALPHA] * len(children))
    
        for child, n in zip(children, noise):
            child.prior = (1 - self.config.DIRICHLET_EPSILON) * child.prior + self.config.DIRICHLET_EPSILON * n
