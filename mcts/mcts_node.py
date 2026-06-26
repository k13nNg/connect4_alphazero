from game_engine.game import GameState
import math

class MCTSNode:
    def __init__(self, state: GameState, parent=None, action_taken=None, prior=0.0):
        self.state = state              # GameState
        self.parent = parent            # parent MCTSNode, None if root
        self.action_taken = action_taken  # which col led to this node
        self.children = {}              # col → MCTSNode

        self.N = 0                      # visit count
        self.W = 0.0                    # total value

        self.prior = prior              # P(s,a) from policy head

    def Q(self):
        '''
        Return the mean action value (calculated by total value / visit count)
        If N == 0, return 0
        '''
        return self.W / self.N if self.N > 0 else 0

    def is_leaf(self):
        '''
        Return True if a node is a leaf node, False otherwise
        '''
        return len(self.children) == 0

    def ucb_score(self, c_puct):
        '''
        Return the UCB score of the current node, calculated using the hyperparameter c_puct
        '''
        return self.Q() + c_puct * self.prior * math.sqrt(self.parent.N) / (1 + self.N)
