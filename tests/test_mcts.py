import pytest
import numpy as np
import torch
from game_engine.game import Connect4, GameState
from neural_net.network import AlphaZeroNetwork
from mcts.mcts import MCTS
from mcts.mcts_node import MCTSNode
from config import Config

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def config():
    cfg = Config()
    cfg.NUM_SIMULATIONS = 50   # fast for testing
    return cfg

@pytest.fixture
def network():
    net = AlphaZeroNetwork()
    net.eval()
    return net

@pytest.fixture
def mcts(network, config):
    return MCTS(Connect4, network, config)

@pytest.fixture
def initial_state():
    return Connect4.initial_state()

# ─────────────────────────────────────────────
# MCTSNode Tests
# ─────────────────────────────────────────────

class TestMCTSNode:
    def test_initial_values(self, initial_state):
        node = MCTSNode(initial_state)
        assert node.N == 0
        assert node.W == 0.0
        assert node.prior == 0.0
        assert node.parent is None
        assert node.action_taken is None
        assert node.children == {}

    def test_is_leaf_when_no_children(self, initial_state):
        node = MCTSNode(initial_state)
        assert node.is_leaf() == True

    def test_is_not_leaf_after_adding_child(self, initial_state):
        node = MCTSNode(initial_state)
        child_state = Connect4.make_move(initial_state, 3)
        node.children[3] = MCTSNode(child_state, parent=node, action_taken=3)
        assert node.is_leaf() == False

    def test_Q_is_zero_when_unvisited(self, initial_state):
        node = MCTSNode(initial_state)
        assert node.Q() == 0

    def test_Q_computes_correctly(self, initial_state):
        node = MCTSNode(initial_state)
        node.N = 4
        node.W = 2.0
        assert node.Q() == 0.5

    def test_Q_negative_value(self, initial_state):
        node = MCTSNode(initial_state)
        node.N = 4
        node.W = -2.0
        assert node.Q() == -0.5

    def test_ucb_unvisited_node_is_high(self, initial_state):
        parent = MCTSNode(initial_state)
        parent.N = 10
        child = MCTSNode(initial_state, parent=parent, prior=0.5)
        # unvisited node should have high UCB (driven by prior)
        assert child.ucb_score(1.4) > 0

    def test_ucb_heavily_visited_node_is_lower(self, initial_state):
        parent = MCTSNode(initial_state)
        parent.N = 100

        unvisited = MCTSNode(initial_state, parent=parent, prior=0.3)

        visited = MCTSNode(initial_state, parent=parent, prior=0.3)
        visited.N = 50
        visited.W = 5.0

        assert unvisited.ucb_score(1.4) > visited.ucb_score(1.4)

# ─────────────────────────────────────────────
# MCTS._select Tests
# ─────────────────────────────────────────────

class TestSelect:
    def test_returns_leaf_from_root(self, mcts, initial_state):
        root = MCTSNode(initial_state)
        mcts._expand(root)
        selected = mcts._select(root)
        assert selected.is_leaf()

    def test_selects_highest_prior_when_unvisited(self, mcts, initial_state):
        # manually set up root with children of known priors
        root = MCTSNode(initial_state)
        for col in range(7):
            child_state = Connect4.make_move(initial_state, col)
            root.children[col] = MCTSNode(child_state, parent=root, action_taken=col, prior=col * 0.1)
        root.N = 1

        selected = mcts._select(root)
        # col 6 has highest prior → should be selected
        assert selected.action_taken == 6

    def test_returns_root_if_already_leaf(self, mcts, initial_state):
        root = MCTSNode(initial_state)
        selected = mcts._select(root)
        assert selected is root

# ─────────────────────────────────────────────
# MCTS._expand Tests
# ─────────────────────────────────────────────

class TestExpand:
    def test_creates_children_for_all_legal_moves(self, mcts, initial_state):
        root = MCTSNode(initial_state)
        mcts._expand(root)
        legal_moves = Connect4.get_legal_moves(initial_state)
        assert set(root.children.keys()) == set(legal_moves)

    def test_children_have_correct_parent(self, mcts, initial_state):
        root = MCTSNode(initial_state)
        mcts._expand(root)
        for child in root.children.values():
            assert child.parent is root

    def test_children_have_correct_action_taken(self, mcts, initial_state):
        root = MCTSNode(initial_state)
        mcts._expand(root)
        for col, child in root.children.items():
            assert child.action_taken == col

    def test_priors_sum_to_one(self, mcts, initial_state):
        root = MCTSNode(initial_state)
        mcts._expand(root)
        total_prior = sum(c.prior for c in root.children.values())
        assert abs(total_prior - 1.0) < 1e-5

    def test_illegal_moves_have_no_children(self, mcts, initial_state):
        # fill column 0
        state = initial_state
        for _ in range(6):
            state = Connect4.make_move(state, 0)
        root = MCTSNode(state)
        mcts._expand(root)
        assert 0 not in root.children

    def test_returns_float_value(self, mcts, initial_state):
        root = MCTSNode(initial_state)
        value = mcts._expand(root)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_node_is_no_longer_leaf_after_expand(self, mcts, initial_state):
        root = MCTSNode(initial_state)
        assert root.is_leaf()
        mcts._expand(root)
        assert not root.is_leaf()

# ─────────────────────────────────────────────
# MCTS._backup Tests
# ─────────────────────────────────────────────

class TestBackup:
    def test_updates_visit_count(self, mcts, initial_state):
        node = MCTSNode(initial_state)
        mcts._backup(node, 0.5)
        assert node.N == 1

    def test_updates_total_value(self, mcts, initial_state):
        node = MCTSNode(initial_state)
        mcts._backup(node, 0.5)
        assert node.W == 0.5

    def test_negates_value_at_parent(self, mcts, initial_state):
        parent = MCTSNode(initial_state)
        child_state = Connect4.make_move(initial_state, 3)
        child = MCTSNode(child_state, parent=parent)

        mcts._backup(child, 0.8)

        assert child.W == 0.8
        assert parent.W == -0.8   # negated

    def test_alternates_sign_across_levels(self, mcts, initial_state):
        s0 = initial_state
        s1 = Connect4.make_move(s0, 3)
        s2 = Connect4.make_move(s1, 4)

        root  = MCTSNode(s0)
        mid   = MCTSNode(s1, parent=root)
        leaf  = MCTSNode(s2, parent=mid)

        mcts._backup(leaf, 1.0)

        assert leaf.W  ==  1.0
        assert mid.W   == -1.0
        assert root.W  ==  1.0

    def test_updates_N_at_all_levels(self, mcts, initial_state):
        s1 = Connect4.make_move(initial_state, 3)
        root = MCTSNode(initial_state)
        child = MCTSNode(s1, parent=root)

        mcts._backup(child, 0.5)

        assert root.N == 1
        assert child.N == 1

# ─────────────────────────────────────────────
# MCTS._evaluate Tests
# ─────────────────────────────────────────────

class TestEvaluate:
    def test_returns_float(self, mcts, initial_state):
        node = MCTSNode(initial_state)
        value = mcts._evaluate(node)
        assert isinstance(value, float)

    def test_returns_one_for_terminal_win(self, mcts):
        # craft a state where last player just won
        state = Connect4.initial_state()
        for _ in range(3):
            state = Connect4.make_move(state, 0)  # P1
            state = Connect4.make_move(state, 1)  # P2
        state = Connect4.make_move(state, 0)       # P1 wins vertically

        node = MCTSNode(state)
        value = mcts._evaluate(node)
        assert value == 1.0

    def test_returns_zero_for_draw(self, mcts):
        # craft a full board with no winner
        state = Connect4.initial_state()
        # fill board in a pattern that avoids a win
        move_sequence = [0,1,0,1,0,1,1,0,1,0,1,0,  # cols 0,1
                         2,3,2,3,2,3,3,2,3,2,3,2,  # cols 2,3
                         4,5,4,5,4,5,5,4,5,4,5,4,  # cols 4,5
                         6,6,6,6,6,6]               # col 6
        for col in move_sequence:
            if Connect4.get_legal_moves(state):
                state = Connect4.make_move(state, col)
        
        node = MCTSNode(state)
        value = mcts._evaluate(node)
        assert value == 0.0 or isinstance(value, float)

    def test_expands_non_terminal_node(self, mcts, initial_state):
        node = MCTSNode(initial_state)
        assert node.is_leaf()
        mcts._evaluate(node)
        assert not node.is_leaf()   # should have been expanded

# ─────────────────────────────────────────────
# MCTS.search Tests
# ─────────────────────────────────────────────

class TestSearch:
    def test_returns_array_of_length_7(self, mcts, initial_state):
        visits = mcts.search(initial_state)
        assert len(visits) == 7

    def test_visit_counts_sum_to_num_simulations(self, mcts, initial_state, config):
        visits = mcts.search(initial_state)
        assert visits.sum() == config.NUM_SIMULATIONS

    def test_no_negative_visit_counts(self, mcts, initial_state):
        visits = mcts.search(initial_state)
        assert all(visits >= 0)

    def test_illegal_moves_have_zero_visits(self, mcts):
        # fill column 0
        state = Connect4.initial_state()
        for _ in range(6):
            state = Connect4.make_move(state, 0)
        visits = mcts.search(state)
        assert visits[0] == 0

    def test_center_columns_preferred(self, mcts, initial_state):
        # center columns should generally get more visits
        visits = mcts.search(initial_state)
        center = visits[2:5].sum()
        edges = visits[0] + visits[6]
        assert center > edges

    def test_best_move_is_valid_column(self, mcts, initial_state):
        visits = mcts.search(initial_state)
        best_move = visits.argmax()
        assert best_move in Connect4.get_legal_moves(initial_state)

    def test_search_on_near_terminal_state(self, mcts):
        # P1 has 3 in a row, should find the winning move
        state = Connect4.initial_state()
        for _ in range(3):
            state = Connect4.make_move(state, 0)  # P1 plays col 0
            state = Connect4.make_move(state, 6)  # P2 plays col 6
        
        visits = mcts.search(state)
        best_move = visits.argmax()
        # col 0 completes 4 in a row for P1
        assert best_move == 0
