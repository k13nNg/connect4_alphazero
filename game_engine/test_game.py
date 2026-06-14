"""
Test suite for the Connect4 game engine.
Run with: python -m pytest game_engine/test_game.py -v
"""

import numpy as np
import torch
import pytest
from game_engine.game import GameState, Connect4, ROWS, COLS


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_state(board_list, current_player=1) -> GameState:
    """Build a GameState from a plain Python 2-D list."""
    return GameState(np.array(board_list, dtype=np.int32), current_player)


def drop(state: GameState, col: int) -> GameState:
    """Convenience wrapper — asserts the move is legal."""
    result = Connect4.make_move(state, col)
    assert result is not False, f"Expected legal move in col {col}"
    return result


# ─────────────────────────────────────────────
# initial_state
# ─────────────────────────────────────────────

class TestInitialState:
    def test_board_is_all_zeros(self):
        s = Connect4.initial_state()
        assert np.all(s.board == 0)

    def test_board_shape(self):
        s = Connect4.initial_state()
        assert s.board.shape == (ROWS, COLS)

    def test_player_1_starts(self):
        s = Connect4.initial_state()
        assert s.current_player == 1


# ─────────────────────────────────────────────
# get_legal_moves
# ─────────────────────────────────────────────

class TestGetLegalMoves:
    def test_all_moves_legal_on_empty_board(self):
        s = Connect4.initial_state()
        assert Connect4.get_legal_moves(s) == list(range(COLS))

    def test_full_column_not_legal(self):
        s = Connect4.initial_state()
        # Fill column 3 completely
        for _ in range(ROWS):
            s = drop(s, 3)
        assert 3 not in Connect4.get_legal_moves(s)

    def test_no_moves_on_full_board(self):
        # Fill every column
        s = Connect4.initial_state()
        # Alternate columns to avoid a win
        fill_order = [0, 1, 2, 3, 4, 5, 6] * ROWS
        for col in fill_order[:ROWS * COLS]:
            result = Connect4.make_move(s, col)
            if result is not False:
                s = result
        legal = Connect4.get_legal_moves(s)
        assert legal == []

    def test_partially_filled_column_still_legal(self):
        s = Connect4.initial_state()
        s = drop(s, 0)  # only one piece in col 0
        assert 0 in Connect4.get_legal_moves(s)


# ─────────────────────────────────────────────
# make_move
# ─────────────────────────────────────────────

class TestMakeMove:
    def test_piece_lands_on_bottom_row(self):
        s = Connect4.initial_state()
        s2 = drop(s, 3)
        assert s2.board[ROWS - 1][3] == 1

    def test_piece_stacks_on_previous(self):
        s = Connect4.initial_state()
        s = drop(s, 3)   # player 1 bottom
        s = drop(s, 3)   # player -1 on top
        assert s.board[ROWS - 1][3] == 1
        assert s.board[ROWS - 2][3] == -1

    def test_player_switches_after_move(self):
        s = Connect4.initial_state()
        assert s.current_player == 1
        s = drop(s, 0)
        assert s.current_player == -1
        s = drop(s, 0)
        assert s.current_player == 1

    def test_illegal_move_returns_false(self):
        s = Connect4.initial_state()
        assert Connect4.make_move(s, -1) is False
        assert Connect4.make_move(s, COLS) is False

    def test_full_column_returns_false(self):
        s = Connect4.initial_state()
        for _ in range(ROWS):
            s = drop(s, 0)
        assert Connect4.make_move(s, 0) is False

    def test_does_not_mutate_original_state(self):
        s = Connect4.initial_state()
        board_before = s.board.copy()
        Connect4.make_move(s, 3)
        assert np.array_equal(s.board, board_before)


# ─────────────────────────────────────────────
# check_win
# ─────────────────────────────────────────────

class TestCheckWin:

    # ── Horizontal ────────────────────────────

    def test_horizontal_win_bottom_row(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        board[5][0:4] = 1
        assert Connect4.check_win(board, 1) is True

    def test_horizontal_win_top_row(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        board[0][3:7] = 1
        assert Connect4.check_win(board, 1) is True

    def test_horizontal_no_win_gap(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        board[5][0] = 1
        board[5][1] = 1
        board[5][2] = 0   # gap
        board[5][3] = 1
        assert Connect4.check_win(board, 1) is False

    def test_horizontal_three_in_a_row_not_win(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        board[5][0:3] = 1
        assert Connect4.check_win(board, 1) is False

    # ── Vertical ──────────────────────────────

    def test_vertical_win(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        board[2:6, 0] = 1
        assert Connect4.check_win(board, 1) is True

    def test_vertical_win_top(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        board[0:4, 3] = 1
        assert Connect4.check_win(board, 1) is True

    def test_vertical_three_not_win(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        board[3:6, 0] = 1
        assert Connect4.check_win(board, 1) is False

    # ── Diagonal ↘ ────────────────────────────

    def test_diagonal_down_right_win(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        for i in range(4):
            board[i][i] = 1
        assert Connect4.check_win(board, 1) is True

    def test_diagonal_down_right_bottom_corner(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        for i in range(4):
            board[2 + i][3 + i] = 1
        assert Connect4.check_win(board, 1) is True

    # ── Diagonal ↙ ────────────────────────────

    def test_diagonal_down_left_win(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        for i in range(4):
            board[i][3 - i] = 1
        assert Connect4.check_win(board, 1) is True

    def test_diagonal_down_left_bottom_corner(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        for i in range(4):
            board[2 + i][3 - i] = 1
        assert Connect4.check_win(board, 1) is True

    # ── Player -1 ─────────────────────────────

    def test_opponent_horizontal_win(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        board[5][0:4] = -1
        assert Connect4.check_win(board, -1) is True

    def test_opponent_win_not_detected_for_player_1(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        board[5][0:4] = -1
        assert Connect4.check_win(board, 1) is False

    # ── No false positives ────────────────────

    def test_empty_board_no_win(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        assert Connect4.check_win(board, 1) is False
        assert Connect4.check_win(board, -1) is False

    def test_mixed_pieces_no_win(self):
        # alternating pieces can never form four in a row
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        for c in range(COLS):
            board[5][c] = 1 if c % 2 == 0 else -1
        assert Connect4.check_win(board, 1) is False
        assert Connect4.check_win(board, -1) is False


# ─────────────────────────────────────────────
# is_terminal
# ─────────────────────────────────────────────

class TestIsTerminal:
    def test_not_terminal_on_empty_board(self):
        s = Connect4.initial_state()
        done, val = Connect4.is_terminal(s, 1)
        assert done is False
        assert val == 0

    def test_win_is_terminal(self):
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        board[5][0:4] = 1
        s = make_state(board, current_player=-1)
        done, val = Connect4.is_terminal(s, 1)
        assert done is True
        assert val == 1

    def test_loss_not_detected_for_wrong_player(self):
        # player -1 has four in a row but we ask about player 1
        board = np.zeros((ROWS, COLS), dtype=np.int32)
        board[5][0:4] = -1
        s = make_state(board, current_player=1)
        done, val = Connect4.is_terminal(s, 1)
        assert done is False

    def test_draw_is_terminal(self):
        # Play out a real draw by filling the board column by column
        # interleaved to avoid any wins: fill col 0 top-to-bottom alternating,
        # then repeat for all cols. We manually build a known draw position.
        # Board that is full with no winner (verified by check_win assertions):
        board = np.array([
            [ 1,  1,  1, -1,  1,  1,  1],
            [-1, -1, -1,  1, -1, -1, -1],
            [ 1,  1,  1, -1,  1,  1,  1],
            [-1, -1, -1,  1, -1, -1, -1],
            [ 1,  1,  1, -1,  1,  1,  1],
            [-1, -1, -1,  1, -1, -1, -1],
        ], dtype=np.int32)
        assert Connect4.check_win(board, 1) is False, "test board has a winner for player 1 — fix the board"
        assert Connect4.check_win(board, -1) is False, "test board has a winner for player -1 — fix the board"
        s = make_state(board)
        done, val = Connect4.is_terminal(s, 1)
        assert done is True
        assert val == 0

    def test_game_in_progress_not_terminal(self):
        s = Connect4.initial_state()
        s = drop(s, 3)
        s = drop(s, 3)
        done, _ = Connect4.is_terminal(s, -1)
        assert done is False


# ─────────────────────────────────────────────
# encode
# ─────────────────────────────────────────────

class TestEncode:
    def test_output_shape(self):
        s = Connect4.initial_state()
        t = Connect4.encode(s)
        assert t.shape == (2, ROWS, COLS)

    def test_empty_board_all_zeros(self):
        s = Connect4.initial_state()
        t = Connect4.encode(s)
        assert torch.all(t == 0)

    def test_player1_piece_in_plane_0(self):
        s = Connect4.initial_state()
        s = drop(s, 3)   # player 1 moves
        t = Connect4.encode(s)
        assert t[0][ROWS - 1][3] == 1.0   # plane 0 has player 1's piece
        assert t[1][ROWS - 1][3] == 0.0   # plane 1 does not

    def test_player2_piece_in_plane_1(self):
        s = Connect4.initial_state()
        s = drop(s, 3)   # player 1
        s = drop(s, 4)   # player -1
        t = Connect4.encode(s)
        assert t[1][ROWS - 1][4] == 1.0   # plane 1 has player -1's piece
        assert t[0][ROWS - 1][4] == 0.0   # plane 0 does not

    def test_planes_are_binary(self):
        s = Connect4.initial_state()
        for col in range(COLS):
            s = drop(s, col)
        t = Connect4.encode(s)
        unique_vals = torch.unique(t)
        for v in unique_vals:
            assert v.item() in (0.0, 1.0)

    def test_planes_dont_overlap(self):
        # A cell can't belong to both players
        s = Connect4.initial_state()
        for col in range(COLS):
            s = drop(s, col)
        t = Connect4.encode(s)
        overlap = (t[0] == 1) & (t[1] == 1)
        assert not torch.any(overlap)

    def test_dtype_is_float32(self):
        s = Connect4.initial_state()
        t = Connect4.encode(s)
        assert t.dtype == torch.float32
