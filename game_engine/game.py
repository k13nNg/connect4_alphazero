import numpy as np
import torch

ROWS = 6
COLS = 7

class GameState:
    '''
    A class that represents a game state, storing:
        - state of the game
        - current player
    '''
    def __init__(self, state, current_player = 1):
        '''
        Initialize a GameState with:
            - board: a 2D numpy array representing the current board
            - current_player: the player whose turn it is (1 or -1)
        '''
        self.board = state
        self.current_player = current_player

    def __repr__(self):
        '''
        Return a string representation of the board, showing:
            - column indices at the top
            - X for player 1, O for player -1, . for empty cells
            - the player to move at the bottom
        '''
        symbols = {1: 'X', -1: 'O', 0: '.'}
        col_numbers = '  ' + ' '.join(str(c) for c in range(COLS))
        rows = '\n'.join('| ' + ' '.join(symbols[v] for v in row) + ' |' for row in self.board)
        player_str = 'X' if self.current_player == 1 else 'O'
        return f"{col_numbers}\n{rows}\nPlayer to move: {player_str}"

class Connect4:
    '''
    A class that represents the game Connect4
    '''

    def initial_state():
        '''
        Return a GameState object, representing the initial state of the game
        '''
        return GameState(np.zeros((ROWS, COLS), dtype=np.int32))

    def get_legal_moves(current_state: GameState) -> list[int]:
        '''
        Return a list of legal next moves based on current_state
        '''

        board = current_state.board
        top_row = board[0]
        result = []

        for col in range(COLS):
            if top_row[col] == 0:
                result.append(col)

        return result

    def make_move(current_state: GameState, col: int) -> GameState | bool:
        '''
        Return: 
            - a GameState represents the state of the game after making move col in current_state, or
            - False if the move is not legal
        '''

        # create a copy of current_state.board so we don't mess with the state's board
        # get current_player from current_state.current_player
        board = current_state.board.copy()
        current_player = current_state.current_player

        # check if the move is legal
        if not (col in Connect4.get_legal_moves(current_state)):
            return False
        
        else:
            # iterate through the rows in a specified column from the bottom up
            # find the first empty cell
            # set the value of that cell to current_state.current_player (mimicking a move made by current_player)
            # set current.current_player = -current.current_player (switching turn) 
            # end loop
            for row in range(ROWS-1, -1, -1):
                if (board[row][col] == 0):
                    board[row][col] = current_player
                    current_player *= -1
                    break
            
            next_state = GameState(board, current_player)

            return next_state

    def check_win(board: np.ndarray, player: int) -> bool:
        '''
        Return True if player "player" wins in the board "board", False otherwise
        '''

        # construct the check matrices
        horizontal_check_matrix = np.array([[1, 1, 1, 1]], dtype=np.int32)
        vertical_check_matrix = np.array([[1],
                                            [1],
                                            [1],
                                            [1]
                                        ], dtype=np.int32)
        diagonal_1_check_matrix = np.identity(4, dtype=np.int32)
        diagonal_2_check_matrix = np.rot90(diagonal_1_check_matrix, k = 1)

        def check_direction(kernel: np.ndarray, player: int) -> bool:
            '''
            Return True if there's a win in the direction that kernel is checking for
            
            Return False otherwise
            '''

            kernel_height, kernel_width = kernel.shape

            # convolve kernel with the board
            for y in range(ROWS - kernel_height + 1):
                for x in range(COLS - kernel_width + 1):
                    window = board[y: y + kernel_height, x: x + kernel_width]

                    # take the Hadamard product between the kernel and the window, then check if all non-zero entries are 
                    # equal to 4*player 
                    # you either have +4 or -4, empty entries do not count
                    if np.sum(window * kernel) == 4 * player:
                        return True
            
            return False

        # check if there's a win for player
        result = check_direction(horizontal_check_matrix, player) or \
                    check_direction(vertical_check_matrix, player) or \
                    check_direction(diagonal_1_check_matrix, player) or \
                    check_direction(diagonal_2_check_matrix, player)

        return result

    def is_terminal(state: GameState, last_player: int) -> tuple[bool, int]:
        '''
        Return:
            - (True, 1) if last_player won the game 
            - (True, 0) if state is a draw_game
            - (False, 0) if the game is still going
        '''

        if (Connect4.check_win(state.board, last_player)):
            return (True, 1)
        
        elif (len(Connect4.get_legal_moves(state)) == 0):
            return (True, 0)
        
        else:
            return (False, 0)

    def encode(state: GameState) -> torch.Tensor:
        '''
        Return a 2x6x7 tensor represents the game state

        Encode the game board into a 2x6x7 tensor
            - Each layer of the tensor has 1 for the pieces of one player and 0 otherwise

        '''
        board = state.board
        result = torch.zeros((2, ROWS, COLS), dtype = torch.float32)

        for row in range(ROWS):
            for col in range(COLS):
                if (board[row][col] == 1):
                    result[0][row][col] = 1

                if (board[row][col] == -1):
                    result[1][row][col] = 1
        
        return result

