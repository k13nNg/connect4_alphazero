import gradio as gr
import numpy as np
import torch
from game_engine.game import Connect4
from neural_net.network import AlphaZeroNetwork
from mcts.mcts import MCTS
from config import Config

# load trained model
config = Config()
config.NUM_SIMULATIONS = 200
network = AlphaZeroNetwork(config.NUM_RES_BLOCKS, config.NUM_CHANNELS)
network.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
network.eval()
mcts = MCTS(Connect4, network, config, torch.device("cpu"))

# game state stored between moves
state = {"current": Connect4.initial_state()}

def render_board(board):
    """Convert board to emoji string for display."""
    symbols = {0: "⚪", 1: "●", -1: "🟡"}
    rows = []
    for row in board:
        rows.append(" ".join(symbols[x] for x in row))
    cols = " ".join(str(i) for i in range(7))
    return "".join(rows) + "" + cols

def player_move(col):
    col = int(col)
    s = state["current"]

    # check legal
    if col not in Connect4.get_legal_moves(s):
        board_str = render_board(s.board)
        return board_str, "Illegal move! Try again."

    # player move
    last_player = s.current_player
    s = Connect4.make_move(s, col)
    state["current"] = s

    done, value = Connect4.is_terminal(s, last_player)
    if done:
        board_str = render_board(s.board)
        state["current"] = Connect4.initial_state()
        return board_str, "You win! 🎉 Board reset." if value == 1 else "Draw! Board reset."

    # AI move
    visits = mcts.search(s)
    ai_col = int(np.argmax(visits))
    last_player = s.current_player
    s = Connect4.make_move(s, ai_col)
    state["current"] = s

    done, value = Connect4.is_terminal(s, last_player)
    board_str = render_board(s.board)

    if done:
        state["current"] = Connect4.initial_state()
        return board_str, f"AI wins! 😈 (played col {ai_col}) Board reset."

    return board_str, f"AI played column {ai_col}"

def reset():
    state["current"] = Connect4.initial_state()
    return render_board(state["current"].board), "Game reset!"

with gr.Blocks() as demo:
    gr.Markdown("# Connect Four — AlphaZero")
    board_display = gr.Textbox(
        value=render_board(Connect4.initial_state().board),
        label="Board",
        lines=8
    )
    status = gr.Textbox(label="Status", value="Your turn! (You are ●)")
    col_input = gr.Number(label="Your column (0-6)", precision=0)
    
    with gr.Row():
        move_btn = gr.Button("Play")
        reset_btn = gr.Button("Reset")
    
    move_btn.click(player_move, inputs=col_input, outputs=[board_display, status])
    reset_btn.click(reset, outputs=[board_display, status])

demo.launch()
