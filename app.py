import json
import gradio as gr
import numpy as np
import torch
from game_engine.game import Connect4
from neural_net.network import AlphaZeroNetwork
from mcts.mcts import MCTS
from config import Config

# ── model setup ───────────────────────────────────────────────────────────────

config = Config()
config.NUM_SIMULATIONS = 200
network = AlphaZeroNetwork(config.NUM_RES_BLOCKS, config.NUM_CHANNELS)
network.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
network.eval()
mcts = MCTS(Connect4, network, config, torch.device("cpu"))

state = {"current": Connect4.initial_state()}

ROWS, COLS = 6, 7


# ── win probability ───────────────────────────────────────────────────────────

def get_ai_win_prob(game_state) -> float:
    encoded = Connect4.encode(game_state).unsqueeze(0).float()
    with torch.no_grad():
        _, value = network(encoded)
    ai_value = -value.item()
    return (ai_value + 1) / 2 * 100


# ── HTML renderers ────────────────────────────────────────────────────────────

# Each renderer returns a self-contained HTML string.
# All animation is done via CSS transitions on inline styles — the browser
# interpolates smoothly whenever the style value changes.

_BOARD_STYLE = """
<style>
  .c4-board {
    display: inline-grid;
    grid-template-columns: repeat(7, 60px);
    grid-template-rows: repeat(6, 60px);
    gap: 6px;
    background: #1a6fbf;
    padding: 12px;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  }
  .c4-cell {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    transition: background-color 0.35s cubic-bezier(0.4, 0, 0.2, 1),
                box-shadow       0.35s ease;
    box-shadow: inset 0 3px 8px rgba(0,0,0,0.4);
  }
  .c4-cell.empty  { background-color: #1c2128; }
  .c4-cell.p1     { background-color: #e63946;
                     box-shadow: inset 0 3px 8px rgba(0,0,0,0.3),
                                 inset -4px -4px 10px rgba(255,255,255,0.15); }
  .c4-cell.p2     { background-color: #ffd166;
                     box-shadow: inset 0 3px 8px rgba(0,0,0,0.3),
                                 inset -4px -4px 10px rgba(255,255,255,0.2); }
  .c4-labels {
    display: grid;
    grid-template-columns: repeat(7, 60px);
    gap: 6px;
    padding: 6px 12px 0;
    text-align: center;
  }
  .c4-labels span {
    font-size: 13px;
    font-weight: bold;
    color: #aaaaaa;
    width: 60px;
    display: inline-block;
  }
</style>
"""

def render_board(board) -> str:
    cells = ""
    for r in range(ROWS):
        for c in range(COLS):
            v = board[r][c]
            cls = "p1" if v == 1 else ("p2" if v == -1 else "empty")
            cells += f'<div class="c4-cell {cls}"></div>\n'

    labels = "".join(f'<span>{c}</span>' for c in range(COLS))

    return f"""
{_BOARD_STYLE}
<div style="display:flex; justify-content:center;">
  <div style="display:inline-block;">
    <div class="c4-board">{cells}</div>
    <div class="c4-labels">{labels}</div>
  </div>
</div>
"""


_WIN_BAR_STYLE = """
<style>
  .win-wrap {
    padding: 8px 0 4px;
  }
  .win-title {
    font-size: 12px;
    font-weight: 600;
    color: #888;
    margin-bottom: 6px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .win-track {
    position: relative;
    width: 100%;
    height: 32px;
    background: #2a2a2a;
    border-radius: 6px;
    overflow: hidden;
  }
  /* player fill grows from left */
  .win-player {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    background: #e63946;
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    border-radius: 6px 0 0 6px;
  }
  /* AI fill grows from right */
  .win-ai {
    position: absolute;
    right: 0; top: 0; bottom: 0;
    background: #ffd166;
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    border-radius: 0 6px 6px 0;
  }
  .win-divider {
    position: absolute;
    left: 50%; top: 0; bottom: 0;
    width: 2px;
    background: #0d1117;
    transform: translateX(-50%);
    z-index: 2;
  }
  .win-label-left, .win-label-right {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    font-size: 12px;
    font-weight: bold;
    z-index: 3;
    pointer-events: none;
  }
  .win-label-left  { left: 8px;  color: #fff; }
  .win-label-right { right: 8px; color: #0d1117; }
</style>
"""

def render_win_bar(prob: float) -> str:
    ai_pct     = round(np.clip(prob, 0, 100), 1)
    player_pct = round(100 - ai_pct, 1)

    return f"""
{_WIN_BAR_STYLE}
<div class="win-wrap">
  <div class="win-title">Winning confidence</div>
  <div class="win-track">
    <div class="win-player" style="width:{player_pct}%"></div>
    <div class="win-ai"     style="width:{ai_pct}%"></div>
    <div class="win-divider"></div>
    <span class="win-label-left">You {player_pct:.0f}%</span>
    <span class="win-label-right">AI {ai_pct:.0f}%</span>
  </div>
</div>
"""


_CONF_STYLE = """
<style>
  .conf-wrap {
    padding: 4px 0;
  }
  .conf-title {
    font-size: 12px;
    font-weight: 600;
    color: #888;
    margin-bottom: 8px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .conf-chart {
    display: flex;
    align-items: flex-end;
    gap: 6px;
    height: 120px;
  }
  .conf-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
    justify-content: flex-end;
    gap: 4px;
  }
  .conf-pct {
    font-size: 10px;
    font-weight: bold;
    color: #ccc;
    min-height: 14px;
  }
  .conf-bar {
    width: 100%;
    border-radius: 4px 4px 0 0;
    transition: height 0.45s cubic-bezier(0.4, 0, 0.2, 1),
                background-color 0.3s ease;
    min-height: 2px;
  }
  .conf-bar.best { background-color: #ffd166; }
  .conf-bar.rest { background-color: #3a7bd5; }
  .conf-label {
    font-size: 11px;
    font-weight: bold;
    color: #aaa;
    padding-top: 4px;
    border-top: 1px solid #2a2a2a;
    width: 100%;
    text-align: center;
  }
</style>
"""

def render_move_confidence(visits: np.ndarray | None) -> str:
    if visits is None or visits.sum() == 0:
        probs = np.zeros(COLS)
        best  = -1
    else:
        probs = visits / visits.sum()
        best  = int(np.argmax(probs))

    max_h = 88   # px — max bar height

    cols_html = ""
    for c in range(COLS):
        p      = probs[c]
        h      = max(2, int(p * max_h))
        cls    = "best" if c == best and p > 0 else "rest"
        pct    = f"{p*100:.0f}%" if p > 0.01 else ""
        cols_html += f"""
      <div class="conf-col">
        <span class="conf-pct">{pct}</span>
        <div class="conf-bar {cls}" style="height:{h}px"></div>
        <div class="conf-label">{c}</div>
      </div>"""

    return f"""
{_CONF_STYLE}
<div class="conf-wrap">
  <div class="conf-title">AI move confidence</div>
  <div class="conf-chart">{cols_html}
  </div>
</div>
"""


# ── game logic ────────────────────────────────────────────────────────────────

def player_move(col: int):
    s = state["current"]

    if col not in Connect4.get_legal_moves(s):
        return (render_board(s.board),
                render_move_confidence(None),
                render_win_bar(get_ai_win_prob(s)),
                "Illegal move! Column is full.")

    # player move
    last_player = s.current_player
    s = Connect4.make_move(s, col)
    state["current"] = s

    done, value = Connect4.is_terminal(s, last_player)
    if done:
        state["current"] = Connect4.initial_state()
        msg = "You win! 🎉 Board reset." if value == 1 else "Draw! Board reset."
        return render_board(s.board), render_move_confidence(None), render_win_bar(0), msg

    # AI move — capture visit distribution before committing
    visits      = mcts.search(s)
    ai_col      = int(np.argmax(visits))
    last_player = s.current_player
    s = Connect4.make_move(s, ai_col)
    state["current"] = s

    done, value = Connect4.is_terminal(s, last_player)

    if done:
        state["current"] = Connect4.initial_state()
        return (render_board(s.board),
                render_move_confidence(visits),
                render_win_bar(100),
                f"AI wins! 😈 (played col {ai_col}) Board reset.")

    return (render_board(s.board),
            render_move_confidence(visits),
            render_win_bar(get_ai_win_prob(s)),
            f"AI played column {ai_col}")


def reset():
    state["current"] = Connect4.initial_state()
    return (
        render_board(state["current"].board),
        render_move_confidence(None),
        render_win_bar(50),
        "Game reset! Your turn. (You are 🔴, AI is 🟡)"
    )


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
.col-btn {
    min-width: 0 !important;
    padding: 8px 2px !important;
    font-size: 1.05rem !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    flex: 1 1 0 !important;
}
"""

with gr.Blocks(theme=gr.themes.Base(), css=CSS) as demo:
    gr.Markdown("# Connect Four — AlphaZero &nbsp;&nbsp; [![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/k13nNg/connect4_alphazero)")

    with gr.Row(equal_height=False):

        # ── left: board + drop buttons ─────────────────────────────────────
        with gr.Column(scale=5):
            board_display = gr.HTML(
                value=render_board(Connect4.initial_state().board)
            )
            with gr.Row(equal_height=True):
                col_buttons = [
                    gr.Button(f"↓ {c}", elem_classes=["col-btn"])
                    for c in range(COLS)
                ]

        # ── right: move confidence, win bar, status, reset ─────────────────
        with gr.Column(scale=3):
            confidence_display = gr.HTML(
                value=render_move_confidence(None)
            )
            win_bar_display = gr.HTML(
                value=render_win_bar(50)
            )
            status = gr.Textbox(
                label="Status",
                value="Your turn! (You are 🔴, AI is 🟡)",
                interactive=False,
                lines=2
            )
            reset_btn = gr.Button("🔄 Reset", variant="secondary")

    outputs = [board_display, confidence_display, win_bar_display, status]

    for c, btn in enumerate(col_buttons):
        btn.click(fn=lambda col=c: player_move(col), outputs=outputs)

    reset_btn.click(fn=reset, outputs=outputs)

demo.launch()
