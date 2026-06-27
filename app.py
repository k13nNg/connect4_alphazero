import gradio as gr
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

# palette
BOARD_COLOR = "#1a6fbf"
EMPTY_COLOR = "#1c2128"   # matches Gradio Base dark app background
P1_COLOR    = "#e63946"   # player — red
P2_COLOR    = "#ffd166"   # AI — yellow
PANEL_BG    = "#1c2128"


# ── board renderer ────────────────────────────────────────────────────────────

def render_board(board):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor(BOARD_COLOR)
    ax.set_facecolor(BOARD_COLOR)

    cell, pad = 1.0, 0.08

    ax.add_patch(patches.FancyBboxPatch(
        (-0.5, -0.5), COLS, ROWS,
        boxstyle="round,pad=0.15", linewidth=0,
        facecolor=BOARD_COLOR, zorder=0
    ))

    for r in range(ROWS):
        for c in range(COLS):
            val = board[r][c]
            y   = ROWS - 1 - r

            # shadow ring
            ax.add_patch(patches.Circle(
                (c, y), cell / 2 - pad + 0.03,
                color="#000000", alpha=0.35, zorder=1
            ))

            color = P1_COLOR if val == 1 else (P2_COLOR if val == -1 else EMPTY_COLOR)
            ax.add_patch(patches.Circle(
                (c, y), cell / 2 - pad,
                color=color, zorder=2
            ))

            # glare on filled discs
            if val != 0:
                ax.add_patch(patches.Circle(
                    (c - 0.12, y + 0.12), (cell / 2 - pad) * 0.35,
                    color="white", alpha=0.25, zorder=3
                ))

    # column labels centred below each column
    for c in range(COLS):
        ax.text(
            c, -0.82, str(c),
            ha="center", va="center",
            fontsize=13, fontweight="bold",
            color="#ffffff"
        )

    ax.set_xlim(-0.6, COLS - 0.4)
    ax.set_ylim(-1.15, ROWS - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig


# ── overall win probability ───────────────────────────────────────────────────

def get_ai_win_prob(game_state) -> float:
    """
    Query the value head. Value is in [-1, 1] from the current player's POV.
    After the AI moves it's the human's turn, so negate to get AI's view,
    then map [-1, 1] → [0, 100].
    """
    encoded = Connect4.encode(game_state).unsqueeze(0).float()
    with torch.no_grad():
        _, value = network(encoded)
    ai_value = -value.item()
    return (ai_value + 1) / 2 * 100


def render_win_gauge(prob: float):
    """
    Horizontal progress bar showing AI win probability (0–100 %).
    Left = player, right = AI. Fill colour shifts green → red with AI confidence.
    """
    fig, ax = plt.subplots(figsize=(4.2, 1.5))
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(PANEL_BG)

    frac = np.clip(prob / 100, 0, 1)
    bar_h = 0.52
    bar_y = 0.5 - bar_h / 2

    # track
    ax.add_patch(patches.FancyBboxPatch(
        (0, bar_y), 100, bar_h,
        boxstyle="round,pad=0.5",
        linewidth=0, facecolor="#2a2a2a", zorder=1
    ))

    # player fill (left side, always red)
    if frac < 1.0:
        ax.add_patch(patches.FancyBboxPatch(
            (0, bar_y), (1 - frac) * 100, bar_h,
            boxstyle="round,pad=0.5",
            linewidth=0, facecolor=P1_COLOR, zorder=2
        ))

    # AI fill (right side, yellow)
    if frac > 0.0:
        ai_x = (1 - frac) * 100
        ax.add_patch(patches.FancyBboxPatch(
            (ai_x, bar_y), frac * 100, bar_h,
            boxstyle="round,pad=0.5",
            linewidth=0, facecolor=P2_COLOR, zorder=2
        ))

    # centre divider line
    ax.plot([50, 50], [bar_y - 0.06, bar_y + bar_h + 0.06],
            color="#0d0d0d", lw=1.5, zorder=3)

    # labels inside the bar
    ax.text(25, 0.5, f"You  {(1-frac)*100:.0f}%",
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="#ffffff", zorder=4)
    ax.text(75, 0.5, f"AI  {prob:.0f}%",
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="#0d1117", zorder=4)

    ax.set_title("Winning confidence", fontsize=10, fontweight="bold",
                 color="#aaaaaa", pad=6)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout(pad=0.3)
    return fig


# ── AlphaGo-style per-move confidence chart ───────────────────────────────────

def render_move_confidence(visits: np.ndarray | None):
    """
    Vertical bar chart — one bar per column, height = share of MCTS visits.
    Best move highlighted in yellow, matching AlphaGo/AlphaZero style.
    """
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(PANEL_BG)

    col_indices = np.arange(COLS)

    if visits is None or visits.sum() == 0:
        probs = np.zeros(COLS)
    else:
        probs = visits / visits.sum()

    best = int(np.argmax(probs)) if probs.sum() > 0 else -1

    bar_colors = [
        P2_COLOR if (c == best and probs[c] > 0) else "#3a7bd5"
        for c in range(COLS)
    ]

    bars = ax.bar(
        col_indices, probs,
        color=bar_colors,
        width=0.62,
        zorder=2,
        edgecolor="none",
    )

    # percentage label above each bar
    for c, (bar, p) in enumerate(zip(bars, probs)):
        if p > 0.01:
            ax.text(
                c, p + 0.012,
                f"{p * 100:.0f}%",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color="#ffffff"
            )

    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, max(probs.max() * 1.25, 0.15))
    ax.set_xlim(-0.6, COLS - 0.4)
    ax.set_xticks(col_indices)
    ax.set_xticklabels([str(c) for c in col_indices],
                       fontsize=11, fontweight="bold", color="#cccccc")
    ax.tick_params(axis="y", colors="#666666", labelsize=8)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color="#2a2a2a", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("AI move confidence", fontsize=11, fontweight="bold",
                 color="#aaaaaa", pad=8)
    ax.set_xlabel("Column", fontsize=9, color="#888888", labelpad=4)

    plt.tight_layout(pad=0.5)
    return fig


# ── game logic ────────────────────────────────────────────────────────────────

def player_move(col: int):
    s = state["current"]

    if col not in Connect4.get_legal_moves(s):
        return (render_board(s.board),
                render_move_confidence(None),
                render_win_gauge(get_ai_win_prob(s)),
                "Illegal move! Column is full.")

    # player move
    last_player = s.current_player
    s = Connect4.make_move(s, col)
    state["current"] = s

    done, value = Connect4.is_terminal(s, last_player)
    if done:
        state["current"] = Connect4.initial_state()
        msg = "You win! 🎉 Board reset." if value == 1 else "Draw! Board reset."
        return render_board(s.board), render_move_confidence(None), render_win_gauge(0), msg

    # AI move — capture visit distribution before committing
    visits      = mcts.search(s)
    ai_col      = int(np.argmax(visits))
    last_player = s.current_player
    s = Connect4.make_move(s, ai_col)
    state["current"] = s

    done, value = Connect4.is_terminal(s, last_player)
    confidence  = render_move_confidence(visits)
    gauge       = render_win_gauge(get_ai_win_prob(s))

    if done:
        state["current"] = Connect4.initial_state()
        return render_board(s.board), confidence, render_win_gauge(100), f"AI wins! 😈 (played col {ai_col}) Board reset."

    return render_board(s.board), confidence, gauge, f"AI played column {ai_col}"


def reset():
    state["current"] = Connect4.initial_state()
    return (
        render_board(state["current"].board),
        render_move_confidence(None),
        render_win_gauge(50),
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
    gr.Markdown("# Connect Four — AlphaZero")

    with gr.Row(equal_height=False):

        # ── left: board + drop buttons ─────────────────────────────────────
        with gr.Column(scale=5):
            board_display = gr.Plot(
                value=render_board(Connect4.initial_state().board),
                label="Board"
            )
            with gr.Row(equal_height=True):
                col_buttons = [
                    gr.Button(f"↓ {c}", elem_classes=["col-btn"])
                    for c in range(COLS)
                ]

        # ── right: confidence chart, win gauge, status, reset ─────────────
        with gr.Column(scale=3):
            confidence_display = gr.Plot(
                value=render_move_confidence(None),
                label="AI move confidence"
            )
            gauge_display = gr.Plot(
                value=render_win_gauge(50),
                label="AI winning confidence"
            )
            status = gr.Textbox(
                label="Status",
                value="Your turn! (You are 🔴, AI is 🟡)",
                interactive=False,
                lines=2
            )
            reset_btn = gr.Button("🔄 Reset", variant="secondary")

    outputs = [board_display, confidence_display, gauge_display, status]

    for c, btn in enumerate(col_buttons):
        btn.click(fn=lambda col=c: player_move(col), outputs=outputs)

    reset_btn.click(fn=reset, outputs=outputs)

demo.launch()
