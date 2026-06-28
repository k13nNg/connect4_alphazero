from game_engine.game import Connect4
from neural_net.network import AlphaZeroNetwork
from mcts.mcts import MCTS
from training.self_play import play_one_game
from training.replay_buffer import ReplayBuffer
from training.trainer import Trainer
from training.evaluator import evaluate
from config import Config

import torch
import os
import json
import pickle

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize
    network = AlphaZeroNetwork(config.NUM_RES_BLOCKS, config.NUM_CHANNELS).to(device)
    buffer  = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
    trainer = Trainer(network, config, device)
    mcts    = MCTS(Connect4, network, config, device)

    # save buffer after each iteration
    with open("checkpoints/buffer.pkl", "wb") as f:
        pickle.dump(buffer, f)

    # create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    # ── resume from checkpoint ──────────────────────────────
    start_iteration = 0
    checkpoint_path = "checkpoints/best.pt"
    progress_path   = "checkpoints/progress.json"
    # load buffer on resume
    buffer_path = "checkpoints/buffer.pkl"
    
    if os.path.exists(buffer_path):
        with open(buffer_path, "rb") as f:
            buffer = pickle.load(f)
        print(f"Loaded replay buffer: {len(buffer)} positions", flush=True)

    if os.path.exists(checkpoint_path):
        network.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")

    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        start_iteration = progress["iteration"] + 1
        print(f"Resuming from iteration {start_iteration}")
    # ────────────────────────────────────────────────────────

    for iteration in range(start_iteration, config.NUM_ITERATIONS):
        print(f"--- Iteration {iteration + 1}/{config.NUM_ITERATIONS} ---", flush = True)

        # 1. self-play: generate training data
        print("Self-play...", flush = True)
        network.eval()
        for _ in range(config.NUM_SELF_PLAY_GAMES):
            game_data = play_one_game(network, mcts, config)
            buffer.add_game(game_data)
        print(f"Buffer size: {len(buffer)}", flush = True)

        # 2. train: update network weights
        if len(buffer) >= config.BATCH_SIZE:
            print("Training...")
            network.train()
            loss = trainer.train(buffer)
            print(f"Loss: {loss:.4f}")

        # 3. evaluate and save checkpoint every N iterations
        if iteration % config.EVAL_EVERY == 0:
            print("Evaluating...")
            # old_network = AlphaZeroNetwork(config.NUM_RES_BLOCKS, config.NUM_CHANNELS)
            old_network = AlphaZeroNetwork(config.NUM_RES_BLOCKS, config.NUM_CHANNELS).to(device)
            old_network.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))

            # save current as best if no checkpoint exists yet
            if not os.path.exists("checkpoints/best.pt"):
                torch.save(network.state_dict(), "checkpoints/best.pt")
                print("Saved initial checkpoint.")
            else:
                is_better = evaluate(network, old_network, config, device)
                if is_better:
                    torch.save(network.state_dict(), "checkpoints/best.pt")
                    print("New network is better! Checkpoint updated.")
                else:
                    print("Old network is better. Keeping old checkpoint.")

        # save iteration checkpoint
        torch.save(network.state_dict(), f"checkpoints/iter_{iteration}.pt")

if __name__ == "__main__":
    main()
