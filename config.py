class Config:
    # Network — smaller
    NUM_RES_BLOCKS = 3
    NUM_CHANNELS = 64

    # MCTS — biggest speedup
    NUM_SIMULATIONS = 100    # was 200
    
    # Self-play — fewer games
    NUM_SELF_PLAY_GAMES = 30  # was 100

    # Training
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 5            # was 10

    # Replay buffer
    REPLAY_BUFFER_SIZE = 20_000  # was 50,000

    # Evaluation
    NUM_EVAL_GAMES = 20       # was 40
    WIN_THRESHOLD = 0.55

    # Main loop
    NUM_ITERATIONS = 50       # was 100
    EVAL_EVERY = 5

    # MCTS
    C_PUCT = 1.4
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25
    TEMPERATURE_THRESHOLD = 10
