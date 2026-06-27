class Config:
    # Network
    NUM_RES_BLOCKS = 5
    NUM_CHANNELS = 256

    # MCTS
    NUM_SIMULATIONS = 200
    C_PUCT = 1.4
    DIRICHLET_ALPHA = 0.3  
    DIRICHLET_EPSILON = 0.25

    # Self-play
    NUM_SELF_PLAY_GAMES = 100
    TEMPERATURE_THRESHOLD = 10

    # Training
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 10

    # Replay buffer
    REPLAY_BUFFER_SIZE = 50_000

    # Evaluation
    NUM_EVAL_GAMES = 40
    WIN_THRESHOLD = 0.55

    # Main loop
    NUM_ITERATIONS = 100
    EVAL_EVERY = 5
