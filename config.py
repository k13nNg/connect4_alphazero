class Config:
    # Network
    NUM_RES_BLOCKS = 5
    NUM_CHANNELS = 256
    
    # MCTS
    NUM_SIMULATIONS = 200
    C_PUCT = 1.4
    DIRICHLET_ALPHA = 0.03
    DIRICHLET_EPSILON = 0.25
