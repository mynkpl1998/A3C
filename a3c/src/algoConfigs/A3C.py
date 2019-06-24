A3C_DEFAULT_DICT = {

    # Gym Environment Parameters
    'env_type': 'gym',
    'env_name': 'CartPole-v0',
    'env_processes': 4,
    'seed_offset': 0,
    'render_env': False,
    'max_episode_length': 1e4,

    # PyTorch Parameters
    'pySeed': 0,

    # Common Policy Parameters
    'memsize': 128,

    # MLP Policy Parameters
    'policy_hiddens': [128],

    # Optimizer Parameters
    'learning_rate': 1e-4,

    # Experiment Configs
    'num_training_frames': 8e7,
    'moving_avg_coef': 0.99,
    'gamma': 0.99,

    # A3C Algorithm Parameters
    'num_steps': 20,
    'gae-lambda': 1.0,
    'value-loss-coef': 0.5,
    'entropy-coef': 0.01,
    'grad-norm': 80
}
