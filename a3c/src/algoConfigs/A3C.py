A3C_DEFAULT_DICT = {

    # Gym Environment Parameters
    'env_type': 'gym',
    'env_name': 'CartPole-v0',
    'env_module': None,
    'env_processes': 16,
    'seed_offset': 0,
    'render_env': False,
    'max_episode_length': 1e4,
    'normalize_state': False,

    # PyTorch Parameters
    'torchSeed': 0,

    # Common Policy Parameters
    'memsize': 256,

    # Policy tyep
    'policy-type': 'mlp',

    # MLP Policy Parameters
    'hidden': 128,

    # Optimizer Parameters
    'learning_rate': 0.0001,

    # Experiment Configs
    'num_training_frames': 8e7,
    'moving_avg_coef': 0.99,
    'gamma': 0.9,

    # A3C Algorithm Parameters
    'num_steps': 10,
    'gae_lambda': 1.0,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'grad_norm': 40,

    # Logging Details
    'exp_name': 'cartpole',
    'log_dir': '/home/'
}
