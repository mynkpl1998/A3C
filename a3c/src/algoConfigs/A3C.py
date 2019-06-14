A3C_DEFAULT_DICT = {

    # Gym Environment Parameters
    'env_type': 'gym',
    'env_name': 'CartPole-v0',
    'env_processes': 4,
    'seed_offset': 0,

    # Common Policy Parameters
    'memsize': 128,

    # MLP Policy Parameters
    'policy_hiddens': [128, 256, 512]
}
