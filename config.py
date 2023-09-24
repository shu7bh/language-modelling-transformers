# WANDB config

config = {
    'method': 'random',
    'name': 'NNLM',
    'metric': {
        'goal': 'minimize',
        'name': 'test_perplexity'
    },
    'parameters': {
        'train_len': {'value': 3*10**4},
        'validation_len': {'value': 10**4},
        'test_len': {'value': 14 * 10**3},
        'learning_rate': {'values': [0.01, 0.005, 0.001]},
        'batch_size': {'value': 128},
        'epochs': {'value': 100},
        'embedding_dim': {'values': [100, 200]},
        'hidden_dim': {'values': [150, 300, 500]},
        'dropout': {'values': [0, 0.2, 0.4]},
        'optimizer': {'values': ['Adam', 'AdamW']}
    }
}