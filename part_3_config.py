# WANDB config

# config = {
#     'method': 'random',
#     'name': 'Transformer',
#     'metric': {
#         'goal': 'minimize',
#         'name': 'test_perplexity'
#     },
#     'parameters': {
#         'train_len': {'value': 3*10**4},
#         'validation_len': {'value': 10**4},
#         'test_len': {'value': 14 * 10**3},
#         'batch_size': {'value': 32},
#         'epochs': {'value': 100},
#         'learning_rate': {'value': 0.001},
#         'dropout': {'value': 0.1},
#         'nhead': {'value': 4},
#         'dim_feedforward': {'value': 1024},
#         'num_layers': {'value': 2},
#         'max_len': {'value': 50},
#         'embedding_dim': {'value': 100},
#         'optimizer': {'value': 'Adam'},
#         'loss': {'value': 'CrossEntropyLoss'},
#     }
# }

config = {
    'method': 'random',
    'name': 'Transformer',
    'metric': {
        'goal': 'minimize',
        'name': 'test_perplexity'
    },
    'parameters': {
        'train_len': {'value': 3*10**4},
        'validation_len': {'value': 10**4},
        'test_len': {'value': 14 * 10**3},
        'batch_size': {'values': [24, 32]},
        'epochs': {'value': 100},
        'learning_rate': {'value': 0.001},
        'dropout': {'values': [0, 0.1, 0.2]},
        'nhead': {'value': 4},
        'dim_feedforward': {'values': [1024, 2048]},
        'num_layers': {'values': [1, 2, 3, 4]},
        'max_len': {'value': 50},
        'embedding_dim': {'values': [100, 200]},
        'optimizer': {'value': 'Adam'},
        'loss': {'value': 'CrossEntropyLoss'},
    }
}