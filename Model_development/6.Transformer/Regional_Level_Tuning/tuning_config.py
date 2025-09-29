SEARCH_SPACE = {
    'batch_size': [16, 32],
    'learning_rate': (1e-4, 3e-4),
    'optimizer': ['adam', 'rmsprop'],
    'lag_days': [1, 2, 3],
    'sequence_length': (7, 30),
    'd_model': [64, 128],
    'num_heads': [2, 4],
    'ff_dim': [64, 128],
    'num_layers': [1, 2],
    'dropout_rate': (0.2, 0.3),
    'month_embed_dim': [4, 8],
    'dow_embed_dim': [2, 4],
}
