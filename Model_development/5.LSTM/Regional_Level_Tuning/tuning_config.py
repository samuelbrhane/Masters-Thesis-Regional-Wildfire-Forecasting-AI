SEARCH_SPACE = {
    'lstm_units': [32, 64, 96],
    'batch_size': [16, 32, 64],
    'dropout_rate': (0.1, 0.5),
    'learning_rate': (1e-5, 1e-2),
    'lag_days': [1, 2, 3],
    'num_layers': [1, 2, 3],
    'sequence_length': (7, 30),
    'activation_function': ['relu', 'tanh'],
    'optimizer': ['adam', 'rmsprop']
}

