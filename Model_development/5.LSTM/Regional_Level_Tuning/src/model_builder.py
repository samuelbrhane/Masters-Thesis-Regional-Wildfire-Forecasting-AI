from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(params, input_shape):
    """
    Build a dynamic LSTM model based on sampled hyperparameters.
    """
    model = Sequential()
    
    num_layers = params['num_layers']
    
    for i in range(num_layers):
        return_seq = (i != num_layers - 1)
        if i == 0:
            model.add(LSTM(params['lstm_units'], activation=params['activation_function'], return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(LSTM(params['lstm_units'], activation=params['activation_function'], return_sequences=return_seq))
        model.add(Dropout(params['dropout_rate']))
    
    model.add(Dense(1))
    return model
