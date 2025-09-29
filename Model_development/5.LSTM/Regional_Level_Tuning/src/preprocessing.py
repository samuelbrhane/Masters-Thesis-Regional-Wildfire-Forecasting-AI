import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from constants import FEATURE_COLS, TARGET_COL

def preprocess_data(df, params):
    """
    Preprocess wildfire data for LSTM
    """
    df['Prev_Num_Fires_Result'] = df[TARGET_COL].shift(params['lag_days'])

    split_1 = int(len(df) * 0.65)
    split_2 = int(len(df) * 0.80)

    df_train = df[:split_1].dropna().reset_index(drop=True)
    df_val = df[split_1:split_2].reset_index(drop=True)
    df_test = df[split_2:].reset_index(drop=True)

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values
    X_val = df_val[FEATURE_COLS].values
    y_val = df_val[TARGET_COL].values
    X_test = df_test[FEATURE_COLS].values
    y_test = df_test[TARGET_COL].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    def create_sequences(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])
        return np.array(X_seq), np.array(y_seq)

    seq_len = params['sequence_length']
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_len)

    return (
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq,
        X_test_seq,
        y_test_seq,
        df_test[seq_len:].reset_index(drop=True),  
        scaler_y
    )
