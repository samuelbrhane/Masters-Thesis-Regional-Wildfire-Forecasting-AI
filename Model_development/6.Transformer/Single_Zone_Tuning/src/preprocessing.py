import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ..constants import FEATURE_COLS, TARGET_COL, DATE_COL

def preprocess_data(df, params):
    """
    Preprocess wildfire data for Transformer model at zone level:
    Includes numeric scaling and categorical time embeddings.
    """

    lag_days = params['lag_days']
    df['Prev_Num_Fires_Result'] = df[TARGET_COL].shift(lag_days)
    df['month'] = df[DATE_COL].dt.month - 3  
    df['day_of_week'] = df[DATE_COL].dt.dayofweek  

    split_1 = int(len(df) * 0.65)
    split_2 = int(len(df) * 0.80)
    df_train = df[:split_1].dropna().reset_index(drop=True)
    df_val = df[split_1:split_2].dropna().reset_index(drop=True)
    df_test = df[split_2:].dropna().reset_index(drop=True)

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values
    X_val = df_val[FEATURE_COLS].values
    y_val = df_val[TARGET_COL].values
    X_test = df_test[FEATURE_COLS].values
    y_test = df_test[TARGET_COL].values

    month_train = df_train['month'].values
    dow_train = df_train['day_of_week'].values
    month_val = df_val['month'].values
    dow_val = df_val['day_of_week'].values
    month_test = df_test['month'].values
    dow_test = df_test['day_of_week'].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    def create_sequences(X, y, month, dow, seq_len):
        X_seq, y_seq, month_seq, dow_seq = [], [], [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len])
            month_seq.append(month[i:i + seq_len])
            dow_seq.append(dow[i:i + seq_len])
        return np.array(X_seq), np.array(y_seq), np.array(month_seq), np.array(dow_seq)

    seq_len = params['sequence_length']
    X_train_seq, y_train_seq, month_train_seq, dow_train_seq = create_sequences(X_train_scaled, y_train_scaled, month_train, dow_train, seq_len)
    X_val_seq, y_val_seq, month_val_seq, dow_val_seq = create_sequences(X_val_scaled, y_val_scaled, month_val, dow_val, seq_len)
    X_test_seq, y_test_seq, month_test_seq, dow_test_seq = create_sequences(X_test_scaled, y_test_scaled, month_test, dow_test, seq_len)

    return (
        X_train_seq, y_train_seq, month_train_seq, dow_train_seq,
        X_val_seq, y_val_seq, month_val_seq, dow_val_seq,
        X_test_seq, y_test_seq, month_test_seq, dow_test_seq,
        df_test, scaler_y
    )
