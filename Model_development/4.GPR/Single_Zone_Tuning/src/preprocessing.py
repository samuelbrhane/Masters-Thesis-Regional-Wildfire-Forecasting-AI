from ..constants import TARGET_COL

def preprocess_data(df, params):
    """
    Preprocess wildfire data for Gaussian Process Regression (GPR):
    """
    fire_lag = params['fire_lag']
    climate_lag = params['climate_lag']
    climate_features = ['Temperature', 'Precipitation', 'Humidity', 'Wind']

    for lag in range(1, fire_lag + 1):
        df[f'Num_Fires_lag_{lag}'] = df[TARGET_COL].shift(lag)

    for col in climate_features:
        for lag in range(1, climate_lag + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    lagged_fire = [f'Num_Fires_lag_{i}' for i in range(1, fire_lag + 1)]
    lagged_climate = [f'{col}_lag_{i}' for col in climate_features for i in range(1, climate_lag + 1)]
    feature_cols = climate_features + lagged_fire + lagged_climate

    split_1 = int(len(df) * 0.65)
    split_2 = int(len(df) * 0.80)
    df_train = df[:split_1].dropna().reset_index(drop=True)
    df_val = df[split_1:split_2].dropna().reset_index(drop=True)
    df_test = df[split_2:].dropna().reset_index(drop=True)

    X_train = df_train[feature_cols].values
    y_train = df_train[TARGET_COL].values
    X_val = df_val[feature_cols].values
    y_val = df_val[TARGET_COL].values
    X_test = df_test[feature_cols].values
    y_test = df_test[TARGET_COL].values

    return X_train, y_train, X_val, y_val, X_test, y_test, df_test
