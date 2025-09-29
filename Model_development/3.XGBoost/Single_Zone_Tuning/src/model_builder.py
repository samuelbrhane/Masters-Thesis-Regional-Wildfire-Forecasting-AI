import xgboost as xgb

def build_model(params, seed):
    """
    Build an XGBoost regression model using sampled hyperparameters and provided seed.
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 6),
        learning_rate=params.get('learning_rate', 0.1),
        subsample=params.get('subsample', 1.0),
        colsample_bytree=params.get('colsample_bytree', 1.0),
        gamma=params.get('gamma', 0),
        reg_alpha=params.get('reg_alpha', 0),
        reg_lambda=params.get('reg_lambda', 1),
        random_state=seed,
        verbosity=0
    )
    return model
