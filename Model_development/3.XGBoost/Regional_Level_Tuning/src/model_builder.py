import xgboost as xgb

def build_model(params, seed):
    """
    Build an XGBoost regression model using sampled hyperparameters and provided seed.
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=seed,
        verbosity=0
    )
    return model
