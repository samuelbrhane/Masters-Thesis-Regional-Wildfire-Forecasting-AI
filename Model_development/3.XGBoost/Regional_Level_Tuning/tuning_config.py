XGBOOST_SEARCH_SPACE = {
    'fire_lag': (7, 30),                   
    'climate_lag': (3, 7),                
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': (0.01, 0.3),
    'subsample': (0.8, 1.0),
    'colsample_bytree': (0.8, 1.0),
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10],
}
