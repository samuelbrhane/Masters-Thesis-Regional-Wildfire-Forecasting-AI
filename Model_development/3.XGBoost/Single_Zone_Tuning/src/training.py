def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the XGBoost model using training data and validate using validation data.
    """
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model
