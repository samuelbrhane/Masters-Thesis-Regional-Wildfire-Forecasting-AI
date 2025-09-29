from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, X, y, scaler_y=None):
    """
    Evaluate a regression model and calculate core metrics across various model types.
    """

    try:
        y_pred, y_std = model.predict(X, return_std=True)
    except TypeError:
        y_pred = model.predict(X)
        y_std = None

    if hasattr(y_pred, "shape") and len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
    else:
        y = y.flatten()

    y_pred = np.clip(np.round(y_pred), 0, None)
    y_true = np.clip(np.round(y), 0, None)

    actual = y_true
    predicted = y_pred

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mean_residual = np.mean(actual - predicted)
    std_residual = np.std(actual - predicted)
    r2 = r2_score(actual, predicted)
    exact_match_percentage = (actual == predicted).sum() / len(actual)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'r2': r2,
        'exact_match_percentage': exact_match_percentage,
    }

    if y_std is not None:
        metrics.update({
            'mean_prediction_std': np.mean(y_std),
            'max_prediction_std': np.max(y_std),
            'min_prediction_std': np.min(y_std),
        })

    return metrics
