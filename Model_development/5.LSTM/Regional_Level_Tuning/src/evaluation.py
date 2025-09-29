from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, X, y, scaler_y):
    """
    Evaluate the model and calculate core regression metrics.
    """

    y_pred = model.predict(X, verbose=0)

    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    y_true_rescaled = scaler_y.inverse_transform(y.reshape(-1, 1))

    y_pred_rescaled = np.clip(np.round(y_pred_rescaled), 0, None)
    y_true_rescaled = np.clip(np.round(y_true_rescaled), 0, None)

    actual = y_true_rescaled.flatten()
    predicted = y_pred_rescaled.flatten()

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

    return metrics
