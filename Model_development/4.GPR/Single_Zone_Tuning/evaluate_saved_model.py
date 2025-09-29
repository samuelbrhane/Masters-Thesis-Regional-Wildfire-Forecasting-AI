import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import joblib

from src.data_loading import load_data
from src.preprocessing import preprocess_data
from src.evaluation import evaluate_model


def plot_model_test(zone_id, model_filename, model_path, params, save_dir="top_candidate_models_evaluation_gpr"):
    output_dir = os.path.join(save_dir, f"zone_{zone_id}")
    os.makedirs(output_dir, exist_ok=True)
    base_name = model_filename.replace('.pkl', '')

    df = load_data(zone_id)
    _, _, _, _, X_test, y_test, df_test = preprocess_data(df, params)
    test_dates = df_test["Date"].reset_index(drop=True)
    model = joblib.load(model_path)

    y_mean, y_std = model.predict(X_test, return_std=True)
    y_pred = np.round(np.clip(y_mean, 0, None))
    y_true = np.round(np.clip(y_test, 0, None))
    residuals = y_true - y_pred

    pred_df = pd.DataFrame({
        "Date": test_dates,
        "y_true": y_true,
        "y_pred_mean": y_mean,
        "y_pred_rounded": y_pred,
        "y_std": y_std,
        "residual": residuals
    })
    csv_path = os.path.join(output_dir, f"test_predictions_{base_name}.csv")
    pred_df.to_csv(csv_path, index=False)

    metrics = evaluate_model(model, X_test, y_test)
