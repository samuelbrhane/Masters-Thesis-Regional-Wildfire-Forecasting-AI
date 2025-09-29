import os
import numpy as np
import pandas as pd
import joblib
from .preprocessing import preprocess_data
from ..Common_Code.data_loading import load_data
from ..Common_Code.evaluation import evaluate_model
from ..Common_Code.plots import save_summary_and_plots


def plot_model_test(model_filename, model_path, params, save_dir, zone_id=None):
    """Plot test results and save evaluation for regional or zone-level models."""
    output_subdir = f"zone_{zone_id}" if zone_id is not None else "regional"
    output_dir = os.path.join(save_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    base_name = model_filename.replace('.pkl', '')

    df = load_data(zone_id) if zone_id is not None else load_data()
    _, _, _, _, X_test, y_test, df_test = preprocess_data(df, params)
    test_dates = df_test["Date"].reset_index(drop=True)

    model = joblib.load(model_path)
    y_pred = np.round(np.clip(model.predict(X_test), 0, None))
    y_true = np.round(np.clip(y_test, 0, None))
    residuals = y_true - y_pred
    metrics = evaluate_model(model, X_test, y_test)

    pred_df = pd.DataFrame({
        "Date": test_dates,
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": residuals
    })
    csv_path = os.path.join(output_dir, f"test_predictions_{base_name}.csv")
    pred_df.to_csv(csv_path, index=False)

    save_summary_and_plots(
        model_filename=model_filename,
        output_dir=output_dir,
        base_name=base_name,
        y_true=y_true,
        y_pred=y_pred,
        residuals=residuals,
        metrics=metrics,
        test_dates=test_dates,
        zone_id=zone_id
    )
