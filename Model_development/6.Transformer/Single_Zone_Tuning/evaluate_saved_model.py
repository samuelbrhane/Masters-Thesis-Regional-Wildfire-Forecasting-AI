import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from src.data_loading import load_data
from src.preprocessing import preprocess_data
from src.evaluation import evaluate_model

def plot_model_test(zone_id, model_filename, model_path, params, save_dir="top_candidate_models_evaluation"):
    output_dir = os.path.join(save_dir, f"zone_{zone_id}")
    os.makedirs(output_dir, exist_ok=True)
    base_name = model_filename.replace('.h5', '')

    df = load_data(zone_id)
    _, _, _, _, _, _, _, _, X_test_seq, y_test_seq, month_test_seq, dow_test_seq, df_test, scaler_y = preprocess_data(df.copy(), params)


    test_dates = df_test["Date"].reset_index(drop=True)
    sequence_length = params["sequence_length"]
    test_dates = test_dates[sequence_length:].reset_index(drop=True)
    date_labels = test_dates.astype(str)

    model = load_model(model_path, custom_objects={"mse": MeanSquaredError()})

    X_test_dict = {
        'numeric_input': X_test_seq,
        'month_input': month_test_seq,
        'dow_input': dow_test_seq
    }

    metrics = evaluate_model(model, X_test_dict, y_test_seq, scaler_y)
    y_true = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(model.predict(X_test_dict, verbose=0)).flatten()

    y_true = np.round(np.clip(y_true, 0, None))
    y_pred = np.round(np.clip(y_pred, 0, None))
    residuals = y_true - y_pred

    pred_df = pd.DataFrame({
        "Date": test_dates,
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": residuals
    })
    csv_path = os.path.join(output_dir, f"test_predictions_{base_name}.csv")
    pred_df.to_csv(csv_path, index=False)

    summary_path = os.path.join(output_dir, f"summary_test_diagnostics_{base_name}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model File: {model_filename}\n")
        f.write(f"Zone ID: {zone_id}\n")
        f.write(f"Number of Test Samples (days evaluated): {len(y_true)}\n\n")
        f.write("Model Performance on Test Data:\n")
        f.write(f" - RMSE: {metrics['rmse']:.2f} fires/day\n")
        f.write(f" - MAE: {metrics['mae']:.2f} fires/day\n")
        f.write(f" - Mean Residual: {metrics['mean_residual']:.2f}\n")
        f.write(f" - Std Residual: {metrics['std_residual']:.2f}\n")
        f.write(f" - R² Score: {metrics['r2']:.3f}\n")
        f.write(f" - Exact Match %: {metrics['exact_match_percentage']:.2%}\n")
    print(f"Saved summary: {summary_path}")

    # Plot: Predicted vs Observed Fires
    x = np.arange(len(y_true))
    zero_offset = 0.02
    y_true_plot = np.where(y_true == 0, zero_offset, y_true)
    y_pred_plot = np.where(y_pred == 0, zero_offset, y_pred)

    plt.figure(figsize=(9, 5))
    plt.plot(x, y_true_plot, label='Observed Fire Counts', color='blue', linewidth=0.6)
    plt.plot(x, y_pred_plot, label='Predicted Fire Counts', color='red', linewidth=0.6)

    plt.title('Predicted vs. Observed Fires', fontsize=10)
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Number of Fires', fontsize=8)

    tick_step = max(1, len(date_labels) // 25)
    xticks = np.arange(0, len(date_labels), tick_step)
    plt.xticks(
        ticks=xticks,
        labels=date_labels[::tick_step],
        rotation=90,
        fontsize=8
    )

    plt.yticks(fontsize=8)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"test_predicted_vs_observed_{base_name}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved curve plot: {plot_path}")

    # Plot: Distribution of Prediction Errors
    plt.figure(figsize=(9, 5))
    min_val = int(np.floor(residuals.min()))
    max_val = int(np.ceil(residuals.max()))
    bin_edges = np.arange(min_val - 0.5, max_val + 1.5, 1)
    histplot = sns.histplot(residuals, bins=bin_edges, color='orange', edgecolor='k', alpha=0.75)

    for patch in histplot.patches:
        height = patch.get_height()
        if height > 0:
            x = patch.get_x() + patch.get_width() / 2
            y = height
            plt.text(x, y + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=6)

    plt.axvline(0, color='black', linestyle='--', linewidth=0.7)
    plt.title('Distribution of Daily Prediction Errors on Test Set', fontsize=10)
    plt.xlabel('Prediction Error (Observed - Predicted Fire Count)', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.grid(True)
    plt.tight_layout()

    plot2_path = os.path.join(output_dir, f"test_prediction_error_distribution_{base_name}.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"Saved histogram plot: {plot2_path}")
