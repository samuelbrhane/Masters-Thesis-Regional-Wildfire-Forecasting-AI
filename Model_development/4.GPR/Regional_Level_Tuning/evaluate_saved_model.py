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


def plot_model_test(model_filename, model_path, params, save_dir="top_candidate_models_evaluation_gpr"):
    output_dir = os.path.join(save_dir, "regional")
    os.makedirs(output_dir, exist_ok=True)
    base_name = model_filename.replace('.pkl', '')

    # Load data and model
    df = load_data()
    _, _, _, _, X_test, y_test, df_test = preprocess_data(df, params)
    test_dates = df_test["Date"].reset_index(drop=True)
    model = joblib.load(model_path)

    # Predict with uncertainty
    y_mean, y_std = model.predict(X_test, return_std=True)
    y_pred = np.round(np.clip(y_mean, 0, None))
    y_true = np.round(np.clip(y_test, 0, None))
    residuals = y_true - y_pred

    # Save prediction data
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

    # Save evaluation summary
    metrics = evaluate_model(model, X_test, y_test)
    summary_path = os.path.join(output_dir, f"summary_test_diagnostics_{base_name}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model File: {model_filename}\n")
        f.write(f"Number of Test Samples: {len(y_true)}\n\n")
        f.write("Model Performance on Test Data:\n")
        f.write(f" - RMSE: {metrics['rmse']:.2f} fires/day\n")
        f.write(f" - MAE: {metrics['mae']:.2f} fires/day\n")
        f.write(f" - Mean Residual: {metrics['mean_residual']:.2f}\n")
        f.write(f" - Std Residual: {metrics['std_residual']:.2f}\n")
        f.write(f" - R² Score: {metrics['r2']:.3f}\n")
        f.write(f" - Exact Match %: {metrics['exact_match_percentage']:.2%}\n")
        f.write(f" - Mean Prediction Std: {metrics.get('mean_prediction_std', y_std.mean()):.2f}\n")
        f.write(f" - Max Prediction Std: {metrics.get('max_prediction_std', y_std.max()):.2f}\n")
    print(f"Saved summary: {summary_path}")

    # Common plotting setup
    x = np.arange(len(y_true))
    date_labels = test_dates.astype(str)
    tick_step = max(1, len(date_labels) // 25)
    xticks = np.arange(0, len(date_labels), tick_step)
    y_true_plot = np.where(y_true == 0, 0.02, y_true)
    y_pred_plot = np.where(y_pred == 0, 0.02, y_pred)
    y_upper_95 = y_mean + 1.96 * y_std
    y_lower_95 = np.maximum(0, y_mean - 1.96 * y_std)

    # === Plot 1: Predicted vs Observed ===
    plt.figure(figsize=(9, 5))
    plt.plot(x, y_true_plot, label='Observed Fire Counts', color='blue', linewidth=0.6)
    plt.plot(x, y_pred_plot, label='Predicted Fire Counts', color='red', linewidth=0.6)
    plt.xticks(ticks=xticks, labels=date_labels[::tick_step], rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('Predicted vs. Observed Fires (No Uncertainty)', fontsize=10)
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Number of Fires', fontsize=8)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plot_path_no_unc = os.path.join(output_dir, f"test_predicted_vs_observed_{base_name}_no_uncertainty.png")
    plt.savefig(plot_path_no_unc, dpi=300)
    plt.close()
    print(f"Saved plot (no uncertainty): {plot_path_no_unc}")

    # === Plot 2: Observed vs. Prediction Confidence Interval (95%) ===
    plt.figure(figsize=(9, 5))
    plt.plot(x, y_true_plot, color='blue', linewidth=0.6, label='Observed Fire Counts')
    plt.fill_between(x, y_lower_95, y_upper_95, color='red', alpha=0.2, label='95% Confidence Interval')

    plt.xticks(ticks=xticks, labels=date_labels[::tick_step], rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('Observed Fires vs. Prediction Confidence Interval (95%)', fontsize=10)
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Number of Fires', fontsize=8)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plot_path_ci = os.path.join(output_dir, f"test_prediction_confidence_interval_{base_name}.png")
    plt.savefig(plot_path_ci, dpi=300)
    plt.close()
    print(f"Saved confidence interval plot: {plot_path_ci}")


    # === Plot 3: Prediction Error Distribution ===
    plt.figure(figsize=(9, 5))
    min_val = int(np.floor(residuals.min()))
    max_val = int(np.ceil(residuals.max()))
    bin_edges = np.arange(min_val - 1, max_val + 2, 2)
    histplot = sns.histplot(residuals, bins=bin_edges, color='orange', edgecolor='k', alpha=0.75)
    for patch in histplot.patches:
        height = patch.get_height()
        if height > 0:
            x_text = patch.get_x() + patch.get_width() / 2
            plt.text(x_text, height + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=6)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.7)
    plt.title('Prediction Error Distribution (Test Set)', fontsize=10)
    plt.xlabel('Prediction Error (Observed - Predicted)', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.grid(True)
    plt.tight_layout()
    plot_path_error = os.path.join(output_dir, f"test_prediction_error_distribution_{base_name}.png")
    plt.savefig(plot_path_error, dpi=300)
    plt.close()
    print(f"Saved error plot: {plot_path_error}")

    # === Plot 4: Std Dev Only ===
    plt.figure(figsize=(9, 4))
    plt.plot(x, y_std, color='purple', linewidth=0.8, label='Prediction Std Dev')
    plt.xticks(ticks=xticks, labels=date_labels[::tick_step], rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('Prediction Uncertainty Over Time (Std Dev)', fontsize=10)
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Std Dev of Prediction', fontsize=8)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plot_path_uncertainty = os.path.join(output_dir, f"test_prediction_uncertainty_std_only_{base_name}.png")
    plt.savefig(plot_path_uncertainty, dpi=300)
    plt.close()
    print(f"Saved uncertainty-only plot: {plot_path_uncertainty}")
