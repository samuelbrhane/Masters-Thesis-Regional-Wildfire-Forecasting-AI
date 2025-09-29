import os
import pandas as pd
import joblib
from .preprocessing import preprocess_data
from .evaluate_selected_models import plot_model_test
from ..Common_Code.data_loading import load_data
from ..Common_Code.evaluation import evaluate_model


def evaluate_top_models(mode, group_name, result_file, save_dir, model_type, zone_id=None, top_n=10):
    """Evaluate and plot test results for top-performing models."""
    if not os.path.exists(result_file):
        print(f"Tuning results not found at {result_file}. Skipping.")
        return

    results_df = pd.read_csv(result_file)
    top_candidates = results_df.sort_values(
        by=[
            "val_exact_match_percentage",
            "val_r2",
            "val_rmse",
            "val_mae",
            "train_exact_match_percentage"
        ],
        ascending=[False, False, True, True, False]
    ).head(top_n)

    df = load_data(zone_id) if mode == "zone" else load_data()
    evaluated_models = []

    for i, model_row in enumerate(top_candidates.to_dict("records")):
        model_path = model_row["model_file"]
        model_filename = os.path.basename(model_path)
        print(f"\nEvaluating model {i + 1}: {model_filename}")

        params = {
            'fire_lag': int(model_row['fire_lag']),
            'climate_lag': int(model_row['climate_lag'])
        }
        if 'fit_intercept' in model_row:
            params['fit_intercept'] = bool(model_row['fit_intercept'])

        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Failed to load model {model_filename}: {e}")
            continue

        _, _, _, _, X_test, y_test, df_test = preprocess_data(df.copy(), params)
        metrics = evaluate_model(model, X_test, y_test)

        print(f"Test metrics for {model_filename}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        evaluated_models.append({
            "model_filename": model_filename,
            "model_path": model_path,
            **params,
            **metrics
        })
        
        plot_model_test(
            model_filename=model_filename,
            model_path=model_path,
            params=params,
            save_dir=save_dir,
            zone_id=zone_id
        )

    if evaluated_models:
        evaluated_models_df = pd.DataFrame(evaluated_models)
        result_name = f"{group_name}_{model_type}_top_models_test_eval.csv"
        result_path = os.path.join(save_dir, result_name)
        evaluated_models_df.to_csv(result_path, index=False)
        print(f"\nSaved test evaluation results to {result_path}")
