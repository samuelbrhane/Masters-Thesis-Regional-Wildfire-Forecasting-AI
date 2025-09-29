import os
import pandas as pd
from xgboost import XGBRegressor
from src.evaluation import evaluate_model
from src.preprocessing import preprocess_data
from src.data_loading import load_data
from evaluate_saved_model import plot_model_test

group_name = "regional"
results_csv_path = os.path.join("results", f"{group_name}_tuning_results.csv")

if not os.path.exists(results_csv_path):
    print(f"Tuning results not found at {results_csv_path}.")
    exit()

# Select top 10 models
results_df = pd.read_csv(results_csv_path)
top_candidates = results_df.sort_values(
    by=[
        "val_exact_match_percentage",
        "val_r2",
        "val_rmse",
        "val_mae",
        "train_exact_match_percentage"
    ],
    ascending=[False, False, True, True, False]
).head(10)

df = load_data()
top_models = top_candidates.to_dict("records")
evaluated_models = []

# Evaluate each top model
for i, model_row in enumerate(top_models):
    model_path = model_row["model_file"]
    model_filename = os.path.basename(model_path)
    print(f"\nEvaluating model {i + 1}: {model_filename}")

    params = {
        'fire_lag': int(model_row['fire_lag']),
        'climate_lag': int(model_row['climate_lag'])
    }
    
    model = XGBRegressor()
    model.load_model(model_path)

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
        params=params
    )


eval_results_path = os.path.join("results", f"{group_name}_top_models_test_eval.csv")
evaluated_models_df = pd.DataFrame(evaluated_models)
evaluated_models_df.to_csv(eval_results_path, index=False)
print(f"\nTest evaluation of top regional models saved to {eval_results_path}")
