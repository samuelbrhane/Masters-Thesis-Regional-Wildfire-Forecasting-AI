import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from src.evaluation import evaluate_model
from src.preprocessing import preprocess_data
from src.data_loading import load_data
from evaluate_saved_model import plot_model_test

# Loop through zones 1 to 8
for zone_id in range(2, 9):
    print(f"\n=== Evaluating Zone {zone_id} ===")

    results_csv_path = os.path.join("results", f"zone_{zone_id}_transformer_tuning_results.csv")
    if not os.path.exists(results_csv_path):
        print(f"Skipping Zone {zone_id} — no tuning results found.")
        continue

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

    df = load_data(zone_id)
    top_models = top_candidates.to_dict("records")
    evaluated_models = []

    for i, model_row in enumerate(top_models):
        model_path = model_row["model_file"]
        model_filename = os.path.basename(model_path)
        print(f"\nEvaluating model {i + 1}: {model_filename}")

        params = {
            'batch_size': model_row['batch_size'],
            'learning_rate': model_row['learning_rate'],
            'dropout_rate': model_row['dropout_rate'],
            'lag_days': model_row['lag_days'],
            'sequence_length': model_row['sequence_length'],
            'd_model': model_row['d_model'],
            'num_heads': model_row['num_heads'],
            'ff_dim': model_row['ff_dim'],
            'num_layers': model_row['num_layers'],
            'month_embed_dim': model_row['month_embed_dim'],
            'dow_embed_dim': model_row['dow_embed_dim'],
            'optimizer': model_row['optimizer']
        }

        custom_objects = {"mse": MeanSquaredError()}
        model = load_model(model_path, custom_objects=custom_objects)

        _, _, _, _, _, _, _, _, X_test_seq, y_test_seq, month_test_seq, dow_test_seq, df_test, scaler_y = preprocess_data(df.copy(), params)


        X_test_dict = {
            'numeric_input': X_test_seq,
            'month_input': month_test_seq,
            'dow_input': dow_test_seq
        }

        metrics = evaluate_model(model, X_test_dict, y_test_seq, scaler_y)
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
            zone_id=zone_id,
            model_filename=model_filename,
            model_path=model_path,
            params=params
        )

    if evaluated_models:
        evaluated_models_df = pd.DataFrame(evaluated_models)
        eval_results_path = os.path.join("results", f"zone_{zone_id}_transformer_top_models_test_eval.csv")
        evaluated_models_df.to_csv(eval_results_path, index=False)
        print(f"Saved evaluation results for Zone {zone_id} to {eval_results_path}")
