import os
import random
import numpy as np
import pandas as pd
import joblib
from constants import DATA_PATH, SEED, RESULTS_DIR, MODELS_DIR
from tuning_config import GPR_SEARCH_SPACE as SEARCH_SPACE
from src.data_loading import load_data
from src.preprocessing import preprocess_data
from src.model_builder import build_and_train_model
from src.evaluation import evaluate_model

random.seed(SEED)
np.random.seed(SEED)

ZONES = range(2, 9)
NUM_TRIALS = 100
START_ZONE = 7
START_TRIAL = 17

def clean_for_python(obj):
    if isinstance(obj, dict):
        return {k: clean_for_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_python(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

for zone_id in ZONES:
    if zone_id < START_ZONE:
        continue

    print(f"\n=== Starting Zone {zone_id} ===")
    model_dir = os.path.join(MODELS_DIR, f"zone_{zone_id}")
    results_csv_path = os.path.join(RESULTS_DIR, f"zone_{zone_id}_tuning_results.csv")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    completed_trials = 0
    if os.path.exists(results_csv_path):
        existing_df = pd.read_csv(results_csv_path)
        completed_trials = len(existing_df)
        print(f"Found {completed_trials} completed trials. Resuming...")

    try:
        df = load_data(zone_id, DATA_PATH)
        print(f"Loaded data for Zone {zone_id}: {len(df)} rows")
    except Exception as e:
        print(f"Skipping Zone {zone_id} due to error: {e}")
        continue

    trial_start_index = max(completed_trials, START_TRIAL)

    for trial in range(trial_start_index, NUM_TRIALS):
        print(f"\n--- Zone {zone_id}, Trial {trial + 1}/{NUM_TRIALS} ---")

        try:
            params = {}
            for key, value in SEARCH_SPACE.items():
                if isinstance(value, list):
                    params[key] = random.choice(value)
                elif isinstance(value, tuple):
                    if all(isinstance(v, int) for v in value):
                        params[key] = random.randint(*value)
                    else:
                        params[key] = random.uniform(*value)
                else:
                    raise ValueError(f"Invalid parameter format: {key} → {value}")

            X_train, y_train, X_val, y_val, X_test, y_test, _ = preprocess_data(df.copy(), params)
            model = build_and_train_model(params, X_train, y_train)

            train_metrics = evaluate_model(model, X_train, y_train)
            val_metrics = evaluate_model(model, X_val, y_val)

            model_filename = f"trial_{trial + 1:03d}.pkl"
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(model, model_path)

            trial_result = {
                **params,
                "model_file": model_path,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test)
            }

            for k, v in train_metrics.items():
                trial_result[f"train_{k}"] = v
            for k, v in val_metrics.items():
                trial_result[f"val_{k}"] = v

            trial_result = clean_for_python(trial_result)
            results_df = pd.DataFrame([trial_result])
            results_df.to_csv(
                results_csv_path,
                mode='a',
                header=not os.path.exists(results_csv_path),
                index=False
            )

        except Exception as e:
            print(f"Trial {trial + 1} failed: {e}")
            continue

    print(f"Finished Zone {zone_id}. Results saved to: {results_csv_path}")

print("\n All GPR tuning complete.")
