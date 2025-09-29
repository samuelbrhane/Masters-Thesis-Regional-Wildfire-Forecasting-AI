import os
import random
import numpy as np
import pandas as pd
from constants import DATA_PATH, RESULTS_DIR, SEED
from tuning_config import XGBOOST_SEARCH_SPACE as SEARCH_SPACE
from src.data_loading import load_data
from src.preprocessing import preprocess_data
from src.model_builder import build_model
from src.training import train_model
from src.evaluation import evaluate_model

random.seed(SEED)
np.random.seed(SEED)

NUM_TRIALS = 300

def clean_for_python(obj):
    if isinstance(obj, dict):
        return {k: clean_for_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_python(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

# Loop over zones 1 to 8
for zone_id in range(1, 9):
    print(f"\n=== Tuning for Zone {zone_id} ===")

    try:
        df = load_data(zone_id, DATA_PATH)
        print(f"Loaded data for Zone {zone_id}: {len(df)} rows")
    except Exception as e:
        print(f"Skipping Zone {zone_id} due to error: {e}")
        continue

    model_dir = os.path.join("models", f"zone_{zone_id}")
    os.makedirs(model_dir, exist_ok=True)

    results = []

    for trial in range(NUM_TRIALS):
        print(f"\nStarting trial {trial + 1}/{NUM_TRIALS}")

        try:
            params = {}
            for key, value in SEARCH_SPACE.items():
                if isinstance(value, list):
                    params[key] = random.choice(value)
                elif isinstance(value, tuple) and len(value) == 2:
                    if all(isinstance(v, int) for v in value):
                        params[key] = random.randint(value[0], value[1])
                    else:
                        params[key] = random.uniform(value[0], value[1])
                else:
                    raise ValueError(f"Unknown parameter format: {key}")

            X_train, y_train, X_val, y_val, X_test, y_test, _ = preprocess_data(df, params)

            model = build_model(params, SEED)
            model = train_model(model, X_train, y_train, X_val, y_val)

            train_metrics = evaluate_model(model, X_train, y_train)
            val_metrics = evaluate_model(model, X_val, y_val)

            model_filename = f"trial_{trial + 1:03d}.json"
            model_path = os.path.join(model_dir, model_filename)
            model.save_model(model_path)

            trial_result = {**params}
            trial_result["model_file"] = model_path
            for k, v in train_metrics.items():
                trial_result[f"train_{k}"] = v
            for k, v in val_metrics.items():
                trial_result[f"val_{k}"] = v

            results.append(trial_result)

        except Exception as e:
            print(f"Skipping trial {trial + 1} due to error: {e}")
            continue

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_py_path = os.path.join(RESULTS_DIR, f"zone_{zone_id}_tuning_results.py")
    results_csv_path = os.path.join(RESULTS_DIR, f"zone_{zone_id}_tuning_results.csv")

    cleaned_results = clean_for_python(results)

    with open(results_py_path, 'w') as f:
        f.write("results = [\n")
        for res in cleaned_results:
            f.write(f"    {res},\n")
        f.write("]\n")

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv_path, index=False)

    print(f"Tuning complete! Results saved to {results_csv_path}")
