import os
import random
import numpy as np
import pandas as pd
import joblib

from constants import RESULTS_DIR, MODELS_DIR, SEED
from tuning_config import GPR_SEARCH_SPACE as SEARCH_SPACE
from src.data_loading import load_data
from src.preprocessing import preprocess_data
from src.model_builder import build_and_train_model
from src.evaluation import evaluate_model

random.seed(SEED)
np.random.seed(SEED)

print("Loading data for all 8 zones (regional model)")
df = load_data()  
print(f"Total rows: {len(df)}")

NUM_TRIALS = 100
group_name = "regional"

model_dir = os.path.join(MODELS_DIR, group_name)
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
                    params[key] = random.randint(*value)
                else:
                    params[key] = random.uniform(*value)
            else:
                raise ValueError(f"Unknown parameter format: {key}")

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

        results.append(trial_result)

    except Exception as e:
        print(f"Skipping trial {trial + 1} due to error: {e}")
        continue

os.makedirs(RESULTS_DIR, exist_ok=True)
results_py_path = os.path.join(RESULTS_DIR, f"{group_name}_tuning_results.py")
results_csv_path = os.path.join(RESULTS_DIR, f"{group_name}_tuning_results.csv")

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

cleaned_results = clean_for_python(results)

with open(results_py_path, 'w') as f:
    f.write("results = [\n")
    for res in cleaned_results:
        f.write(f"    {res},\n")
    f.write("]\n")

results_df = pd.DataFrame(results)
results_df.to_csv(results_csv_path, index=False)

print(f"\nTuning complete for regional GPR model. Results saved to {results_csv_path}")
