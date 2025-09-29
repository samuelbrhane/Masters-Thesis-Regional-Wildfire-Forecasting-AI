import os
import random
import numpy as np
import pandas as pd
import joblib
from ...Common_Code.constants import DATA_PATH, SEED
from ...Common_Code.tuning_config import LINEAR_REGRESSION_SEARCH_SPACE as SEARCH_SPACE
from ...Common_Code.data_loading import load_data
from ...Common_Code.evaluation import evaluate_model
from ...Common_Code.services import clean_for_python, sample_params
from ..preprocessing import preprocess_data
from ..model_build_train import build_and_train_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

random.seed(SEED)
np.random.seed(SEED)

NUM_TRIALS = 20

for zone_id in range(1, 9):
    print(f"\n=== Tuning for Zone {zone_id} ===")
    try:
        df = load_data(zone_id, DATA_PATH)
    except Exception as e:
        print(f"Skipping Zone {zone_id} due to error: {e}")
        continue

    model_dir = os.path.join(MODELS_DIR, f"zone_{zone_id}")
    os.makedirs(model_dir, exist_ok=True)
    results = []

    for trial in range(NUM_TRIALS):
        print(f"\nStarting trial {trial + 1}/{NUM_TRIALS}")
        try:
            params = sample_params(SEARCH_SPACE)
            X_train, y_train, X_val, y_val, X_test, y_test, _ = preprocess_data(df, params)

            model = build_and_train_model(X_train, y_train, params)

            train_metrics = evaluate_model(model, X_train, y_train)
            val_metrics = evaluate_model(model, X_val, y_val)

            model_filename = f"trial_{trial + 1:03d}.pkl"
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(model, model_path)

            trial_result = {
                **params,
                "model_file": model_path,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            }

            results.append(trial_result)
        except Exception as e:
            print(f"Skipping trial {trial + 1} due to error: {e}")
            continue

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_py_path = os.path.join(RESULTS_DIR, f"zone_{zone_id}_linear_results.py")
    results_csv_path = os.path.join(RESULTS_DIR, f"zone_{zone_id}_linear_results.csv")

    with open(results_py_path, 'w') as f:
        f.write("results = [\n")
        for res in clean_for_python(results):
            f.write(f"    {res},\n")
        f.write("]\n")

    pd.DataFrame(results).to_csv(results_csv_path, index=False)
    print(f"Tuning complete! Results saved to {results_csv_path}")
