import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from constants import DATA_PATH, RESULTS_DIR, SEED
from tuning_config import SEARCH_SPACE
from src.data_loading import load_data
from src.preprocessing import preprocess_data
from src.model_builder import build_model
from src.training import train_model
from src.evaluation import evaluate_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

zone_id = 8
NUM_TRIALS = 300
model_dir = os.path.join("models", f"zone_{zone_id}")
results_csv_path = os.path.join(RESULTS_DIR, f"zone_{zone_id}_tuning_results.csv")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
df = load_data(zone_id, DATA_PATH)
print(f"Loaded data for Zone {zone_id}: {len(df)} rows")

search_space = SEARCH_SPACE

def clean_for_python(obj):
    """Convert NumPy types to native Python types."""
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

# Trial loop
for trial in range(NUM_TRIALS):
    print(f"\nStarting trial {trial + 1}/{NUM_TRIALS}")

    try:
        params = {}
        for key, value in search_space.items():
            if isinstance(value, list):
                params[key] = random.choice(value)
            elif isinstance(value, tuple) and len(value) == 2:
                if all(isinstance(v, int) for v in value):
                    params[key] = random.randint(*value)
                else:
                    params[key] = random.uniform(*value)
            else:
                raise ValueError(f"Unknown parameter format for {key}: {value}")

        X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, df_test, scaler_y = preprocess_data(df, params)
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

        model = build_model(params, input_shape)
        model, history = train_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, params)

        train_metrics = evaluate_model(model, X_train_seq, y_train_seq, scaler_y)
        val_metrics = evaluate_model(model, X_val_seq, y_val_seq, scaler_y)

        model_filename = f"trial_{trial + 1:03d}.h5"
        model_path = os.path.join(model_dir, model_filename)
        model.save(model_path)

        trial_result = {**params}
        for k, v in train_metrics.items():
            trial_result[f"train_{k}"] = v
        for k, v in val_metrics.items():
            trial_result[f"val_{k}"] = v
        trial_result['model_file'] = model_path

        cleaned_result = clean_for_python(trial_result)
        results_df = pd.DataFrame([cleaned_result])
        results_df.to_csv(
            results_csv_path,
            mode='a',
            header=not os.path.exists(results_csv_path),
            index=False
        )

    except tf.errors.ResourceExhaustedError:
        print(f"Skipping trial {trial + 1}: Resource Exhausted (OOM)")
        continue

    except Exception as e:
        print(f"Trial {trial + 1} failed due to: {e}")
        continue

    finally:
        K.clear_session()

print(f"\nTuning complete! Incremental results saved to {results_csv_path}")
