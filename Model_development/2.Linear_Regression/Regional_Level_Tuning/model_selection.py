from pathlib import Path
from ..evaluation_runner import evaluate_top_models

BASE_DIR = Path(__file__).resolve().parent
TUNING_RESULTS_DIR = BASE_DIR / "results"
MODEL_SELECTION_DIR = BASE_DIR / "model_selection_results"
MODEL_SELECTION_DIR.mkdir(exist_ok=True)

evaluate_top_models(
    mode="regional",
    group_name="regional",
    result_file=str(TUNING_RESULTS_DIR / "regional_linear_results.csv"),
    save_dir=str(MODEL_SELECTION_DIR),
    model_type="linear"
)
