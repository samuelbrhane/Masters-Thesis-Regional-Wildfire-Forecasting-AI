from pathlib import Path
from ..evaluation_runner import evaluate_top_models
BASE_DIR = Path(__file__).resolve().parent
TUNING_RESULTS_DIR = BASE_DIR / "results"
MODEL_SELECTION_DIR = BASE_DIR / "model_selection_results"
MODEL_SELECTION_DIR.mkdir(exist_ok=True)

for zone_id in range(1, 9):
    result_file = TUNING_RESULTS_DIR / f"zone_{zone_id}_linear_results.csv"
    evaluate_top_models(
        mode="zone",
        group_name=f"zone_{zone_id}",
        result_file=str(result_file),
        save_dir=str(MODEL_SELECTION_DIR),
        model_type="linear",
        zone_id=zone_id
    )
