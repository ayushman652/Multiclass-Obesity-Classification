from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DATASET_PATH = (
    WORKSPACE_ROOT / "datasets" / "obesity" / "Obesity_level_prediction_dataset.csv"
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TARGET = "NObeyesdad"

