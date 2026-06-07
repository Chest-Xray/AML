from pathlib import Path

BATCH_SIZE: int = 4  # play around with this on Habrok
SEED: int = 42
NUM_WORKERS: int = 6    # play around with this on Habrok
K_FOLDS: int = 4
NUM_EPOCHS: int = 25
MODEL_PATH: Path = Path(__file__).parent.parent / "data" / "models"
