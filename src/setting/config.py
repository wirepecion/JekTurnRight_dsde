from __future__ import annotations
from pathlib import Path


def _detect_project_root() -> Path:
    """
    Try to detect the project root in a way that works from:
      - normal .py files (where __file__ exists)
      - notebooks / REPL (where __file__ does NOT exist)
    Strategy:
      1. Start from __file__ if available, else from CWD
      2. Walk upwards until we see something that looks like the repo root:
         - a 'data' directory AND a 'src' directory
         OR a '.git' folder
    """
    try:
        here = Path(__file__).resolve()
    except NameError:
        # Jupyter / IPython / REPL
        here = Path.cwd().resolve()

    candidates = [here] + list(here.parents)
    for parent in candidates:
        has_data_and_src = (parent / "data").is_dir() and (parent / "src").is_dir()
        has_git = (parent / ".git").is_dir()
        if has_data_and_src or has_git:
            return parent

    # Fallback: whatever we started from
    return here


PROJECT_ROOT: Path = _detect_project_root()

DATA_DIR: Path = PROJECT_ROOT / "data"

RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL: Path = DATA_DIR / "external"
MODEL_DIR: Path = DATA_DIR / "model"

CLEANED_TRAFFY_PATH: Path = PROCESSED_DIR / "traffy_clean.parquet"
FLOOD_TS_PATH: Path = PROCESSED_DIR / "flood_daily_by_district.parquet"


def ensure_dirs() -> None:
    """
    Create standard data dirs if they don't exist yet.
    Safe to call multiple times.
    """
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, EXTERNAL, OUTPUT_DIR, CLEANED_TRAFFY_PATH.parent, FLOOD_TS_PATH.parent]:
        d.mkdir(parents=True, exist_ok=True)
