from pathlib import Path
from src.common.config import RAW_DIR
from src.de.extract.download_traffy import download_from_gdrive

FILE_ID = "19QkF8i1my99gjbyHe7de_qZNwgrca6R5"

def main():
    output_path = RAW_DIR / "bangkok_traffy.csv"
    download_from_gdrive(FILE_ID, output_path)

if __name__ == "__main__":
    main()
# uv run python -m pipelines.jobs.download_raw_data
