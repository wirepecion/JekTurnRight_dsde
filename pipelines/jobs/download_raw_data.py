import gdown
from pathlib import Path

FILE_ID = "19QkF8i1my99gjbyHe7de_qZNwgrca6R5"
OUTPUT_PATH = Path("data/raw/bangkok_traffy.csv")

def file_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Check exist
    if file_exists(OUTPUT_PATH):
        print(f"[Download] Skipped — file already exists: {OUTPUT_PATH}")
        print(f"[Download] Size: {OUTPUT_PATH.stat().st_size/1_000_000:.2f} MB")
        return
    
    # URL
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print(f"[Download] File not found, downloading now...")
    print(f"[Download] Target: {OUTPUT_PATH}")

    gdown.download(url, str(OUTPUT_PATH), quiet=False)

    if file_exists(OUTPUT_PATH):
        print(f"[Download] Completed: {OUTPUT_PATH}")
    else:
        print("[Download] Error: file not downloaded properly!")


if __name__ == "__main__":
    run()

# uv run python pipelines/jobs/download_raw_data.py