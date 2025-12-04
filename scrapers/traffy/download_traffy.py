from __future__ import annotations
import gdown
from pathlib import Path
import time

def file_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def download_from_gdrive(file_id: str, output_path: Path, retries: int = 3) -> None:
    """Download a file from Google Drive and save it to output_path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if exists
    if file_exists(output_path):
        print(f"[Download] Skipped — file already exists: {output_path}")
        print(f"[Download] Size: {output_path.stat().st_size/1_000_000:.2f} MB")
        return

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[Download] Starting download...")
    print(f"[Download] URL: {url}")
    print(f"[Download] Target: {output_path}")

    for attempt in range(1, retries + 1):
        try:
            gdown.download(url, str(output_path), quiet=False)

            if file_exists(output_path):
                print(f"[Download] Completed: {output_path}")
                return

            print(f"[Download] Attempt {attempt}: File not downloaded properly.")

        except Exception as e:
            print(f"[Download] Attempt {attempt}: Error occurred -> {e}")

        if attempt < retries:
            time.sleep(2)

    raise RuntimeError(f"[Download] Failed after {retries} attempts.")
