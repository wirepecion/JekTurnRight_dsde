from pathlib import Path
from src.common.config import RAW_DIR, EXTERNAL
from src.de.extract.web_scraper.station_name_scraping import parse_locations_file


def main() -> None:
    # Where the raw JS array is stored
    input_path = RAW_DIR / "bma_rain_station_js_array.txt"

    # Where you want to save cleaned metadata
    output_path = EXTERNAL / "station_metadata.csv"

    print(f"[StationMetadata] Reading raw JS array from: {input_path}")
    station_df = parse_locations_file(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    station_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[StationMetadata] ✅ Saved cleaned metadata to: {output_path}")
    print(f"[StationMetadata] Total stations processed: {len(station_df)}")
    print("[StationMetadata] Sample:")
    print(station_df.head().to_string(index=False))


if __name__ == "__main__":
    main()

# uv run python -m pipelines.jobs.station_name_scraping
