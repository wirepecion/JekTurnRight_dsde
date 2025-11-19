# pipelines/jobs/scrape_water_levels.py
from pathlib import Path
from src.common.config import EXTERNAL  # or PROCESSED_DIR, your choice
from src.de.extract.web_scraper.water_level_scraper import scrape_range

def main():
    start = "01/01/2022"
    end = "10/01/2022"
    df = scrape_range(start, end)

    output_path = EXTERNAL / f"rain_water_{start.replace('/', '-')}_to_{end.replace('/', '-')}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

# uv run python -m pipelines.jobs.scrape_water_levels
