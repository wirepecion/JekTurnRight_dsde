# JekTurnRight_dsde

External Repo :
Huggingface Model : https://huggingface.co/sirasira/flood-lstm-v1/tree/main
FASTAPI (Huggingface Space) : https://huggingface.co/spaces/sirasira/bangkok-flood-api/blob/main/app.py

## Make Ikernel
```bash
uv venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
uv pip install -e .
uv pip install ipykernel
python -m ipykernel install --user --name traffy-dsde
```

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw data files (excluded from git)
│   └── processed/        # Processed data files (excluded from git)
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── ds/              # Data Science modules
│   ├── de/              # Data Engineering modules
│   │   ├── extract/
│   │   │   └── web_scraper/  # Web scraping utilities
│   │   ├── transform/        # Data transformation logic
│   │   ├── load/            # Data loading utilities
│   │   └── spark_jobs/      # PySpark ETL jobs
│   └── common/          # Shared utilities
├── pipelines/
│   └── jobs/            # Pipeline job definitions
├── tests/               # Test files
└── pyproject.toml       # Project configuration and dependencies
```

## Setup with uv

### Install uv

If you don't have `uv` installed, install it using one of these methods:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using pip
pip install uv
```

### Create Virtual Environment and Install Dependencies

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install project dependencies
uv pip install -e .

# Install development dependencies (optional)
uv pip install -e ".[dev]"
```

### Alternative: Use uv without explicit venv activation

`uv` can manage the virtual environment automatically:

```bash
# Install dependencies directly
uv pip install -e .

# Run Python scripts
uv run python src/de/extract/web_scraper/scraper.py

# Run any command in the uv environment
uv run pytest
```

## Dependencies

The project includes the following core dependencies:

- **pandas** (>=2.0.0): Data manipulation and analysis
- **numpy** (>=1.24.0): Numerical computing
- **requests** (>=2.31.0): HTTP library for web scraping
- **pyspark** (>=3.5.0): Big data processing with Apache Spark
- **scikit-learn** (>=1.3.0): Machine learning library

## Example Usage

### Web Scraper

```bash
# Run the web scraper example
uv run python src/de/extract/web_scraper/scraper.py
```

### Download Job (Skip if Exists)

```bash
# Run the download job
uv run python pipelines/jobs/download_job.py
```

### Spark ETL Job

```bash
# Run the Spark ETL example
uv run python src/de/spark_jobs/etl_example.py
```

## Development

### Running Tests

```bash
# Install dev dependencies first
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/
```

### Code Formatting

```bash
# Format code with black
uv run black src/ tests/

# Lint with ruff
uv run ruff check src/ tests/
```

## Data Management

- Raw data should be placed in `data/raw/`
- Processed data should be saved to `data/processed/`
- Both directories are excluded from version control via `.gitignore`
- Only `.gitkeep` files are tracked to preserve directory structure

## Notes

- This project uses a `src/` layout for better package organization
- `uv` provides faster dependency resolution compared to pip
- PySpark jobs may require Java to be installed on your system
