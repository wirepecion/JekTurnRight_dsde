"""
Download job that skips downloading if file already exists.
"""
import os
import requests
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DownloadJob:
    """
    A job for downloading files with skip-if-exists functionality.
    """

    def __init__(self, download_dir: str = "data/raw"):
        """
        Initialize the download job.

        Args:
            download_dir: Directory to save downloaded files
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download_file(
        self,
        url: str,
        filename: Optional[str] = None,
        skip_if_exists: bool = True,
        chunk_size: int = 8192
    ) -> Optional[Path]:
        """
        Download a file from a URL.

        Args:
            url: URL to download from
            filename: Optional filename to save as (defaults to URL filename)
            skip_if_exists: Skip download if file already exists
            chunk_size: Size of chunks for streaming download

        Returns:
            Path to downloaded file, or None if download failed
        """
        # Determine filename
        if filename is None:
            filename = url.split("/")[-1]
        
        filepath = self.download_dir / filename

        # Check if file exists
        if skip_if_exists and filepath.exists():
            logger.info(f"File already exists, skipping download: {filepath}")
            return filepath

        try:
            logger.info(f"Downloading {url} to {filepath}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress for large files
                        if total_size > 0 and downloaded % (chunk_size * 100) == 0:
                            progress = (downloaded / total_size) * 100
                            logger.debug(f"Download progress: {progress:.1f}%")

            logger.info(f"Successfully downloaded to {filepath} ({downloaded} bytes)")
            return filepath

        except requests.RequestException as e:
            logger.error(f"Error downloading {url}: {e}")
            # Clean up partial download
            if filepath.exists():
                filepath.unlink()
            return None

    def download_multiple(
        self,
        urls: list[str],
        skip_if_exists: bool = True
    ) -> list[Optional[Path]]:
        """
        Download multiple files.

        Args:
            urls: List of URLs to download
            skip_if_exists: Skip download if files already exist

        Returns:
            List of paths to downloaded files (None for failed downloads)
        """
        results = []
        for url in urls:
            result = self.download_file(url, skip_if_exists=skip_if_exists)
            results.append(result)
        return results


def example_download():
    """
    Example usage of the DownloadJob class.
    """
    job = DownloadJob("data/raw")
    
    # Example: Download a sample CSV file
    sample_urls = [
        "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv",
    ]
    
    print("Starting downloads...")
    results = job.download_multiple(sample_urls, skip_if_exists=True)
    
    print(f"\nDownload Summary:")
    for url, path in zip(sample_urls, results):
        if path:
            print(f"✓ {url.split('/')[-1]} - Downloaded to {path}")
        else:
            print(f"✗ {url.split('/')[-1]} - Failed")


if __name__ == "__main__":
    example_download()
