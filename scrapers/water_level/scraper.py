"""
Web scraper module for extracting data from websites.
"""
import requests
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """
    A simple web scraper for fetching and parsing web content.
    """

    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the web scraper.

        Args:
            base_url: Base URL for the scraper
            headers: Optional HTTP headers
        """
        self.base_url = base_url
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def fetch_page(self, endpoint: str = "", params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Fetch a page from the web.

        Args:
            endpoint: Endpoint to append to base URL
            params: Optional query parameters

        Returns:
            HTML content as string, or None if request failed
        """
        url = f"{self.base_url}/{endpoint}".rstrip("/")
        try:
            logger.info(f"Fetching URL: {url}")
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully fetched {url}")
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def fetch_json(self, endpoint: str = "", params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch JSON data from an API endpoint.

        Args:
            endpoint: Endpoint to append to base URL
            params: Optional query parameters

        Returns:
            JSON data as dictionary, or None if request failed
        """
        url = f"{self.base_url}/{endpoint}".rstrip("/")
        try:
            logger.info(f"Fetching JSON from: {url}")
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully fetched JSON from {url}")
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching JSON from {url}: {e}")
            return None


def scrape_example():
    """
    Example usage of the WebScraper class.
    """
    # Example: Scraping a public API
    scraper = WebScraper("https://api.github.com")
    data = scraper.fetch_json("repos/python/cpython")
    if data:
        print(f"Repository: {data.get('name')}")
        print(f"Stars: {data.get('stargazers_count')}")
        print(f"Description: {data.get('description')}")


if __name__ == "__main__":
    scrape_example()
