"""
Tests for common utilities.
"""
import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.utils import get_timestamp, ensure_dir


def test_get_timestamp():
    """Test timestamp generation."""
    timestamp = get_timestamp()
    assert isinstance(timestamp, str)
    assert len(timestamp) > 0


def test_ensure_dir(tmp_path):
    """Test directory creation."""
    test_dir = tmp_path / "test_dir"
    result = ensure_dir(test_dir)
    assert result.exists()
    assert result.is_dir()


def test_ensure_dir_existing(tmp_path):
    """Test ensure_dir with existing directory."""
    test_dir = tmp_path / "existing_dir"
    test_dir.mkdir()
    result = ensure_dir(test_dir)
    assert result.exists()
    assert result == test_dir
