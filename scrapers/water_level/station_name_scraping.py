from __future__ import annotations

from pathlib import Path
import ast
import pandas as pd

STATION_COLUMNS = [
    "Latitude",
    "Longitude",
    "StationName_Short",
    "Value_1_Rain_3Hrs",
    "Value_2_Rain_6Hrs",
    "Timestamp",
    "StatusClass",
    "DistrictCode_from_JS",
    "DistrictName",
    "Icon",
    "StatusText",
    "Value_4_Rain_12Hrs",
    "Value_5_Rain_24Hrs",
    "Value_6",
    "Value_7",
    "Value_8",
    "StationCode",
    "StatusID",
]


def parse_locations_data(raw_js_array: str) -> pd.DataFrame:
    """
    Convert a raw JS-like array string into a clean station metadata DataFrame.

    Parameters
    ----------
    raw_js_array : str
        The content of the JavaScript array (e.g. "[[13.7,100.5,'name',...], ...]")

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per station and selected clean columns.
    """
    # 1) Safely convert JS array string -> Python list
    try:
        locations_data = ast.literal_eval(raw_js_array)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse JS array string: {e}") from e

    # 2) Full DataFrame with all columns
    df = pd.DataFrame(locations_data, columns=STATION_COLUMNS)

    # 3) Keep only the useful subset
    final_df = (
        df[
            [
                "StationCode",
                "StationName_Short",
                "DistrictName",
                "Latitude",
                "Longitude",
                "Timestamp",
                "DistrictCode_from_JS",
            ]
        ]
        .copy()
        .rename(columns={"Timestamp": "LastReadingTime"})
    )

    return final_df


def parse_locations_file(path: Path) -> pd.DataFrame:
    """
    Read the JS array content from a file and return a clean station metadata DataFrame.
    """
    raw_js_array = path.read_text(encoding="utf-8")
    return parse_locations_data(raw_js_array)
