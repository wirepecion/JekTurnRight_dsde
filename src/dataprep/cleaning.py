import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def drop_nan_rows(df: pd.DataFrame, subset_cols: list) -> pd.DataFrame:
    """Removes rows where specific columns are missing."""
    return df.dropna(subset=subset_cols).copy()

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Converts timestamp columns to datetime objects."""
    df = df.copy()
    # Use coerce to handle bad formats gracefully (turns errors into NaT)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')
    df['last_activity'] = pd.to_datetime(df['last_activity'], format='ISO8601', errors='coerce')
    return df

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts Year, Month, Day from timestamp."""
    df = df.copy()
    df['year_timestamp'] = df['timestamp'].dt.year
    df['month_timestamp'] = df['timestamp'].dt.month
    df['days_timestamp'] = df['timestamp'].dt.day
    return df

def parse_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Splits '100.5,13.7' string into Longitude/Latitude float columns."""
    df = df.copy()
    coords = df['coords'].str.split(',', expand=True).astype(float)
    df['longitude'] = coords[0]
    df['latitude'] = coords[1]
    return df

def clean_province_name(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes Bangkok province name and filters for it."""
    df = df.copy()
    # Vectorized string replacement is faster than apply
    mask = df['province'].str.contains("กรุงเทพ", na=False)
    df.loc[mask, 'province'] = "กรุงเทพมหานคร"
    return df[df["province"] == "กรุงเทพมหานคร"]

def parse_type_column(df: pd.DataFrame, explode: bool = False) -> pd.DataFrame:
    """Cleans the 'type' column (e.g., '{flood,traffic}')."""
    def _parse_string(text):
        if not isinstance(text, str) or pd.isna(text):
            return []
        text = text.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
        return [item.strip() for item in text.split(',') if item.strip()]

    df = df[df['type'] != '{}'].copy()
    df['type_list'] = df['type'].apply(_parse_string)
    
    if explode:
        return df.explode('type_list')
    return df

def impute_missing(df: pd.DataFrame, columns: list, strategy: str = 'most_frequent', fill_value=None) -> pd.DataFrame:
    """Imputes missing values using Sklearn SimpleImputer."""
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    df[columns] = imputer.fit_transform(df[columns])
    return df

def build_flood_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes merged dataframe (reports × rainfall per subdistrict)
    and constructs ML features:
        • rainfall memory (API)
        • seasonality
        • binary flood target
    """

    # parse
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['subdistrict', 'date']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])

    # rolling mean — physics inspired soil memory
    for w in [30, 60, 90]:
        df[f'API_{w}d'] = (
            df.groupby('subdistrict')['rainfall']
              .transform(lambda x: x.rolling(w, min_periods=1).mean())
              .bfill()
        )

    # cyclic month embedding
    df['month_timestamp'] = df['date'].dt.month
    df['month_sin'] = np.sin(2*np.pi * df['month_timestamp']/12)
    df['month_cos'] = np.cos(2*np.pi * df['month_timestamp']/12)

    # binary flood label
    if 'number_of_report_flood' in df.columns:
        df['target'] = (df['number_of_report_flood'] > 0).astype(int)
        df.drop(columns=['number_of_report_flood'], inplace=True)

    # remove unused identifier columns if present
    cols_to_drop = [col for col in ('district', 'station_code') if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df