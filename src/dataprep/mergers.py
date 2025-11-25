import pandas as pd
from .cleaning import impute_missing

def clean_station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes station metadata column names."""
    df = df.rename(columns={
        'StationCode': 'station_code', 'StationName_Short': 'station_name',
        'DistrictName': 'district', 'Latitude': 'latitude', 'Longitude': 'longitude'
    })
    df['district'] = df['district'].replace({
        'ป้อมปราบฯ': 'ป้อมปราบศัตรูพ่าย',
        'ราษฏร์บูรณะ': 'ราษฎร์บูรณะ'
    })
    mask = df['station_code'].str.len() == 9
    df.loc[mask, 'station_code'] = df.loc[mask, 'station_code'].str[3:]
    return df

def clean_rainfall_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans rainfall data and melts it from wide to long format."""
    df = df.drop(columns=['NKM.03','LSI.02','MBR.03','NJK.04','SPK.01'], errors='ignore')
    df = impute_missing(df, df.columns[df.isna().any()].tolist(), strategy='most_frequent')
    df = df.rename(columns={'Date': 'date'})
    
    # Melt to long format
    df_long = df.melt(id_vars='date', var_name='station_code', value_name='rainfall')
    df_long['date'] = pd.to_datetime(df_long['date'], format='mixed').dt.date
    return df_long

def expand_rainfall_to_subdistricts(rain_df: pd.DataFrame, shape_with_station: pd.DataFrame) -> pd.DataFrame:
    """
    Joins Rainfall (by Station) with Shapefile Info (by Subdistrict).
    Result: A dataframe with 1 row per Date per Subdistrict.
    """
    # Merge rain with the subdistrict map on 'station_code'
    # shape_with_station has columns: [subdistrict, district, latitude, longitude, station_code]
    expanded_df = rain_df.merge(
        shape_with_station,
        on='station_code',
        how='left'
    )
    # Drop rows where mapping failed (no subdistrict found)
    return expanded_df.dropna(subset=['subdistrict', 'district'])

def merge_rainfall_with_reports(expanded_rain_df: pd.DataFrame, report_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the Subdistrict-level Rainfall with the Aggregated Flood Reports.
    """
    # 1. Aggregate Reports by Date AND Location
    # We group by subdistrict/district to match the expanded rain data
    report_agg = report_df.groupby(['date', 'district', 'subdistrict']).agg(
        number_of_report_flood=('type_list', lambda x: x.apply(lambda l: 'น้ำท่วม' in l).sum()),
        total_report=('ticket_id', 'count')
    ).reset_index()
    
    # 2. Merge (Left Join on Rain data ensures we keep days with 0 reports)
    final_df = expanded_rain_df.merge(
        report_agg, 
        on=['date', 'district', 'subdistrict'], 
        how='left'
    )
    
    # 3. Fill NaN (No report = 0 floods)
    final_df[['number_of_report_flood', 'total_report']] = final_df[['number_of_report_flood', 'total_report']].fillna(0)
    
    # 4. Select & Order Columns to match your expectation
    cols = [
        'date', 'district', 'subdistrict', 'station_code', 
        'latitude', 'longitude', 
        'number_of_report_flood', 'total_report', 'rainfall'
    ]
    return final_df[cols]