import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

#=============== Load Data ===============
def get_raw_data(file_path):
    return pd.read_csv(file_path)

def get_cleaned_data(file_path):
    raw_data = get_raw_data(file_path)
    return clean_data(raw_data)

def get_shape_file(file_path):
    return gpd.read_file(file_path)

#=============== Clean Data ===============#

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_not_used_columns(df)   #Drop not used columns
    df = drop_nan_rows(df)           #Drop nan value (rows) 
    df = convert_data_types(df)      #Convert data types
    df = add_crucial_cols(df)        #Add crucial columns
    df = clean_type_columns(df)      #Clean 'type' column

    return df
#--------------------------------------------------------------------------------------------------------------
def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:

    #Convert data types
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df['last_activity'] = pd.to_datetime(df['last_activity'], format='ISO8601')
    
    return df
#--------------------------------------------------------------------------------------------------------------
def add_crucial_cols(df: pd.DataFrame) -> pd.DataFrame:

    #Create new timestamp's year month and date columns
    df['year_timestamp'] = df['timestamp'].dt.year
    df['month_timestamp'] = df['timestamp'].dt.month
    df['days_timestamp'] = df['timestamp'].dt.days_in_month

    #Extract longitude and latitude from 'coords' column
    df[['longitude', 'latitude']] = df['coords'].str.split(',', expand=True).astype(float)
    
    return df
#--------------------------------------------------------------------------------------------------------------
def drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:

    #Drop row which contain nan value 
    cols = ['ticket_id', 'type', 'organization', 'coords', 'timestamp', 'last_activity']
    df = df.dropna(subset=cols, inplace=False)
    
    return df
#--------------------------------------------------------------------------------------------------------------
def drop_not_used_columns(df: pd.DataFrame) -> pd.DataFrame:

    #Drop not-used column
    not_used_cols = ['photo', 'photo_after', 'star']
    df = df.drop(columns=not_used_cols, axis='columns')  
    
    return df
#--------------------------------------------------------------------------------------------------------------
def clean_type_columns(df: pd.DataFrame, explode: bool = False) -> pd.DataFrame:
    
    df = df[df['type'] != '{}']

    if(explode):
        df['type_list'] = df['type'].apply(parse_type_string) 
        df = df.explode['type_list']
    
    return df

#--------------------------------------------------------------------------------------------------------------
def clean_address(df: pd.DataFrame, shape: gpd.GeoDataFrame, drop_missed_value: bool = False) -> pd.DataFrame:
    
    df_col  = 'subdistrict'
    BMA_col = 'SUBDISTR_1'
    tag     = 'น้ำท่วม'

#--------------------------------------------------------------------------------------------------------------
def check_subdistrict_and_district(check:str,sub_dist:str):
    '''
    check = path of .shp ,that want to check
    sub_dist = path of .csv that contain true subdistrict and district
    '''
    subdiswithdis = pd.read_csv(sub_dist)
    improve = gpd.read_file(check)
    # แก้ชื่อ subdistrict
    # แก้ชื่อ district
    improve = check_subdistrict(improve,subdiswithdis)
    improve = check_district(improve,subdiswithdis)
    return improve

#--------------------------------------------------------------------------------------------------------------
def check_subdistrict(check:gpd.GeoDataFrame,subdistrict:pd.DataFrame):
    check['SUBDISTRIC'] =check['SUBDISTRIC'].astype('int')
    if(len(check.loc[~check['SUBDISTR_1'].isin(subdistrict['sname']),'SUBDISTR_1']) != 0):
        subdistrict_dict = dict(zip(subdistrict['scode'], subdistrict['sname']))
        check['SUBDISTR_1'] = check['SUBDISTRIC'].map(subdistrict_dict)
    return check

#--------------------------------------------------------------------------------------------------------------
def check_district(check:gpd.GeoDataFrame,district:pd.DataFrame):
    check['DISTRICT_I'] =check['DISTRICT_I'].astype('int')
    if(len(check.loc[~check['DISTRICT_N'].isin(district['dname']),'DISTRICT_N'])!=0): 
        district_dict = dict(zip(district['dcode'], district['dname']))
        check['DISTRICT_N'] = check['DISTRICT_I'].map(district_dict)
    return check

#--------------------------------------------------------------------------------------------------------------
def filter_year(df:pd.DataFrame,start,stop):
    df = df[(df['year_timestamp']>=start)&(df['year_timestamp']<=stop)]
    return df

#--------------------------------------------------------------------------------------------------------------
def drop_not_use_province(df:pd.DataFrame):
    df.loc[df['province'].str.contains("กรุงเทพ"), 'province'] = "กรุงเทพมหานคร"
    df = df[(df["province"] == "กรุงเทพมหานคร")]
    return df

#--------------------------------------------------------------------------------------------------------------
def verify_coordination_with_address(df:pd.DataFrame,verify_df:gpd.GeoDataFrame):
    df_points = gpd.GeoDataFrame(df,geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],crs="EPSG:4326")
    if verify_df.crs != "EPSG:4326":
        verify_df = verify_df.to_crs("EPSG:4326")
    joined = gpd.sjoin(df_points, verify_df, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep='first')]
    result = joined.drop(columns=['index_right','OBJECTID','AREA_CAL','AREA_BMA','PERIMETER','ADMIN_ID','SUBDISTRIC','DISTRICT_I','CHANGWAT_I','CHANGWAT_N'])
    result['checkDis'] = (result['DISTRICT_N'] == result['district'])
    result['checkSub'] = (result['SUBDISTR_1'] == result['subdistrict'])
    result = result[(((result['checkDis']) & (result['checkSub'])))]
    return joined
 

#=============== Insight ===============#

def Co_occurrence_analysis(df, correlation_check__tag):
    s = df['type'].value_counts()

    s_roads = s[s.index.str.contains(correlation_check__tag)]
    co_occurrence_counts = {}

    for tag_set_str, count in s_roads.items():
    
        tags = tag_set_str.strip('{}').split(',')
        if len(tags) == 1:
            co_occurrence_counts[correlation_check__tag] = co_occurrence_counts.get(correlation_check__tag, 0) + count
            
        else:
            for tag in tags:
                if tag != correlation_check__tag:
                    co_occurrence_counts[tag] = co_occurrence_counts.get(tag, 0) + count

    correlation_series = pd.Series(co_occurrence_counts).sort_values(ascending=False)
    return correlation_series

def explode_columns(df):
    df_cleaned = df.dropna(subset=['type'])
    df_cleaned['type_list'] = df_cleaned['type'].apply(parse_type_string)
    df_cleaned = df_cleaned.explode('type_list')
    return df_cleaned
    
def parse_type_string(text):
    if not isinstance(text, str) or pd.isna(text):
        return []
    text = text.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
    items = text.split(',')
    cleaned_items = [item.strip() for item in items if item.strip()]
    return cleaned_items

def get_heat_map(df: pd.DataFrame, shape: gpd.GeoDataFrame, drop_missed_value: bool = False) -> pd.DataFrame:
    
    map_df_col  = 'subdistrict'
    map_BMA_col = 'SUBDISTR_1'
    tag         = 'น้ำท่วม'
    
    flood_reports = df[df['type_list'] == tag]
    flood_counts = flood_reports.groupby(map_df_col).size().reset_index(name='flood_report_count')

    map_with_flood_counts = shape.merge(flood_counts, left_on=map_BMA_col, right_on=map_df_col, how='left')
    map_with_flood_counts['flood_report_count'] = map_with_flood_counts['flood_report_count'].fillna(0)

    map_with_flood_counts.explore(
    column='flood_report_count',
    cmap='Blues',
    legend=True
)