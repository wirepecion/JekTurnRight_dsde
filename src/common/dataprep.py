import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.impute import SimpleImputer
from sklearn.neighbors import BallTree

#=============== Load Data ===============
def get_raw_data(file_path:str):
    return pd.read_csv(file_path)

def get_cleaned_data(file_path:str,shape_path:str,csv_to_check_shape_path:str = None):
    raw_data = get_raw_data(file_path)
    return clean_data(raw_data,shape_path,csv_to_check_shape_path)

def get_shape_file(file_path:str):
    gdf = gpd.read_file(file_path)
    gdf.rename(columns={
        'SUBDISTRIC':'subdistrict_id',
        'SUBDISTR_1':'subdistrict',
        'DISTRICT_I':'district_id',
        'DISTRICT_N':'district',
    	'CHANGWAT_N':'changwat',
    	'Shape_Area':'shape_area'
    },inplace=True)
    gdf.drop(columns=['OBJECTID','AREA_CAL','AREA_BMA','PERIMETER',	'ADMIN_ID', 'CHANGWAT_I','Shape_Leng'],inplace=True)
    return gdf

#=============== Clean Data ===============#

def clean_data(df: pd.DataFrame,shape_path:str,csv_to_check_shape_path:str = None) -> pd.DataFrame:
    df = drop_nan_rows(df)           #Drop nan value (rows) 
    df = convert_data_types(df)      #Convert data types
    df = add_crucial_cols(df)        #Add crucial columns
    df = filter_year(df,start=2022,stop=2024)
    df = drop_not_use_province(df)
    
    shape_gdf =get_shape_file(shape_path)
    if(csv_to_check_shape_path is not None):
        check_df = get_raw_data(csv_to_check_shape_path)
        df = verify_coordination_with_address(df=df,verify_df=shape_gdf,check=check_df)
    else:
        df = verify_coordination_with_address(df=df,verify_df=shape_gdf)
    cleaned = clean_type_columns(df)      #Clean 'type' column
    
    return cleaned
#=============== Merge Data From Scrape ===============#
def merge_with_scraped_data(data:pd.DataFrame,station_path:str,rainfall_path:str,shape_path:str,csv_to_check_shape_path:str = None):
    df = data.copy()
    station_df = get_raw_data(station_path)
    station_df = clean_station(station_df)
    rainfall_df = get_raw_data(rainfall_path)
    rainfall_df = clean_rainfall(rainfall_df)
    shape_df = get_shape_file(shape_path)
    shape_df = get_centroid(shape_df)
    df = get_report_detail(df)
    df = get_centroid_of_subdistrict(df,shape_df)
    df = make_station_info_nearest(df,station_df)
    shape_df = make_station_info_nearest(shape_df,station_df)
    rain_and_station_code = get_station_code_for_rain(rainfall_df,shape_df)
    result = get_result_of_merge(df,rain_and_station_code)
    return result

#--------------------------------------------------------------------------------------------------------------
def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    #Convert data types
    df =df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df['last_activity'] = pd.to_datetime(df['last_activity'], format='ISO8601')
    return df

def convert_to_date(df:pd.DataFrame)-> pd.DataFrame:
    df =df.copy()
    df['timestamp'] = df['timestamp'].dt.date
    df.rename(columns={'timestamp':'date'},inplace=True)
    df['last_activity'] = df['last_activity'].dt.date
    return df
#--------------------------------------------------------------------------------------------------------------
def add_crucial_cols(df: pd.DataFrame) -> pd.DataFrame:
    #Create new timestamp's year month and date columns
    df =df.copy()
    df['year_timestamp'] = df['timestamp'].dt.year
    df['month_timestamp'] = df['timestamp'].dt.month
    df['days_timestamp'] = df['timestamp'].dt.day
    # df['date_timestamp'] = df['timestamp'].dt.date

    #Extract longitude and latitude from 'coords' column
    df[['longitude', 'latitude']] = df['coords'].str.split(',', expand=True).astype(float)
    return df
    

#--------------------------------------------------------------------------------------------------------------
def drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    df =df.copy()
    #Drop row which contain nan value 
    cols = ['ticket_id', 'type', 'organization', 'coords', 'province', 'timestamp', 'last_activity']
    df = df.dropna(subset=cols)
    return df

#--------------------------------------------------------------------------------------------------------------
def drop_not_used_columns(df: pd.DataFrame,cols:list) -> pd.DataFrame:
    #Drop not-used column
    df = df.copy()
    df = df.drop(columns=cols, axis='columns')  
    return df

#--------------------------------------------------------------------------------------------------------------
def clean_type_columns(df: pd.DataFrame, explode: bool = False) -> pd.DataFrame:
    df =df.copy()
    df = df[df['type'] != '{}']
    df['type_list'] = df['type'].apply(parse_type_string) 

    if(explode):
        df = df.explode('type_list')
    
    return df
#--------------------------------------------------------------------------------------------------------------
def verify_geopandas(shape:gpd.GeoDataFrame,check_df:pd.DataFrame=None) -> gpd.GeoDataFrame:
    '''
    shape_ = gdf of .shp ,that want to check
    check_df = df that contain true subdistrict and district
    '''
    if check_df is not None:
        # แก้ชื่อ subdistrict
        shape = check_subdistrict(shape,check_df)
        # แก้ชื่อ district
        shape = check_district(shape,check_df)
    return shape

#--------------------------------------------------------------------------------------------------------------
def check_subdistrict(shape_gdf:gpd.GeoDataFrame,subdistrict:pd.DataFrame) -> gpd.GeoDataFrame:
    shape_gdf['subdistrict_id'] =shape_gdf['subdistrict_id'].astype('int')
    if(len(shape_gdf.loc[~shape_gdf['subdistrict'].isin(subdistrict['sname']),'subdistrict']) != 0):
        subdistrict_dict = dict(zip(subdistrict['scode'], subdistrict['sname']))
        shape_gdf['subdistrict'] = shape_gdf['subdistrict_id'].map(subdistrict_dict)
    return shape_gdf

#--------------------------------------------------------------------------------------------------------------
def check_district(shape_gdf:gpd.GeoDataFrame,district:pd.DataFrame) -> gpd.GeoDataFrame:
    shape_gdf['district_id'] =shape_gdf['district_id'].astype('int')
    if(len(shape_gdf.loc[~shape_gdf['district'].isin(district['dname']),'district'])!=0): 
        district_dict = dict(zip(district['dcode'], district['dname']))
        shape_gdf['district'] = shape_gdf['district_id'].map(district_dict)
    return shape_gdf

#--------------------------------------------------------------------------------------------------------------
def filter_year(df:pd.DataFrame,start,stop) -> pd.DataFrame:
    df = df[(df['year_timestamp']>=start)&(df['year_timestamp']<=stop)]
    return df


#--------------------------------------------------------------------------------------------------------------
def drop_not_use_province(df:pd.DataFrame) -> pd.DataFrame:
    # df.dropna(subset='province') 
    df.loc[df['province'].str.contains("กรุงเทพ"), 'province'] = "กรุงเทพมหานคร"
    df = df[(df["province"] == "กรุงเทพมหานคร")]
    return df

#--------------------------------------------------------------------------------------------------------------
def verify_coordination_with_address(df:pd.DataFrame,verify_df:gpd.GeoDataFrame,check:pd.DataFrame=None) -> pd.DataFrame:
    df_points = gpd.GeoDataFrame(df,geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],crs="EPSG:4326")
    if check is not None:
        verify_df =verify_geopandas(verify_df,check)
    if verify_df.crs != "EPSG:4326":
        verify_df = verify_df.to_crs("EPSG:4326")
    joined = gpd.sjoin(df_points, verify_df, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep='first')]
    joined['check_dis'] = (joined['district_left'] == joined['district_right'])
    joined['check_sub'] = (joined['subdistrict_left'] == joined['subdistrict_right'])
    joined = joined[(((joined['check_dis']) & (joined['check_sub'])))]
    joined = drop_not_used_columns(df=joined,cols=['index_right','subdistrict_id','district_id','changwat','geometry', 'subdistrict_right', 'district_right','shape_area', 'check_dis', 'check_sub'])
    joined.rename(columns={'district_left':'district','subdistrict_left':'subdistrict'},inplace=True)
    return joined
 
#--------------------------------------------------------------------------------------------------------------
def use_simple_imputer(strategy:str,data:pd.DataFrame,column:list,fill_value = None)->pd.DataFrame:
    if (strategy=='constant'):
        imp = SimpleImputer(strategy=strategy,fill_value=fill_value)
    else: 
        imp = SimpleImputer(strategy=strategy)
    df =data.copy()
    df[column] =imp.fit_transform(df[column])
    return df

#--------------------------------------------------------------------------------------------------------------
def clean_station(station_df:pd.DataFrame)->pd.DataFrame:
    df = station_df.copy()
    df.rename(columns={
        'StationCode':'station_code',
        'StationName_Short':'station_name',
        'DistrictName':'district',
        'Latitude':'latitude',
        'Longitude':'longitude',
        'LastReadingTime':'last_reading_time',
        'DistrictCode_from_JS':'districtCode_from_js'
    },inplace=True)
    district_replacements = {
    'ป้อมปราบฯ': 'ป้อมปราบศัตรูพ่าย',
    'ราษฏร์บูรณะ': 'ราษฎร์บูรณะ',}
    df['district'] = df['district'].replace(district_replacements)
    df = df[df['district'] != 'อำเภอเมืองสมุทรปราการ'].copy()
    if df['station_code'].str.len().eq(9).all():
        df.loc[:,'station_code'] = df['station_code'].str[3:]
    df = df[df['station_code']!='PYT.02'].copy()
    return df

#--------------------------------------------------------------------------------------------------------------
def clean_rainfall(rain_df:pd.DataFrame)->pd.DataFrame:
    df = rain_df.copy()
    not_use_station = ['NKM.03','LSI.02','MBR.03','NJK.04','SPK.01']
    df = drop_not_used_columns(df,not_use_station)
    nan_cols = df.columns[df.isna().any()].tolist()
    df = use_simple_imputer(strategy='most_frequent',data=df,column=nan_cols)
    df.rename(columns={'Date':'date'},inplace=True)
    rain_expand_df = df.melt(id_vars='date', var_name='station_code', value_name='rainfall')
    rain_expand_df['date'] = pd.to_datetime(rain_expand_df['date'],format='mixed')
    rain_expand_df['date'] = rain_expand_df['date'].dt.date
    return rain_expand_df

#--------------------------------------------------------------------------------------------------------------
def get_centroid(shape_df:gpd.GeoDataFrame)->pd.DataFrame:
    shape_centers = shape_df.copy()
    shape_centers['centroid'] = shape_centers.geometry.centroid.to_crs(epsg=4326)
    shape_centers['latitude'] = shape_centers['centroid'].y
    shape_centers['longitude'] = shape_centers['centroid'].x
    return shape_centers

#--------------------------------------------------------------------------------------------------------------
def get_report_detail(df:pd.DataFrame)->pd.DataFrame:
    data = df.copy()
    data['flood'] = data['type'].str.contains("น้ำท่วม")
    data = (
    data.copy().groupby(['date','year_timestamp', 'month_timestamp', 'day_timestamp','district', 'subdistrict'])
      .agg(
          number_of_report_flood=('flood', 'sum'), 
          total_report=('flood', 'count')           
      )
      .reset_index()
    )
    return data

#--------------------------------------------------------------------------------------------------------------
def get_centroid_of_subdistrict(df:pd.DataFrame,shape_df:gpd.GeoDataFrame) -> pd.DataFrame:
    data = df.merge(
        shape_df[['subdistrict','district','latitude','longitude']],
        how='left',
        on=['subdistrict','district']
    )
    return data

#--------------------------------------------------------------------------------------------------------------
def make_station_info_nearest(df: pd.DataFrame, station_df: pd.DataFrame):
    df = df.copy()
    n_rows = len(df)
    station_map = (
        station_df.groupby('district')
        .apply(lambda g: g[['station_code', 'latitude', 'longitude']].to_dict('records'))
        .to_dict()
    )
    df['station_list'] = df['district'].map(station_map)
    df['station_code'] = np.nan
    df_rad = np.radians(df[['latitude','longitude']].to_numpy())
    positions = np.arange(n_rows)
    for district, stations in station_map.items():
        # Positional indices of rows in this district
        idx = positions[df['district'].to_numpy() == district]
        if len(idx) == 0:
            continue
        
        if len(stations) == 1:
            df.loc[df.index[idx], 'station_code'] = stations[0]['station_code']
        else:
            st_coords = np.radians(np.array([[s['latitude'], s['longitude']] for s in stations]))
            tree = BallTree(st_coords, metric='haversine')
            dist, nearest_idx = tree.query(df_rad[idx], k=1)
            nearest_idx = nearest_idx.flatten()
            df.loc[df.index[idx], 'station_code'] = [stations[i]['station_code'] for i in nearest_idx]
    df = df.drop(columns=['station_list'],axis='columns')
    return df

#--------------------------------------------------------------------------------------------------------------
def get_station_code_for_rain(rain_df:pd.DataFrame,shape_df:pd.DataFrame)->pd.DataFrame:
    data =rain_df.merge(
        shape_df[['subdistrict','district','station_code','latitude','longitude']],
        how='left',
        on='station_code')
    data.dropna(subset=['subdistrict','district'],inplace=True)
    return data

#--------------------------------------------------------------------------------------------------------------
def get_result_of_merge(report_df:pd.DataFrame,rain_df:pd.DataFrame):
    data = rain_df.merge(
    report_df[['date','station_code','number_of_report_flood','total_report','district','subdistrict',]],
    on=['date', 'station_code','district','subdistrict'],
    how='left'
    )
    data =use_simple_imputer(strategy='constant',data=data,column=['number_of_report_flood','total_report'],fill_value=0,)
    data = data[['date','district','subdistrict','station_code','latitude','longitude','number_of_report_flood','total_report','rainfall']]
    return data

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

def get_heat_map(df: pd.DataFrame, shape: gpd.GeoDataFrame, drop_missed_value: bool = False):
    
    map_df_col  = 'subdistrict'
    map_BMA_col = 'subdistrict'
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