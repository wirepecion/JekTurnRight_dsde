import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd

# Import your functional workers
from . import io, cleaning, geo, mergers

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Holds all configuration paths."""
    traffy_path: str
    shape_path: str
    station_path: str
    rain_path: str
    output_path: Optional[str] = None
    check_path: Optional[str] = None
    year_start: int = 2022
    year_end: int = 2024

class FloodDataPipeline:
    """
    The Orchestrator Class.
    Coordinates data flow between IO, Cleaning, Geo, and Mergers.
    """
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def run(self) -> pd.DataFrame:
        try:
            # --- STEP 1: LOAD ---
            logger.info(">>> [1/7] Loading Data...")
            raw_traffy = io.load_csv(self.cfg.traffy_path)
            shape_gdf = io.load_shapefile(self.cfg.shape_path)
            station_df = io.load_csv(self.cfg.station_path)
            rain_df = io.load_csv(self.cfg.rain_path)
            if self.cfg.check_path is not None:
                check_df = io.load_csv(self.cfg.check_path)

            # --- STEP 2: CLEAN ---
            logger.info(">>> [2/7] Cleaning Data...")
            # Clean Traffy
            df = cleaning.drop_nan_rows(raw_traffy, ['ticket_id', 'type', 'organization', 'coords', 'province', 'timestamp', 'last_activity'])
            df = cleaning.convert_types(df)
            df = cleaning.extract_time_features(df)
            df = df[(df['year_timestamp'] >= self.cfg.year_start) & (df['year_timestamp'] <= self.cfg.year_end)]
            df = cleaning.parse_coordinates(df)
            df = cleaning.clean_province_name(df)
            df = cleaning.parse_type_column(df)
                
            #Clean shape file
            if self.cfg.check_path is not None:
                shape_gdf = geo.verify_shape_gdf(shape_gdf,check_df)
            
            # Clean External
            station_df = mergers.clean_station_metadata(station_df)
            rain_df = mergers.clean_rainfall_data(rain_df)

            # --- STEP 3: SPATIAL VERIFY (Traffy Points) ---
            logger.info(">>> [3/7] Verifying Traffy Coordinates...")
            df = geo.spatial_join_verification(df, shape_gdf)
            # Ensure date column exists
            df['date'] = df['timestamp'].dt.date

            # --- STEP 4: PREPARE SUBDISTRICT MAP (The Missing Link) ---
            logger.info(">>> [4/7] Mapping Subdistricts to Stations...")
            # A. Get Centroids of every subdistrict
            subdistrict_centroids = geo.get_subdistrict_centroids(shape_gdf)
            
            # B. Find nearest station for every subdistrict centroid
            # This gives us a table: [subdistrict, district, lat, lon, station_code]
            shape_with_station = geo.get_nearest_station(subdistrict_centroids, station_df)

            # --- STEP 5: EXPAND RAINFALL ---
            logger.info(">>> [5/7] Expanding Rainfall to Subdistricts...")
            # Creates rows for every Date * Subdistrict
            expanded_rain = mergers.expand_rainfall_to_subdistricts(rain_df, shape_with_station)

            # --- STEP 6: FINAL MERGE ---
            logger.info(">>> [6/7] Merging Reports...")
            final_df = mergers.merge_rainfall_with_reports(
                expanded_rain_df=expanded_rain,
                report_df=df
            )

            # --- STEP 7: FEATURE ENGINEERING ---
            logger.info(">>> [7/8] Generating ML features...")
            final_df = cleaning.build_flood_features(final_df)

            # --- STEP 8: SAVE ---
            if self.cfg.output_path:
                logger.info(f"Saving result to {self.cfg.output_path}")
                final_df.to_csv(self.cfg.output_path, index=False)

            logger.info(">>> Pipeline Finished Successfully.")
            return final_df

        except Exception as e:
            logger.error(f"!!! Pipeline Failed: {e}", exc_info=True)
            raise e
        
class VisualizationDataPipeline:
    """
    A separate pipeline for preparing data for visualization purposes.
    Could include aggregation, filtering, or formatting steps specific to visualization needs.
    """
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def run(self) -> pd.DataFrame:
        # --- STEP 1: LOAD ---
        logger.info(">>> [1/3] Loading Data...")
        raw_traffy = io.load_csv(self.cfg.traffy_path)
        shape_gdf = io.load_shapefile(self.cfg.shape_path)

        # --- STEP 2: CLEAN ---
        logger.info(">>> [2/3] Cleaning Data...")
        # Clean Traffy
        df = cleaning.drop_nan_rows(raw_traffy, ['coords', 'timestamp', 'type'])
        df = df[df['type'] != '{}']
        df = cleaning.convert_types(df)
        df = cleaning.extract_time_features(df)
        df = df[(df['year_timestamp'] >= self.cfg.year_start) & (df['year_timestamp'] <= self.cfg.year_end)]
        df = cleaning.parse_coordinates(df)
        df = cleaning.clean_province_name(df)
        df = cleaning.parse_type_column(df)

        # --- STEP 3: SPATIAL VERIFY (Traffy Points) ---
        logger.info(">>> [3/3] Verifying Traffy Coordinates...")
        df = geo.spatial_join_verification(df, shape_gdf)
        # Ensure date column exists
        df['date'] = df['timestamp'].dt.date

        return df
    