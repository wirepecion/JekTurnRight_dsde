"""
src/dataprep/pipeline.py
------------------------
Orchestrator Class for the Flood Data Pipeline.
"""
import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd

# Import your functional workers
from . import io, cleaning, geo, mergers

# Setup Logger
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """
    Holds all configuration paths to avoid passing 4 arguments to every function.
    """
    traffy_path: str
    shape_path: str
    station_path: str
    rain_path: str
    output_path: Optional[str] = None
    year_start: int = 2022
    year_end: int = 2024

class FloodDataPipeline:
    """
    The Orchestrator.
    Responsibility: Coordinate the data flow between IO, Cleaning, and Geo modules.
    """
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        logger.info(f"Pipeline initialized for years {self.cfg.year_start}-{self.cfg.year_end}")

    def run(self) -> pd.DataFrame:
        """
        Executes the full End-to-End pipeline.
        Returns: The final cleaned DataFrame.
        """
        try:
            # --- STEP 1: LOAD RAW DATA ---
            logger.info(">>> [1/6] Loading Raw Data...")
            raw_traffy = io.load_csv(self.cfg.traffy_path)
            shape_gdf = io.load_shapefile(self.cfg.shape_path)

            # --- STEP 2: CLEAN TRAFFY DATA ---
            logger.info(">>> [2/6] Cleaning Traffy Data...")
            # Note: We pass data explicitly. We do NOT store 'self.df' to avoid state confusion.
            df = cleaning.drop_nan_rows(raw_traffy, ['coords', 'timestamp'])
            df = cleaning.convert_types(df)
            df = cleaning.extract_time_features(df)
            
            # Filter Years
            df = df[
                (df['year_timestamp'] >= self.cfg.year_start) & 
                (df['year_timestamp'] <= self.cfg.year_end)
            ]
            
            df = cleaning.parse_coordinates(df)
            df = cleaning.clean_province_name(df)
            df = cleaning.parse_type_column(df)

            # --- STEP 3: SPATIAL VERIFICATION ---
            logger.info(">>> [3/6] Verifying Spatial Data...")
            # Ensure points are actually inside Bangkok districts
            df = geo.spatial_join_verification(df, shape_gdf)

            # --- STEP 4: PREPARE EXTERNAL DATA ---
            logger.info(">>> [4/6] Preparing External Data (Station & Rain)...")
            station_df = io.load_csv(self.cfg.station_path)
            station_df = mergers.clean_station_metadata(station_df)

            rain_df = io.load_csv(self.cfg.rain_path)
            rain_df = mergers.clean_rainfall_data(rain_df)

            # --- STEP 5: GEOSPATIAL MAPPING (Heavy Lift) ---
            logger.info(">>> [5/6] Mapping to Nearest Stations...")
            df_mapped = geo.get_nearest_station(df, station_df)

            # --- STEP 6: FINAL MERGE ---
            logger.info(">>> [6/6] Merging Rainfall Data...")
            # Create a date column for joining
            df_mapped['date'] = pd.to_datetime(df_mapped['timestamp']).dt.date
            
            final_df = mergers.merge_rainfall_with_reports(df_mapped, rain_df,station_df)

            # --- OPTIONAL: SAVE ---
            if self.cfg.output_path:
                logger.info(f"Saving result to {self.cfg.output_path}")
                final_df.to_parquet(self.cfg.output_path, index=False)

            logger.info(">>> Pipeline Finished Successfully.")
            return final_df

        except Exception as e:
            logger.error(f"!!! Pipeline Failed: {e}", exc_info=True)
            raise e