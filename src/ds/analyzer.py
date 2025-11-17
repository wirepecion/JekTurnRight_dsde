"""
Example data science module for basic data analysis.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    A simple data analyzer for exploratory data analysis.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame.

        Args:
            df: Input pandas DataFrame
        """
        self.df = df
        logger.info(f"Initialized DataAnalyzer with {len(df)} rows")

    def summary_stats(self) -> pd.DataFrame:
        """
        Get summary statistics for the dataset.

        Returns:
            DataFrame with summary statistics
        """
        logger.info("Generating summary statistics")
        return self.df.describe()

    def check_missing(self) -> pd.Series:
        """
        Check for missing values.

        Returns:
            Series with count of missing values per column
        """
        missing = self.df.isnull().sum()
        logger.info(f"Total missing values: {missing.sum()}")
        return missing

    def prepare_features(
        self,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features for machine learning.

        Args:
            target_col: Name of the target column
            test_size: Proportion of dataset for test set
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Preparing features with target: {target_col}")
        
        # Separate features and target
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        # Select only numeric columns
        X = X.select_dtypes(include=[np.number])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        return X_train, X_test, y_train, y_test


def example_analysis():
    """
    Example data analysis workflow.
    """
    # Create sample data
    np.random.seed(42)
    data = {
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)

    # Analyze
    analyzer = DataAnalyzer(df)
    print("\nSummary Statistics:")
    print(analyzer.summary_stats())
    
    print("\nMissing Values:")
    print(analyzer.check_missing())

    # Prepare for ML
    X_train, X_test, y_train, y_test = analyzer.prepare_features('target')
    print(f"\nPrepared {len(X_train)} training samples and {len(X_test)} test samples")


if __name__ == "__main__":
    example_analysis()
