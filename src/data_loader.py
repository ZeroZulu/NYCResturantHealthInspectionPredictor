"""
Data Loader Module
==================
Handles data ingestion from NYC Open Data API or local CSV files.
Performs initial cleaning and validation.


Author: [Shril Patel]
Date: [Dec 13, 2025]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from datetime import datetime

# Optional: NYC Open Data API client
try:
    from sodapy import Socrata
    SODAPY_AVAILABLE = True
except ImportError:
    SODAPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# NYC Open Data dataset identifier
NYC_DATASET_ID = "43nn-pn8j"
NYC_DOMAIN = "data.cityofnewyork.us"


class NYCRestaurantDataLoader:
    """
    Loads and performs initial cleaning on NYC Restaurant Inspection data.
    
    Can load from:
    - Local CSV file
    - NYC Open Data API (if sodapy is installed)
    
    Example:
        loader = NYCRestaurantDataLoader()
        df = loader.load_from_csv("path/to/data.csv")
        df_clean = loader.clean_data(df)
    """
    
    # Expected columns in the dataset
    EXPECTED_COLUMNS = [
        'camis', 'dba', 'boro', 'building', 'street', 'zipcode', 'phone',
        'cuisine_description', 'inspection_date', 'action', 'violation_code',
        'violation_description', 'critical_flag', 'score', 'grade',
        'grade_date', 'record_date', 'inspection_type', 'latitude', 'longitude'
    ]
    
    # Columns that should be numeric
    NUMERIC_COLUMNS = ['camis', 'zipcode', 'score', 'latitude', 'longitude']
    
    # Columns that should be dates
    DATE_COLUMNS = ['inspection_date', 'grade_date', 'record_date']
    
    def __init__(self, app_token: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            app_token: Optional NYC Open Data app token for higher rate limits
        """
        self.app_token = app_token
        
        # Create data directories if they don't exist
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load data from a local CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Raw DataFrame
        """
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(
            filepath,
            low_memory=False,
            dtype={'ZIPCODE': str, 'PHONE': str}  # Keep as strings
        )
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        return df
    
    def load_from_api(self, limit: int = 500000) -> pd.DataFrame:
        """
        Load data directly from NYC Open Data API.
        
        Args:
            limit: Maximum number of records to fetch
            
        Returns:
            Raw DataFrame
        """
        if not SODAPY_AVAILABLE:
            raise ImportError(
                "sodapy is required for API access. Install with: pip install sodapy"
            )
        
        logger.info(f"Fetching data from NYC Open Data API (limit: {limit:,})")
        
        client = Socrata(NYC_DOMAIN, self.app_token)
        results = client.get(NYC_DATASET_ID, limit=limit)
        
        df = pd.DataFrame.from_records(results)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        logger.info(f"Fetched {len(df):,} rows from API")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform initial data cleaning.
        
        Steps:
        1. Remove exact duplicates
        2. Convert data types
        3. Handle missing values
        4. Standardize text fields
        5. Filter invalid records
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df = df.copy()
        initial_rows = len(df)
        
        # 1. Remove exact duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df):,} duplicate rows")
        
        # 2. Convert date columns
        for col in self.DATE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 3. Convert numeric columns
        if 'score' in df.columns:
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
        
        if 'camis' in df.columns:
            df['camis'] = pd.to_numeric(df['camis'], errors='coerce')
        
        # Convert lat/long to numeric
        for col in ['latitude', 'longitude']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 4. Standardize text fields
        text_columns = ['dba', 'boro', 'cuisine_description', 'grade', 'critical_flag']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                df[col] = df[col].replace(['NAN', 'NONE', ''], np.nan)
        
        # 5. Filter invalid records
        # Remove rows without inspection date
        if 'inspection_date' in df.columns:
            valid_dates = df['inspection_date'].notna()
            df = df[valid_dates]
            logger.info(f"Removed {(~valid_dates).sum():,} rows with missing inspection dates")
        
        # Remove future dates (data quality issue)
        if 'inspection_date' in df.columns:
            future_mask = df['inspection_date'] > datetime.now()
            df = df[~future_mask]
            if future_mask.sum() > 0:
                logger.info(f"Removed {future_mask.sum():,} rows with future dates")
        
        # Remove invalid boroughs
        valid_boroughs = ['MANHATTAN', 'BROOKLYN', 'QUEENS', 'BRONX', 'STATEN ISLAND']
        if 'boro' in df.columns:
            df = df[df['boro'].isin(valid_boroughs)]
        
        # 6. Create cleaned zipcode (5 digits only)
        if 'zipcode' in df.columns:
            df['zipcode'] = df['zipcode'].astype(str).str[:5]
            df['zipcode'] = df['zipcode'].replace(['NAN', 'NONE', 'nan'], np.nan)
        
        logger.info(f"Cleaning complete. {len(df):,} rows remaining ({len(df)/initial_rows*100:.1f}%)")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate the cleaned data for common issues.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Tuple of (is_valid, list of issues found)
        """
        issues = []
        
        # Check for required columns
        required = ['camis', 'inspection_date', 'boro', 'cuisine_description']
        missing_cols = [col for col in required if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for minimum data
        if len(df) < 1000:
            issues.append(f"Very few records ({len(df)}). Expected at least 1,000.")
        
        # Check date range
        if 'inspection_date' in df.columns:
            date_range = df['inspection_date'].max() - df['inspection_date'].min()
            if date_range.days < 365:
                issues.append(f"Date range is only {date_range.days} days. Need at least 1 year.")
        
        # Check for score distribution
        if 'score' in df.columns:
            score_missing = df['score'].isna().sum() / len(df)
            if score_missing > 0.5:
                issues.append(f"High % of missing scores: {score_missing*100:.1f}%")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a summary of the dataset.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'unique_restaurants': df['camis'].nunique() if 'camis' in df.columns else None,
            'date_range': None,
            'boroughs': None,
            'top_cuisines': None,
            'missing_values': df.isnull().sum().to_dict()
        }
        
        if 'inspection_date' in df.columns:
            summary['date_range'] = {
                'min': str(df['inspection_date'].min()),
                'max': str(df['inspection_date'].max())
            }
        
        if 'boro' in df.columns:
            summary['boroughs'] = df['boro'].value_counts().to_dict()
        
        if 'cuisine_description' in df.columns:
            summary['top_cuisines'] = df['cuisine_description'].value_counts().head(10).to_dict()
        
        return summary


def main():
    """Main function to run data loading pipeline."""
    
    loader = NYCRestaurantDataLoader()
    
    # Check if local file exists
    local_file = DATA_RAW / "DOHMH_New_York_City_Restaurant_Inspection_Results.csv"
    
    if local_file.exists():
        df = loader.load_from_csv(str(local_file))
    else:
        print(f"\nNo local file found at: {local_file}")
        print("\nOptions:")
        print("1. Place your CSV file in the data/raw/ directory")
        print("2. Download from: https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j")
        print("\nOr run with API access (requires internet):")
        print("   df = loader.load_from_api(limit=100000)")
        return
    
    # Clean the data
    df_clean = loader.clean_data(df)
    
    # Validate
    is_valid, issues = loader.validate_data(df_clean)
    if not is_valid:
        logger.warning(f"Data validation issues: {issues}")
    
    # Print summary
    summary = loader.get_data_summary(df_clean)
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"Total Records: {summary['total_rows']:,}")
    print(f"Unique Restaurants: {summary['unique_restaurants']:,}")
    print(f"Date Range: {summary['date_range']['min'][:10]} to {summary['date_range']['max'][:10]}")
    print(f"\nRecords by Borough:")
    for boro, count in summary['boroughs'].items():
        print(f"  {boro}: {count:,}")
    
    # Save cleaned data
    output_path = DATA_PROCESSED / "inspections_cleaned.csv"
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")


if __name__ == "__main__":
    main()
