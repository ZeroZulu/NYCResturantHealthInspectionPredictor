"""
Feature Engineering Module
==========================
Creates predictive features from raw inspection data.
This is where the data science magic happens!

Features are grouped into categories:
1. Restaurant-level historical features
2. Temporal/time-based features
3. Geographic features
4. Cuisine-based risk features
5. Inspection pattern features

Author: [Shril Patel]
Date: [Dec 13, 2025]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


class FeatureEngineer:
    """
    Creates features for the restaurant inspection prediction model.
    
    The key insight: We want to predict FUTURE inspection outcomes
    using only information available BEFORE the inspection.
    
    This means we must be careful about temporal leakage!
    
    Example:
        fe = FeatureEngineer()
        df_features = fe.create_all_features(df_clean)
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.cuisine_risk_scores = None
        self.borough_risk_scores = None
        self.zipcode_risk_scores = None
    
    def create_all_features(
        self, 
        df: pd.DataFrame,
        target_col: str = 'inspection_failed'
    ) -> pd.DataFrame:
        """
        Master function to create all features.
        
        Args:
            df: Cleaned inspection DataFrame
            target_col: Name of target column to create
            
        Returns:
            DataFrame with all features
        """
        logger.info("Starting feature engineering pipeline...")
        df = df.copy()
        
        # Sort by restaurant and date (critical for temporal features)
        df = df.sort_values(['camis', 'inspection_date']).reset_index(drop=True)
        
        # Step 1: Create target variable
        logger.info("Creating target variable...")
        df = self._create_target(df, target_col)
        
        # Step 2: Create restaurant-level historical features
        logger.info("Creating historical features...")
        df = self._create_historical_features(df)
        
        # Step 3: Create temporal features
        logger.info("Creating temporal features...")
        df = self._create_temporal_features(df)
        
        # Step 4: Create geographic risk features
        logger.info("Creating geographic features...")
        df = self._create_geographic_features(df)
        
        # Step 5: Create cuisine risk features
        logger.info("Creating cuisine features...")
        df = self._create_cuisine_features(df)
        
        # Step 6: Create inspection pattern features
        logger.info("Creating inspection pattern features...")
        df = self._create_inspection_pattern_features(df)
        
        # Step 7: Create violation severity features
        logger.info("Creating violation features...")
        df = self._create_violation_features(df)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        
        return df
    
    def _create_target(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Create the target variable for prediction.
        
        We define "failed" as:
        - Score >= 14 (Grade B or worse), OR
        - Any critical violation, OR
        - Grade is 'C' or worse
        
        You can adjust this threshold based on your analysis!
        """
        df = df.copy()
        
        # Primary target: Score-based failure
        score_fail = (df['score'] >= 14) if 'score' in df.columns else False
        
        # Alternative: Critical violation present
        critical_fail = (df['critical_flag'] == 'CRITICAL') if 'critical_flag' in df.columns else False
        
        # Alternative: Grade-based
        grade_fail = df['grade'].isin(['C', 'Z', 'P']) if 'grade' in df.columns else False
        
        # Combine (any of the above = fail)
        df[target_col] = (score_fail | critical_fail | grade_fail).astype(int)
        
        # Log class distribution
        if target_col in df.columns:
            fail_rate = df[target_col].mean()
            logger.info(f"Target variable '{target_col}' created. Failure rate: {fail_rate:.1%}")
        
        return df
    
    def _create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on restaurant's inspection history.
        
        These are the most predictive features - past behavior predicts future!
        
        IMPORTANT: We use expanding windows to avoid data leakage.
        Each feature only uses information from BEFORE the current inspection.
        """
        df = df.copy()
        
        # Group by restaurant
        grouped = df.groupby('camis')
        
        # 1. Count of previous inspections (experience with the system)
        df['prev_inspection_count'] = grouped.cumcount()
        
        # 2. Historical violation rate (expanding mean, shifted to avoid leakage)
        if 'inspection_failed' in df.columns:
            df['historical_fail_rate'] = (
                grouped['inspection_failed']
                .expanding()
                .mean()
                .shift(1)  # CRITICAL: shift to avoid leakage
                .reset_index(level=0, drop=True)
            )
        
        # 3. Historical average score
        if 'score' in df.columns:
            df['historical_avg_score'] = (
                grouped['score']
                .expanding()
                .mean()
                .shift(1)
                .reset_index(level=0, drop=True)
            )
            
            # Historical max score (worst performance)
            df['historical_max_score'] = (
                grouped['score']
                .expanding()
                .max()
                .shift(1)
                .reset_index(level=0, drop=True)
            )
            
            # Historical score trend (is it improving or getting worse?)
            df['score_trend'] = (
                grouped['score']
                .diff()
                .shift(1)
                .reset_index(level=0, drop=True)
            )
        
        # 4. Previous inspection score (most recent)
        if 'score' in df.columns:
            df['prev_score'] = grouped['score'].shift(1)
        
        # 5. Previous inspection outcome
        if 'inspection_failed' in df.columns:
            df['prev_failed'] = grouped['inspection_failed'].shift(1)
        
        # 6. Count of critical violations historically
        if 'critical_flag' in df.columns:
            df['is_critical'] = (df['critical_flag'] == 'CRITICAL').astype(int)
            df['historical_critical_count'] = (
                grouped['is_critical']
                .expanding()
                .sum()
                .shift(1)
                .reset_index(level=0, drop=True)
            )
        
        # 7. Streak features (consecutive passes or fails)
        if 'inspection_failed' in df.columns:
            df['consecutive_fails'] = self._calculate_streak(df, 'inspection_failed', 1)
            df['consecutive_passes'] = self._calculate_streak(df, 'inspection_failed', 0)
        
        # Fill NaN for first inspection (no history)
        history_cols = [
            'historical_fail_rate', 'historical_avg_score', 'historical_max_score',
            'prev_score', 'prev_failed', 'historical_critical_count',
            'score_trend', 'consecutive_fails', 'consecutive_passes'
        ]
        
        for col in history_cols:
            if col in df.columns:
                # For first inspections, use global average or 0
                if 'rate' in col or 'avg' in col:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(0)
        
        return df
    
    def _calculate_streak(
        self, 
        df: pd.DataFrame, 
        col: str, 
        value: int
    ) -> pd.Series:
        """Calculate consecutive streak of a value within each restaurant."""
        
        def streak_counter(group):
            streaks = []
            current_streak = 0
            prev_val = None
            
            for val in group[col].shift(1):  # Shift to avoid leakage
                if pd.isna(val):
                    streaks.append(0)
                elif val == value:
                    current_streak += 1
                    streaks.append(current_streak)
                else:
                    current_streak = 0
                    streaks.append(0)
                prev_val = val
            
            return pd.Series(streaks, index=group.index)
        
        return df.groupby('camis', group_keys=False).apply(streak_counter)
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Key insight from EDA:
        - Summer months (June-July) have highest failure rates (~63%)
        - April-May have lowest (~53%)
        - January also high (~61%) - possibly post-holiday rush
        - Failure rates trending up from 49% (2022) to 61% (2025)
        """
        df = df.copy()
        
        if 'inspection_date' not in df.columns:
            return df
        
        # 1. Basic date components
        df['inspection_year'] = df['inspection_date'].dt.year
        df['inspection_month'] = df['inspection_date'].dt.month
        df['inspection_day_of_week'] = df['inspection_date'].dt.dayofweek
        df['inspection_day_of_year'] = df['inspection_date'].dt.dayofyear
        
        # 2. Seasonal features (cyclical encoding)
        # Use sin/cos to capture cyclical nature of months
        df['month_sin'] = np.sin(2 * np.pi * df['inspection_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['inspection_month'] / 12)
        
        # 3. High-risk season indicators (from EDA)
        # Summer is highest risk (June-July ~63%)
        df['is_summer'] = df['inspection_month'].isin([6, 7, 8]).astype(int)
        df['is_peak_summer'] = df['inspection_month'].isin([6, 7]).astype(int)  # Specifically Jun-Jul
        
        # Winter also elevated (Jan ~61%)
        df['is_winter'] = df['inspection_month'].isin([12, 1, 2]).astype(int)
        df['is_january'] = (df['inspection_month'] == 1).astype(int)  # Jan is notably high
        
        # Low-risk spring (Apr-May ~53%)
        df['is_spring'] = df['inspection_month'].isin([4, 5]).astype(int)
        
        # 4. Monthly risk score (pre-computed from EDA)
        MONTHLY_RISK = {
            1: 0.613,   # January - high
            2: 0.576,
            3: 0.561,
            4: 0.534,   # April - low
            5: 0.537,   # May - low
            6: 0.625,   # June - high
            7: 0.633,   # July - highest
            8: 0.573,
            9: 0.561,
            10: 0.563,
            11: 0.597,
            12: 0.547,
        }
        df['monthly_risk_score'] = df['inspection_month'].map(MONTHLY_RISK)
        
        # 5. Days since last inspection (per restaurant)
        df['days_since_last_inspection'] = (
            df.groupby('camis')['inspection_date']
            .diff()
            .dt.days
        )
        df['days_since_last_inspection'] = df['days_since_last_inspection'].fillna(
            df['days_since_last_inspection'].median()
        )
        
        # 6. Is this inspection "overdue"? (> 365 days since last)
        df['inspection_overdue'] = (df['days_since_last_inspection'] > 365).astype(int)
        
        # 7. Day of week effects (from EDA: Saturday lowest at 42%, Sunday highest at 67%)
        df['is_weekend'] = df['inspection_day_of_week'].isin([5, 6]).astype(int)
        df['is_monday'] = (df['inspection_day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['inspection_day_of_week'] == 4).astype(int)
        
        # 8. Restaurant age (days since first inspection)
        first_inspection = df.groupby('camis')['inspection_date'].transform('min')
        df['restaurant_age_days'] = (df['inspection_date'] - first_inspection).dt.days
        
        # 9. Year trend feature (failure rates increasing over time)
        # Normalize year to 0-1 scale for recent years
        min_year = 2022
        max_year = 2025
        df['year_normalized'] = (df['inspection_year'] - min_year) / (max_year - min_year)
        df['year_normalized'] = df['year_normalized'].clip(0, 1)
        
        return df
    
    def _create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create location-based risk features.
        
        Key insight from EDA:
        - ZIP codes vary from 29% (UES) to 80% (parts of Brooklyn/Bronx)
        - Manhattan has lowest failure rate (53.7%)
        - Staten Island/Queens/Bronx have highest (~60%)
        
        Your data also has: community_board, council_district, nta (Neighborhood Tabulation Area)
        """
        df = df.copy()
        
        # 1. Borough-level risk score (pre-computed from EDA)
        BOROUGH_RISK = {
            'STATEN ISLAND': 0.606,
            'QUEENS': 0.604,
            'BRONX': 0.602,
            'BROOKLYN': 0.590,
            'MANHATTAN': 0.537,
        }
        
        if 'boro' in df.columns:
            df['borough_risk_score'] = df['boro'].str.upper().map(BOROUGH_RISK)
            df['borough_risk_score'] = df['borough_risk_score'].fillna(0.575)  # Overall average
            
            # Manhattan is notably lower risk
            df['is_manhattan'] = (df['boro'].str.upper() == 'MANHATTAN').astype(int)
        
        # 2. Zipcode-level risk score (calculate from data if target available)
        if 'zipcode' in df.columns and 'inspection_failed' in df.columns:
            # Only use zipcodes with enough data
            zipcode_counts = df['zipcode'].value_counts()
            valid_zipcodes = zipcode_counts[zipcode_counts >= 20].index
            
            self.zipcode_risk_scores = (
                df[df['zipcode'].isin(valid_zipcodes)]
                .groupby('zipcode')['inspection_failed']
                .mean()
                .to_dict()
            )
            
            df['zipcode_risk_score'] = df['zipcode'].map(self.zipcode_risk_scores)
            
            # Fill missing with borough average
            df['zipcode_risk_score'] = df['zipcode_risk_score'].fillna(
                df['borough_risk_score'] if 'borough_risk_score' in df.columns 
                else 0.575
            )
        
        # 3. NTA (Neighborhood Tabulation Area) risk - more granular than ZIP
        if 'nta' in df.columns and 'inspection_failed' in df.columns:
            nta_counts = df['nta'].value_counts()
            valid_ntas = nta_counts[nta_counts >= 20].index
            
            nta_risk = (
                df[df['nta'].isin(valid_ntas)]
                .groupby('nta')['inspection_failed']
                .mean()
                .to_dict()
            )
            
            df['nta_risk_score'] = df['nta'].map(nta_risk)
            df['nta_risk_score'] = df['nta_risk_score'].fillna(
                df['zipcode_risk_score'] if 'zipcode_risk_score' in df.columns
                else df['borough_risk_score'] if 'borough_risk_score' in df.columns
                else 0.575
            )
        
        # 4. Community Board risk (another geographic granularity)
        if 'community_board' in df.columns and 'inspection_failed' in df.columns:
            cb_counts = df['community_board'].value_counts()
            valid_cbs = cb_counts[cb_counts >= 20].index
            
            cb_risk = (
                df[df['community_board'].isin(valid_cbs)]
                .groupby('community_board')['inspection_failed']
                .mean()
                .to_dict()
            )
            
            df['community_board_risk_score'] = df['community_board'].map(cb_risk)
            df['community_board_risk_score'] = df['community_board_risk_score'].fillna(0.575)
        
        # 5. Borough one-hot encoding
        if 'boro' in df.columns:
            # Clean borough names first
            df['boro_clean'] = df['boro'].str.upper().str.strip()
            boro_dummies = pd.get_dummies(df['boro_clean'], prefix='boro')
            df = pd.concat([df, boro_dummies], axis=1)
            df = df.drop('boro_clean', axis=1)
        
        return df
    
    def _create_cuisine_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cuisine-based risk features.
        
        Key insight from EDA: Cuisine type is HIGHLY predictive!
        - Thai/Indian: ~77% failure rate
        - Donuts/Hamburgers: ~26-31% failure rate
        - That's a 50+ percentage point spread!
        """
        df = df.copy()
        
        if 'cuisine_description' not in df.columns:
            return df
        
        # 1. Pre-computed cuisine risk scores from EDA analysis
        # These are based on actual failure rates in the dataset
        KNOWN_CUISINE_RISK = {
            'THAI': 0.774,
            'INDIAN': 0.769,
            'ASIAN/ASIAN FUSION': 0.716,
            'MIDDLE EASTERN': 0.696,
            'CHINESE': 0.685,
            'CARIBBEAN': 0.665,
            'MEXICAN': 0.651,
            'SPANISH': 0.637,
            'LATIN AMERICAN': 0.620,
            'JAPANESE': 0.614,
            'KOREAN': 0.613,
            'BAGELS/PRETZELS': 0.613,
            'JEWISH/KOSHER': 0.600,
            'BAKERY PRODUCTS/DESSERTS': 0.591,
            'JUICE, SMOOTHIES, FRUIT SALADS': 0.581,
            'PIZZA': 0.546,
            'AMERICAN': 0.513,
            'COFFEE/TEA': 0.496,
            'MEDITERRANEAN': 0.509,
            'FROZEN DESSERTS': 0.475,
            'SANDWICHES': 0.338,
            'TEX-MEX': 0.322,
            'HAMBURGERS': 0.309,
            'DONUTS': 0.260,
        }
        
        # Apply known risk scores
        df['cuisine_risk_score'] = df['cuisine_description'].str.upper().map(KNOWN_CUISINE_RISK)
        
        # For unknown cuisines, calculate from data if available, else use median
        if 'inspection_failed' in df.columns:
            unknown_mask = df['cuisine_risk_score'].isna()
            if unknown_mask.any():
                # Calculate risk for unknown cuisines
                cuisine_counts = df['cuisine_description'].value_counts()
                valid_cuisines = cuisine_counts[cuisine_counts >= 50].index
                
                calculated_risk = (
                    df[df['cuisine_description'].isin(valid_cuisines)]
                    .groupby('cuisine_description')['inspection_failed']
                    .mean()
                    .to_dict()
                )
                
                # Fill unknown with calculated or overall mean
                df.loc[unknown_mask, 'cuisine_risk_score'] = (
                    df.loc[unknown_mask, 'cuisine_description']
                    .str.upper()
                    .map(calculated_risk)
                )
            
            # Final fallback: use overall failure rate
            df['cuisine_risk_score'] = df['cuisine_risk_score'].fillna(
                df['inspection_failed'].mean()
            )
        else:
            # No target available, use median of known scores
            df['cuisine_risk_score'] = df['cuisine_risk_score'].fillna(0.575)
        
        # 2. High-risk cuisine flag (from EDA: Thai, Indian, Asian, Middle Eastern, Chinese)
        high_risk_cuisines = ['THAI', 'INDIAN', 'ASIAN/ASIAN FUSION', 'MIDDLE EASTERN', 'CHINESE']
        df['is_high_risk_cuisine'] = (
            df['cuisine_description'].str.upper().isin(high_risk_cuisines)
        ).astype(int)
        
        # 3. Low-risk cuisine flag (from EDA: Donuts, Hamburgers, Tex-Mex, Sandwiches)
        low_risk_cuisines = ['DONUTS', 'HAMBURGERS', 'TEX-MEX', 'SANDWICHES']
        df['is_low_risk_cuisine'] = (
            df['cuisine_description'].str.upper().isin(low_risk_cuisines)
        ).astype(int)
        
        # 4. Cuisine category (group similar cuisines)
        cuisine_categories = {
            'AMERICAN': ['AMERICAN', 'HAMBURGERS', 'SANDWICHES', 'HOTDOGS', 'CHICKEN'],
            'ASIAN': ['CHINESE', 'JAPANESE', 'KOREAN', 'THAI', 'VIETNAMESE', 'ASIAN'],
            'LATIN': ['MEXICAN', 'LATIN AMERICAN', 'SPANISH', 'CARIBBEAN', 'CUBAN', 'PERUVIAN'],
            'EUROPEAN': ['ITALIAN', 'FRENCH', 'GERMAN', 'GREEK', 'POLISH', 'RUSSIAN'],
            'BAKERY_CAFE': ['BAKERY', 'COFFEE/TEA', 'DONUTS', 'BAGELS', 'ICE CREAM', 'FROZEN DESSERTS'],
            'PIZZA': ['PIZZA', 'PIZZA/ITALIAN'],
            'INDIAN': ['INDIAN', 'BANGLADESHI', 'PAKISTANI'],
            'MIDDLE_EASTERN': ['MIDDLE EASTERN', 'TURKISH', 'EGYPTIAN', 'MOROCCAN'],
        }
        
        def categorize_cuisine(cuisine):
            cuisine_upper = str(cuisine).upper()
            for category, keywords in cuisine_categories.items():
                if any(kw in cuisine_upper for kw in keywords):
                    return category
            return 'OTHER'
        
        df['cuisine_category'] = df['cuisine_description'].apply(categorize_cuisine)
        
        # 5. One-hot encode cuisine categories
        cuisine_dummies = pd.get_dummies(df['cuisine_category'], prefix='cuisine')
        df = pd.concat([df, cuisine_dummies], axis=1)
        
        return df
    
    def _create_inspection_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on inspection type and patterns.
        
        Key insight: Re-inspections behave differently than initial inspections.
        """
        df = df.copy()
        
        if 'inspection_type' in df.columns:
            # 1. Inspection type indicators
            df['is_initial_inspection'] = df['inspection_type'].str.contains(
                'INITIAL|CYCLE', case=False, na=False
            ).astype(int)
            
            df['is_reinspection'] = df['inspection_type'].str.contains(
                'RE-INSPECTION|REINSPECTION', case=False, na=False
            ).astype(int)
            
            df['is_complaint_inspection'] = df['inspection_type'].str.contains(
                'COMPLAINT', case=False, na=False
            ).astype(int)
        
        # 2. Which inspection in the cycle (1st, 2nd, 3rd, etc.)
        df['inspection_number_this_year'] = (
            df.groupby(['camis', 'inspection_year']).cumcount() + 1
            if 'inspection_year' in df.columns
            else df.groupby('camis').cumcount() + 1
        )
        
        return df
    
    def _create_violation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on violation codes and descriptions.
        
        Key insight from EDA: Top violations include:
        - 10F (523): Unclean non-food surfaces
        - 08A (396): Rodent/insect harborage conditions
        - 02B (272): Hot food temperature violations
        - 04L (246): Evidence of mice
        - 04M: Live roaches
        """
        df = df.copy()
        
        if 'violation_code' not in df.columns:
            return df
        
        # 1. Count violations per inspection
        violation_counts = df.groupby(['camis', 'inspection_date']).size()
        violation_counts.name = 'violation_count_this_inspection'
        df = df.merge(
            violation_counts.reset_index(), 
            on=['camis', 'inspection_date'], 
            how='left'
        )
        
        # 2. High-risk violation codes (from EDA - these are the most common/serious)
        # Pest-related (very serious)
        pest_codes = ['04L', '04M', '04N', '08A']  # Mice, roaches, flies, harborage
        
        # Temperature violations (food safety critical)
        temp_codes = ['02B', '02G']  # Hot food cold, cold food warm
        
        # Contamination/hygiene
        contamination_codes = ['04A', '04H', '06C', '06D', '06F']
        
        # All high-risk combined
        high_risk_codes = pest_codes + temp_codes + ['04A', '04H']
        
        if 'violation_code' in df.columns:
            # Individual category flags
            df['has_pest_violation'] = df['violation_code'].isin(pest_codes).astype(int)
            df['has_temp_violation'] = df['violation_code'].isin(temp_codes).astype(int)
            df['has_high_risk_violation'] = df['violation_code'].isin(high_risk_codes).astype(int)
            
            # Aggregate to inspection level
            for col in ['has_pest_violation', 'has_temp_violation', 'has_high_risk_violation']:
                agg_col = col.replace('has_', 'any_')
                agg_data = df.groupby(['camis', 'inspection_date'])[col].max()
                df = df.merge(
                    agg_data.reset_index().rename(columns={col: agg_col}),
                    on=['camis', 'inspection_date'],
                    how='left'
                )
        
        # 3. Critical violation flag (already in data but ensure numeric)
        if 'critical_flag' in df.columns:
            df['is_critical'] = (df['critical_flag'].str.upper() == 'CRITICAL').astype(int)
            
            # Count critical violations per inspection
            critical_count = df.groupby(['camis', 'inspection_date'])['is_critical'].sum()
            df = df.merge(
                critical_count.reset_index().rename(columns={'is_critical': 'critical_count_this_inspection'}),
                on=['camis', 'inspection_date'],
                how='left'
            )
        
        return df
    
    def get_feature_list(self) -> dict:
        """
        Return lists of features by category for documentation.
        """
        return {
            'historical': [
                'prev_inspection_count', 'historical_fail_rate', 'historical_avg_score',
                'historical_max_score', 'prev_score', 'prev_failed', 
                'historical_critical_count', 'consecutive_fails', 'consecutive_passes',
                'score_trend'
            ],
            'temporal': [
                'inspection_month', 'inspection_day_of_week', 'month_sin', 'month_cos',
                'is_summer', 'is_winter', 'days_since_last_inspection',
                'inspection_overdue', 'is_monday', 'is_friday', 'restaurant_age_days'
            ],
            'geographic': [
                'borough_risk_score', 'zipcode_risk_score',
                'boro_MANHATTAN', 'boro_BROOKLYN', 'boro_QUEENS', 'boro_BRONX', 'boro_STATEN ISLAND'
            ],
            'cuisine': [
                'cuisine_risk_score', 'cuisine_category'
            ],
            'inspection_pattern': [
                'is_initial_inspection', 'is_reinspection', 'is_complaint_inspection',
                'inspection_number_this_year'
            ],
            'violation': [
                'violation_count_this_inspection', 'any_high_risk_violation'
            ]
        }
    
    def prepare_model_data(
        self, 
        df: pd.DataFrame,
        target_col: str = 'inspection_failed',
        test_year: int = 2024
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for model training with proper train/test split.
        
        Uses temporal split to avoid data leakage:
        - Train: All data before test_year
        - Test: Data from test_year onwards
        
        Args:
            df: Feature-engineered DataFrame
            target_col: Name of target column
            test_year: Year to use as test set
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Columns to exclude (IDs, dates, raw text, and target)
        exclude_cols = [
            # IDs and raw data
            'camis', 'dba', 'boro', 'building', 'street', 'zipcode', 'phone',
            'cuisine_description', 'inspection_date', 'action', 'violation_code',
            'violation_description', 'critical_flag', 'score', 'grade',
            'grade_date', 'record_date', 'inspection_type', 'latitude', 'longitude',
            # Target and intermediate columns
            target_col, 'is_critical', 'has_high_risk_violation', 'cuisine_category',
            'inspection_year',
            # Extra geographic columns (text-based)
            'community_board', 'council_district', 'census_tract', 'bin', 'bbl',
            'nta', 'location', 'boro_clean',
            # Any has_ columns that are intermediate
            'has_pest_violation', 'has_temp_violation'
        ]
        
        # Get only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to numeric columns not in exclude list
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Using {len(feature_cols)} features for modeling")
        
        # Temporal split
        if 'inspection_year' not in df.columns:
            df['inspection_year'] = df['inspection_date'].dt.year
            
        train_mask = df['inspection_year'] < test_year
        test_mask = df['inspection_year'] >= test_year
        
        X_train = df.loc[train_mask, feature_cols].copy()
        X_test = df.loc[test_mask, feature_cols].copy()
        y_train = df.loc[train_mask, target_col].copy()
        y_test = df.loc[test_mask, target_col].copy()
        
        # Handle any remaining NaN values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Handle any remaining inf values
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Train set: {len(X_train):,} samples ({y_train.mean():.1%} positive)")
        logger.info(f"Test set: {len(X_test):,} samples ({y_test.mean():.1%} positive)")
        
        return X_train, X_test, y_train, y_test


def main():
    """Run feature engineering pipeline."""
    
    # Load cleaned data
    input_path = DATA_PROCESSED / "inspections_cleaned.csv"
    
    if not input_path.exists():
        print(f"Error: Cleaned data not found at {input_path}")
        print("Run data_loader.py first!")
        return
    
    df = pd.read_csv(input_path, parse_dates=['inspection_date'])
    logger.info(f"Loaded {len(df):,} rows from cleaned data")
    
    # Create features
    fe = FeatureEngineer()
    df_features = fe.create_all_features(df)
    
    # Save featured data
    output_path = DATA_PROCESSED / "inspections_featured.csv"
    df_features.to_csv(output_path, index=False)
    logger.info(f"Saved featured data to {output_path}")
    
    # Print feature summary
    print("\n" + "="*50)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*50)
    
    feature_list = fe.get_feature_list()
    for category, features in feature_list.items():
        present = [f for f in features if f in df_features.columns]
        print(f"\n{category.upper()} ({len(present)} features):")
        for f in present[:5]:  # Show first 5
            print(f"  - {f}")
        if len(present) > 5:
            print(f"  ... and {len(present) - 5} more")


if __name__ == "__main__":
    main()
