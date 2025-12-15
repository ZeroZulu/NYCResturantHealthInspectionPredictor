"""
Utility Functions
=================
Helper functions used across the project.

Author: [Shril Patel]
Date: [Dec 13, 2025]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default if division by zero.
    
    Args:
        numerator: The number to divide
        denominator: The number to divide by
        default: Value to return if denominator is 0
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def get_grade_from_score(score: float) -> str:
    """
    Convert inspection score to letter grade.
    
    NYC Grading:
    - A: 0-13 points
    - B: 14-27 points
    - C: 28+ points
    
    Args:
        score: Inspection score
        
    Returns:
        Letter grade (A, B, or C)
    """
    if pd.isna(score):
        return None
    
    if score <= 13:
        return 'A'
    elif score <= 27:
        return 'B'
    else:
        return 'C'


def format_large_number(number: int) -> str:
    """
    Format large numbers with commas.
    
    Args:
        number: Integer to format
        
    Returns:
        Formatted string (e.g., "1,234,567")
    """
    return f"{number:,}"


def calculate_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced classification.
    
    Args:
        y: Target variable
        
    Returns:
        Dictionary mapping class to weight
    """
    class_counts = y.value_counts()
    total = len(y)
    n_classes = len(class_counts)
    
    weights = {}
    for cls, count in class_counts.items():
        weights[cls] = total / (n_classes * count)
    
    return weights


def get_risk_category(probability: float) -> Dict[str, str]:
    """
    Categorize risk based on failure probability.
    
    Args:
        probability: Predicted failure probability (0-1)
        
    Returns:
        Dictionary with risk level, color, and emoji
    """
    if probability >= 0.7:
        return {
            'level': 'HIGH',
            'color': '#ff4444',
            'emoji': '🔴',
            'description': 'Immediate attention recommended'
        }
    elif probability >= 0.4:
        return {
            'level': 'MEDIUM',
            'color': '#ffaa00',
            'emoji': '🟡',
            'description': 'Monitor closely'
        }
    else:
        return {
            'level': 'LOW',
            'color': '#00C851',
            'emoji': '🟢',
            'description': 'Normal risk profile'
        }


def validate_camis(camis: Any) -> bool:
    """
    Validate a CAMIS (restaurant ID) value.
    
    Args:
        camis: Value to validate
        
    Returns:
        True if valid CAMIS
    """
    try:
        camis_int = int(camis)
        return 10000000 <= camis_int <= 99999999
    except (ValueError, TypeError):
        return False


def get_nyc_coordinates() -> Dict[str, tuple]:
    """
    Get approximate center coordinates for NYC boroughs.
    
    Returns:
        Dictionary mapping borough to (lat, lon)
    """
    return {
        'MANHATTAN': (40.7831, -73.9712),
        'BROOKLYN': (40.6782, -73.9442),
        'QUEENS': (40.7282, -73.7949),
        'BRONX': (40.8448, -73.8648),
        'STATEN ISLAND': (40.5795, -74.1502)
    }


def create_directory_structure(base_path: Path) -> None:
    """
    Create the standard project directory structure.
    
    Args:
        base_path: Root path for the project
    """
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'notebooks',
        'src',
        'streamlit_app',
        'streamlit_app/pages',
        'streamlit_app/components',
        'tests',
        'assets'
    ]
    
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {full_path}")


def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Log summary information about a DataFrame.
    
    Args:
        df: DataFrame to summarize
        name: Name to use in logging
    """
    logger.info(f"\n{name} Summary:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"  Missing values: {df.isnull().sum().sum():,}")
    logger.info(f"  Columns: {list(df.columns)}")


class Timer:
    """
    Context manager for timing code blocks.
    
    Example:
        with Timer("Data loading"):
            df = load_data()
    """
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        logger.info(f"Starting: {self.name}")
        return self
    
    def __exit__(self, *args):
        import time
        elapsed = time.time() - self.start_time
        logger.info(f"Completed: {self.name} ({elapsed:.2f}s)")
