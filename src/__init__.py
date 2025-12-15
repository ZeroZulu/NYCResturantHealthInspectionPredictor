"""
NYC Restaurant Health Inspection Predictor
==========================================

A machine learning system for predicting restaurant inspection outcomes.

Modules:
- data_loader: Data ingestion and cleaning
- feature_engineering: Feature creation pipeline
- model_training: ML model training and evaluation
- utils: Helper functions

Example usage:
    from src.data_loader import NYCRestaurantDataLoader
    from src.feature_engineering import FeatureEngineer
    from src.model_training import ModelTrainer
    
    # Load and clean data
    loader = NYCRestaurantDataLoader()
    df = loader.load_from_csv("data/raw/inspections.csv")
    df_clean = loader.clean_data(df)
    
    # Create features
    fe = FeatureEngineer()
    df_features = fe.create_all_features(df_clean)
    
    # Train models
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = fe.prepare_model_data(df_features)
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
"""

__version__ = "1.0.0"
__author__ = "[Your Name]"

from .data_loader import NYCRestaurantDataLoader
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .utils import (
    get_risk_category,
    get_grade_from_score,
    validate_camis,
    Timer
)

__all__ = [
    'NYCRestaurantDataLoader',
    'FeatureEngineer',
    'ModelTrainer',
    'get_risk_category',
    'get_grade_from_score',
    'validate_camis',
    'Timer'
]
