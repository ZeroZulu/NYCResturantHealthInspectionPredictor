"""
Model Training Module
=====================
Trains and evaluates multiple classification models for inspection prediction.

Models implemented:
1. Logistic Regression (interpretable baseline)
2. Random Forest (robust ensemble)
3. XGBoost (typically best performance)

Includes:
- Hyperparameter tuning with cross-validation
- Handling class imbalance (SMOTE)
- Comprehensive evaluation metrics
- SHAP explainability
- Model persistence

Author: [Shril Patel]
Date: [Dec 13, 2025]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging
import joblib
from datetime import datetime
import warnings

# ML Libraries
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")

# SMOTE for imbalanced data
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    warnings.warn("imbalanced-learn not installed. Install with: pip install imbalanced-learn")

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


class ModelTrainer:
    """
    Trains, evaluates, and explains ML models for inspection prediction.
    
    Example:
        trainer = ModelTrainer()
        results = trainer.train_all_models(X_train, X_test, y_train, y_test)
        trainer.explain_model('xgboost', X_test)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_names = None
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        use_smote: bool = True
    ) -> Dict[str, Dict]:
        """
        Train and evaluate all models.
        
        Args:
            X_train, X_test: Feature DataFrames
            y_train, y_test: Target Series
            use_smote: Whether to use SMOTE for class imbalance
            
        Returns:
            Dictionary of results for each model
        """
        self.feature_names = list(X_train.columns)
        
        # Handle class imbalance
        if use_smote and SMOTE_AVAILABLE:
            logger.info("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=self.random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE: {len(X_train_balanced):,} samples")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Scale features for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Train each model
        results = {}
        
        # 1. Logistic Regression
        logger.info("Training Logistic Regression...")
        results['logistic_regression'] = self._train_logistic_regression(
            X_train_scaled, X_test_scaled, y_train_balanced, y_test
        )
        
        # 2. Random Forest
        logger.info("Training Random Forest...")
        results['random_forest'] = self._train_random_forest(
            X_train_balanced, X_test, y_train_balanced, y_test
        )
        
        # 3. XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost...")
            results['xgboost'] = self._train_xgboost(
                X_train_balanced, X_test, y_train_balanced, y_test
            )
        
        self.results = results
        return results
    
    def _train_logistic_regression(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """Train Logistic Regression with hyperparameter tuning."""
        
        # Hyperparameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['saga']
        }
        
        # Grid search
        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        self.models['logistic_regression'] = best_model
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        return self._compute_metrics(y_test, y_pred, y_prob, 'Logistic Regression')
    
    def _train_random_forest(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """Train Random Forest with hyperparameter tuning."""
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Use RandomizedSearchCV for faster tuning
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        self.models['random_forest'] = best_model
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        return self._compute_metrics(y_test, y_pred, y_prob, 'Random Forest')
    
    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """Train XGBoost with hyperparameter tuning."""
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        self.models['xgboost'] = best_model
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        return self._compute_metrics(y_test, y_pred, y_prob, 'XGBoost')
    
    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        model_name: str
    ) -> Dict:
        """Compute comprehensive evaluation metrics."""
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred)
        }
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.3f}")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall:    {metrics['recall']:.3f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.3f}")
        logger.info(f"  AUC-ROC:   {metrics['roc_auc']:.3f}")
        
        return metrics
    
    def get_feature_importance(self, model_name: str = 'xgboost') -> pd.DataFrame:
        """
        Get feature importance from a trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with feature importances
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train it first!")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            raise ValueError(f"Model '{model_name}' doesn't have feature importances")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def explain_with_shap(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        n_samples: int = 100
    ) -> Optional[Any]:
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            model_name: Name of the model to explain
            X_test: Test features
            n_samples: Number of samples to explain
            
        Returns:
            SHAP explainer object
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
            return None
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        X_sample = X_test.iloc[:n_samples]
        
        logger.info(f"Computing SHAP values for {model_name}...")
        
        if model_name == 'logistic_regression':
            # Scale features for logistic regression
            X_sample_scaled = self.scalers['standard'].transform(X_sample)
            explainer = shap.LinearExplainer(model, X_sample_scaled)
            shap_values = explainer.shap_values(X_sample_scaled)
        else:
            # Tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'X_sample': X_sample
        }
    
    def plot_feature_importance(
        self,
        model_name: str = 'xgboost',
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """Plot feature importance."""
        
        importance_df = self.get_feature_importance(model_name).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(
            data=importance_df,
            x='importance',
            y='feature',
            palette='viridis',
            ax=ax
        )
        ax.set_title(f'Top {top_n} Feature Importances - {model_name.title()}')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.close()
        return fig
    
    def plot_roc_curves(self, y_test: pd.Series, X_test: pd.DataFrame, save_path: Optional[str] = None):
        """Plot ROC curves for all models."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            if model_name == 'logistic_regression':
                X_scaled = self.scalers['standard'].transform(X_test)
                y_prob = model.predict_proba(X_scaled)[:, 1]
            else:
                y_prob = model.predict_proba(X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            
            ax.plot(fpr, tpr, label=f'{model_name.title()} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved ROC curves to {save_path}")
        
        plt.close()
        return fig
    
    def plot_confusion_matrix(
        self,
        model_name: str,
        y_test: pd.Series,
        X_test: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix for a model."""
        
        model = self.models[model_name]
        
        if model_name == 'logistic_regression':
            X_scaled = self.scalers['standard'].transform(X_test)
            y_pred = model.predict(X_scaled)
        else:
            y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Pass', 'Fail'],
            yticklabels=['Pass', 'Fail'],
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_name.title()}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.close()
        return fig
    
    def save_models(self, version: str = "v1"):
        """Save all trained models to disk."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            filename = MODELS_DIR / f"{model_name}_{version}_{timestamp}.joblib"
            joblib.dump(model, filename)
            logger.info(f"Saved {model_name} to {filename}")
        
        # Save scalers
        scaler_file = MODELS_DIR / f"scalers_{version}_{timestamp}.joblib"
        joblib.dump(self.scalers, scaler_file)
        logger.info(f"Saved scalers to {scaler_file}")
        
        # Save feature names
        features_file = MODELS_DIR / f"feature_names_{version}_{timestamp}.joblib"
        joblib.dump(self.feature_names, features_file)
        logger.info(f"Saved feature names to {features_file}")
    
    def load_model(self, model_path: str) -> Any:
        """Load a saved model."""
        return joblib.load(model_path)
    
    def print_comparison_table(self):
        """Print a comparison table of all model results."""
        
        if not self.results:
            logger.warning("No results to display. Train models first!")
            return
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
        print("-"*70)
        
        for model_name, metrics in self.results.items():
            print(f"{metrics['model_name']:<25} "
                  f"{metrics['accuracy']:<10.3f} "
                  f"{metrics['precision']:<10.3f} "
                  f"{metrics['recall']:<10.3f} "
                  f"{metrics['f1_score']:<10.3f} "
                  f"{metrics['roc_auc']:<10.3f}")
        
        print("="*70)


def main():
    """Run model training pipeline."""
    
    from feature_engineering import FeatureEngineer
    
    # Load featured data
    input_path = DATA_PROCESSED / "inspections_featured.csv"
    
    if not input_path.exists():
        print(f"Error: Featured data not found at {input_path}")
        print("Run feature_engineering.py first!")
        return
    
    df = pd.read_csv(input_path, parse_dates=['inspection_date'])
    logger.info(f"Loaded {len(df):,} rows")
    
    # Prepare model data
    fe = FeatureEngineer()
    X_train, X_test, y_train, y_test = fe.prepare_model_data(df, test_year=2024)
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Print comparison
    trainer.print_comparison_table()
    
    # Get feature importance
    print("\n" + "="*50)
    print("TOP 10 FEATURES (XGBoost)")
    print("="*50)
    importance = trainer.get_feature_importance('xgboost')
    print(importance.head(10).to_string(index=False))
    
    # Save plots
    plots_dir = PROJECT_ROOT / "assets"
    plots_dir.mkdir(exist_ok=True)
    
    trainer.plot_feature_importance('xgboost', save_path=plots_dir / 'feature_importance.png')
    trainer.plot_roc_curves(y_test, X_test, save_path=plots_dir / 'roc_curves.png')
    trainer.plot_confusion_matrix('xgboost', y_test, X_test, save_path=plots_dir / 'confusion_matrix.png')
    
    # Save models
    trainer.save_models()
    
    print("\n✅ Model training complete!")
    print(f"   Models saved to: {MODELS_DIR}")
    print(f"   Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
