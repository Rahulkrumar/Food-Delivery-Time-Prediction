"""
Model Training Module
Trains and evaluates delivery time prediction models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeliveryTimeModel:
    """Train and evaluate delivery time prediction models"""
    
    def __init__(self, data_path='data/processed/features.csv'):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.best_model = None
        
    def load_data(self):
        """Load feature data"""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} records with {len(self.df.columns)} features")
        return self
    
    def prepare_data(self, target_col='delivery_time'):
        """Prepare X and y"""
        logger.info("Preparing features and target...")
        
        # Exclude non-feature columns
        exclude_cols = [
            target_col,
            'ID', 'Delivery_person_ID', 'Order_Date',
            'Time_Orderd', 'Time_Order_picked',
            'Restaurant_latitude', 'Restaurant_longitude',
            'Delivery_location_latitude', 'Delivery_location_longitude',
            'Type_of_order', 'Type_of_vehicle',
            'Road_traffic_density', 'Weather_conditions',
            'multiple_deliveries', 'Festival', 'City',
            'Delivery_person_Age', 'Delivery_person_Ratings'
        ]
        
        # Get feature columns
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Remove columns with too many nulls
        null_pct = self.df[feature_cols].isnull().mean()
        feature_cols = [col for col in feature_cols if null_pct[col] < 0.2]
        
        logger.info(f"Using {len(feature_cols)} features")
        
        X = self.df[feature_cols].fillna(0)
        y = self.df[target_col]
        
        # Remove any remaining NaN targets
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"Final dataset: {X.shape}")
        
        return X, y, feature_cols
    
    def split_data(self, X, y, test_size=0.2):
        """Split into train and test sets"""
        logger.info(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"Train: {len(X_train)} records")
        logger.info(f"Test: {len(X_test)} records")
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression (baseline)"""
        logger.info("Training Linear Regression...")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        self.models['linear_regression'] = model
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest"""
        logger.info("Training Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost"""
        logger.info("Training XGBoost...")
        
        model = XGBRegressor(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate model performance"""
        logger.info(f"\nEvaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"{model_name} Performance:")
        logger.info(f"  MAE: {mae:.2f} minutes")
        logger.info(f"  RMSE: {rmse:.2f} minutes")
        logger.info(f"  R²: {r2:.4f}")
        
        return {'MAE': mae, 'RMSE': rmse, 'R²': r2}
    
    def save_model(self, model, model_name, output_dir='models'):
        """Save trained model"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / f'{model_name}_model.pkl'
        joblib.dump(model, model_path)
        logger.info(f"Saved {model_name} to {model_path}")
    
    def train_all(self):
        """Complete training pipeline"""
        logger.info("="*80)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*80)
        
        # Load and prepare
        self.load_data()
        X, y, feature_cols = self.prepare_data()
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        results = {}
        
        # Train models
        lr_model = self.train_linear_regression(X_train, y_train)
        results['Linear Regression'] = self.evaluate_model(lr_model, X_test, y_test, "Linear Regression")
        
        rf_model = self.train_random_forest(X_train, y_train)
        results['Random Forest'] = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        
        xgb_model = self.train_xgboost(X_train, y_train)
        results['XGBoost'] = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        
        # Find best model (lowest MAE)
        best_model_name = min(results, key=lambda x: results[x]['MAE'])
        self.best_model = self.models[best_model_name.lower().replace(' ', '_')]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"BEST MODEL: {best_model_name} (MAE: {results[best_model_name]['MAE']:.2f} min)")
        logger.info(f"{'='*80}")
        
        # Save best model
        self.save_model(self.best_model, 'best_model')
        
        # Save feature columns
        joblib.dump(feature_cols, 'models/feature_columns.pkl')
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        
        return results


if __name__ == "__main__":
    trainer = DeliveryTimeModel('data/processed/features.csv')
    results = trainer.train_all()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            if metric_name == 'R²':
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value:.2f} minutes")
