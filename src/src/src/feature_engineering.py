"""
Feature Engineering Module
Creates features for delivery time prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from math import radians, sin, cos, sqrt, atan2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for model training"""
    
    def __init__(self, df):
        self.df = df.copy()
        
    def calculate_distance(self, lat1_col, lon1_col, lat2_col, lon2_col):
        """
        Calculate distance using Haversine formula
        Returns distance in kilometers
        """
        logger.info("Calculating distances using Haversine formula...")
        
        def haversine(lat1, lon1, lat2, lon2):
            """Haversine formula for distance calculation"""
            R = 6371  # Earth radius in km
            
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            
            return R * c
        
        # Apply to dataframe
        self.df['distance_km'] = self.df.apply(
            lambda row: haversine(
                row[lat1_col], row[lon1_col],
                row[lat2_col], row[lon2_col]
            ), axis=1
        )
        
        logger.info(f"Distance range: {self.df['distance_km'].min():.2f} - {self.df['distance_km'].max():.2f} km")
        return self
    
    def create_time_features(self, time_col):
        """Extract time-based features"""
        logger.info("Creating time features...")
        
        # Convert to datetime if needed
        if self.df[time_col].dtype == 'object':
            self.df[time_col] = pd.to_datetime(self.df[time_col])
        
        # Extract hour
        self.df['order_hour'] = self.df[time_col].dt.hour
        
        # Peak hours: Lunch (11-14) and Dinner (18-21)
        self.df['is_peak_hour'] = self.df['order_hour'].apply(
            lambda x: 1 if (11 <= x <= 14) or (18 <= x <= 21) else 0
        )
        
        # Weekend indicator
        self.df['is_weekend'] = self.df[time_col].dt.dayofweek.apply(
            lambda x: 1 if x >= 5 else 0
        )
        
        logger.info("Time features created!")
        return self
    
    def encode_traffic(self, traffic_col):
        """Encode traffic density"""
        logger.info("Encoding traffic...")
        
        traffic_mapping = {
            'Low': 1,
            'Medium': 2,
            'High': 3,
            'Jam': 4
        }
        
        self.df['traffic_encoded'] = self.df[traffic_col].map(traffic_mapping)
        
        # Fill any unmapped values with median
        if self.df['traffic_encoded'].isnull().any():
            self.df['traffic_encoded'].fillna(2, inplace=True)
        
        return self
    
    def encode_weather(self, weather_col):
        """Encode weather conditions"""
        logger.info("Encoding weather...")
        
        weather_mapping = {
            'Sunny': 0,
            'Clear': 0,
            'Cloudy': 1,
            'Fog': 2,
            'Rain': 3,
            'Storm': 4,
            'Sandstorms': 4
        }
        
        self.df['weather_encoded'] = self.df[weather_col].map(weather_mapping)
        
        # Fill any unmapped values
        if self.df['weather_encoded'].isnull().any():
            self.df['weather_encoded'].fillna(0, inplace=True)
        
        return self
    
    def create_interaction_features(self):
        """Create interaction features"""
        logger.info("Creating interaction features...")
        
        # Distance × Traffic
        if 'distance_km' in self.df.columns and 'traffic_encoded' in self.df.columns:
            self.df['distance_traffic'] = self.df['distance_km'] * self.df['traffic_encoded']
        
        # Peak hour × Traffic
        if 'is_peak_hour' in self.df.columns and 'traffic_encoded' in self.df.columns:
            self.df['peak_traffic'] = self.df['is_peak_hour'] * self.df['traffic_encoded']
        
        return self
    
    def create_all_features(self, config):
        """Create all features based on config"""
        logger.info("="*60)
        logger.info("CREATING ALL FEATURES")
        logger.info("="*60)
        
        initial_cols = len(self.df.columns)
        
        # Distance
        if all(k in config for k in ['lat1', 'lon1', 'lat2', 'lon2']):
            self.calculate_distance(
                config['lat1'], config['lon1'],
                config['lat2'], config['lon2']
            )
        
        # Time features
        if 'time_col' in config:
            self.create_time_features(config['time_col'])
        
        # Traffic encoding
        if 'traffic_col' in config:
            self.encode_traffic(config['traffic_col'])
        
        # Weather encoding
        if 'weather_col' in config:
            self.encode_weather(config['weather_col'])
        
        # Interactions
        self.create_interaction_features()
        
        final_cols = len(self.df.columns)
        
        logger.info("="*60)
        logger.info(f"Initial columns: {initial_cols}")
        logger.info(f"Final columns: {final_cols}")
        logger.info(f"New features: {final_cols - initial_cols}")
        logger.info("="*60)
        
        return self.df
    
    def save_features(self, output_path='data/processed/features.csv'):
        """Save engineered features"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Features saved to: {output_path}")


if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('data/processed/processed_data.csv')
    
    # Feature engineering config
    # Update column names based on your actual dataset
    config = {
        'lat1': 'Restaurant_latitude',
        'lon1': 'Restaurant_longitude',
        'lat2': 'Delivery_location_latitude',
        'lon2': 'Delivery_location_longitude',
        'time_col': 'Order_Date',
        'traffic_col': 'Road_traffic_density',
        'weather_col': 'Weather_conditions'
    }
    
    # Create features
    engineer = FeatureEngineer(df)
    df_features = engineer.create_all_features(config)
    
    # Save
    engineer.save_features('data/processed/features.csv')
    
    print("\nFeature Engineering Complete!")
    print(f"Total features: {len(df_features.columns)}")
    print("\nNew features:")
    print(df_features.columns.tolist())
