"""
Data Processing Module
Cleans and validates food delivery data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Clean and process delivery data"""
    
    def __init__(self, data_path='data/raw/food_delivery_time_dataset.csv'):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """Load CSV file"""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} records")
        logger.info(f"Columns: {self.df.columns.tolist()}")
        return self
    
    def check_missing_values(self):
        """Check for missing values"""
        logger.info("Checking missing values...")
        missing = self.df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            # Fill or drop
            self.df.dropna(inplace=True)
        else:
            logger.info("No missing values!")
        return self
    
    def remove_outliers(self):
        """Remove outliers from delivery time"""
        logger.info("Removing outliers...")
        
        # Remove extreme delivery times
        if 'delivery_time' in self.df.columns:
            q1 = self.df['delivery_time'].quantile(0.01)
            q99 = self.df['delivery_time'].quantile(0.99)
            
            initial_len = len(self.df)
            self.df = self.df[
                (self.df['delivery_time'] >= q1) & 
                (self.df['delivery_time'] <= q99)
            ]
            logger.info(f"Removed {initial_len - len(self.df)} outlier records")
        
        return self
    
    def validate_coordinates(self):
        """Validate latitude and longitude"""
        logger.info("Validating coordinates...")
        
        # Valid lat: -90 to 90, long: -180 to 180
        coord_cols = [col for col in self.df.columns if 'latitude' in col.lower() or 'longitude' in col.lower()]
        
        for col in coord_cols:
            if 'latitude' in col.lower():
                self.df = self.df[(self.df[col] >= -90) & (self.df[col] <= 90)]
            elif 'longitude' in col.lower():
                self.df = self.df[(self.df[col] >= -180) & (self.df[col] <= 180)]
        
        logger.info(f"Valid records: {len(self.df)}")
        return self
    
    def process_all(self, save_path='data/processed'):
        """Run complete pipeline"""
        logger.info("="*60)
        logger.info("STARTING DATA PROCESSING")
        logger.info("="*60)
        
        self.load_data()
        self.check_missing_values()
        self.remove_outliers()
        self.validate_coordinates()
        
        # Save processed data
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        output_file = save_path / 'processed_data.csv'
        self.df.to_csv(output_file, index=False)
        logger.info(f"Saved to: {output_file}")
        
        logger.info("="*60)
        logger.info(f"Final records: {len(self.df)}")
        logger.info(f"Columns: {len(self.df.columns)}")
        logger.info("="*60)
        
        return self.df


if __name__ == "__main__":
    processor = DataProcessor('data/raw/food_delivery_time_dataset.csv')
    df = processor.process_all()
    print("\nProcessed Data:")
    print(df.head())
    print(f"\nShape: {df.shape}")
