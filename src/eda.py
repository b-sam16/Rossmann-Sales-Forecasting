# src/eda.py

import pandas as pd
from src.logger import Logger

class EDA:
    def __init__(self, train_path: str, store_path: str):
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.train_path = train_path
        self.store_path = store_path
        self.df = None  # Combined DataFrame

    def load_and_merge_data(self):
        """Load train and store datasets and merge them on 'Store' column."""
        try:
            self.logger.info("Loading train and store datasets...")
            train_df = pd.read_csv(self.train_path)
            store_df = pd.read_csv(self.store_path)
            self.logger.info("Datasets loaded successfully. Merging them...")
            self.df = pd.merge(train_df, store_df, on='Store', how='left')
            self.logger.info("Datasets merged successfully.")
            return self.df
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error during data loading or merging: {e}")
            raise

    def explore_data(self):
        """Explore basic details of the dataset - data types, first few rows, info, and basic statistics."""
        try:
            self.logger.info("Exploring dataset...")
            
            # Data Types
            data_types = self.df.dtypes
            self.logger.info(f"Data Types:\n{data_types}")
            
            # First 5 Rows
            head = self.df.head()
            self.logger.info(f"First 5 Rows:\n{head}")
            
            # Data Info (non-null count, memory usage)
            info = self.df.info()
            self.logger.info("Data Info collected successfully.")
            
            # Basic Statistics
            stats = self.df.describe()
            self.logger.info(f"Basic Statistics:\n{stats}")
            
            return data_types, head, info, stats
        
        except Exception as e:
            self.logger.error(f"Error exploring data: {e}")
            raise

    def check_missing_values(self):
        """Check for missing values in the dataset."""
        try:
            self.logger.info("Checking for missing values in the dataset...")
            missing_values = self.df.isnull().sum()
            missing_percentage = (missing_values / len(self.df)) * 100
            missing_df = pd.DataFrame({
                'Missing Values': missing_values,
                'Percentage': missing_percentage
            }).sort_values(by='Missing Values', ascending=False)
            
            self.logger.info("Missing values calculated successfully.")
            self.logger.info(f"Missing Values Summary:\n{missing_df[missing_df['Missing Values'] > 0]}")
            
            return missing_df[missing_df['Missing Values'] > 0]
        
        except Exception as e:
            self.logger.error(f"Error while checking missing values: {e}")
            raise
