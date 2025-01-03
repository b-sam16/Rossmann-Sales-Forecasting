import sys
sys.path.append('../src')
import pandas as pd
from logger import Logger

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
            
            # First 5 Rows
            head = self.df.head()
            self.logger.info(f"First 5 Rows:\n{head}")
            
            # Data Info (non-null count, memory usage)
            info = self.df.info()
            self.logger.info("Data Info collected successfully.")
            
            # Basic Statistics
            stats = self.df.describe()
            self.logger.info(f"Basic Statistics:\n{stats}")
            
            return head, info, stats
        
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
            
            # Filter only columns with missing data
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            
            self.logger.info("Missing values calculated successfully.")
            self.logger.info(f"Missing Values Summary:\n{missing_df}")
            
            return missing_df
        except Exception as e:
            self.logger.error(f"Error while checking missing values: {e}")
            raise
    
    def process_and_clean_data(self):
        # Log the start of the data cleaning process
        self.logger.info("Processing and cleaning data: Handling missing values, converting data types, and checking for negative values...")

        # 1. Handle Missing Values
        for column in self.df.columns:
            if self.df[column].dtype == 'object':  # Categorical columns
                missing_count = self.df[column].isnull().sum()
                if missing_count > 0:
                    self.df[column].fillna('Unknown', inplace=True)
                    self.logger.info(f"Missing values in '{column}' filled with 'Unknown'.")
            else:  # Numeric columns
                missing_count = self.df[column].isnull().sum()
                if missing_count > 0:
                    self.df[column].fillna(self.df[column].median(), inplace=True)
                    self.logger.info(f"Missing values in '{column}' filled with median value.")

        # 2. Convert Data Types using a loop
        dtype_conversions = {
            'StateHoliday': 'category',
            'StoreType': 'category',
            'Assortment': 'category',
            'PromoInterval': 'category',
            'Date': 'datetime',
            'CompetitionOpenSinceMonth': 'Int64',
            'CompetitionOpenSinceYear': 'Int64'
        }

        for column, dtype in dtype_conversions.items():
            if column in self.df.columns:
                if dtype == 'datetime':
                    self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
                else:
                    self.df[column] = self.df[column].astype(dtype, errors='ignore')
                self.logger.info(f"Converted '{column}' to {dtype}.")

        # 3. Check for Negative Values
        for column in self.df.select_dtypes(include=['float64', 'int64']).columns:
            negative_values = self.df[column] < 0
            if negative_values.any():
                negative_count = negative_values.sum()
                self.df.loc[negative_values, column] = 0  # Replace negative values with zero or an appropriate value
                self.logger.info(f"Replaced {negative_count} negative values in '{column}' with zero.")

        # Log the completion of the data cleaning process
        self.logger.info("Data processing and cleaning completed.")

        return self.df