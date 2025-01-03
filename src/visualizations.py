import sys
sys.path.append('../src')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from logger import Logger
import matplotlib.dates as mdates

# Initialize Logger
logger = Logger('SalesVisualizer').get_logger()


class SalesVisualizer:
    """
    Class for generating sales visualizations and answering key analysis questions.
    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Initialize with training and test datasets.
        """
        self.train_df = train_df
        self.test_df = test_df
        logger.info("SalesVisualizer initialized with training and test data.")

    def compare_promo_distribution(self):
        """
        Compare promotion distribution between training and test datasets.
        """
        try:
            logger.info("Comparing promotion distribution between training and test datasets.")
            plt.figure(figsize=(10, 6))
            sns.histplot(self.train_df['Promo'], color='blue', label='Train', kde=True)
            sns.histplot(self.test_df['Promo'], color='orange', label='Test', kde=True)
            plt.title('Promotion Distribution in Train vs Test Data')
            plt.legend()
            plt.show()
            logger.info("Promotion distribution plot generated successfully.")
        except Exception as e:
            logger.error(f"Error in compare_promo_distribution: {e}")

    def sales_during_holidays(self):
        """
        Analyze sales behavior before, during, and after holidays.
        """
        try:
            logger.info("Analyzing sales behavior before, during, and after holidays using a line plot with markers.")
        
            # Ensure 'Date' column is in datetime format
            self.train_df['Date'] = pd.to_datetime(self.train_df['Date'])
        
            # Group by date and calculate average sales
            daily_sales = self.train_df.groupby(['Date', 'SchoolHoliday'])['Sales'].mean().reset_index()
        
            plt.figure(figsize=(14, 7))
            sns.lineplot(data=daily_sales, x='Date', y='Sales', hue='SchoolHoliday', palette='tab10')
        
            # Highlight holiday periods
            holiday_dates = daily_sales[daily_sales['SchoolHoliday'] == 1]['Date']
            plt.vlines(holiday_dates, ymin=daily_sales['Sales'].min(), ymax=daily_sales['Sales'].max(),
                   colors='red', linestyles='dashed', alpha=0.5, label='Holiday')
        
            plt.title('Sales Behavior Before, During, and After Holidays')
            plt.xlabel('Date')
            plt.ylabel('Average Sales')
            plt.legend(title='Holiday Status')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
            logger.info("Sales behavior during holidays plot generated successfully using line plot with markers.")
        except Exception as e:
            logger.error(f"Error in sales_during_holidays: {e}")

    def seasonal_behavior(self):
        """
        Analyze seasonal sales patterns.
        """
        try:
            logger.info("Analyzing seasonal sales behavior.")
            self.train_df['Date'] = pd.to_datetime(self.train_df['Date'])
            self.train_df['Month'] = self.train_df['Date'].dt.month
            monthly_sales = self.train_df.groupby('Month')['Sales'].mean().reset_index()
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o')
            plt.title('Average Monthly Sales Trend')
            plt.show()
            logger.info("Seasonal behavior plot generated successfully.")
        except Exception as e:
            logger.error(f"Error in seasonal_behavior: {e}")

    def sales_customer_correlation(self):
        """
        Analyze correlation between sales and number of customers.
        """
        try:
            logger.info("Analyzing correlation between sales and customers.")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=self.train_df, x='Customers', y='Sales', alpha=0.5)
            plt.title('Correlation Between Sales and Number of Customers')
            plt.show()
            logger.info("Sales and customers correlation plot generated successfully.")
        except Exception as e:
            logger.error(f"Error in sales_customer_correlation: {e}")