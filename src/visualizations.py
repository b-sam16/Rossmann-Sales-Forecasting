import sys
sys.path.append('../src')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from logger import Logger
import matplotlib.dates as mdates
import logging

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
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
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

    def promo_impact_on_sales_and_customers(self):
        """
        Analyze how promo affects sales and whether it attracts new customers.
        """
        try:
            logger.info("Analyzing promo impact on sales and customer behavior.")
        
            # Compare sales and customer behavior during promo vs non-promo periods
            g = sns.FacetGrid(self.train_df, col='Promo', hue='Promo', height=5, aspect=1.5)
            g.map(sns.scatterplot, 'Customers', 'Sales', alpha=.7)
        
            g.add_legend()
            g.set_axis_labels('Number of Customers', 'Sales')
            plt.tight_layout()
            plt.show()
        
            logger.info("Promo impact on sales and customers plot generated successfully.")
        except Exception as e:
            logger.error(f"Error in promo_impact_on_sales_and_customers: {e}")
    def effective_promo_deployment(self):
        """
        Analyze where promos could be deployed more effectively.
        """
        try:
            logger.info("Analyzing effective promo deployment.")
        
            # Scatter plot comparing sales and promo status by store
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=self.train_df, x='Store', y='Sales', hue='Promo', style='Promo', palette='deep', markers=["o", "s"])
        
            plt.title('Promo Deployment Effectiveness by Store')
            plt.xlabel('Store ID')
            plt.ylabel('Sales')
            plt.tight_layout()
            plt.show()
        
            logger.info("Effective promo deployment plot generated successfully.")
        except Exception as e:
            logger.error(f"Error in effective_promo_deployment: {e}")

    def customer_behavior_during_opening_closing(self):
        """Visualize customer behavior during store opening and closing times.
        """
        try:
            self.logger.info("Analyzing customer behavior during store opening and closing times.")

            # Check for customer behavior based on the 'Open' column (0 = closed, 1 = open)
            open_data = self.df[self.df['Open'] == 1]  # Data when the store is open
            closed_data = self.df[self.df['Open'] == 0]  # Data when the store is closed

            # Visualize customer behavior during opening and closing times
            plt.figure(figsize=(12, 6))

            # Plot customer count during store open and closed times
            plt.plot(open_data['Date'], open_data['CustomerCount'], label='Customer Count during Opening', color='green', linestyle='--')
            plt.plot(closed_data['Date'], closed_data['CustomerCount'], label='Customer Count during Closing', color='red', linestyle=':')

            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Customer Count')
            plt.title('Customer Behavior During Store Opening and Closing Times')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)  # Rotate x-axis labels for readability
            plt.show()

            self.logger.info("Customer behavior during store opening and closing times visualized successfully.")

        except Exception as e:
            self.logger.error(f"Error in customer_behavior_during_opening_closing: {e}")
        raise


    def stores_open_all_weekdays(self):
        """Analyze stores open on all weekdays and their sales on weekends."""
        try:
            self.logger.info("Analyzing stores open on all weekdays and their sales on weekends.")
        
            # Identify stores that are open on all weekdays (Monday to Friday)
            weekday_open_stores = self.df.groupby('Store')['Open'].apply(lambda x: set(x[:5]) == {1}).reset_index()
            weekday_open_stores = weekday_open_stores[weekday_open_stores['Open'] == True]

            # Merge back with the original data to get sales data for the selected stores
            weekday_open_data = self.df[self.df['Store'].isin(weekday_open_stores['Store'])]

            # Analyze weekend sales for stores that are open on weekdays
            weekend_data = weekday_open_data[weekday_open_data['dayofweek'] >= 5]  # Saturday (5) and Sunday (6)

            # Visualize the sales behavior on weekends for stores open all weekdays
            plt.figure(figsize=(12, 6))

            # Plot sales for the weekend (Saturday and Sunday)
            plt.plot(weekend_data['Date'], weekend_data['Sales'], label='Weekend Sales for Weekday Open Stores', color='blue')

            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.title('Sales Behavior of Stores Open on All Weekdays    (Weekend Analysis)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.show()

            self.logger.info("Sales behavior for stores open on all weekdays visualized successfully.")

        except Exception as e:
            self.logger.error(f"Error in stores_open_all_weekdays: {e}")
        raise



    def assortment_type_sales_impact(self):
        """
        Analyze how assortment type affects sales.
        """
        try:
            logger.info("Analyzing how assortment type affects sales.")
        
            # Box plot to compare sales across different assortment types
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.train_df, x='Assortment', y='Sales', palette='Set2')
        
            plt.title('Sales by Assortment Type')
            plt.xlabel('Assortment Type')
            plt.ylabel('Sales')
            plt.tight_layout()
            plt.show()
        
            self.logger.info("Sales by assortment type plot generated successfully.")
        except Exception as e:
            self.logger.error(f"Error in assortment_type_sales_impact: {e}")


    def competitor_distance_sales_impact(self):
        """
        Analyze how the distance to the next competitor affects sales.
        """
        try:
            self.logger.info("Analyzing the impact of competitor distance on sales.")
        
            # Create a column to distinguish city center stores (assumed logic for city center classification)
            self.train_df['CityCenter'] = self.train_df['CompetitionDistance'].apply(lambda x: 1 if x < 1 else 0)
        
            # Scatter plot for distance to next competitor and sales
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=self.train_df, x='CompetitionDistance', y='Sales', hue='CityCenter', palette='viridis', alpha=0.7)
        
            plt.title('Sales vs. Competitor Distance')
            plt.xlabel('Competitor Distance (km)')
            plt.ylabel('Sales')
            plt.tight_layout()
            plt.show()
        
            self.logger.info("Competitor distance vs. sales plot generated successfully.")
        except Exception as e:
            self.logger.error(f"Error in competitor_distance_sales_impact: {e}")

    def competitor_reopening_sales_impact(self):
        """ Analyze how the reopening of competitors impacts store sales. """
        try:
            self.logger.info("Analyzing the impact of competitor re-opening on store sales.")
        
            # Create a column to classify stores that have had a competitor open/reopened
            self.train_df['CompetitorReopened'] = self.train_df.apply(
            lambda row: 1 if row['CompetitionOpenSinceMonth'] != 0 and row['CompetitionOpenSinceYear'] != 0 else 0, axis=1)

            # Plot the sales comparison
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='CompetitorReopened', y='Sales', data=self.train_df, palette='coolwarm')
            plt.title('Sales Impact of Competitor Reopening')
            plt.xlabel('Competitor Reopened (1: Yes, 0: No)')
            plt.ylabel('Sales')
            plt.xticks([0, 1], ['No', 'Yes'])
            plt.tight_layout()
            plt.show()

            self.logger.info("Competitor reopening vs. sales plot generated successfully.")
    
        except Exception as e:
            self.logger.error(f"Error in competitor_reopening_sales_impact: {e}")
            raise RuntimeError("An error occurred in competitor_reopening_sales_impact") from e
