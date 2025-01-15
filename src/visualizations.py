import sys
sys.path.append('../src')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from logger import Logger
import matplotlib.dates as mdates


class SalesVisualizer:
    """
    Class for generating sales visualizations and answering key analysis questions.
    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Initialize with training and test datasets.
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.train_df = train_df
        self.test_df = test_df
        self.logger.info("SalesVisualizer initialized with training and test data.")

    def compare_promo_distribution(self):
        """
        Compare promotion distribution between training and test datasets.
        """
        try:
            self.logger.info("Comparing promotion distribution between training and test datasets.")

            train_promos = self.train_df['Promo'].value_counts(normalize=True) 
            test_promos = self.test_df['Promo'].value_counts(normalize=True) 
            comparison_df = pd.DataFrame({'Train': train_promos, 'Test': test_promos})
                                                                    
            plt.figure(figsize=(10, 6))
            comparison_df.plot(kind='bar')
            plt.title('Promotion Distribution in Train vs Test Data')
            plt.xlabel('Promo') 
            plt.ylabel('Frequency')
            plt.show()
            self.logger.info("Promotion distribution plot generated successfully.")
        except Exception as e:
            self.logger.error(f"Error in compare_promo_distribution: {e}")

    def sales_behavior_around_holidays(self):
        """
        Compare sales behavior before, during, and after holidays.
        """
        try:
            self.logger.info("Comparing sales behavior before, during, and after holidays.")
        
            # Filter rows with holidays
            holiday_sales = self.train_df[self.train_df['StateHoliday'] != '0'].copy()
        
            # Ensure the dataset is sorted by Date for accurate time-based analysis
            self.train_df.sort_values(by='Date', inplace=True)
            self.train_df.reset_index(drop=True, inplace=True)
        
            # Create 'HolidayPeriod' column
            self.train_df['HolidayPeriod'] = 'Normal'
            self.train_df.loc[self.train_df['StateHoliday'] != '0', 'HolidayPeriod'] = 'During Holiday'
            self.train_df.loc[self.train_df['StateHoliday'] == '0','HolidayPeriod'] = 'Normal'
        
            # Label 'Before Holiday' and 'After Holiday'
            for idx in holiday_sales.index:
                if idx > 0:
                    self.train_df.at[idx - 1, 'HolidayPeriod'] = 'Before Holiday'
                if idx < len(self.train_df) - 1:
                    self.train_df.at[idx + 1, 'HolidayPeriod'] = 'After Holiday'
        
            # Group by HolidayPeriod and calculate average sales
            holiday_period_sales = self.train_df.groupby('HolidayPeriod')['Sales'].mean().reset_index()

            # Plotting
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=holiday_period_sales,
                x='HolidayPeriod',
                y='Sales',
                palette={'Before Holiday': 'skyblue', 'During Holiday': 'orange', 'After Holiday': 'green', 'Normal': 'gray'}
            )
        
            plt.title('Sales Behavior Before, During, and After Holidays')
            plt.xlabel('Holiday Period')
            plt.ylabel('Average Sales')
            plt.show()
        
        except Exception as e:
            self.logger.error(f"Error analyzing sales behavior around holidays: {e}")

    def seasonal_behavior(self):
        """
        Analyze seasonal sales patterns.
        """
        try:
            self.logger.info("Analyzing seasonal sales behavior.")
            self.train_df['Date'] = pd.to_datetime(self.train_df['Date'])
            self.train_df['Month'] = self.train_df['Date'].dt.month
            monthly_sales = self.train_df.groupby('Month')['Sales'].mean().reset_index()
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o')
            plt.title('Average Monthly Sales Trend')
            plt.show()
            self.logger.info("Seasonal behavior plot generated successfully.")
        except Exception as e:
            self.logger.error(f"Error analyzing seasonal sales behavior: {e}")

    def sales_customer_correlation(self):
        """
        Analyze correlation between sales and number of customers.
        """
        try:
            self.logger.info("Analyzing correlation between sales and customers.")

            # Check correlation coefficient
            correlation = self.train_df['Sales'].corr(self.train_df['Customers'])
            self.logger.info(f"Correlation between Sales and Customers: {correlation:.2f}")

            # Plotting scatter plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=self.train_df, x='Customers', y='Sales', alpha=0.5)
            plt.title('Correlation Between Sales and Number of Customers')
            plt.show()
            self.logger.info("Sales and customers correlation plot generated successfully.")
        except Exception as e:
            self.logger.error(f"Error in sales_customer_correlation: {e}")

    def promo_impact_on_sales_and_customers(self):
        """
        Analyze how promo affects sales and whether it attracts new customers.
        """
        try:
            self.logger.info("Analyzing promo impact on sales and customer behavior.")
        
            # Average sales and customers during promo and non-promo days
            promo_summary = self.train_df.groupby('Promo').agg({
                'Sales': 'mean',
                'Customers': 'mean'
            }).reset_index()
        
            # Melt the data for visualization
            promo_summary_melted = promo_summary.melt(
                id_vars='Promo',
                value_vars=['Sales', 'Customers'],
                var_name='Metric',
                value_name='Average'
            )
        
            # Plotting average sales and customers with and without promo
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(
                data=promo_summary_melted,
                x='Metric',
                y='Average',
                hue='Promo',
                palette='coolwarm'
            )
            plt.title('Average Sales and Customer Count During Promo and Non-Promo Days')
            plt.xlabel('Metric')
            plt.ylabel('Average Value')
            
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=['No Promo', 'Promo'], title='Promo')
            plt.show()
        
            # Promo trends over time
            plt.figure(figsize=(14, 7))
            ax = sns.lineplot(
                data=self.train_df,
                x='Date',
                y='Sales',
                hue='Promo'
            )
            plt.title('Sales Trends Over Time with Promo and Without Promo')
            plt.xlabel('Date')
            plt.ylabel('Sales')
            handles, labels = ax.get_legend_handles_labels()

            ax.legend(handles=handles, labels=['No Promo', 'Promo'], title='Promo')
            plt.show()
        except Exception as e:
            self.logger.error(f"Error in promo_impact_on_sales_and_customers: {e}")


    def effective_promo_deployment(self):
        """
        Analyze where promos could be deployed more effectively.
        """
        try:
            self.logger.info("Analyzing promo effectiveness across stores.")
        
            # Calculate average sales with and without promo for each store
            promo_store = self.train_df.groupby(['Store', 'Promo']).agg({
            'Sales': 'mean',
            'Customers': 'mean'
            }).reset_index()
        
            # Pivot the data for easier comparison
            promo_pivot = promo_store.pivot(index='Store', columns='Promo', values=['Sales', 'Customers'])
            promo_pivot.columns = ['Sales_No_Promo', 'Sales_Promo', 'Customers_No_Promo', 'Customers_Promo']
            promo_pivot.reset_index(inplace=True)
        
            # Calculate promo effectiveness
            promo_pivot['Sales_Change_%'] = ((promo_pivot['Sales_Promo'] - promo_pivot['Sales_No_Promo']) / 
                                         promo_pivot['Sales_No_Promo']) * 100
            promo_pivot['Customers_Change_%'] = ((promo_pivot['Customers_Promo'] - promo_pivot['Customers_No_Promo']) / 
                                             promo_pivot['Customers_No_Promo']) * 100
        
            # Top-performing stores (High Sales and Customer Change %)
            top_stores = promo_pivot.sort_values(by='Sales_Change_%', ascending=False).head(10)
        
            # Underperforming stores (Low or Negative Sales Change %)
            low_stores = promo_pivot.sort_values(by='Sales_Change_%').head(10)
        
            # Plotting top-performing stores
            plt.figure(figsize=(14, 6))
            sns.barplot(data=top_stores, x='Store', y='Sales_Change_%', palette='Blues_d')
            plt.title('Top 10 Stores with Highest Sales Increase During Promo')
            plt.xlabel('Store')
            plt.ylabel('Sales Change (%)')
            plt.xticks(rotation=45)
            plt.show()
        
            # Plotting underperforming stores
            plt.figure(figsize=(14, 6))
            sns.barplot(data=low_stores, x='Store', y='Sales_Change_%', palette='Reds_d')
            plt.title('Bottom 10 Stores with Lowest Sales Change During Promo')
            plt.xlabel('Store')
            plt.ylabel('Sales Change (%)')
            plt.xticks(rotation=45)
            plt.show()
        except Exception as e:
            self.logger.error(f"Error in effective_promo_deployment: {e}")

    def customer_behavior_during_opening_closing(self):
        """Visualize customer behavior during store opening and closing times.
        """
        try:
            self.logger.info("Analyzing customer behavior during store opening and closing times.")
        
            # Extract hour from the Date column (assuming store open/close times are consistent)
            self.train_df['Day'] = self.train_df['Date'].dt.day_name()
            self.train_df['Hour'] = self.train_df['Date'].dt.hour if 'Hour' in self.train_df.columns else 0
        
            # Filter only open stores
            open_stores = self.train_df[self.train_df['Open'] == 1]
        
            # Average Customers by Day of the Week
            daily_customers = open_stores.groupby('Day')['Customers'].mean().reset_index()
            daily_customers = daily_customers.sort_values(by='Customers', ascending=False)
        
            # Plot Average Customers by Day of the Week
            plt.figure(figsize=(12, 6))
            sns.barplot(data=daily_customers, x='Day', y='Customers', palette='viridis')
            plt.title('Average Number of Customers by Day of the Week')
            plt.xlabel('Day of the Week')
            plt.ylabel('Average Customers')
            plt.xticks(rotation=45)
            plt.show()
        
            # Customers during opening vs closing hours
            open_closing_customers = open_stores.groupby('Day')[['Sales', 'Customers']].mean().reset_index()
        
            # Plot Sales vs Customers by Day
            fig, ax1 = plt.subplots(figsize=(14, 6))
        
            sns.barplot(data=open_closing_customers, x='Day', y='Sales', color='skyblue', ax=ax1, label='Sales')
            ax2 = ax1.twinx()
            sns.lineplot(data=open_closing_customers, x='Day', y='Customers', color='orange', marker='o', ax=ax2, label='Customers')
        
            ax1.set_xlabel('Day of the Week')
            ax1.set_ylabel('Average Sales', color='skyblue')
            ax2.set_ylabel('Average Customers', color='orange')
            plt.title('Sales and Customer Trends by Day of the Week')
            fig.legend(loc='upper right')
            plt.show()

        except Exception as e:
            self.logger.error(f"Error in customer_behavior_during_opening_closing: {e}")

    def analyze_weekday_vs_weekend_sales(self):
        """
        Identify stores consistently open on all weekdays and compare their sales behavior 
        on weekdays vs weekends.
        """
        try:
            self.logger.info("Analyzing stores open on all weekdays and their sales on weekends.")
        
            # Step 1: Identify stores open on all weekdays (Mon-Fri)
            weekdays_data = self.train_df[self.train_df['DayOfWeek'].isin([1, 2, 3, 4, 5])]
        
            weekday_open_stores = weekdays_data.groupby('Store').agg(
                weekdays_open=('Open', lambda x: (x == 1).all())
            ).reset_index()
        
            # Filter stores consistently open all weekdays
            open_all_weekdays_stores = weekday_open_stores[weekday_open_stores['weekdays_open']]['Store'].tolist()
        
            print("Stores consistently open on weekdays:", open_all_weekdays_stores)
        
            if not open_all_weekdays_stores:
                self.logger.warning("No stores found that are consistently open on weekdays.")
                return
        
            # Step 2: Analyze sales performance (Weekdays vs Weekends)
            stores_data = self.train_df[self.train_df['Store'].isin(open_all_weekdays_stores)].copy()
        
            # Add DayType column
            stores_data['DayType'] = stores_data['DayOfWeek'].apply(
                lambda x: 'Weekday' if x in [1, 2, 3, 4, 5] else 'Weekend'
            )
        
            # Calculate average sales by DayType
            sales_by_daytype = stores_data.groupby('DayType')['Sales'].mean().reset_index()
        
            # Visualization
            plt.figure(figsize=(10, 6))
            sns.barplot(data=sales_by_daytype, x='DayType', y='Sales', palette='muted', hue='DayType', legend=False)
            plt.title('Average Sales on Weekdays vs Weekends for Stores Open All Weekdays')
            plt.ylabel('Average Sales')
            plt.xlabel('Day Type')
            plt.show()
        
            self.logger.info("Sales behavior for stores open on all weekdays visualized successfully.")
    
        except Exception as e:
            self.logger.error(f"Error in analyzing weekday vs weekend sales: {e}")
            raise


    def assortment_type_sales_impact(self):
        """
        Analyze how assortment type affects sales.
        """
        try:
            self.logger.info("Analyzing how assortment type affects sales.")
            # Calculate average sales per assortment type
            assortment_sales = (
                self.train_df[self.train_df['Open'] == 1]
                .groupby('Assortment')['Sales']
                .mean()
                .reset_index()
                .sort_values(by='Sales', ascending=False)
            )
        
            # Map assortment types for better readability
            assortment_mapping = {'a': 'Basic', 'b': 'Extra', 'c': 'Extended'}
            assortment_sales['Assortment'] = assortment_sales['Assortment'].map(assortment_mapping)


            plt.figure(figsize=(10, 6))
            sns.barplot(data=assortment_sales, x='Assortment', y='Sales', palette='muted', hue='Assortment')
            plt.title('Effect of Assortment Type on Sales')
            plt.xlabel('Assortment Type')
            plt.ylabel('Average Sales')
            plt.show()
            self.logger.info("Sales by assortment type plot generated successfully.")
        except Exception as e:
            self.logger.error(f"Error in assortment_type_sales_impact: {e}")

    def analyze_competition_distance_effect(self):
        """
        Analyze how the distance to the next competitor affects sales, and 
        compare the effect between city center and non-city center stores.
        """
        try:
            self.logger.info("Analyzing the effect of competitor distance on sales.")
        
            # Step 1: Data Preparation 
            self.train_df = self.train_df.dropna(subset=['CompetitionDistance']) 

            # Step 2: Correlation Analysis 
            overall_correlation = self.train_df['Sales'].corr(self.train_df['CompetitionDistance']) 
            print(f"Overall Correlation between Sales and CompetitionDistance: {overall_correlation:.2f}") 

            # Step 3: Visualization 
            plt.figure(figsize=(10, 6)) 
            sns.scatterplot(data=self.train_df, x='CompetitionDistance', y='Sales') 
            plt.title(f'Sales vs Competition Distance (Corr: {overall_correlation:.2f})') 
            plt.xlabel('Competition Distance') 
            plt.ylabel('Sales')
        
            self.logger.info("Competitor distance analysis completed successfully.")
    
        except Exception as e:
            self.logger.error(f"Error in analyzing competitor distance effect: {e}")
            raise



    def analyze_competition_impact(self, df):
        """
        Analyzes the impact of competition on store sales by comparing the average sales before and after
        the competitor opened.
        """
        self.logger.info("Starting analyze_competition_impact method.")
        
        # Filter stores with NA in CompetitionDistance
        stores_with_na = df[df['CompetitionDistance'].isna()]
        stores_with_na_ids = stores_with_na['Store'].unique()
        
        self.logger.info(f"Identified stores with no competitors: {stores_with_na_ids.tolist()}")
        
        # Check if these stores later receive valid CompetitionDistance values
        for store_id in stores_with_na_ids:
            store_data = df[df['Store'] == store_id]
            valid_distance = store_data['CompetitionDistance'].dropna()
            
            if valid_distance.empty:
                self.logger.info(f"Store {store_id} did not have a competitor open later.")
            else:
                self.logger.info(f"Store {store_id} had a competitor open later with CompetitionDistance values: {valid_distance.tolist()}")
        
        # Preprocess the CompetitionOpenSinceYear and remove unnecessary repeated steps
        df['CompetitionOpenSinceYear'] = pd.to_numeric(df['CompetitionOpenSinceYear'], errors='coerce').fillna(0).astype(int)
        
        # List to store average sales for each category
        avg_sales_before, avg_sales_after, avg_sales_no_competitor = [], [], []
        
        # Process each store with a valid competition distance
        for store_id in df[df['CompetitionDistance'].notna()]['Store'].unique():
            store_data = df[df['Store'] == store_id].copy()
            
            # Get the earliest competitor opening year
            competitor_open_year = store_data['CompetitionOpenSinceYear'].min()
            
            if competitor_open_year == 0:
                continue  # Skip if no competitor information
            
            # Split data into before and after competitor opening
            before_open = store_data[store_data['Date'] < pd.to_datetime(f"{competitor_open_year}-01-01")]
            after_open = store_data[store_data['Date'] >= pd.to_datetime(f"{competitor_open_year}-01-01")]
            
            # Store with no competitor (CompetitionDistance = NA)
            no_competitor = df[df['CompetitionDistance'].isna()]
            
            # Calculate average sales for each group
            avg_sales_before.append(before_open['Sales'].mean() if not before_open.empty else 0)
            avg_sales_after.append(after_open['Sales'].mean() if not after_open.empty else 0)
            avg_sales_no_competitor.append(no_competitor['Sales'].mean() if not no_competitor.empty else 0)
        
        # Create a bar chart for average sales comparison
        categories = ['Before Competitor Opened', 'After Competitor Opened', 'No Competitor']
        avg_sales = [
            sum(avg_sales_before) / len(avg_sales_before), 
            sum(avg_sales_after) / len(avg_sales_after),
            sum(avg_sales_no_competitor) / len(avg_sales_no_competitor)
        ]
        
        self.logger.info("Generated average sales data for each category.")
        
        # Plotting the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(categories, avg_sales, color=['red', 'green', 'blue'])
        plt.title('Average Sales Comparison: Competitor Opening Impact')
        plt.ylabel('Average Sales')
        plt.xlabel('Store Categories')
        plt.show()

        self.logger.info("Displayed bar chart for sales comparison.")
