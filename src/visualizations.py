import sys
sys.path.append('../src')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from logger import Logger

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