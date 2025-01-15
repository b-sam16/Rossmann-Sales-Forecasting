import sys
sys.path.append('./src')
import os
import pytest
import pandas as pd
import numpy as np
from eda import EDA


@pytest.fixture
def mock_data(tmp_path):
    """Fixture to create mock train and store data CSVs."""
    # Mock train data
    train_data = {
        'Store': [1, 2, 3],
        'DayOfWeek': [1, 2, 3],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'Sales': [1000, 2000, 3000],
        'Customers': [10, 20, 30],
        'Open': [1, 0, 1],
        'Promo': [1, 0, 1],
        'StateHoliday': ['0', 'a', '0'],
        'SchoolHoliday': [0, 1, 0]
    }
    train_df = pd.DataFrame(train_data)
    train_path = tmp_path / "train.csv"
    train_df.to_csv(train_path, index=False)

    # Mock store data
    store_data = {
        'Store': [1, 2, 3],
        'StoreType': ['a', 'b', 'c'],
        'Assortment': ['basic', 'extended', 'extra'],
        'CompetitionDistance': [100.0, 200.0, np.nan],
        'CompetitionOpenSinceMonth': [1, 2, 3],
        'CompetitionOpenSinceYear': [2010, 2011, np.nan],
        'Promo2': [0, 1, 0],
        'Promo2SinceWeek': [np.nan, 1, np.nan],
        'Promo2SinceYear': [np.nan, 2020, np.nan],
        'PromoInterval': [np.nan, 'Jan,Apr,Jul,Oct', np.nan]
    }
    store_df = pd.DataFrame(store_data)
    store_path = tmp_path / "store.csv"
    store_df.to_csv(store_path, index=False)

    return str(train_path), str(store_path)


def test_load_and_merge_data(mock_data):
    """Test loading and merging datasets."""
    train_path, store_path = mock_data
    eda = EDA(train_path, store_path)
    merged_df = eda.load_and_merge_data()

    # Assert merged DataFrame is not None
    assert merged_df is not None, "Merged DataFrame should not be None."

    # Assert column count is correct
    assert 'StoreType' in merged_df.columns, "Merged DataFrame should contain 'StoreType' column."

    # Assert rows match train data
    assert len(merged_df) == 3, "Merged DataFrame row count should match train dataset."


def test_explore_data(mock_data):
    """Test basic exploration of the dataset."""
    train_path, store_path = mock_data
    eda = EDA(train_path, store_path)
    eda.load_and_merge_data()

    head, info, stats = eda.explore_data()

    # Assert basic exploration details
    assert len(head) == 3, "Head should return 3 rows."
    assert stats is not None, "Statistics DataFrame should not be None."


def test_check_missing_values(mock_data):
    """Test detection of missing values."""
    train_path, store_path = mock_data
    eda = EDA(train_path, store_path)
    eda.load_and_merge_data()

    missing_df = eda.check_missing_values()

    # Assert missing value detection
    assert 'CompetitionDistance' in missing_df.index, "Missing values should include 'CompetitionDistance'."
    assert missing_df.loc['CompetitionDistance', 'Missing Values'] == 1, "CompetitionDistance should have 1 missing value."


def test_process_and_clean_data(mock_data):
    """Test processing and cleaning the dataset."""
    train_path, store_path = mock_data
    eda = EDA(train_path, store_path)
    eda.load_and_merge_data()

    cleaned_df = eda.process_and_clean_data()

    # Assert missing values are handled
    assert cleaned_df['CompetitionDistance'].isnull().sum() == 0, "All missing values should be handled."

    # Assert no negative values
    assert (cleaned_df.select_dtypes(include=['float64', 'int64']) < 0).sum().sum() == 0, "There should be no negative values."

    # Assert correct data type conversion
    assert pd.api.types.is_categorical_dtype(cleaned_df['StateHoliday']), "'StateHoliday' should be categorical."


def test_handle_outliers(mock_data):
    """Test outlier detection and handling."""
    train_path, store_path = mock_data
    eda = EDA(train_path, store_path)
    eda.load_and_merge_data()

    cleaned_df = eda.process_and_clean_data()
    outlier_handled_df = eda.handle_outliers()

    # Assert outliers are capped
    for column in ['Sales', 'Customers']:
        assert not ((outlier_handled_df[column] < cleaned_df[column].quantile(0.25) - 1.5 * (cleaned_df[column].quantile(0.75) - cleaned_df[column].quantile(0.25))) |
                    (outlier_handled_df[column] > cleaned_df[column].quantile(0.75) + 1.5 * (cleaned_df[column].quantile(0.75) - cleaned_df[column].quantile(0.25)))).any(), \
            f"Outliers in '{column}' should be capped."


def test_save_separated_data(mock_data, tmp_path):
    """Test saving separated train and store data."""

    train_path, store_path = mock_data
    eda = EDA(train_path, store_path)
    eda.load_and_merge_data()
    cleaned_df = eda.process_and_clean_data()

    # Ensure the processed directory exists under tmp_path
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Call the save_separated_data method with the correct path
    eda.save_separated_data(cleaned_df, save_path=processed_dir)

    # Check if the files were saved in the temporary directory
    assert (processed_dir / 'cleaned_store_data.csv').exists()
    assert (processed_dir / 'cleaned_train_data.csv').exists()
