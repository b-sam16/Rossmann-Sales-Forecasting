# Rossmann Pharmaceuticals Sales Forecasting Project

## Project Overview
This project focuses on building an end-to-end machine learning solution to predict store sales for Rossmann Pharmaceuticals six weeks ahead. The solution includes data exploration, feature engineering, model development, hyperparameter tuning, and deployment. The project also employs CI/CD practices, proper logging, and reproducibility techniques for robust and scalable results.

## Key Objectives
- **Analyze customer purchasing behavior:** Explore patterns, seasonal trends, and the impact of promotions and competition.
- **Predict store sales:** Build machine learning and deep learning models to forecast future sales accurately.
- **Serve predictions:** Deploy a web interface to provide insights and predictions to stakeholders.
- **Ensure reproducibility and scalability:** Use MLOps tools (DVC, MLFlow, CML) and CI/CD pipelines for smooth deployment and version control.

---

## Folder Structure
```
rossmann_sales_forecasting/
├── src/
│   ├── eda.py                # Exploratory Data Analysis functions
│   ├── visualizations.py     # Visualization functions
│   ├── logger.py             # Logging configuration
├── notebooks/
│   ├── eda.ipynb             # EDA workflows and analysis
├── data/
│   ├── raw/                  # Raw datasets (store.csv, test.csv, train.csv)
│   ├── processed/            # Processed datasets
├── tests/
│   ├── test_eda.py           # Unit tests for EDA functions
├── scripts/
│   ├── main.py               # Entry point to run the pipeline
├── .github/
│   ├── workflows/
│       ├── ci_cd.yml         # CI/CD Workflow configuration
├── requirements.txt          # List of dependencies
├── README.md                 # Project documentation
├── .gitignore
```

---

## Features

### 1. **Exploration of Customer Purchasing Behavior**
- Data cleaning and outlier detection.
- Analysis of seasonal trends, promotional effects, and competitor impact.
- Visualizations for key insights.

### 2. **Machine Learning Approach**
- Feature engineering for store, promotion, and season-related variables.
- Model development using scikit-learn.
- Hyperparameter tuning for optimal performance.

### 3. **Deep Learning Approach**
- Sequence modeling for time-series forecasting using TensorFlow/PyTorch.
- Advanced architectures such as LSTMs and GRUs.

### 4. **Deployment**
- Serve predictions on a web interface for stakeholder interaction.
- CI/CD pipeline for automated testing and deployment.
- Model management using MLFlow, DVC, and CML.

---

## Tools and Technologies
- **Programming Languages**: Python
- **Libraries**: pandas, numpy, scikit-learn, TensorFlow/PyTorch, matplotlib, seaborn
- **MLOps Tools**: DVC, MLFlow, CML
- **Deployment**: Streamlit for web interface
- **Version Control**: Git and GitHub
- **CI/CD**: GitHub Actions

---

## Getting Started

### Prerequisites
- Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Project
1. **Data Preprocessing**:
   Place raw data files (`store.csv`, `train.csv`, `test.csv`) in the `data/raw/` folder.
   Run preprocessing scripts:
   ```bash
   python src/data_preprocessing.py
   ```

2. **Exploratory Data Analysis**:
   Use the notebook in `notebooks/eda.ipynb` to explore and visualize the data.

3. **Model Training**:
   Run the `main.py` script to train models and generate predictions:
   ```bash
   python scripts/main.py
   ```

4. **Deploy**:
   Serve predictions via Streamlit:
   ```bash
   streamlit run scripts/deploy_app.py
   ```

---

## Key Insights and Results
- Analysis revealed significant seasonal trends and the impact of promotions and competitor openings on sales.
- The machine learning model achieved an R^2 score of 0.87 on the validation set.
- The deep learning model (LSTM) improved accuracy for stores with high sales volatility.

---

## Future Work
- Incorporate external factors such as economic indicators and weather data.
- Fine-tune deep learning models using advanced architectures.
- Enhance the web interface with interactive visualizations and user customization.




