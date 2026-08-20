# Gold Price Prediction Platform

A time-series forecasting platform for gold price prediction across short-term and long-term forecasting horizons. The system leverages multiple forecasting models, including statistical, Bayesian, and deep learning approaches, to generate and compare price forecasts through an interactive visualization dashboard.

## Overview

This project focuses on developing forecasting models for gold prices to support better timing of buy and sell decisions.

The system provides forecasts across different time horizons:

- Short-term: 1 hour to 1 week
- Long-term: 1 month to 6 months

The platform evaluates model performance using historical data and Walk-Forward Backtesting to provide a more realistic assessment of forecasting performance.

## Project Objectives

The main objectives of this project are:

1. Develop gold price forecasting models for both short-term and long-term horizons.
2. Compare the performance of multiple time-series forecasting approaches.
3. Analyze historical market data and identify trends, seasonality, and cyclical patterns.
4. Evaluate forecasting accuracy using multiple performance metrics.
5. Provide an interactive dashboard for viewing historical prices, forecasts, and market trends.
6. Design a workflow that can be retrained with new incoming market data.

## System Workflow

```text
Market Data
    |
    v
Data Ingestion
    |
    v
Spark ETL
    |
    v
Data Quality Check
    |
    v
Data Quality Decision
    |
    +---- Fail ----> Stop Pipeline
    |
    +---- Pass
          |
          v
      Train Models
          |
          v
     Generate Forecasts
          |
          v
      Forecast Files
          |
          v
       Flask Web UI
````

The workflow consists of data ingestion, ETL processing, data quality validation, model training, forecasting, and visualization through a Flask-based web interface.

## Dataset

The project uses financial market data sourced from [Yahoo Finance](https://finance.yahoo.com/).

Two datasets are used to support different forecasting horizons:

* **Short-term:** 729 days of hourly data with 19,715 records
* **Long-term:** 5 years of daily data with 1,303 records

### Data Sources

The financial assets used in the project include:

* [Gold (GC=F)](https://finance.yahoo.com/quote/GC%3DF/)
* [Bitcoin (BTC-USD)](https://finance.yahoo.com/quote/BTC-USD/)
* [S&P 500 (^GSPC)](https://finance.yahoo.com/quote/%5EGSPC/)
* [Silver (SLV)](https://finance.yahoo.com/quote/SLV/)
* [EUR/USD (EURUSD=X)](https://finance.yahoo.com/quote/EURUSD=X/)
* [Dow Jones Industrial Average (^DJI)](https://finance.yahoo.com/quote/%5EDJI/)

The data is analyzed to identify trend, seasonality, cyclical behavior, and residual patterns before model development.

## Technologies

### Programming Language

* Python

### Data Processing

* Pandas
* Apache Spark

### Statistical Modeling

* Statsmodels
* ARIMAX
* Bayesian VAR

### Time-Series Forecasting

* Prophet

### Deep Learning

* TensorFlow
* Keras
* LSTM

### Backend

* Flask

### Data Visualization

* TradingView
* Vega-Lite
* Altair

### Frontend

* HTML
* CSS
* JavaScript

## Forecasting Models

The project evaluates four forecasting approaches:

### ARIMAX

ARIMAX is used to model the relationship between the target variable and external explanatory variables while accounting for time-series dependencies.

### Prophet

Prophet is used as an interpretable forecasting model for capturing trend and seasonal behavior in the time series.

### LSTM

Long Short-Term Memory is a recurrent neural network architecture designed to capture sequential dependencies and high-frequency patterns.

The custom LSTM architecture consists of:

* LSTM layer with 32 units
* Dropout layer with a rate of 0.2
* Dense output layer

The model is designed for multi-output prediction.

### Bayesian VAR

Bayesian Vector Autoregression is used to model relationships among multiple time-series variables and their historical dependencies.

## Data Preprocessing

The preprocessing stage includes analysis of:

* Trend
* Seasonality
* Cyclical patterns
* Residual behavior
* Stationarity

The Augmented Dickey-Fuller (ADF) test was applied to the gold closing price. The analysis indicated that the original gold price series was non-stationary, so first-order differencing was required before subsequent modeling.

The analysis also identified increasing volatility in the later part of the series, indicating that simple additive decomposition may not fully capture the market dynamics.

## Model Evaluation

The models are evaluated using Walk-Forward Backtesting rather than relying only on a conventional train-test split.

This approach simulates how the forecasting models would operate over time in a real-world deployment scenario.

### Evaluation Metrics

#### MAE

Mean Absolute Error measures the average absolute difference between predicted and actual values.

#### RMSE

Root Mean Squared Error measures prediction error while giving greater weight to larger errors.

#### sMAPE

Symmetric Mean Absolute Percentage Error evaluates prediction accuracy using percentage-based errors.

#### Directional Accuracy

Measures how frequently the model correctly predicts the direction of price movement.

#### Inference Time

Measures the average time required by a trained model to generate a forecast.

## Model Performance

The models are evaluated across multiple forecasting targets and horizons, including:

* Long-term close price
* Long-term high spread
* Long-term low spread
* Short-term close price
* Short-term high spread
* Short-term low spread

For long-term close-price forecasting, ARIMAX achieved the lowest reported MAE and RMSE among the evaluated models, while also achieving the highest directional accuracy in the comparison.

For short-term forecasting, ARIMAX and Prophet generally demonstrated lower prediction errors than LSTM and Bayesian VAR for the close-price target.

Model selection is based on a balanced evaluation of prediction error, directional accuracy, and operational efficiency rather than relying on a single metric.

## Model Adaptability

The system is designed to support retraining when new market data becomes available.

The training workflow can:

1. Download the latest market data.
2. Incorporate new data into the dataset.
3. Retrain the forecasting models.
4. Generate updated forecasts.

This allows the forecasting system to adapt to changing market conditions and reduce the risk of relying on outdated models.

## Web Dashboard

The project includes an interactive web dashboard built with Flask for visualizing historical gold prices, model forecasts, and market trends.

The dashboard supports:

* Real-time gold price visualization
* Historical price analysis
* Multi-model forecast comparison
* Multiple forecasting horizons
* Market trend indicators
* Interactive charts
* Chart export

The available forecasting models include:

* ARIMAX
* Prophet
* LSTM
* Bayesian VAR

## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── arimax/
│   ├── prophet/
│   ├── lstm/
│   └── bvar/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   └── forecasting/
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   └── js/
│
├── app.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

Make sure the following are installed:

* Python 3.x
* pip
* Git

### Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The Flask application will then be available through the local development server.

## Evaluation Approach

The overall evaluation workflow can be summarized as:

```text
Historical Data
      |
      v
Data Preprocessing
      |
      v
Feature Preparation
      |
      v
Model Training
      |
      v
Walk-Forward Backtesting
      |
      v
Performance Metrics
      |
      +------------------+
      |                  |
      v                  v
Prediction Error    Directional Accuracy
      |                  |
      +---------+--------+
                |
                v
         Model Selection
```

## Key Findings

Based on the experiments documented in the project:

* The original gold price series is non-stationary.
* Gold prices show a pronounced upward trend in the analyzed period.
* Volatility increases during parts of the later period.
* ARIMAX and Prophet generally provide competitive forecasting performance.
* LSTM can capture complex sequential patterns but requires greater computational time.
* Bayesian VAR provides a multivariate modeling approach but does not consistently achieve the lowest prediction error.
* Directional Accuracy is an important metric when evaluating models for potential trading applications.
* Walk-Forward Backtesting provides a more realistic evaluation than a single static train-test split.

## Limitations

This project is intended for academic and experimental purposes.

Forecasting financial markets involves substantial uncertainty, and model predictions should not be interpreted as guaranteed future prices or financial advice.

Model performance may change as market conditions, volatility, and relationships between financial assets evolve.

## Future Improvements

Potential improvements include:

* Incorporating additional macroeconomic variables
* Testing additional forecasting architectures
* Improving hyperparameter optimization
* Implementing more advanced volatility models
* Improving real-time data ingestion
* Expanding the dashboard with additional analytical features
* Deploying the system as a production service
* Adding automated model monitoring and retraining
