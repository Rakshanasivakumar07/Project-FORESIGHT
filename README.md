# Project Foresight — Retail Sales Forecasting

Project Foresight is a sales forecasting system built on the **Online Retail II** dataset (UK-based online retailer, 2009–2011). It predicts future sales trends using time-series modeling and serves results through an interactive **Streamlit dashboard**.

## Overview

- **Dataset**: Online Retail II (transactional e-commerce data, Dec 2009 – Dec 2011, 739 days)
- **Goal**: Forecast sales for the next 30 days based on historical trend and weekly seasonality
- **Approach**: Compare a seasonal-naive baseline against a Linear Regression model (trend + weekly seasonality features)
- **Delivery**: Results are served via a Streamlit web app for interactive exploration

## Model Performance

Backtested on the last 30 days of the dataset.

| Model | WAPE |
|---|---|
| Baseline (Seasonal Naive) | 36.65% |
| Linear Regression (trend + weekly seasonality) | 31.36% |

**Conclusion**: The model beats the seasonal-naive baseline by 5.29 percentage points, showing that trend and weekly-seasonality features add real forecasting value.

**Forecast horizon**: Next 30 days (2011-12-10 to 2012-01-08)

## Project Structure

```
project-foresight/
├── forecast_model.py        # Model training script
├── forecast_model.pkl       # Trained model (not tracked in git)
├── streamlit_app.py         # Streamlit dashboard app
├── prediction_results.csv   # Generated forecast output
├── model_performance.md     # Detailed performance report
├── requirements.txt         # Python dependencies
└── online_retail_II.xlsx    # Raw dataset (not tracked in git)
```

## Setup & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rakshanasivakumar07/project-foresight.git
   cd project-foresight
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

3. **Add the dataset**
   Place `online_retail_II.xlsx` in the project root (not included in this repo due to size).

4. **Train the model**
   ```bash
   python forecast_model.py
   ```

5. **Run the Streamlit dashboard**
   ```bash
   streamlit run streamlit_app.py
   ```

## Tech Stack

- Python
- Pandas / NumPy — data processing
- Scikit-learn — Linear Regression model
- Streamlit — interactive dashboard

## Author

Rakshana Sivakumar
