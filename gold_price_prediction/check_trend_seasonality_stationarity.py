# Monthly data from 1 Jan 2011 to 1 Dec 2025. Check for trend, seasonality and stationarity. If data non stationary with
# trend but no seasonality, ARIMA but if data non stationary with trend and seasonality,
# use SARIMA. Else, try Prophet, TBATS, SARIMAX, ML/DL models
# Data source for gold: https://www.macrotrends.net/1333/historical-gold-prices-100-year-chart
# Data source for silver: https://www.macrotrends.net/1470/historical-silver-prices-100-year-chart

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from utils import read_process_csv

def check_trend_seasonality(timeseries, period):
    """
    Decomposes a time series into trend, seasonality, and residuals, and plots them.

    Args:
        timeseries (pd.Series): The input time series data (must have a DatetimeIndex).
        period (int): The period of the seasonality (e.g., 12 for monthly data, 7 for daily data).
    """
    # Use multiplicative model if magnitude of seasonality changes with trend, otherwise additive
    # Review a plot to decide the model
    model = 'multiplicative'
    decomposition = seasonal_decompose(timeseries, model=model, period=period)

    fig, axes = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(timeseries)
    axes[0].set_title('Original Data')
    axes[1].plot(decomposition.trend)
    axes[1].set_title('Trend')
    axes[2].plot(decomposition.seasonal)
    axes[2].set_title('Seasonality')
    axes[3].plot(decomposition.resid)
    axes[3].set_title('Residuals')
    plt.tight_layout()
    plt.show()

    print(f"\n--- Decomposition Summary ({model} model) ---")
    # Trend is present if there is an increasing or decreasing slope in the trend plot.
    # Seasonality is present if there is a distinct, repeating pattern in the seasonality plot.


def check_stationarity(timeseries):
    """
    Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity.

    Args:
        timeseries (pd.Series or np.array): The input time series data.
    """
    print('--- Results of Augmented Dickey-Fuller Test ---')
    adf_result = adfuller(timeseries.dropna())  # drops NaN values which can occur after decomposition or differencing

    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print(f'\t{key}: {value}')

    if adf_result[1] < 0.05:
        print(
            "\nConclusion: The p-value is less than 0.05. We reject the null hypothesis (H0) and conclude the time series is stationary.")
    else:
        print(
            "\nConclusion: The p-value is greater than or equal to 0.05. We fail to reject the null hypothesis (H0) and conclude the time series is non-stationary (it has a unit root).")

if __name__ == "__main__":
    df = read_process_csv("gold_monthly_prices.csv", "MonthlyGoldPrice")
    # 2. Check for Trend and Seasonality
    # Period is 12 for monthly data
    check_trend_seasonality(df['MonthlyGoldPrice'], period=12)
    # the seasonal plot has a seasonal trend so we need to fit a seasonal model
    # the trend is exponential
    # 3. Check for Stationarity
    check_stationarity(df['MonthlyGoldPrice'])
    # --- Results of Augmented Dickey-Fuller Test ---
    # ADF Statistic: -6.830734053802615
    # p-value: 1.89665209175494e-09
    # Critical Values:
    # 	1%: -3.4674201432469816
    # 	5%: -2.877826051844538
    # 	10%: -2.575452082332012
    # Conclusion: The p-value is less than 0.05. We reject the null hypothesis (H0) and conclude the time series is stationary.
    # Should fit 2 models: Holt-Winters Exponential Smoothing (for data with trend and seasonality) with triple
    # exponential smoothing to capture trend, seasonality and random errors
    # Model 2: Apply log transformation to data to convert exponential trend into linear (additive) trend
    # and then fit a SARIMA model
    # Model extensions: VARIMAX or SARIMAX where we know gold prices are affected by prices of other commodities and
    # other exogenous factors. Fit both models in models.py