import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

if __name__ == "__main__":
    df = pd.read_csv("gold_monthly_prices.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index("Timestamp")
    # Drop the $ from the monthly gold price, we know it is dollars per ounce
    df['MonthlyGoldPrice'] = df['MonthlyGoldPrice'].str.replace('$', '')
    df['MonthlyGoldPrice'] = df['MonthlyGoldPrice'].str.replace(',', '')
    df['MonthlyGoldPrice'] = df['MonthlyGoldPrice'].astype("float64")
    df = df.sort_index()

    model = ExponentialSmoothing(
        df['MonthlyGoldPrice'],
        trend='mul',  # Multiplicative trend
        seasonal='add', # Multiplicative seasonality
        seasonal_periods=12 # 12 months in a year
    )

    # Fit the model to the data
    # The `fit()` method estimates the optimal smoothing parameters (alpha, beta, gamma)
    fitted_model = model.fit()

    # Forecast the next 12 periods (e.g., the next year)
    forecast_periods = 12
    forecast = fitted_model.forecast(steps=forecast_periods)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.plot(df['MonthlyGoldPrice'].index, df['MonthlyGoldPrice'], label='Observed Data')
    plt.plot(forecast.index, forecast, label='Forecast', color='red', linestyle='--')
    plt.title('Holt-Winters Triple Exponential Smoothing Forecast')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

    # Print the forecasted values
    print("Forecasted values from Triple Exponential Smoothing Holt-Winters:")
    print(forecast)

    # Model 2: SARIMA
    # Apply log transformation to the target column
    df['log_data'] = np.log(df['MonthlyGoldPrice'])

    model2 = SARIMAX(df['log_data'],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12))
    results = model2.fit()

    # Print model summary to check p-values and AIC
    print(results.summary())

    # Forecast for the next 12 steps
    forecast_log = results.forecast(steps=12)

    # Reverse the log transformation using np.exp()
    forecast_original_scale = np.exp(forecast_log)
    print("SARIMAX 12-month forecast:")
    print(forecast_original_scale)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.plot(df['MonthlyGoldPrice'].index, df['MonthlyGoldPrice'], label='Observed Data')
    plt.plot(forecast_original_scale.index, forecast_original_scale, label='Forecast', color='red', linestyle='--')
    plt.title('SARIMA(1,1,1)x(1,1,1,12) Forecast')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

    # AR, MA non-seasonal coefficients and AR seasonal coefficient are not statistically significant
    # So fitting SARIMA(0,1,0)x(0,1,1,12) model:

    model3 = SARIMAX(df['log_data'],
                     order=(0, 1, 0),
                     seasonal_order=(0, 1, 1, 12))
    results2 = model3.fit()

    # Print model summary to check p-values and AIC
    print(results2.summary())

    # Forecast for the next 12 steps
    forecast_log2 = results2.forecast(steps=12)

    # Reverse the log transformation using np.exp()
    forecast_original_scale2 = np.exp(forecast_log2)
    print("SARIMA(0,1,0)x(0,1,1,12) 12-month forecast:")
    print(forecast_original_scale2)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.plot(df['MonthlyGoldPrice'].index, df['MonthlyGoldPrice'], label='Observed Data')
    plt.plot(forecast_original_scale2.index, forecast_original_scale2, label='Forecast', color='red', linestyle='--')
    plt.title('SARIMA(0,1,0)x(0,1,1,12) Forecast')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.show()