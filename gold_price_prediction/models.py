import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from utils import read_process_csv

def model_gold(filename: str, colname: str):
    df = read_process_csv(filename, colname)

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

def model_gold_silver(filename_gold: str, colname_gold: str, filename_silver: str, colname_silver: str):
    """
    Acc to gold_silver_comovement.py, model gold and silver using VECM (vector error correction model)
    :param filename_gold:
    :param colname_gold:
    :param filename_silver:
    :param colname_silver:
    :return:
    """
    print("Modelling gold and silver now...")
    df_gold = read_process_csv(filename_gold, colname_gold)
    df_silver = read_process_csv(filename_silver, colname_silver)
    x = df_gold['MonthlyGoldPrice'].to_numpy()
    y = df_silver['MonthlySilverPrice'].to_numpy()
    df = pd.DataFrame({'Series_Gold': x, 'Series_Silver': y})
    # Assuming df contains two non-stationary (I(1)) cointegrated series
    # Find optimal lag order for the underlying VAR
    lag_selection = select_order(data=df, maxlags=10, deterministic="ci")
    print(lag_selection.summary())

    # Use the AIC-recommended lag order
    opt_lag = lag_selection.aic

    # Test for cointegration rank (r)
    rank_test = select_coint_rank(df, det_order=0, k_ar_diff=opt_lag, method="trace")
    print(rank_test.summary())

    # r_0 = 1 suggests one cointegrating relationship
    coint_rank = rank_test.rank

    # Fit the VECM
    # k_ar_diff is the number of lags in the VECM (level lags - 1)
    model = VECM(df, k_ar_diff=opt_lag, coint_rank=coint_rank, deterministic="ci")
    vecm_res = model.fit()

    print(vecm_res.summary())
    # Coefficients are statistically significant (p values small)

    # Forecast the next 10 steps
    forecast = vecm_res.predict(steps=10)
    print("VECM forecast for next 10 months for gold and silver:")
    print(forecast)

if __name__ == "__main__":
    model_gold("gold_monthly_prices.csv", "MonthlyGoldPrice")
    model_gold_silver("gold_monthly_prices.csv", "MonthlyGoldPrice",
                      "silver_monthly_px.csv", "MonthlySilverPrice")