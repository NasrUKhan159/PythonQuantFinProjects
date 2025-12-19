import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from utils import read_process_csv

def rolling_window_corr_analysis(window_size: int, gold_series: pd.Series, silver_series: pd.Series):
    """
    Compute `window_size`-day window correlation
    :param window_size:
    :param gold_series:
    :param silver_series:
    :return:
    """
    rolling_corr = gold_series.rolling(window=window_size).corr(silver_series)
    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_corr, label=f'{window_size}-Day Rolling Correlation', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8) # Baseline at zero
    plt.title('Rolling Window Correlation Between Gold and Silver')
    plt.ylabel('Correlation Coefficient')
    plt.legend()
    plt.show()

def test_cointegration(gold_series: pd.Series, silver_series: pd.Series):
    """
    Test for co-integration using the Engle-Granger test which is recommended for two time series
    and the Johansen test which is good if we have a large enough dataset (greater than 100 timepoints)
    and don't want bias of pre-selecting dependent variable. Engle-Granger is good if we know one
    variable is dependent variable - there is one interpretation that as a safer asset, gold sets the
    benchmark so gold may affect the price of silver but good to use both tests.
    :param gold_series:
    :param silver_series:
    :return:
    """
    x = gold_series.to_numpy()
    y = silver_series.to_numpy()
    # Perform Engle-Granger test
    score, pvalue, _ = coint(x, y)

    print(f'Cointegration Test Statistic: {score}')
    print(f'P-value: {pvalue}')

    if pvalue < 0.05:
        print("According to Engle-Granger, the series are likely cointegrated.")
    else:
        print("According to Engle-Granger, no evidence of cointegration.")

    # Combine gold and silver data into a DataFrame
    df = pd.DataFrame({'x': x, 'y': y})

    # Perform Johansen test (det_order=0 for constant, k_ar_diff=1 for lag)
    result = coint_johansen(df, det_order=0, k_ar_diff=1)

    # Compare Trace Statistic against 95% Critical Value
    trace_stat = result.lr1[0]
    critical_val_95 = result.cvt[0, 1] # Index [0, 1] is for rank 0 at 95% confidence

    print(f'Johansen Test - Trace Statistic: {trace_stat}')
    print(f'Johansen Test - 95% Critical Value: {critical_val_95}')

    if trace_stat > critical_val_95:
        print("Reject H0: The series are cointegrated acc to Johansen Test")

if __name__ == "__main__":
    df_silver = read_process_csv("silver_monthly_px.csv", "MonthlySilverPrice")
    df_gold = read_process_csv("gold_monthly_prices.csv", "MonthlyGoldPrice")
    rolling_window_corr_analysis(20, df_gold['MonthlyGoldPrice'], df_silver['MonthlySilverPrice'])
    test_cointegration(df_gold['MonthlyGoldPrice'], df_silver['MonthlySilverPrice'])
    # If there is reciprocal feedback (both variables affect each other), Engle-Granger may fail to find a relationship
    # that the system-based Johansen test correctly identifies with large enough sample size (greater than 100) which is
    # the case here. Therefore, we should follow the Johansen test and model gold and silver in models.py
    # using the VECM model which will treat both variables as endogenous