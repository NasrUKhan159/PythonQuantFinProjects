import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def check_stationarity(df: pd.DataFrame):
    """
    Performs the Augmented Dickey-Fuller test on all columns of a pandas DataFrame
    and prints the results.
    """
    print("--- Augmented Dickey-Fuller Test Results ---")
    
    for column in df.columns:
        print(f"\nResults for column: **{column}**")
        
        # Perform ADF test
        # autolag='AIC' automatically determines the optimal lag length
        adf_result = adfuller(df[column].dropna(), autolag='AIC')
        
        # Extract and interpret results
        test_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]
        
        print(f'  ADF Statistic: {test_statistic:.4f}')
        print(f'  p-value: {p_value:.4f}')
        print(f'  Critical Values (1%): {critical_values["1%"]:.4f}')
        print(f'  Critical Values (5%): {critical_values["5%"]:.4f}')
        print(f'  Critical Values (10%): {critical_values["10%"]:.4f}')
        
        # Interpretation based on p-value and a significance level of 0.05
        if p_value <= 0.05:
            print(f'  **Conclusion:** The p-value is less than 0.05. We reject the null hypothesis (H0). The series is **stationary**.')
        else:
            print(f'  **Conclusion:** The p-value is greater than 0.05. We fail to reject the null hypothesis (H0). The series is **non-stationary**.')
    
def transform_nonstationary_cols(df: pd.DataFrame) -> pd.DataFrame:
    print(2)

def fit_dfm(df_merged: pd.DataFrame) -> pd.Series:
    """
    Invert indicators where lower values mean more stress
     higher interbank spread (LendingRateOverall, LendingRateStocks), higher NPL ratio, inflation rate, 
    and 'Price' (Price measures USDPKR spot) suggests greater financial stress.
    A lower value of NEER suggests weaker currency relative to trading partners so 
    potentially greater financial stress. However, an abnormally high NEER can also suggest 
    economic strengthenining because of non productive gains. Same phenomenon applies for SBP
    benchmark rate. Therefore, need to construct deviation measures out of NEER and SBP benchmark before
    including them.
    A strictly lower FX reserves value, KSE100 index, remittances, balance of trade, Price (i.e.
    USDPKR spot) indicates greater financial stress so need to invert these for PCA fit
    This approach is similar to PCA but accounts for time-varying nature and dependencies
    b/w financial indicators, especially correlations between different market segment changes during
    periods of stress
    Key consideration in DFM implementations: data stationarity 
    """
    neer_spread = df_merged['NEER'].mean()
    # calculate the squared deviation from the mean, forcing the indicator to be unidirectional i.e. 
    # higher sq deviation implies higher stress
    df_merged['NEER_SqDev'] = (df_merged['NEER'] - neer_spread)**2
    # doing the same for SBP Benchmark rate
    sbp_benchmark_spread = df_merged['BenchmarkRate'].mean()
    df_merged['BenchmarkRate_SqDev'] = (df_merged['BenchmarkRate'] - sbp_benchmark_spread)**2
    df_merged['FXReserves-USDMil'] = -df_merged['FXReserves-USDMil']
    df_merged['KSE100Idx'] = -df_merged['KSE100Idx']
    df_merged['TotalRemittances'] = -df_merged['TotalRemittances']
    df_merged['BalanceOfTrade_PKRMil'] = -df_merged['BalanceOfTrade_PKRMil']
    df_merged['Price'] = -df_merged['Price']
    df_merged_fit = df_merged.copy()
    df_merged_fit = df_merged_fit.drop(columns=['NEER', 'BenchmarkRate', 'Date'])
    # See here that apart from KSE100Idx and BenchmarkRate_SqDev, all features are non-stationary so need to implement this
    df_merged_fit = transform_nonstationary_cols(df_merged_fit)
    check_stationarity(df_merged_fit)
    # Define and Fit the Dynamic Factor Model using the transformed data
    model = sm.tsa.DynamicFactorMQ(
        endog=df_model_input,  # Use the data with the squared deviation feature
        factors=1,
        factor_orders=1,
        idiosyncratic_ar1=True
        )       
    results = model.fit(maxiter=100, disp=False)
    # Extract and Visualize the Estimated Financial Stress Index (the common factor)
    fsi_data = results.factors.filtered
    fsi = pd.Series(fsi_data, index=df_merged_fit['Date'], name='Pakistan Financial Stress Index (DFM)')
    return fsi