import pandas as pd
from functools import reduce
from typing import Tuple

def convert_strdate_todatetime_col(df: pd.DataFrame) -> pd.DataFrame:
    df['Observation Date'] = pd.to_datetime(df['Observation Date'], format="%d-%b-%Y")
    df = df.rename(columns={'Observation Date': 'Date'})
    return df

def convert_to_start_of_month(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    return df

def preprocess_interest_rate_spreads(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_lending_overall = df[df['Series Display Name'] == '. Weighted Average Lending Deposit Rates  Lending Marginal (Overall)'][['Date', 'Observation Value']]
    df_lending_stocks = df[df['Series Display Name'] == '. Weighted Average Lending Deposit Rates  Lending Stocks (Overall)'][['Date', 'Observation Value']]
    df_lending_overall = df_lending_overall.rename(columns={'Observation Value': 'LendingRateOverall'})
    df_lending_stocks = df_lending_stocks.rename(columns={'Observation Value': 'LendingRateStocks'})
    return df_lending_overall, df_lending_stocks


def read_datasets(fx_reserves_file: str, inflation_data_file: str, 
                  interest_rate_spreads_file: str, kse100_file: str,
                  nom_effective_xr_file: str, npl_file: str,
                  remittances_file: str, sbp_benchmark_rates_file: str,
                  trade_balances_file: str, usdpkr_spot_file: str) -> pd.DataFrame:
    fx_reserves = pd.read_excel("data/" + fx_reserves_file)
    inflation_data = pd.read_excel("data/" + inflation_data_file)
    interest_rate_spreads = pd.read_csv("data/" + interest_rate_spreads_file)
    kse100_idx = pd.read_excel("data/" + kse100_file)
    nom_effective_xr = pd.read_csv("data/" + nom_effective_xr_file)
    npl_data = pd.read_excel("data/" + npl_file)
    remittances = pd.read_csv("data/" + remittances_file)
    sbp_benchmark_rates = pd.read_excel("data/" + sbp_benchmark_rates_file)
    trade_balances = pd.read_excel("data/" + trade_balances_file)
    usdpkr_spot_rates = pd.read_excel("data/" + usdpkr_spot_file)

    # Convert 'Observation Date' columns in interest rate spreads, nom effective xr 
    # and remittances from string to pd.Timestamp
    interest_rate_spreads = convert_strdate_todatetime_col(interest_rate_spreads)
    nom_effective_xr = convert_strdate_todatetime_col(nom_effective_xr)
    remittances = convert_strdate_todatetime_col(remittances)

    # rename KSE1O0 index data from 'Price' to 'KSE100Idx' to avoid mixing with USDPKR spot rates
    kse100_idx = kse100_idx.rename(columns={'Price': 'KSE100Idx'})

    # interpolate using cubic spline for the quarterly NPL data to estimate monthly NPL data
    npl_data = npl_data.set_index("EndDate")
    monthly_npl = npl_data.resample('MS').interpolate(
        method='spline', order=3).reset_index().rename(
        columns={"EndDate": "Date"})
    
    # filter interest_rate_spread data
    interest_rate_spreads_lending_overall, interest_rate_spreads_lending_stocks = preprocess_interest_rate_spreads(interest_rate_spreads)
    
    # filter NEER and remittances data
    neer_filtered = nom_effective_xr[nom_effective_xr['Series Display Name']=='a. NEER (Base 2010=100)'][['Date', 'Observation Value']]
    neer_filtered = neer_filtered.rename(columns={'Observation Value': 'NEER'})
    remittances_filtered = remittances[remittances['Series Display Name']=='. Total (I+II)'][['Date', 'Observation Value']]
    remittances_filtered = remittances_filtered.rename(columns={'Observation Value': 'TotalRemittances'})

    # Convert end of month timestamps on lending rates, NEER and remittances to start of month
    interest_rate_spreads_lending_overall = convert_to_start_of_month(interest_rate_spreads_lending_overall)
    interest_rate_spreads_lending_stocks = convert_to_start_of_month(interest_rate_spreads_lending_stocks)
    neer_filtered = convert_to_start_of_month(neer_filtered)
    remittances_filtered = convert_to_start_of_month(remittances_filtered)

    # Inner join all datasets together by time
    # List of yosur dataframes
    dfs = [fx_reserves, interest_rate_spreads_lending_overall, inflation_data, interest_rate_spreads_lending_stocks, kse100_idx,
           neer_filtered, monthly_npl, remittances_filtered, sbp_benchmark_rates, trade_balances, usdpkr_spot_rates]

    # Merge them all on the 'Date' column
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='Date', how='inner'), dfs)
    return df_merged


if __name__ == "__main__":
    fx_reserves_file = "fx_reserves.xlsx"
    inflation_data_file = "inflation_data.xlsx"
    interest_rate_spreads_file = "InterestRateSpread_WeightedAvgLendingDepositRate.csv"
    kse100_file = "kse100_index.xlsx"
    nom_effective_xr_file = "NEER_Dataset.csv"
    npl_file = "NPL_Data.xlsx"
    remittances_file = "RemittancesData.csv"
    sbp_benchmark_rates_file = "SBP_BenchmarkInterestRate.xlsx"
    trade_balances_file = "trade_balance_data.xlsx"
    usdpkr_spot_file = "USDPKR_SpotRates.xlsx"
    df_merged = read_datasets(fx_reserves_file, inflation_data_file, interest_rate_spreads_file, kse100_file,
                  nom_effective_xr_file, npl_file, remittances_file, sbp_benchmark_rates_file,
                  trade_balances_file, usdpkr_spot_file)
