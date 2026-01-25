from data_preprocessing.read_data import read_datasets
from modelling.principal_component_analysis import fit_pca
from modelling.dynamic_factor_model import fit_dfm

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
    print("Method 1: Building FSI (financial stress index) using dynamic factor model")
    fsi_dfm = fit_dfm(df_merged)
    print("Method 2: Building FSI (financial stress index) using PCA")
    df_merged_fsi_pca = fit_pca(df_merged)