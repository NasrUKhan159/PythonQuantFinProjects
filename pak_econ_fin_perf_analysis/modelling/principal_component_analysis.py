from data_preprocessing.read_data import read_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def fit_pca(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Invert indicators where lower values mean more stress
     higher interbank spread (LendingRateOverall, LendingRateStocks), higher NPL ratio, inflation rate, 
    and 'Price' (Price measures USDPKR spot) suggests greater financial stress.
    A lower value of NEER suggests weaker currency relative to trading partners so 
    potentially greater financial stress. However, an abnormally high NEER can also suggest 
    economic strengthenining because of non productive gains. Same phenomenon applies for SBP
    benchmark rate. Therefore, remove SBP benchmark rate and NEER for PCA fitting.
    A strictly lower FX reserves value, KSE100 index, remittances, balance of trade, Price (i.e.
    USDPKR spot) indicates greater financial stress so need to invert these for PCA fit
    Downside of PCA method is it assumes monotonicity of stress indication i.e. a lower value of
    the feature variable indicates higher stress and vice versa. However, realistically a very low
    benchmark rate and a very high benchmark rate both can indicate increased stress.
    """
    df_merged = df_merged.drop(columns=['NEER', 'BenchmarkRate', 'Date'])
    df_merged['FXReserves-USDMil'] = -df_merged['FXReserves-USDMil']
    df_merged['KSE100Idx'] = -df_merged['KSE100Idx']
    df_merged['TotalRemittances'] = -df_merged['TotalRemittances']
    df_merged['BalanceOfTrade_PKRMil'] = -df_merged['BalanceOfTrade_PKRMil']
    df_merged['Price'] = -df_merged['Price']
    # standardize the data (Z-score)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_merged)
    # apply PCA to extract the first component
    pca = PCA(n_components=1)
    fsi_scores = pca.fit_transform(df_scaled)
    # one measure of the FSI (financial stress index) is the first principal component
    df_merged['FSI'] = fsi_scores
    # standardize the FSI for easier interpretation (mean 0, variance 1)
    # this way, positive FSI indicates above-avg stress, negative FSI indicates below-avg stress
    df_merged['FSI_Normalized'] = StandardScaler().fit_transform(df_merged[['FSI']])
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
    df_merged = fit_pca(df_merged)