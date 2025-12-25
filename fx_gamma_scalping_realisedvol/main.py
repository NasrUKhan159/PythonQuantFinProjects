"""
Conceptual implementation of gamma scalping realised volatility modelling for FX options
We apply in backtesting historical data here instead of live prices
Gamma scalping is an options strategy where a trader profits if the realized volatility of the underlying asset is
higher than the implied volatility used to price the option. In Python, this involves continuously calculating the
position's delta and rebalancing with trades in the underlying asset whenever a predetermined delta threshold is crossed.
The strategy itself involves:
Establishing a position: Start with a long options position (e.g., a long straddle or strangle) and a corresponding
short/long stock position to achieve a delta-neutral portfolio.
Calculating Greeks: Periodically calculate the delta and gamma of your options position. This often requires an option
pricing model like Black-Scholes and implied volatility calculation using libraries such as scipy.stats or dedicated f
inancial modeling libraries.
Rebalancing: As the underlying price moves, the position's delta will shift due to the positive gamma. When the net
delta crosses a specified threshold, you buy or sell the underlying to bring the delta back to neutral, capturing
profits in the process.
Data source in spot_rates.csv: Google Finance (FX spot rates from 16-17 Dec 2025)
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

def read_eurusd_data(file: str):
    df = pd.read_csv(file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')
    df_eurusd = df['EURUSD']
    return df_eurusd

# --- Black-Scholes & Greeks Calculation ---
def garman_kohlhagen_greeks(S, K, T, r_domestic, r_foreign, vol, option_type='call'):
    """
    Calculate Greeks (Delta and Gamma) using the Garman-Kohlhagen model.

    S: spot price (e.g., EUR/USD)
    K: strike price
    T: time to expiration (years)
    r_domestic: domestic risk-free rate (e.g., USD rate if trading EURUSD from the US)
    r_foreign: foreign risk-free rate (e.g., EUR rate)
    vol: volatility (annualized)
    option_type: 'call' or 'put'
    """
    if T <= 0:
        return 0, 0

    # The key adjustment in G-K: r is replaced by (r_domestic - r_foreign)
    d1 = (np.log(S / K) + (r_domestic - r_foreign + 0.5 * vol**2) * T) / (vol * T**0.5)
    d2 = d1 - vol * T**0.5

    if option_type == 'call':
        # Delta formula in G-K incorporates exp(-r_foreign * T)
        delta = np.exp(-r_foreign * T) * norm.cdf(d1)
    else:
        # Put delta formula
        delta = -np.exp(-r_foreign * T) * norm.cdf(-d1)

    # Gamma formula is slightly different from standard BS due to the exp(-r_foreign * T) term
    gamma = np.exp(-r_foreign * T) * norm.pdf(d1) / (S * vol * T**0.5)

    return delta, gamma

# --- Gamma Scalping Strategy (Conceptual Rebalancing Loop) ---

def backtest_gamma_scalping(historical_series, K, T_years, r_domestic, r_foreign, implied_vol, initial_option_qty):
    """
    Backtests the gamma scalping strategy using EURUSD historical prices
    """

    initial_price = historical_series.iloc[0]

    # 1. Initial Position: Establish delta-neutrality
    d, g = garman_kohlhagen_greeks(initial_price, K, T_years, r_domestic, r_foreign, implied_vol)
    current_stock_pos = -(initial_option_qty * d)

    trades = []

    print(f"Starting backtest with initial price {initial_price:.4f}, delta hedge of {current_stock_pos:.4f} units.")

    # 2. Simulation Loop: Iterate through historical prices (daily)
    for i in range(1, len(historical_series)):
        current_price = historical_series.iloc[i]

        # Calculate time remaining (assuming daily data for simplicity)
        # This part requires a more precise date calculation in a real backtest
        # using the index dates and an expiration date.
        T_remaining = max(0.001, T_years - (i / 252)) # Ensure T is positive

        # Calculate current Greeks
        delta, gamma = garman_kohlhagen_greeks(current_price, K, T_remaining, r_domestic, r_foreign, implied_vol)
        portfolio_delta = (initial_option_qty * delta) + current_stock_pos

        # 3. Rebalancing Logic (Scalping)
        delta_threshold = 0.001 # Rebalance if delta moves by 0.1 cents (this can be calibrated)
        if abs(portfolio_delta) > delta_threshold:

            trade_qty = -portfolio_delta # Amount of underlying to trade

            # Update position and log trade
            current_stock_pos += trade_qty
            trades.append({'time': historical_series.index[i], 'price': current_price, 'qty': trade_qty})

            print(f"Rebalance at {historical_series.index[i].date()}: Price {current_price:.4f}, Trade Qty {trade_qty:.4f}, New Pos {current_stock_pos:.4f}")

    print(f"\nBacktest finished. Total rebalancing trades executed: {len(trades)}")
    return trades

if __name__ == "__main__":

    # read historical EURUSD data
    df_eurusd = read_eurusd_data("spot_rates.csv")

    # Define strategy parameters
    FX_SYMBOL = 'EURUSD'
    STRIKE_PRICE = 1.1100 # Example strike
    TIME_TO_EXPIRATION_YEARS = 0.1 # Approx 36 days expiration
    RISK_FREE_RATE_DOMESTIC = 0.04
    RISK_FREE_RATE_FOREIGN = 0.02
    IMPLIED_VOLATILITY = 0.10 # Assuming an IV for option pricing
    INITIAL_OPTION_QTY = 1 # Long 1 call option
    NO_LOTS = 1000

    # Run the backtest
    trade_log = backtest_gamma_scalping(
        df_eurusd,
        STRIKE_PRICE,
        TIME_TO_EXPIRATION_YEARS,
        RISK_FREE_RATE_DOMESTIC,
        RISK_FREE_RATE_FOREIGN,
        IMPLIED_VOLATILITY,
        INITIAL_OPTION_QTY
    )
    print("Trades:")
    print(trade_log)