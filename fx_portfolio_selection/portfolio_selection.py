# Personal project for academic interest: using mixed integer optimisation to
# model portfolio selection across a set of highly liquid currency pairs across
# the G4: EURUSD, GBPUSD, USDJPY.
# Spot rates data source: Google Finance

import pandas as pd
from typing import Tuple
import numpy as np
import pulp
from scipy.optimize import milp, LinearConstraint, Bounds

def compute_returns(file: str, abs_or_raw_returns: str)-> Tuple[pd.Series, list]:
    ccy_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    df = pd.read_csv(file)
    df = df.dropna()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # df = df[(df['Timestamp']<pd.Timestamp('2025-12-18 00:00:00.000')) & (df['Timestamp']>=pd.Timestamp('2025-12-17 00:00:00.000'))]
    # Compute the mean absolute returns or mean raw returns
    if abs_or_raw_returns == "abs":
        returns = df[['EURUSD', 'GBPUSD', 'USDJPY']].pct_change().abs().dropna().mean()
    elif abs_or_raw_returns == "raw":
        returns = df[['EURUSD', 'GBPUSD', 'USDJPY']].pct_change().dropna().mean()
    else:
        raise AssertionError("Wrong value for abs_or_raw_returns param!!!!!")
    return returns, ccy_pairs

def mixed_int_opt_scipy(returns: pd.Series, ccy_pairs: list, start_date: str, end_date: str, budget: int,
                  lot_size: int, max_pos: int):
    """
    Mixed integer optimisation problem for deriving optimal position sizes
    in EURUSD, GBPUSD and USDJPY in a FX portfolio
    :param returns:
    :param ccy_pairs:
    :param start_date:
    :param end_date:
    :param budget: Total capital allocated in USD
    :param lot_size: FX is often traded in increments, where standard increment = 100k,
    mini increment = 10k, micro increment = 1k (minimum tradable lot)
    :param max_pos: The number of pairs that can be traded at maximum
    :return:
    """
    M = budget / lot_size  # Big-M: Upper bound for lots per pair

    # 2. Variable Definition
    # Decision Variables: [lots_1, lots_2, lots_3, active_1, active_2, active_3]
    # Index 0-2: Integer lots (x); Index 3-5: Binary active status (y)

    # Objective: Minimize -Returns (SciPy minimizes, so negate for maximization)
    c = np.concatenate([-returns * lot_size, np.zeros(3)])

    # 3. Constraint Matrix (A @ x <= upper_bounds)
    # (A) Budget: sum(lots * lot_size) <= budget
    a_budget = np.concatenate([np.full(3, lot_size), np.zeros(3)])
    # (B) Max Positions: sum(active_y) <= max_pos
    a_max_pos = np.concatenate([np.zeros(3), np.ones(3)])
    # (C) Big-M: lots_i - (M * active_i) <= 0 (forces active_i=1 if lots_i > 0)
    a_big_m = np.hstack([np.eye(3), -np.eye(3) * M])

    # Combine all constraints into one matrix
    A = np.vstack([a_budget, a_max_pos, a_big_m])
    lower_bound_cons = np.full(A.shape[0], -np.inf)
    upper_bound_cons = np.array([budget, max_pos, 0, 0, 0])
    constraints = LinearConstraint(A, lower_bound_cons, upper_bound_cons)

    # 4. Variable Bounds and Integrality
    # Integrality: 1 = integer/binary, 0 = continuous
    integrality = np.ones(6)
    bounds = Bounds(lb=0, ub=np.concatenate([np.full(3, M), np.ones(3)]))

    # 5. Solve
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

    if res.success:
        print(f"Optimal Allocation for {start_date} to {end_date} at 30 minute increments using Scipy:")
        for i, t in enumerate(ccy_pairs):
            lots = int(res.x[i])
            if lots > 0:
                print(f"Pair: {t}, Lots: {lots}, Investment: ${lots * lot_size:,.2f}")
    else:
        print("Optimization failed:", res.message)

def mixed_int_opt_pulp(returns: pd.Series, ccy_pairs: list, start_date: str, end_date: str, budget: int,
                       lot_size: int, max_pos: int):
    # Initialize Problem
    prob = pulp.LpProblem("FX_Historical_Portfolio_Optimization", pulp.LpMaximize)

    # Decision Variables
    x = pulp.LpVariable.dicts("lots", ccy_pairs, lowBound=0, cat='Integer')
    y = pulp.LpVariable.dicts("active", ccy_pairs, cat='Binary')

    # Objective Function: Maximize Expected Return
    prob += pulp.lpSum([returns[p] * x[p] * lot_size for p in ccy_pairs])

    # Constraints
    # Total investment within budget
    prob += pulp.lpSum([x[p] * lot_size for p in ccy_pairs]) <= budget

    # Link x and y (Big-M constraint)
    for p in ccy_pairs:
        prob += x[p] <= (budget / lot_size) * y[p]

    # Limit the total number of active positions
    prob += pulp.lpSum([y[p] for p in ccy_pairs]) <= max_pos

    # --- 3. Solve and Display Results ---
    prob.solve()

    print(f"Status: {pulp.LpStatus[prob.status]}")
    print(f"\f\nOptimal Allocation from {start_date} to {end_date} at 30 minute increments using PuLP:")
    for p in ccy_pairs:
        if x[p].varValue > 0:
            print(f"Pair: {p}, Lots: {int(x[p].varValue)}, Investment: ${x[p].varValue * lot_size:,.2f}")

if __name__ == "__main__":
    start_date = '2025-12-16'
    end_date = '2025-12-17'
    budget = 100000
    lot_size = 10000
    max_pos = 2
    abs_or_raw_returns = "raw" # can take value "abs" too
    returns, ccy_pairs = compute_returns("spot_rates.csv", abs_or_raw_returns)
    mixed_int_opt_scipy(returns, ccy_pairs, start_date, end_date, budget, lot_size, max_pos)
    mixed_int_opt_pulp(returns, ccy_pairs, start_date, end_date, budget, lot_size, max_pos)