"""Backtest og strategi-simulation - vektoriseret"""
import numpy as np
import pandas as pd
from analysis import (
    get_indicators, technical_score_vectorized, recommendation_label
)


def run_backtest(hist_full, holding_days=90, sample_freq=5):
    df = get_indicators(hist_full)
    if len(df) < 250 + holding_days:
        return None

    scores = technical_score_vectorized(df).values
    close = df["Close"].values

    start_idx = 250
    end_idx = len(df) - holding_days
    indices = np.arange(start_idx, end_idx, sample_freq)

    entry_prices = close[indices]
    exit_prices = close[indices + holding_days]
    score_values = scores[indices]

    valid = ~(np.isnan(score_values) | np.isnan(entry_prices)
              | np.isnan(exit_prices) | (entry_prices <= 0))
    indices = indices[valid]
    entry_prices = entry_prices[valid]
    exit_prices = exit_prices[valid]
    score_values = score_values[valid]

    if len(indices) == 0:
        return None

    returns = (exit_prices / entry_prices - 1) * 100
    recs = np.where(score_values >= 75, "STÆRKT KØB",
            np.where(score_values >= 60, "KØB",
             np.where(score_values >= 45, "HOLD",
              np.where(score_values >= 30, "SÆLG", "STÆRKT SÆLG"))))

    df_results = pd.DataFrame({
        "date": df.index[indices],
        "score": score_values,
        "recommendation": recs,
        "entry_price": entry_prices,
        "exit_price": exit_prices,
        "return_pct": returns,
    })

    stats = {}
    grouped = df_results.groupby("recommendation")["return_pct"]
    for rec in ["STÆRKT KØB", "KØB", "HOLD", "SÆLG", "STÆRKT SÆLG"]:
        if rec in grouped.groups:
            sub = grouped.get_group(rec)
            stats[rec] = {
                "count": len(sub),
                "avg_return": sub.mean(),
                "median_return": sub.median(),
                "win_rate": (sub > 0).sum() / len(sub) * 100,
                "best": sub.max(),
                "worst": sub.min(),
            }
        else:
            stats[rec] = None

    bh_return = (close[end_idx] / close[start_idx] - 1) * 100

    return {
        "results": df_results, "stats": stats,
        "buy_hold_return": bh_return, "holding_days": holding_days,
        "n_trades": len(df_results),
        "start_date": df.index[start_idx], "end_date": df.index[end_idx],
    }


def simulate_strategy(hist_full, buy_threshold=60, sell_threshold=30, sample_freq=5):
    df = get_indicators(hist_full)
    if len(df) < 250:
        return None

    scores = technical_score_vectorized(df).values
    close = df["Close"].values
    start_idx = 250
    initial_cash = 10000.0
    cash = initial_cash
    shares = 0
    in_position = False

    equity_curve = []
    trades = []

    for i in range(start_idx, len(df), sample_freq):
        score = scores[i]
        if np.isnan(score):
            continue
        price = close[i]
        if np.isnan(price) or price <= 0:
            continue

        if score >= buy_threshold and not in_position:
            shares = cash / price
            cash = 0
            in_position = True
            trades.append({"date": df.index[i], "action": "KØB",
                          "price": price, "score": score})
        elif score <= sell_threshold and in_position:
            cash = shares * price
            shares = 0
            in_position = False
            trades.append({"date": df.index[i], "action": "SÆLG",
                          "price": price, "score": score})

        equity = cash + shares * price
        equity_curve.append({"date": df.index[i], "strategy": equity, "price": price})

    if in_position:
        cash = shares * close[-1]
        shares = 0

    final_value = cash
    strategy_return = (final_value / initial_cash - 1) * 100

    bh_start_price = close[start_idx]
    bh_end_price = close[-1]
    bh_final = (initial_cash / bh_start_price) * bh_end_price
    bh_return = (bh_final / initial_cash - 1) * 100

    df_eq = pd.DataFrame(equity_curve)
    if len(df_eq) > 0:
        df_eq["buy_hold"] = initial_cash * (df_eq["price"] / bh_start_price)

    return {
        "equity_curve": df_eq, "trades": pd.DataFrame(trades),
        "final_value": final_value, "strategy_return": strategy_return,
        "buy_hold_return": bh_return,
        "outperformance": strategy_return - bh_return,
        "n_trades": len(trades), "initial_cash": initial_cash,
    }
