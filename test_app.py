"""
Backend tests for Aktie Dashboard
Kør med: pytest test_app.py -v
Eller:   python -m pytest test_app.py -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Tilføj projektmappen til path så vi kan importere app.py funktioner
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock streamlit før import (så vi kan teste uden Streamlit context)
class MockSecrets:
    def get(self, key, default=None):
        return os.environ.get(key, default)

class MockStreamlit:
    secrets = MockSecrets()
    session_state = {}
    def cache_data(self, **kwargs):
        def decorator(func):
            return func
        return decorator
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

sys.modules['streamlit'] = MockStreamlit()

# Nu kan vi importere fra app.py
from app import (
    safe, fundamental_score, technical_score, add_indicators,
    dcf_valuation, risk_metrics, monte_carlo, recommendation,
    fetch_yahoo, fetch_finnhub, fetch_stooq, fetch_data
)


# ============ HJÆLPEFUNKTIONER ============

def make_fake_hist(days=300, start_price=100, volatility=0.02):
    """Lav syntetiske prisdata til test"""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
    returns = np.random.normal(0.0005, volatility, days)
    prices = start_price * np.cumprod(1 + returns)
    df = pd.DataFrame({
        "Open": prices * (1 + np.random.normal(0, 0.005, days)),
        "High": prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
        "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, days)
    }, index=dates)
    return df


# ============ TESTS: HJÆLPEFUNKTIONER ============

class TestSafe:
    def test_safe_with_dict(self):
        assert safe({"a": 1}, "a") == 1

    def test_safe_with_missing_key(self):
        assert safe({"a": 1}, "b") is None

    def test_safe_with_default(self):
        assert safe({"a": 1}, "b", 99) == 99

    def test_safe_with_none_dict(self):
        assert safe(None, "a") is None

    def test_safe_with_nan(self):
        assert safe({"a": float('nan')}, "a", 0) == 0


# ============ TESTS: SCORING ============

class TestFundamentalScore:
    def test_empty_info(self):
        score, det = fundamental_score({})
        assert score == 50
        assert det == []

    def test_excellent_company(self):
        info = {
            "trailingPE": 12, "pegRatio": 0.8, "priceToBook": 2,
            "returnOnEquity": 0.25, "debtToEquity": 20, "profitMargins": 0.25,
            "revenueGrowth": 0.25, "freeCashflow": 5e9, "currentRatio": 2.0
        }
        score, det = fundamental_score(info)
        assert score >= 75, f"Excellent company should score 75+, got {score}"

    def test_terrible_company(self):
        info = {
            "trailingPE": 80, "priceToBook": 15,
            "returnOnEquity": -0.10, "debtToEquity": 300, "profitMargins": -0.05,
            "revenueGrowth": -0.10, "freeCashflow": -2e9
        }
        score, det = fundamental_score(info)
        assert score <= 30, f"Terrible company should score 30 or less, got {score}"

    def test_score_bounds(self):
        """Score skal altid være mellem 0 og 100"""
        for _ in range(10):
            info = {
                "trailingPE": np.random.uniform(5, 100),
                "returnOnEquity": np.random.uniform(-0.3, 0.5),
                "profitMargins": np.random.uniform(-0.2, 0.4)
            }
            score, _ = fundamental_score(info)
            assert 0 <= score <= 100


class TestTechnicalScore:
    def test_indicators_added(self):
        df = make_fake_hist(300)
        df = add_indicators(df)
        for col in ["SMA50", "SMA200", "RSI", "MACD", "BB_high", "BB_low", "ADX", "ATR"]:
            assert col in df.columns, f"Missing indicator: {col}"

    def test_score_bounds(self):
        df = make_fake_hist(300)
        df = add_indicators(df)
        score, _ = technical_score(df)
        assert 0 <= score <= 100

    def test_uptrend_scores_higher(self):
        """Stigende trend bør score højere end faldende trend"""
        np.random.seed(1)
        days = 300
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')

        # Klart opadgående trend
        up_prices = np.linspace(50, 150, days) + np.random.normal(0, 1, days)
        df_up = pd.DataFrame({"Open": up_prices, "High": up_prices*1.01,
                             "Low": up_prices*0.99, "Close": up_prices,
                             "Volume": [1e6]*days}, index=dates)

        # Klart nedadgående trend
        down_prices = np.linspace(150, 50, days) + np.random.normal(0, 1, days)
        df_down = pd.DataFrame({"Open": down_prices, "High": down_prices*1.01,
                               "Low": down_prices*0.99, "Close": down_prices,
                               "Volume": [1e6]*days}, index=dates)

        score_up, _ = technical_score(add_indicators(df_up))
        score_down, _ = technical_score(add_indicators(df_down))

        assert score_up > score_down, f"Uptrend ({score_up}) should score higher than downtrend ({score_down})"


# ============ TESTS: DCF ============

class TestDCF:
    def test_dcf_basic(self):
        info = {"freeCashflow": 1e9, "sharesOutstanding": 1e8,
                "totalDebt": 5e8, "totalCash": 2e8}
        fair = dcf_valuation(info, g=0.10, dr=0.10, tg=0.025)
        assert fair is not None
        assert fair > 0

    def test_dcf_negative_fcf_returns_none(self):
        info = {"freeCashflow": -1e9, "sharesOutstanding": 1e8}
        assert dcf_valuation(info, 0.10, 0.10, 0.025) is None

    def test_dcf_missing_data_returns_none(self):
        assert dcf_valuation({}, 0.10, 0.10, 0.025) is None

    def test_dcf_higher_growth_higher_value(self):
        info = {"freeCashflow": 1e9, "sharesOutstanding": 1e8,
                "totalDebt": 0, "totalCash": 0}
        low = dcf_valuation(info, g=0.05, dr=0.10, tg=0.025)
        high = dcf_valuation(info, g=0.20, dr=0.10, tg=0.025)
        assert high > low, "Higher growth should give higher fair value"


# ============ TESTS: RISIKO ============

class TestRiskMetrics:
    def test_risk_keys_present(self):
        df = make_fake_hist(500)
        risk = risk_metrics(df)
        for key in ["ann_r", "ann_v", "sharpe", "sortino", "max_dd", "var95"]:
            assert key in risk

    def test_volatility_positive(self):
        df = make_fake_hist(500)
        risk = risk_metrics(df)
        assert risk["ann_v"] > 0

    def test_max_drawdown_negative(self):
        df = make_fake_hist(500)
        risk = risk_metrics(df)
        assert risk["max_dd"] <= 0, "Max drawdown should be 0 or negative"

    def test_low_volatility_data(self):
        """Test med næsten flade data"""
        np.random.seed(0)
        days = 200
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        prices = 100 + np.random.normal(0, 0.1, days).cumsum()
        df = pd.DataFrame({"Open": prices, "High": prices, "Low": prices,
                          "Close": prices, "Volume": [1e6]*days}, index=dates)
        risk = risk_metrics(df)
        assert risk["ann_v"] < 0.5  # Lav vol


# ============ TESTS: MONTE CARLO ============

class TestMonteCarlo:
    def test_monte_carlo_shape(self):
        df = make_fake_hist(300)
        sims, lp = monte_carlo(df, days=252, sims=100)
        assert sims.shape == (100, 252)
        assert lp == df["Close"].iloc[-1]

    def test_monte_carlo_positive_prices(self):
        df = make_fake_hist(300)
        sims, _ = monte_carlo(df, days=100, sims=50)
        # Næsten alle simulationer skal have positive priser
        assert (sims > 0).mean() > 0.99


# ============ TESTS: ANBEFALINGER ============

class TestRecommendation:
    def test_strong_buy(self):
        anb, _ = recommendation(80)
        assert "STÆRKT KØB" in anb

    def test_buy(self):
        anb, _ = recommendation(65)
        assert "KØB" in anb

    def test_hold(self):
        anb, _ = recommendation(50)
        assert "HOLD" in anb

    def test_sell(self):
        anb, _ = recommendation(35)
        assert "SÆLG" in anb

    def test_strong_sell(self):
        anb, _ = recommendation(15)
        assert "STÆRKT SÆLG" in anb


# ============ TESTS: DATAKILDER (LIVE) ============

@pytest.mark.live
class TestDataSources:
    """Tests der kræver internet - kør med: pytest -m live"""

    def test_yahoo_aapl(self):
        result = fetch_yahoo("AAPL", period="1y")
        if result == "RATE_LIMIT":
            pytest.skip("Yahoo rate limited")
        assert result is not None
        assert "info" in result
        assert "hist" in result
        assert not result["hist"].empty

    def test_stooq_aapl(self):
        result = fetch_stooq("AAPL", period="1y")
        # Stooq kan være langsomt/upålideligt - skip hvis fejler
        if result is None:
            pytest.skip("Stooq unavailable")
        assert "hist" in result
        assert not result["hist"].empty

    def test_hybrid_fallback(self):
        """Test at hybrid funktion altid returnerer noget for AAPL"""
        # Cache decorator er fjernet i mock, så det kører rigtigt
        result = fetch_data("AAPL", period="1y")
        if result is None:
            pytest.skip("Alle datakilder utilgængelige")
        assert "source" in result
        assert "hist" in result


# ============ INTEGRATION TEST ============

class TestFullPipeline:
    def test_full_analysis_pipeline(self):
        """Test at hele analyse-flowet virker med syntetiske data"""
        # Lav fake data
        hist = make_fake_hist(500)
        info = {
            "trailingPE": 20, "priceToBook": 3, "returnOnEquity": 0.15,
            "debtToEquity": 50, "profitMargins": 0.10, "revenueGrowth": 0.08,
            "freeCashflow": 2e9, "sharesOutstanding": 1e9,
            "totalDebt": 5e8, "totalCash": 1e9
        }

        # Indikatorer
        df = add_indicators(hist)
        assert "RSI" in df.columns

        # Scoring
        f_score, f_det = fundamental_score(info)
        t_score, t_det = technical_score(df)
        assert 0 <= f_score <= 100
        assert 0 <= t_score <= 100

        # DCF
        fair = dcf_valuation(info, 0.10, 0.10, 0.025)
        assert fair is not None

        # Risiko
        risk = risk_metrics(hist)
        assert risk["sharpe"] is not None

        # Monte Carlo
        sims, lp = monte_carlo(hist, sims=50)
        assert sims.shape[0] == 50

        # Anbefalinger
        overall = f_score * 0.6 + t_score * 0.4
        anb, color = recommendation(overall)
        assert anb is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
