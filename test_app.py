"""
Backend tests for Aktie Dashboard
Kør med: pytest test_app.py -v
Eller:   python -m pytest test_app.py -v
Live-tests: pytest test_app.py -v -m live
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Tilføj projektmappen til path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock streamlit FØR vi importerer app.py
class MockSecrets:
    def get(self, key, default=None):
        return os.environ.get(key, default)

class MockColumn:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class MockStreamlit:
    secrets = MockSecrets()
    session_state = {}

    def cache_data(self, *args, **kwargs):
        # Kan bruges som @st.cache_data eller @st.cache_data(ttl=...)
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

    def set_page_config(self, **kwargs): pass
    def markdown(self, *args, **kwargs): pass
    def caption(self, *args, **kwargs): pass
    def title(self, *args, **kwargs): pass
    def write(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def success(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def spinner(self, *args, **kwargs): return MockColumn()
    def columns(self, n, **kwargs):
        if isinstance(n, int):
            return [MockColumn() for _ in range(n)]
        return [MockColumn() for _ in range(len(n))]
    def text_input(self, *args, **kwargs): return kwargs.get("value", "")
    def selectbox(self, *args, **kwargs):
        opts = args[1] if len(args) > 1 else kwargs.get("options", [])
        idx = kwargs.get("index", 0)
        return opts[idx] if opts else None
    def button(self, *args, **kwargs): return False
    def slider(self, *args, **kwargs): return kwargs.get("value", args[3] if len(args) > 3 else 0)
    def expander(self, *args, **kwargs): return MockColumn()
    def sidebar(self): return MockColumn()
    def tabs(self, names): return [MockColumn() for _ in names]
    def plotly_chart(self, *args, **kwargs): pass
    def dataframe(self, *args, **kwargs): pass
    def metric(self, *args, **kwargs): pass
    def stop(self): pass
    def rerun(self): pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

mock_st = MockStreamlit()
mock_st.sidebar = MockColumn()
sys.modules['streamlit'] = mock_st

# Nu kan vi importere fra app.py
from app import (
    safe, fundamental_score, technical_score, add_indicators,
    dcf_valuation, risk_metrics, monte_carlo, recommendation,
    fetch_yahoo, fetch_stooq, fetch_data, to_stooq_ticker
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


class TestTickerMapping:
    def test_us_ticker(self):
        assert to_stooq_ticker("AAPL") == "aapl.us"

    def test_us_ticker_lowercase(self):
        assert to_stooq_ticker("MSFT") == "msft.us"

    def test_danish_ticker(self):
        assert to_stooq_ticker("NOVO-B.CO") == "novob.co"

    def test_danish_no_dash(self):
        assert to_stooq_ticker("DSV.CO") == "dsv.co"

    def test_german_ticker(self):
        assert to_stooq_ticker("SAP.DE") == "sap.de"

    def test_dutch_ticker(self):
        assert to_stooq_ticker("ASML.AS") == "asml.nl"

    def test_swiss_ticker(self):
        assert to_stooq_ticker("NESN.SW") == "nesn.ch"

    def test_french_ticker(self):
        assert to_stooq_ticker("MC.PA") == "mc.fr"


# ============ TESTS: FUNDAMENTAL SCORING ============

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
        np.random.seed(0)
        for _ in range(20):
            info = {
                "trailingPE": np.random.uniform(5, 100),
                "returnOnEquity": np.random.uniform(-0.3, 0.5),
                "profitMargins": np.random.uniform(-0.2, 0.4),
                "debtToEquity": np.random.uniform(0, 400),
                "revenueGrowth": np.random.uniform(-0.3, 0.5),
                "priceToBook": np.random.uniform(0.5, 12),
            }
            score, _ = fundamental_score(info)
            assert 0 <= score <= 100, f"Score out of bounds: {score}"

    def test_details_returned(self):
        info = {"trailingPE": 12, "returnOnEquity": 0.25}
        score, det = fundamental_score(info)
        assert len(det) >= 2
        for item in det:
            assert "label" in item
            assert "value" in item
            assert "impact" in item


# ============ TESTS: TEKNISKE INDIKATORER ============

class TestIndicators:
    def test_indicators_added(self):
        df = make_fake_hist(300)
        df = add_indicators(df)
        for col in ["SMA50", "SMA200", "RSI", "MACD", "MACD_signal",
                    "BB_high", "BB_low", "STOCH_K", "ADX", "ATR"]:
            assert col in df.columns, f"Missing indicator: {col}"

    def test_rsi_in_valid_range(self):
        df = make_fake_hist(300)
        df = add_indicators(df)
        rsi_valid = df["RSI"].dropna()
        assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()

    def test_sma_relationship(self):
        """SMA50 og SMA200 bør være tæt på Close prisen"""
        df = make_fake_hist(500)
        df = add_indicators(df)
        last = df.iloc[-1]
        assert abs(last["SMA50"] - last["Close"]) / last["Close"] < 0.5


# ============ TESTS: TECHNICAL SCORING ============

class TestTechnicalScore:
    def test_score_bounds(self):
        df = make_fake_hist(300)
        df = add_indicators(df)
        score, _ = technical_score(df)
        assert 0 <= score <= 100

    def test_uptrend_scores_higher(self):
        """Klart opadgående trend skal score højere end faldende"""
        np.random.seed(1)
        days = 300
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')

        up_prices = np.linspace(50, 150, days) + np.random.normal(0, 1, days)
        df_up = pd.DataFrame({"Open": up_prices, "High": up_prices*1.01,
                             "Low": up_prices*0.99, "Close": up_prices,
                             "Volume": [1e6]*days}, index=dates)

        down_prices = np.linspace(150, 50, days) + np.random.normal(0, 1, days)
        df_down = pd.DataFrame({"Open": down_prices, "High": down_prices*1.01,
                               "Low": down_prices*0.99, "Close": down_prices,
                               "Volume": [1e6]*days}, index=dates)

        score_up, _ = technical_score(add_indicators(df_up))
        score_down, _ = technical_score(add_indicators(df_down))

        assert score_up > score_down, f"Uptrend ({score_up}) should beat downtrend ({score_down})"

    def test_details_returned(self):
        df = make_fake_hist(300)
        df = add_indicators(df)
        _, det = technical_score(df)
        assert len(det) > 0
        for item in det:
            assert "label" in item
            assert "impact" in item


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

    def test_dcf_no_shares_returns_none(self):
        info = {"freeCashflow": 1e9}
        assert dcf_valuation(info, 0.10, 0.10, 0.025) is None

    def test_dcf_higher_growth_higher_value(self):
        info = {"freeCashflow": 1e9, "sharesOutstanding": 1e8,
                "totalDebt": 0, "totalCash": 0}
        low = dcf_valuation(info, g=0.05, dr=0.10, tg=0.025)
        high = dcf_valuation(info, g=0.20, dr=0.10, tg=0.025)
        assert high > low, "Higher growth should give higher fair value"

    def test_dcf_lower_discount_higher_value(self):
        info = {"freeCashflow": 1e9, "sharesOutstanding": 1e8,
                "totalDebt": 0, "totalCash": 0}
        v_low_dr = dcf_valuation(info, g=0.10, dr=0.07, tg=0.025)
        v_high_dr = dcf_valuation(info, g=0.10, dr=0.15, tg=0.025)
        assert v_low_dr > v_high_dr, "Lower discount rate should give higher value"

    def test_dcf_debt_reduces_value(self):
        info_no_debt = {"freeCashflow": 1e9, "sharesOutstanding": 1e8,
                        "totalDebt": 0, "totalCash": 0}
        info_with_debt = {"freeCashflow": 1e9, "sharesOutstanding": 1e8,
                         "totalDebt": 5e9, "totalCash": 0}
        fair_no_debt = dcf_valuation(info_no_debt, 0.10, 0.10, 0.025)
        fair_with_debt = dcf_valuation(info_with_debt, 0.10, 0.10, 0.025)
        assert fair_no_debt > fair_with_debt


# ============ TESTS: RISIKO METRICS ============

class TestRiskMetrics:
    def test_risk_keys_present(self):
        df = make_fake_hist(500)
        risk = risk_metrics(df)
        for key in ["ann_r", "ann_v", "sharpe", "sortino", "max_dd", "var95", "dd_series"]:
            assert key in risk

    def test_volatility_positive(self):
        df = make_fake_hist(500)
        risk = risk_metrics(df)
        assert risk["ann_v"] > 0

    def test_max_drawdown_negative_or_zero(self):
        df = make_fake_hist(500)
        risk = risk_metrics(df)
        assert risk["max_dd"] <= 0

    def test_var95_negative(self):
        df = make_fake_hist(500)
        risk = risk_metrics(df)
        # VaR 95% bør være negativ (worst 5% afkast)
        assert risk["var95"] <= 0

    def test_low_volatility_data(self):
        np.random.seed(0)
        days = 200
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        prices = 100 + np.random.normal(0, 0.1, days).cumsum()
        df = pd.DataFrame({"Open": prices, "High": prices, "Low": prices,
                          "Close": prices, "Volume": [1e6]*days}, index=dates)
        risk = risk_metrics(df)
        assert risk["ann_v"] < 0.5


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
        assert (sims > 0).mean() > 0.99

    def test_monte_carlo_starts_from_last_price(self):
        df = make_fake_hist(300)
        sims, lp = monte_carlo(df, days=252, sims=100)
        # Første dag bør være tæt på sidste pris
        first_day_avg = sims[:, 0].mean()
        assert abs(first_day_avg - lp) / lp < 0.10

    def test_monte_carlo_variation(self):
        """Monte Carlo skal give forskellige resultater"""
        df = make_fake_hist(300)
        sims, _ = monte_carlo(df, days=252, sims=100)
        final_prices = sims[:, -1]
        # Standardafvigelse skal være > 0
        assert final_prices.std() > 0


# ============ TESTS: ANBEFALINGER ============

class TestRecommendation:
    def test_strong_buy(self):
        anb, color = recommendation(80)
        assert "STÆRKT KØB" in anb
        assert color.startswith("#")

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

    def test_thresholds(self):
        """Test eksakte tærskler"""
        assert "STÆRKT KØB" in recommendation(75)[0]
        assert "KØB" in recommendation(60)[0]
        assert "HOLD" in recommendation(45)[0]
        assert "SÆLG" in recommendation(30)[0]


# ============ LIVE TESTS (kræver internet) ============

@pytest.mark.live
class TestDataSources:
    """Tests der kræver internet - kør med: pytest -m live"""

    def test_yahoo_aapl(self):
        result = fetch_yahoo("AAPL", period="1y")
        if result == "RATE_LIMIT":
            pytest.skip("Yahoo rate limited")
        if result is None:
            pytest.skip("Yahoo unavailable")
        assert "info" in result
        assert "hist" in result
        assert not result["hist"].empty

    def test_stooq_aapl(self):
        result = fetch_stooq("AAPL", period="1y")
        if result is None:
            pytest.skip("Stooq unavailable")
        assert "hist" in result
        assert not result["hist"].empty

    def test_stooq_danish(self):
        result = fetch_stooq("NOVO-B.CO", period="1y")
        if result is None:
            pytest.skip("Stooq unavailable for danish stocks")
        assert "hist" in result
        assert not result["hist"].empty

    def test_hybrid_returns_data(self):
        """Hybrid funktion bør altid returnere noget for AAPL"""
        result = fetch_data("AAPL", period="1y")
        if result is None:
            pytest.skip("Alle datakilder utilgængelige")
        assert "source" in result
        assert "hist" in result


# ============ INTEGRATION TEST ============

class TestFullPipeline:
    def test_full_analysis_pipeline(self):
        """End-to-end test af hele analyseflowet med syntetiske data"""
        hist = make_fake_hist(500)
        info = {
            "trailingPE": 20, "priceToBook": 3, "returnOnEquity": 0.15,
            "debtToEquity": 50, "profitMargins": 0.10, "revenueGrowth": 0.08,
            "freeCashflow": 2e9, "sharesOutstanding": 1e9,
            "totalDebt": 5e8, "totalCash": 1e9
        }

        # 1. Tilføj indikatorer
        df = add_indicators(hist)
        assert "RSI" in df.columns

        # 2. Scoring
        f_score, f_det = fundamental_score(info)
        t_score, t_det = technical_score(df)
        assert 0 <= f_score <= 100
        assert 0 <= t_score <= 100
        assert len(f_det) > 0
        assert len(t_det) > 0

        # 3. DCF
        fair = dcf_valuation(info, 0.10, 0.10, 0.025)
        assert fair is not None
        assert fair > 0

        # 4. Risiko
        risk = risk_metrics(hist)
        assert risk["sharpe"] is not None
        assert risk["max_dd"] <= 0

        # 5. Monte Carlo
        sims, lp = monte_carlo(hist, sims=50)
        assert sims.shape == (50, 252)

        # 6. Anbefalinger
        overall = f_score * 0.6 + t_score * 0.4
        anb, color = recommendation(overall)
        assert anb is not None
        assert color.startswith("#")

    def test_pipeline_with_minimal_info(self):
        """Test at pipeline ikke crasher med mangelfulde data"""
        hist = make_fake_hist(250)
        info = {}  # Tom info dict

        df = add_indicators(hist)
        f_score, _ = fundamental_score(info)
        t_score, _ = technical_score(df)
        assert f_score == 50  # Default når ingen data
        assert 0 <= t_score <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
