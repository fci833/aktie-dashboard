"""News sentiment analyzer.

Henter nyheder fra Finnhub + yfinance og analyserer sentiment med VADER.
Gratis, ingen API-key til VADER, lynhurtigt.
"""
import os
import time
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# VADER sentiment - lazy import for at undgå crash hvis ikke installeret
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    _vader = None
    print("[news_sentiment] vaderSentiment ikke installeret - kør: pip install vaderSentiment")


# Finansielle keyword-justeringer for VADER
# (VADER er trænet på generel tekst, så vi booster financial keywords)
FINANCIAL_LEXICON = {
    "beat": 3.0, "beats": 3.0, "beaten": 2.5, "exceeded": 3.0, "exceeds": 3.0,
    "surge": 3.0, "surges": 3.0, "surged": 3.0, "soar": 3.5, "soars": 3.5, "soared": 3.5,
    "rally": 2.5, "rallied": 2.5, "jumped": 2.5, "jumps": 2.5,
    "upgrade": 2.5, "upgraded": 2.5, "outperform": 2.5,
    "buy": 2.0, "bullish": 3.0, "strong": 1.5, "growth": 1.5, "profit": 2.0,
    "record": 2.0, "high": 1.0, "gains": 2.0, "gained": 2.0,

    "miss": -3.0, "missed": -3.0, "missing": -2.5, "disappoints": -3.0, "disappointed": -3.0,
    "plunge": -3.5, "plunges": -3.5, "plunged": -3.5, "tumble": -3.0, "tumbled": -3.0,
    "crash": -3.5, "crashed": -3.5, "crashes": -3.5, "drop": -2.0, "dropped": -2.0,
    "downgrade": -2.5, "downgraded": -2.5, "underperform": -2.5,
    "sell": -2.0, "bearish": -3.0, "weak": -2.0, "loss": -2.5, "losses": -2.5,
    "lawsuit": -2.5, "investigation": -2.0, "probe": -2.0, "fraud": -3.5,
    "warning": -2.5, "concerns": -1.5, "concerned": -1.5, "fears": -2.0,
    "decline": -2.0, "declines": -2.0, "declined": -2.0,
    "cut": -1.5, "slashed": -2.5, "layoffs": -2.5, "layoff": -2.5,
}


def _enhance_vader():
    """Tilføj finansielle ord til VADER's lexicon"""
    if _vader and VADER_AVAILABLE:
        _vader.lexicon.update(FINANCIAL_LEXICON)


# Kør én gang ved import
_enhance_vader()


# ============================================================
# NYHEDS-HENTNING
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)  # 15 min cache
def fetch_finnhub_news(ticker: str, days: int = 7):
    """Hent nyheder fra Finnhub (gratis tier: 60 calls/min)"""
    api_key = os.getenv("FINNHUB_API_KEY") or st.secrets.get("FINNHUB_API_KEY", None)
    if not api_key:
        return []

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker.upper(),
        "from": start_date,
        "to": end_date,
        "token": api_key,
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, list):
            return []

        articles = []
        for item in data[:30]:  # max 30 nyheder
            articles.append({
                "title": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("source", "Finnhub"),
                "date": datetime.fromtimestamp(item.get("datetime", 0)),
                "image": item.get("image", ""),
            })
        return articles
    except Exception as e:
        print(f"[fetch_finnhub_news] {e}")
        return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_yfinance_news(ticker: str):
    """Hent nyheder fra yfinance (fallback)"""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        news = t.news or []

        articles = []
        for item in news[:20]:
            content = item.get("content", item)
            articles.append({
                "title": content.get("title", ""),
                "summary": content.get("summary", "") or content.get("description", ""),
                "url": (content.get("canonicalUrl", {}) or {}).get("url", "") or content.get("link", ""),
                "source": (content.get("provider", {}) or {}).get("displayName", "Yahoo Finance"),
                "date": _parse_yf_date(content),
                "image": _parse_yf_image(content),
            })
        return articles
    except Exception as e:
        print(f"[fetch_yfinance_news] {e}")
        return []


def _parse_yf_date(content):
    """Parse yfinance date format"""
    try:
        ts = content.get("pubDate") or content.get("providerPublishTime")
        if isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
        elif isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts)
    except Exception:
        pass
    return datetime.now()


def _parse_yf_image(content):
    """Parse yfinance image"""
    try:
        thumb = content.get("thumbnail") or {}
        resolutions = thumb.get("resolutions") or []
        if resolutions:
            return resolutions[0].get("url", "")
    except Exception:
        pass
    return ""


def fetch_all_news(ticker: str, days: int = 7):
    """Hent fra både Finnhub og yfinance, dedupliker, sortér efter dato"""
    all_news = []

    # Prøv Finnhub først
    finnhub_news = fetch_finnhub_news(ticker, days)
    all_news.extend(finnhub_news)

    # Hvis vi har under 5 fra Finnhub, supplér med yfinance
    if len(finnhub_news) < 5:
        yf_news = fetch_yfinance_news(ticker)
        all_news.extend(yf_news)

    # Dedupliker baseret på titel
    seen_titles = set()
    deduped = []
    for art in all_news:
        title_key = art["title"][:80].lower().strip()
        if title_key and title_key not in seen_titles:
            seen_titles.add(title_key)
            deduped.append(art)

    # Sortér efter dato (nyeste først)
    deduped.sort(key=lambda x: x["date"], reverse=True)
    return deduped[:25]  # Returnér max 25


# ============================================================
# SENTIMENT ANALYSE
# ============================================================

def analyze_sentiment(text: str) -> dict:
    """Analysér en tekst med VADER + return scores 0-100"""
    if not VADER_AVAILABLE or not text:
        return {"score": 50, "compound": 0, "label": "NEUTRAL", "color": "#6b7280"}

    scores = _vader.polarity_scores(text)
    compound = scores["compound"]  # -1 til +1

    # Konverter til 0-100 skala
    score_100 = (compound + 1) * 50

    if compound >= 0.5:
        label = "MEGET POSITIV"
        color = "#16a34a"
        emoji = "🚀"
    elif compound >= 0.15:
        label = "POSITIV"
        color = "#22c55e"
        emoji = "✅"
    elif compound > -0.15:
        label = "NEUTRAL"
        color = "#6b7280"
        emoji = "➖"
    elif compound > -0.5:
        label = "NEGATIV"
        color = "#f97316"
        emoji = "⚠️"
    else:
        label = "MEGET NEGATIV"
        color = "#ef4444"
        emoji = "🔴"

    return {
        "score": round(score_100, 1),
        "compound": round(compound, 3),
        "label": label,
        "color": color,
        "emoji": emoji,
        "positive": round(scores["pos"] * 100, 1),
        "negative": round(scores["neg"] * 100, 1),
        "neutral": round(scores["neu"] * 100, 1),
    }


def analyze_articles(articles: list) -> list:
    """Tilføj sentiment til alle artikler"""
    for art in articles:
        # Vægt titel højere end summary (titel er mere koncentreret signal)
        title_text = art.get("title", "")
        summary_text = art.get("summary", "")
        combined = f"{title_text}. {title_text}. {summary_text}"  # Titel tæller dobbelt
        art["sentiment"] = analyze_sentiment(combined)
    return articles


def aggregate_sentiment(articles: list) -> dict:
    """
    Beregn samlet sentiment-score på 0-100 for hele nyhedsfeed.
    
    Vægter nyere artikler højere (eksponentielt henfald).
    """
    if not articles:
        return {
            "score": 50,
            "label": "INGEN NYHEDER",
            "color": "#6b7280",
            "emoji": "❓",
            "count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "score_adjustment": 0,
        }

    now = datetime.now()
    total_weight = 0
    weighted_score = 0

    pos_count = neg_count = neu_count = 0

    for art in articles:
        if "sentiment" not in art:
            continue
        sent = art["sentiment"]
        score = sent["score"]

        # Tids-vægt: nyere artikler vejer mere
        try:
            age_days = max(0, (now - art["date"]).days)
        except Exception:
            age_days = 0
        time_weight = max(0.3, 1.0 - (age_days / 14))  # 14 dage til min vægt

        weighted_score += score * time_weight
        total_weight += time_weight

        # Tæl op
        if sent["compound"] >= 0.15:
            pos_count += 1
        elif sent["compound"] <= -0.15:
            neg_count += 1
        else:
            neu_count += 1

    if total_weight == 0:
        avg_score = 50
    else:
        avg_score = weighted_score / total_weight

    # Label
    if avg_score >= 70:
        label, color, emoji = "MEGET POSITIV", "#16a34a", "🚀"
    elif avg_score >= 58:
        label, color, emoji = "POSITIV", "#22c55e", "✅"
    elif avg_score >= 42:
        label, color, emoji = "NEUTRAL", "#eab308", "➖"
    elif avg_score >= 30:
        label, color, emoji = "NEGATIV", "#f97316", "⚠️"
    else:
        label, color, emoji = "MEGET NEGATIV", "#ef4444", "🔴"

    # Score adjustment til overall: -5 til +5 point
    # 50 = neutral = 0 adjustment
    # 100 = max positiv = +5
    # 0 = max negativ = -5
    score_adjustment = round((avg_score - 50) / 10, 1)

    return {
        "score": round(avg_score, 1),
        "label": label,
        "color": color,
        "emoji": emoji,
        "count": len(articles),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "score_adjustment": score_adjustment,
    }


# ============================================================
# UI RENDERING
# ============================================================

def render_sentiment_summary(agg: dict):
    """Render samlet sentiment som card"""
    color = agg["color"]
    emoji = agg["emoji"]

    summary_html = (
        f"<div style='background:linear-gradient(90deg, {color}33 0%, {color}11 100%);"
        f"padding:1rem 1.5rem;border-radius:12px;"
        f"border-left:5px solid {color};margin:0.5rem 0'>"
        f"<div style='display:flex;align-items:center;gap:1rem;flex-wrap:wrap'>"
        f"<div style='font-size:2.5rem'>{emoji}</div>"
        f"<div style='flex:1;min-width:200px'>"
        f"<div style='color:{color};font-weight:bold;font-size:1.3rem'>"
        f"{agg['label']} SENTIMENT"
        f"</div>"
        f"<div style='color:#aaa;font-size:0.9rem;margin-top:0.2rem'>"
        f"Baseret på <b>{agg['count']}</b> nyhedsartikler "
        f"(✅ {agg['positive_count']} pos · ➖ {agg['neutral_count']} neu · "
        f"⚠️ {agg['negative_count']} neg)"
        f"</div>"
        f"</div>"
        f"<div style='text-align:right'>"
        f"<div style='font-size:0.8rem;color:#888'>SCORE</div>"
        f"<div style='font-size:1.5rem;font-weight:bold;color:{color}'>"
        f"{agg['score']:.0f}<small style='color:#888;font-size:0.8rem'>/100</small>"
        f"</div>"
        f"</div>"
        f"<div style='text-align:right;border-left:1px solid #444;padding-left:1rem'>"
        f"<div style='font-size:0.8rem;color:#888'>SCORE EFFEKT</div>"
        f"<div style='font-size:1.2rem;font-weight:bold;color:{color}'>"
        f"{agg['score_adjustment']:+.1f}"
        f"</div>"
        f"</div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(summary_html, unsafe_allow_html=True)


def render_news_feed(articles: list, max_show: int = 10):
    """Render nyheder med sentiment-farver"""
    if not articles:
        st.info("📭 Ingen nyheder fundet for denne aktie i de seneste 7 dage.")
        return

    st.markdown(f"### 📰 Seneste nyheder ({len(articles)})")

    show_all = st.checkbox("Vis alle nyheder", value=False, key=f"news_show_all_{id(articles)}")
    n_show = len(articles) if show_all else min(max_show, len(articles))

    for i, art in enumerate(articles[:n_show]):
        sent = art.get("sentiment", {})
        color = sent.get("color", "#6b7280")
        emoji = sent.get("emoji", "➖")
        label = sent.get("label", "NEUTRAL")
        score = sent.get("score", 50)

        try:
            date_str = art["date"].strftime("%d. %b %Y · %H:%M")
            age_days = (datetime.now() - art["date"]).days
            if age_days == 0:
                age_str = "🕐 I dag"
            elif age_days == 1:
                age_str = "🕐 I går"
            else:
                age_str = f"🕐 {age_days}d siden"
        except Exception:
            date_str = "Ukendt"
            age_str = ""

        title = art.get("title", "Ingen titel")
        source = art.get("source", "Ukendt")
        url = art.get("url", "#")
        summary = art.get("summary", "")
        if len(summary) > 250:
            summary = summary[:250] + "..."

        article_html = (
            f"<div style='background:{color}11;padding:0.8rem 1rem;border-radius:8px;"
            f"border-left:4px solid {color};margin-bottom:0.6rem'>"
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap'>"
            f"<div style='flex:1;min-width:300px'>"
            f"<div style='font-size:1rem;font-weight:bold;margin-bottom:0.3rem'>"
            f"<a href='{url}' target='_blank' style='color:#fff;text-decoration:none'>{title}</a>"
            f"</div>"
            f"<div style='color:#aaa;font-size:0.85rem;margin-bottom:0.4rem'>"
            f"📰 {source} · {date_str} · {age_str}"
            f"</div>"
            f"<div style='color:#bbb;font-size:0.9rem'>{summary}</div>"
            f"</div>"
            f"<div style='text-align:right;min-width:120px'>"
            f"<div style='background:{color}33;padding:0.3rem 0.6rem;border-radius:6px;display:inline-block'>"
            f"<div style='font-size:0.7rem;color:#aaa'>SENTIMENT</div>"
            f"<div style='color:{color};font-weight:bold;font-size:0.9rem'>{emoji} {label}</div>"
            f"<div style='color:{color};font-size:1.1rem;font-weight:bold'>{score:.0f}/100</div>"
            f"</div>"
            f"</div>"
            f"</div>"
            f"</div>"
        )
        st.markdown(article_html, unsafe_allow_html=True)


# ============================================================
# HOVED-FUNKTION
# ============================================================

def get_news_sentiment(ticker: str, days: int = 7) -> dict:
    """
    Henter nyheder + analyserer sentiment + returnerer alt.

    Returns:
        dict med:
            articles: list af artikler med sentiment
            aggregate: samlet sentiment dict
            score_adjustment: -5 til +5 (til at justere overall score)
    """
    if not VADER_AVAILABLE:
        return {
            "articles": [],
            "aggregate": {
                "score": 50, "label": "VADER IKKE INSTALLERET",
                "color": "#6b7280", "emoji": "❓", "count": 0,
                "positive_count": 0, "negative_count": 0, "neutral_count": 0,
                "score_adjustment": 0,
            },
            "score_adjustment": 0,
            "error": "vaderSentiment ikke installeret. Kør: pip install vaderSentiment",
        }

    articles = fetch_all_news(ticker, days=days)
    articles = analyze_articles(articles)
    agg = aggregate_sentiment(articles)

    return {
        "articles": articles,
        "aggregate": agg,
        "score_adjustment": agg["score_adjustment"],
    }
