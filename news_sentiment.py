"""
News Sentiment Analysis
=======================
Henter nyhedsartikler og analyserer sentiment for aktier.

Krav:
- yfinance (allerede installeret) — primær nyhedskilde
- vaderSentiment (anbefales) — sentiment-analyse
  → pip install vaderSentiment
- finnhub (valgfri) — fallback nyhedskilde

Hvis VADER ikke er installeret bruges keyword-baseret fallback.
"""
import streamlit as st
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List

# ============ VADER SENTIMENT (anbefalet) ============
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False
    _vader = None

# ============ YFINANCE ============
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


# ============ FINANCE KEYWORDS ============
# Custom finance-specifik vægtning (boost VADER score)

POSITIVE_KEYWORDS = {
    # Earnings/performance
    "beat", "beats", "exceed", "exceeds", "outperform", "outperforms", "tops",
    "record", "all-time high", "ath", "rally", "surge", "soar", "jump",
    # Growth
    "growth", "expansion", "expanding", "boom", "thriving", "robust",
    # Analyst actions
    "upgrade", "upgraded", "buy rating", "overweight", "outperform rating",
    # Corporate actions
    "buyback", "dividend increase", "dividend hike", "acquisition", "merger",
    "partnership", "deal", "contract win", "approval", "fda approval",
    # Sentiment words
    "strong", "bullish", "optimistic", "confident", "innovative", "breakthrough",
    "profit", "gains", "rose", "rises", "boosts", "winning", "successful",
}

NEGATIVE_KEYWORDS = {
    # Earnings/performance
    "miss", "misses", "missed", "underperform", "underperforms", "disappoint",
    "plunge", "tumble", "crash", "slump", "slide", "fall", "drop",
    # Decline
    "decline", "shrink", "weak", "weakness", "slowdown", "recession",
    # Analyst actions
    "downgrade", "downgraded", "sell rating", "underweight",
    # Bad news
    "lawsuit", "investigation", "probe", "fraud", "scandal", "recall",
    "warning", "guidance cut", "earnings cut", "layoffs", "fired", "resign",
    "bankruptcy", "default", "fine", "penalty", "settlement",
    # Sentiment words
    "bearish", "pessimistic", "concerns", "fears", "worry", "uncertain",
    "loss", "losses", "fell", "drops", "dropped", "missed",
}


def _keyword_boost(text: str) -> float:
    """Beregn finance-keyword boost (-0.3 til +0.3)"""
    if not text:
        return 0.0
    text_lower = text.lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    if pos == 0 and neg == 0:
        return 0.0
    total = pos + neg
    return ((pos - neg) / max(total, 1)) * 0.3


def _analyze_sentiment(text: str) -> dict:
    """Analyser sentiment med VADER + finance keyword boost"""
    if not text or not text.strip():
        return {"score": 0.0, "label": "Neutralt", "emoji": "🟡"}

    # VADER base score
    if VADER_AVAILABLE:
        scores = _vader.polarity_scores(text)
        base_score = scores["compound"]  # -1 til +1
    else:
        # Fallback: kun keyword-baseret (boost mere)
        base_score = _keyword_boost(text) * 3.0

    # Tilføj finance-specifik boost
    boost = _keyword_boost(text)
    final_score = max(-1.0, min(1.0, base_score + boost))

    if final_score >= 0.15:
        label = "Positivt"
        emoji = "🟢"
    elif final_score <= -0.15:
        label = "Negativt"
        emoji = "🔴"
    else:
        label = "Neutralt"
        emoji = "🟡"

    return {"score": final_score, "label": label, "emoji": emoji}


# ============ NYHEDSKILDER ============

def _fetch_yfinance_news(ticker: str, limit: int = 20) -> List[Dict]:
    """Hent nyheder via yfinance.Ticker.news"""
    if not YF_AVAILABLE:
        return []

    try:
        tk = yf.Ticker(ticker)
        news_raw = tk.news or []
        articles = []

        for item in news_raw[:limit]:
            # yfinance returnerer enten gammel eller ny format
            content = item.get("content", item)

            title = content.get("title") or item.get("title", "")
            if not title:
                continue

            # URL
            url = ""
            if isinstance(content.get("canonicalUrl"), dict):
                url = content["canonicalUrl"].get("url", "")
            elif isinstance(content.get("clickThroughUrl"), dict):
                url = content["clickThroughUrl"].get("url", "")
            else:
                url = item.get("link", "") or content.get("link", "")

            # Publisher
            publisher = ""
            if isinstance(content.get("provider"), dict):
                publisher = content["provider"].get("displayName", "")
            else:
                publisher = item.get("publisher", "") or content.get("publisher", "")

            # Published time
            pub_time = None
            if "pubDate" in content:
                try:
                    pub_time = datetime.fromisoformat(
                        content["pubDate"].replace("Z", "+00:00")
                    )
                except Exception:
                    pass
            elif "providerPublishTime" in item:
                try:
                    pub_time = datetime.fromtimestamp(
                        item["providerPublishTime"], tz=timezone.utc
                    )
                except Exception:
                    pass

            # Summary
            summary = (
                content.get("summary")
                or content.get("description")
                or item.get("summary", "")
                or ""
            )

            articles.append({
                "title": title.strip(),
                "publisher": (publisher or "Unknown").strip(),
                "url": url,
                "published": pub_time,
                "summary": summary[:400],
            })

        return articles
    except Exception as e:
        print(f"[news_sentiment] yfinance fejl for {ticker}: {e}")
        return []


def _fetch_finnhub_news(ticker: str, limit: int = 20) -> List[Dict]:
    """Hent nyheder via Finnhub (kun US-tickers)"""
    if "." in ticker:  # ikke-US ticker
        return []

    try:
        from data_sources import get_api_keys
        finnhub_key, _ = get_api_keys()
        if not finnhub_key:
            return []

        import finnhub
        client = finnhub.Client(api_key=finnhub_key)

        end = datetime.now()
        start = end - timedelta(days=14)

        news = client.company_news(
            ticker,
            _from=start.strftime("%Y-%m-%d"),
            to=end.strftime("%Y-%m-%d"),
        )

        articles = []
        for item in news[:limit]:
            pub_time = None
            if item.get("datetime"):
                try:
                    pub_time = datetime.fromtimestamp(
                        item["datetime"], tz=timezone.utc
                    )
                except Exception:
                    pass

            articles.append({
                "title": (item.get("headline") or "").strip(),
                "publisher": (item.get("source") or "Finnhub").strip(),
                "url": item.get("url", ""),
                "published": pub_time,
                "summary": (item.get("summary") or "")[:400],
            })

        return [a for a in articles if a["title"]]
    except Exception as e:
        print(f"[news_sentiment] Finnhub fejl for {ticker}: {e}")
        return []


# ============ HOVED-FUNKTION ============

@st.cache_data(ttl=900, show_spinner=False)  # Cache 15 min
def get_news_sentiment(
    ticker: str,
    company_name: Optional[str] = None,
    limit: int = 20
) -> Optional[Dict]:
    """
    Henter nyheder for ticker og analyserer sentiment.

    Returns:
        Dict med sentiment-score, artikler og statistik, eller None ved fejl.
    """
    if not ticker:
        return None

    # Forsøg yfinance først
    articles = _fetch_yfinance_news(ticker, limit=limit)
    source = "Yahoo Finance"

    # Fallback til Finnhub
    if not articles:
        articles = _fetch_finnhub_news(ticker, limit=limit)
        source = "Finnhub"

    if not articles:
        return {
            "ticker": ticker,
            "company_name": company_name or ticker,
            "article_count": 0,
            "sentiment_score": 0.0,
            "label": "Ingen data",
            "label_emoji": "❓",
            "color": "#888888",
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "articles": [],
            "source": "Ingen kilder fungerede",
            "fetched_at": datetime.now(),
        }

    # Analyser hver artikel
    pos_count = 0
    neu_count = 0
    neg_count = 0
    weighted_sum = 0.0
    total_weight = 0.0

    for art in articles:
        # Kombiner titel + summary for bedre context
        text = (art["title"] or "") + ". " + (art.get("summary") or "")
        sent = _analyze_sentiment(text)
        art["sentiment_score"] = sent["score"]
        art["sentiment_label"] = sent["label"]
        art["sentiment_emoji"] = sent["emoji"]

        # Vægt efter alder (nyere = højere vægt)
        weight = 0.7  # default for ukendt alder
        if art.get("published"):
            try:
                age_hours = (
                    datetime.now(timezone.utc) - art["published"]
                ).total_seconds() / 3600
                if age_hours < 24:
                    weight = 1.0
                elif age_hours < 72:
                    weight = 0.85
                elif age_hours < 168:  # 7 dage
                    weight = 0.65
                elif age_hours < 720:  # 30 dage
                    weight = 0.45
                else:
                    weight = 0.25
            except Exception:
                weight = 0.7

        weighted_sum += sent["score"] * weight
        total_weight += weight

        if sent["label"] == "Positivt":
            pos_count += 1
        elif sent["label"] == "Negativt":
            neg_count += 1
        else:
            neu_count += 1

    # Samlet vægtet sentiment
    overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    if overall_score >= 0.15:
        overall_label = "Positivt"
        overall_emoji = "🟢"
        overall_color = "#16a34a"
    elif overall_score <= -0.15:
        overall_label = "Negativt"
        overall_emoji = "🔴"
        overall_color = "#ef4444"
    else:
        overall_label = "Neutralt"
        overall_emoji = "🟡"
        overall_color = "#eab308"

    return {
        "ticker": ticker,
        "company_name": company_name or ticker,
        "article_count": len(articles),
        "sentiment_score": overall_score,
        "label": overall_label,
        "label_emoji": overall_emoji,
        "color": overall_color,
        "positive_count": pos_count,
        "neutral_count": neu_count,
        "negative_count": neg_count,
        "articles": articles,
        "source": source,
        "fetched_at": datetime.now(),
        "vader_available": VADER_AVAILABLE,
    }


# ============ UI RENDERERS ============

def render_sentiment_summary(data: Optional[Dict], compact: bool = False) -> None:
    """
    Renderer sentiment-summary i Streamlit.

    Args:
        data: Output fra get_news_sentiment()
        compact: Hvis True, vises i kompakt 4-kolonne layout (analyse-view)
                 Hvis False, vises fuldt med chart (Nyheder-tab)
    """
    if data is None or data.get("article_count", 0) == 0:
        st.info(
            "📰 **Ingen nyheder fundet.** Mulige årsager:\n"
            "- Ticker er for niche (få store nyheder)\n"
            "- Yahoo Finance / Finnhub har midlertidige issues\n"
            "- Tjek `vaderSentiment` er installeret: `pip install vaderSentiment`"
        )
        return

    score = data["sentiment_score"]
    label = data["label"]
    emoji = data["label_emoji"]
    color = data["color"]
    n = data["article_count"]
    pos = data["positive_count"]
    neu = data["neutral_count"]
    neg = data["negative_count"]

    if compact:
        # Kompakt: 4 kolonner
        cols = st.columns([2, 1, 1, 1])
        cols[0].markdown(
            f"<div style='background:{color}22;padding:0.8rem;border-radius:10px;"
            f"border-left:5px solid {color}'>"
            f"<small style='color:#888'>📰 NYHEDS-SENTIMENT (sidste {n} artikler)</small>"
            f"<h3 style='margin:0.2rem 0;color:{color}'>{emoji} {label}</h3>"
            f"<small>Score: {score:+.2f} (-1 til +1) · {data.get('source', '?')}</small>"
            f"</div>",
            unsafe_allow_html=True
        )
        cols[1].metric("🟢 Positive", pos, f"{pos/n*100:.0f}%" if n else "")
        cols[2].metric("🟡 Neutrale", neu, f"{neu/n*100:.0f}%" if n else "")
        cols[3].metric("🔴 Negative", neg, f"{neg/n*100:.0f}%" if n else "")

    else:
        # Fuld visning
        st.markdown(
            f"<div style='background:{color}15;padding:1.2rem;border-radius:12px;"
            f"border-left:5px solid {color};margin-bottom:1rem'>"
            f"<small style='color:#888'>📰 SAMLET NYHEDS-SENTIMENT (vægtet efter alder)</small>"
            f"<h2 style='margin:0.5rem 0;color:{color}'>{emoji} {label}</h2>"
            f"<h3 style='margin:0.3rem 0'>Score: {score:+.3f} "
            f"<small style='color:#888;font-size:1rem'>(-1 til +1)</small></h3>"
            f"<small>📊 {n} artikler analyseret · 🔍 Kilde: {data.get('source', '?')}"
            + (f" · ⚙️ VADER: {'✅' if data.get('vader_available') else '⚠️ keyword fallback'}")
            + f"</small></div>",
            unsafe_allow_html=True
        )

        cols = st.columns(3)
        cols[0].metric("🟢 Positive", pos, f"{pos/n*100:.0f}% af artikler" if n else "")
        cols[1].metric("🟡 Neutrale", neu, f"{neu/n*100:.0f}% af artikler" if n else "")
        cols[2].metric("🔴 Negative", neg, f"{neg/n*100:.0f}% af artikler" if n else "")

        # Distribution chart
        try:
            import plotly.graph_objects as go
            fig = go.Figure(data=[
                go.Bar(
                    x=["🟢 Positive", "🟡 Neutrale", "🔴 Negative"],
                    y=[pos, neu, neg],
                    marker_color=["#16a34a", "#eab308", "#ef4444"],
                    text=[pos, neu, neg],
                    textposition="outside",
                    textfont=dict(size=14, color="white"),
                )
            ])
            fig.update_layout(
                template="plotly_dark",
                height=280,
                title="Sentiment-fordeling i artikler",
                showlegend=False,
                margin=dict(t=50, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


def render_news_feed(
    data: Optional[Dict],
    filter_type: str = "Alle",
    sort_by: str = "Nyeste først",
    max_items: int = 15,
) -> None:
    """
    Renderer artikel-feed med filtre.

    Args:
        data: Output fra get_news_sentiment()
        filter_type: "Alle", "🟢 Kun positive", "🔴 Kun negative", "🟡 Kun neutrale"
        sort_by: "Nyeste først", "Mest positive", "Mest negative"
        max_items: Max artikler at vise
    """
    if data is None or not data.get("articles"):
        st.info("Ingen artikler at vise.")
        return

    articles = list(data["articles"])  # kopi

    # Filter
    ft = filter_type.lower()
    if "positive" in ft:
        articles = [a for a in articles if a.get("sentiment_label") == "Positivt"]
    elif "negative" in ft:
        articles = [a for a in articles if a.get("sentiment_label") == "Negativt"]
    elif "neutral" in ft:
        articles = [a for a in articles if a.get("sentiment_label") == "Neutralt"]

    # Sort
    if sort_by == "Mest positive":
        articles.sort(key=lambda x: x.get("sentiment_score", 0), reverse=True)
    elif sort_by == "Mest negative":
        articles.sort(key=lambda x: x.get("sentiment_score", 0))
    else:  # Nyeste først
        def _sort_key(a):
            pub = a.get("published")
            if pub is None:
                return datetime.min.replace(tzinfo=timezone.utc)
            return pub
        articles.sort(key=_sort_key, reverse=True)

    # Limit
    articles = articles[:max_items]

    if not articles:
        st.info("Ingen artikler matcher filteret.")
        return

    st.caption(f"📑 Viser **{len(articles)} artikler**")

    # Render
    for art in articles:
        sent_label = art.get("sentiment_label", "Neutralt")
        sent_color = (
            "#16a34a" if sent_label == "Positivt"
            else "#ef4444" if sent_label == "Negativt"
            else "#eab308"
        )
        sent_emoji = art.get("sentiment_emoji", "🟡")
        sent_score = art.get("sentiment_score", 0)

        # Tid siden
        time_str = "?"
        if art.get("published"):
            try:
                age = datetime.now(timezone.utc) - art["published"]
                if age.days > 30:
                    time_str = f"{age.days // 30}m siden"
                elif age.days > 0:
                    time_str = f"{age.days}d siden"
                elif age.seconds > 3600:
                    time_str = f"{age.seconds // 3600}t siden"
                elif age.seconds > 60:
                    time_str = f"{age.seconds // 60}m siden"
                else:
                    time_str = "lige nu"
            except Exception:
                time_str = "?"

        title = art.get("title", "Ingen titel")
        publisher = art.get("publisher", "?")
        summary = art.get("summary", "")
        url = art.get("url", "")

        # Trunkér summary
        if len(summary) > 250:
            summary_display = summary[:250].rsplit(" ", 1)[0] + "..."
        else:
            summary_display = summary

        # Link-knap
        link_html = ""
        if url:
            link_html = (
                f"<div style='margin-top:0.6rem'>"
                f"<a href='{url}' target='_blank' "
                f"style='color:#0099ff;font-size:0.85rem;text-decoration:none'>"
                f"🔗 Læs hele artiklen →</a></div>"
            )

        st.markdown(
            f"<div style='background:{sent_color}10;padding:1rem;border-radius:10px;"
            f"border-left:4px solid {sent_color};margin-bottom:0.6rem'>"
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:1rem'>"
            f"<div style='flex:1'>"
            f"<div style='font-weight:bold;font-size:1.05rem;margin-bottom:0.3rem;line-height:1.3'>"
            f"{sent_emoji} {title}</div>"
            f"<div style='color:#aaa;font-size:0.85rem;margin-bottom:0.4rem'>"
            f"📰 {publisher} · ⏰ {time_str}</div>"
            f"<div style='color:#ccc;font-size:0.9rem;line-height:1.4'>{summary_display}</div>"
            f"{link_html}"
            f"</div>"
            f"<div style='text-align:right;min-width:80px'>"
            f"<div style='background:{sent_color}33;padding:0.4rem 0.7rem;border-radius:8px;"
            f"font-weight:bold;color:{sent_color};font-size:0.95rem'>{sent_score:+.2f}</div>"
            f"<small style='color:#888;font-size:0.75rem'>{sent_label}</small>"
            f"</div></div></div>",
            unsafe_allow_html=True
        )
