import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os

# Load API key: check Streamlit secrets first (Cloud or local secrets.toml), then .env
try:
    _api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    if _api_key:
        os.environ["ANTHROPIC_API_KEY"] = _api_key
except Exception:
    pass
if "ANTHROPIC_API_KEY" not in os.environ:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env", override=True)
    except ImportError:
        pass

# Data loading: use live Rader Joint locally, Dropbox API for Streamlit Cloud
_local_data = Path(r"C:\Users\Trader\Dropbox\Python\SMB Python\Rader Joint")
_use_dropbox = not _local_data.exists()

if _use_dropbox:
    from dropbox_loader import read_csv as _dbx_read_csv, read_parquet as _dbx_read_parquet
    DATA_DIR = None  # not used when loading from Dropbox
else:
    DATA_DIR = _local_data

st.set_page_config(page_title="Market Dashboard", layout="wide")

# DEBUG: show secrets keys (remove after confirming)
if _use_dropbox:
    st.sidebar.caption(f"Secrets keys: {list(st.secrets.keys())}")
    st.sidebar.caption(f"Env ANTHROPIC: {'SET' if os.environ.get('ANTHROPIC_API_KEY') else 'NOT SET'}")

# --- Portrait mode: constrain content to left half ---
st.markdown("""
<style>
    [data-testid="stAppViewBlockContainer"] {
        max-width: 55% !important;
        margin-left: 0 !important;
        margin-right: auto !important;
    }
    [data-testid="stSidebar"] {
        min-width: 220px !important;
        max-width: 220px !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading ---
# Each loader reads from local Rader Joint folder or Dropbox API depending on environment.

@st.cache_data
def load_vix():
    df = _dbx_read_csv("VIX_data.csv", parse_dates=["Date"]) if _use_dropbox else pd.read_csv(DATA_DIR / "VIX_data.csv", parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df.reset_index(drop=True)


@st.cache_data
def load_pcr():
    df = _dbx_read_csv("Historical_put_call_ratio.csv", parse_dates=["date"]) if _use_dropbox else pd.read_csv(DATA_DIR / "Historical_put_call_ratio.csv", parse_dates=["date"])
    df.sort_values("date", inplace=True)
    return df.reset_index(drop=True)


@st.cache_data
def load_ad():
    df = _dbx_read_csv("Advance_Decline.csv", parse_dates=["date"]) if _use_dropbox else pd.read_csv(DATA_DIR / "Advance_Decline.csv", parse_dates=["date"])
    df.sort_values("date", inplace=True)
    return df.reset_index(drop=True)


@st.cache_data
def load_ty():
    df = _dbx_read_csv("Treasury_Yields.csv", parse_dates=["Date"]) if _use_dropbox else pd.read_csv(DATA_DIR / "Treasury_Yields.csv", parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df.reset_index(drop=True)


@st.cache_data
def load_pivot():
    df = _dbx_read_csv("sp500_pivot_breadth.csv", parse_dates=["date"]) if _use_dropbox else pd.read_csv(DATA_DIR / "sp500_pivot_breadth.csv", parse_dates=["date"])
    df.sort_values("date", inplace=True)
    return df.reset_index(drop=True)


@st.cache_data
def load_ema_breadth():
    """Load sp500_daily_technicals and pre-compute % of stocks above each EMA per day."""
    df = _dbx_read_parquet("sp500_daily_technicals.parquet") if _use_dropbox else pd.read_parquet(DATA_DIR / "sp500_daily_technicals.parquet")
    df["date"] = pd.to_datetime(df["date"])
    ema_cols = ["EMA_5", "EMA_8", "EMA_20", "EMA_50", "EMA_200"]
    result = pd.DataFrame({"date": df["date"].unique()}).sort_values("date").reset_index(drop=True)
    for ema in ema_cols:
        above = (df["close"] > df[ema]).groupby(df["date"]).mean()
        result = result.merge(above.rename(f"pct_above_{ema}").reset_index(), on="date", how="left")
    return result


@st.cache_data
def load_ofi():
    df = _dbx_read_parquet("SPY_order_flow_daily.parquet") if _use_dropbox else pd.read_parquet(DATA_DIR / "SPY_order_flow_daily.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    return df.reset_index(drop=True)


@st.cache_data
def load_gex():
    df = _dbx_read_csv("SPY_historical_gamma_2005_2025.csv", parse_dates=["Date"]) if _use_dropbox else pd.read_csv(DATA_DIR / "SPY_historical_gamma_2005_2025.csv", parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df.reset_index(drop=True)


@st.cache_data
def load_spy():
    df = _dbx_read_csv("SPY_historical_data.csv", parse_dates=["date"]) if _use_dropbox else pd.read_csv(DATA_DIR / "SPY_historical_data.csv", parse_dates=["date"])
    df.sort_values("date", inplace=True)
    return df.reset_index(drop=True)


@st.cache_data
def load_etf():
    df = _dbx_read_parquet("etf_historical_data.parquet") if _use_dropbox else pd.read_parquet(DATA_DIR / "etf_historical_data.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["symbol", "date"], inplace=True)
    return df.reset_index(drop=True)


SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care",
    "XLY": "Cons. Discretionary", "XLP": "Cons. Staples", "XLE": "Energy",
    "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication",
}
COMMODITY_ETFS = {
    "GLD": "Gold", "SLV": "Silver", "USO": "Oil (WTI)", "BNO": "Oil (Brent)",
    "UNG": "Natural Gas", "CPER": "Copper", "CORN": "Corn", "SOYB": "Soybeans",
    "WEAT": "Wheat", "DBA": "Agriculture", "PDBC": "Broad Commodity",
}


import numpy as np

def find_swing_points(series, order=5):
    """Find local maxima and minima using a rolling window comparison.
    order = number of bars on each side that must be lower/higher."""
    highs = []
    lows = []
    vals = series.values
    for i in range(order, len(vals) - order):
        if np.isnan(vals[i]):
            continue
        window = vals[i - order:i + order + 1]
        if np.any(np.isnan(window)):
            continue
        if vals[i] == np.max(window):
            highs.append(i)
        if vals[i] == np.min(window):
            lows.append(i)
    return highs, lows


def detect_rsi_divergences(df, price_col="close", rsi_col="SPY_RSI_14", order=5):
    """Detect regular bullish and bearish RSI divergences.

    Bullish: price makes lower low, RSI makes higher low
    Bearish: price makes higher high, RSI makes lower high

    Returns lists of (date, price, rsi) tuples for each divergence type.
    """
    price = df[price_col].reset_index(drop=True)
    rsi = df[rsi_col].reset_index(drop=True)
    dates = df["date"].reset_index(drop=True)

    price_highs, price_lows = find_swing_points(price, order)
    rsi_highs, rsi_lows = find_swing_points(rsi, order)

    bullish = []  # bearish price + bullish RSI at lows
    bearish = []  # bullish price + bearish RSI at highs

    # Bullish divergence: compare consecutive price lows
    for i in range(1, len(price_lows)):
        idx_prev, idx_curr = price_lows[i - 1], price_lows[i]
        # Price made lower low
        if price.iloc[idx_curr] < price.iloc[idx_prev]:
            # Find RSI low closest to each price low
            rsi_near_prev = [r for r in rsi_lows if abs(r - idx_prev) <= order + 2]
            rsi_near_curr = [r for r in rsi_lows if abs(r - idx_curr) <= order + 2]
            if rsi_near_prev and rsi_near_curr:
                rsi_prev = rsi.iloc[min(rsi_near_prev, key=lambda r: abs(r - idx_prev))]
                rsi_curr = rsi.iloc[min(rsi_near_curr, key=lambda r: abs(r - idx_curr))]
                # RSI made higher low (divergence)
                if rsi_curr > rsi_prev:
                    bullish.append({
                        "date": dates.iloc[idx_curr],
                        "price": price.iloc[idx_curr],
                        "rsi": rsi_curr,
                    })

    # Bearish divergence: compare consecutive price highs
    for i in range(1, len(price_highs)):
        idx_prev, idx_curr = price_highs[i - 1], price_highs[i]
        # Price made higher high
        if price.iloc[idx_curr] > price.iloc[idx_prev]:
            # Find RSI high closest to each price high
            rsi_near_prev = [r for r in rsi_highs if abs(r - idx_prev) <= order + 2]
            rsi_near_curr = [r for r in rsi_highs if abs(r - idx_curr) <= order + 2]
            if rsi_near_prev and rsi_near_curr:
                rsi_prev = rsi.iloc[min(rsi_near_prev, key=lambda r: abs(r - idx_prev))]
                rsi_curr = rsi.iloc[min(rsi_near_curr, key=lambda r: abs(r - idx_curr))]
                # RSI made lower high (divergence)
                if rsi_curr < rsi_prev:
                    bearish.append({
                        "date": dates.iloc[idx_curr],
                        "price": price.iloc[idx_curr],
                        "rsi": rsi_curr,
                    })

    return bullish, bearish


# --- Persistent disk cache for API calls ---
import hashlib

CACHE_DIR = Path(__file__).parent / ".api_cache"
CACHE_DIR.mkdir(exist_ok=True)

def disk_cache_get(prefix: str, *args) -> str | None:
    """Check if a cached result exists for the given inputs."""
    key = hashlib.md5("".join(args).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{prefix}_{key}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    return None

def disk_cache_timestamp(prefix: str) -> str | None:
    """Return the last modified time of the most recent cache file for a prefix."""
    from datetime import datetime
    files = list(CACHE_DIR.glob(f"{prefix}_*.txt"))
    if files:
        mtime = max(f.stat().st_mtime for f in files)
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %I:%M %p")
    return None

def disk_cache_set(prefix: str, result: str, *args):
    """Save a result to disk cache."""
    key = hashlib.md5("".join(args).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{prefix}_{key}.txt"
    cache_file.write_text(result, encoding="utf-8")
    # Clean old cache files for this prefix (keep only latest)
    for f in CACHE_DIR.glob(f"{prefix}_*.txt"):
        if f != cache_file:
            f.unlink()


def check_scheduled_cache_clear():
    """Clear API cache at 8:45 AM on weekdays so next load gets fresh AI analysis."""
    from datetime import datetime, date
    now = datetime.now()
    # Only on weekdays (Mon=0 to Fri=4)
    if now.weekday() > 4:
        return
    marker_file = CACHE_DIR / f"cleared_{date.today().isoformat()}.marker"
    # Already cleared today
    if marker_file.exists():
        return
    # Not yet 8:45 AM
    if now.hour < 8 or (now.hour == 8 and now.minute < 45):
        return
    # Clear all cached regime + strategy files
    for f in CACHE_DIR.glob("regime_*.txt"):
        f.unlink()
    for f in CACHE_DIR.glob("strategies_*.txt"):
        f.unlink()
    # Also clear Streamlit's in-memory cache so data reloads
    st.cache_data.clear()
    # Write marker so we don't clear again today
    marker_file.write_text(now.strftime("%H:%M:%S"))
    # Clean old markers
    for f in CACHE_DIR.glob("cleared_*.marker"):
        if f != marker_file:
            f.unlink()

# Run on every page load
check_scheduled_cache_clear()


# --- Shared chart layout ---
CHART_MARGIN = dict(t=40, b=50, l=60, r=20)
CHART_LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)

def stamp_last_date(fig, last_date):
    """Add last-date annotation below chart and standardize legend."""
    fig.add_annotation(
        text=f"Last: {last_date.strftime('%Y-%m-%d')}",
        xref="paper", yref="paper", x=1.0, y=-0.15,
        showarrow=False, font=dict(size=10, color="gray"),
        xanchor="right", yanchor="top",
    )
    fig.update_layout(legend=CHART_LEGEND)
    return fig


# --- Actionable Strategies ---
def get_ai_strategy_recommendations(signals_json: str, regime_summary: str = "") -> str:
    """Call Claude API to generate actionable strategy recommendations aligned with regime summary. Disk-cached."""
    cache_key = signals_json + regime_summary
    cached = disk_cache_get("strategies", cache_key)
    if cached:
        return cached
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except (KeyError, AttributeError):
            return "**API key not configured.** Set ANTHROPIC_API_KEY in secrets or .env file."
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""You are a senior systematic trading strategist specializing in SPY equity and options strategies. Based on the 5-day trailing market signals below, provide specific, actionable strategy recommendations.

CRITICAL: The regime summary below has ALREADY been generated and presented to the user. Your strategy recommendations MUST be consistent with this regime classification. Do NOT contradict it. If the regime says "breakout favored", your stock strategies must recommend breakouts. If it says "mean-reversion", recommend fades. Build on the regime analysis, don't re-analyze from scratch.

REGIME SUMMARY (already shown to user — align with this):
{regime_summary}

Structure your response EXACTLY as follows:

## STOCK STRATEGIES

Based on the regime summary above, recommend the specific stock strategy that fits:
- **Breakout** — if breadth is expanding, pivot breakouts are holding (low R1 failure rate), OFI is positive, and EMA structure is bullish
- **Mean-Reversion / Fade** — if VIX is elevated, R1 failure rate is high, gamma is positive (dealers suppressing moves), and breadth is mixed
- **Directional Short / Breakdown** — if breadth is deteriorating, support breakdowns are holding (low S1 failure rate), OFI is negative, gamma is negative
- **Sit Out / Reduce Size** — if signals are deeply conflicting with no clear edge

Explain WHY this regime is favored with specific signal references. Include entry/exit guidance.

---

## OPTIONS STRATEGIES

Consider the INTERACTION between these factors:
1. **IV Environment**: VIX level and z-scores tell you if options are cheap or expensive
2. **Directional Bias**: OFI, breadth, EMA structure tell you the likely direction
3. **Gamma Environment**: Positive gamma = mean-reversion (sell premium), negative gamma = momentum (buy premium for directional)
4. **Breakout/Breakdown Quality**: Pivot breadth failure rates tell you if directional moves stick

Based on the interaction, recommend specific options structures:

**High IV + Mean-Reversion Regime (positive gamma, high R1 failure)**:
→ Sell premium: iron condors, strangles, butterflies, credit spreads

**High IV + Directional Regime (negative gamma, breakouts/breakdowns working)**:
→ If bullish: bull put spreads (credit), call debit spreads
→ If bearish: bear call spreads (credit), put debit spreads

**Low IV + Directional Regime**:
→ Buy premium: long calls/puts, debit spreads (cheap options, expect vol expansion)

**Low IV + Range-Bound**:
→ Calendar spreads, diagonal spreads (benefit from vol expansion while range-bound)

For each recommended structure, explain:
- Why THIS structure fits the current regime
- Suggested tenor (weeklies vs monthlies based on VIX term structure)
- Strike selection guidance relative to hedge wall and pivot levels
- Key risk to watch that would invalidate the trade

---

## RISK MANAGEMENT

One paragraph on position sizing and risk given the current volatility regime. Reference VIX level and gamma environment.

Use **bold** for key terms. Be specific and actionable — this is for an experienced systematic trader, not a beginner.

5-Day Trailing Market Signals:
{signals_json}"""
        }]
    )
    result = message.content[0].text
    disk_cache_set("strategies", result, cache_key)
    return result


# --- News / RSS ---
@st.cache_data(ttl=600)  # refresh headlines every 10 minutes
def fetch_rss_headlines():
    """Fetch macro/financial headlines from major free RSS feeds."""
    import xml.etree.ElementTree as ET
    import urllib.request
    from datetime import datetime

    feeds = [
        ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
        ("Reuters Markets", "https://feeds.reuters.com/reuters/marketsNews"),
        ("CNBC Top News", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
        ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("MarketWatch Top", "https://feeds.marketwatch.com/marketwatch/topstories/"),
        ("MarketWatch Markets", "https://feeds.marketwatch.com/marketwatch/marketpulse/"),
    ]

    headlines = []
    for source, url in feeds:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                tree = ET.parse(resp)
                root = tree.getroot()
                for item in root.iter("item"):
                    title = item.findtext("title", "").strip()
                    pub_date = item.findtext("pubDate", "").strip()
                    link = item.findtext("link", "").strip()
                    desc = item.findtext("description", "").strip()
                    if title:
                        headlines.append({
                            "source": source,
                            "title": title,
                            "description": desc[:200] if desc else "",
                            "date": pub_date,
                            "link": link,
                        })
        except Exception:
            continue  # skip failed feeds silently

    return headlines


@st.cache_data(ttl=1800)  # re-curate headlines every 30 minutes
def get_ai_curated_headlines(headlines_json: str, signals_json: str) -> str:
    """Use Claude to filter headlines for regime-relevant macro/geopolitical events."""
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except (KeyError, AttributeError):
            return "**API key not configured.** Set ANTHROPIC_API_KEY in secrets or .env file."
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""You are a senior macro strategist. From the raw news headlines below, select the 5-8 most important MACRO and GEOPOLITICAL headlines that would affect market regime (risk-on/risk-off, volatility, sector rotation, monetary policy).

IGNORE: individual stock earnings, celebrity news, sports, entertainment, minor corporate news, product launches.
PRIORITIZE: Fed/central bank policy, tariffs/trade war, geopolitical conflict, economic data releases (jobs, CPI, GDP), fiscal policy, sovereign debt, oil/commodity shocks, major market moves.

For each selected headline, format EXACTLY like this (with the horizontal rule separator):

**Headline text here** (source.com)

Regime impact sentence here.

---

Use the actual domain of the source (e.g. reuters.com, cnbc.com, marketwatch.com). Each headline MUST be separated by a horizontal rule (---). Do not use bullet points or numbered lists.

Current market context for reference (use to assess impact):
{signals_json}

If no regime-relevant headlines are found, say so.

Raw Headlines:
{headlines_json}"""
        }]
    )
    return message.content[0].text


# --- AI Regime Summary ---
def get_ai_regime_summary(signals_json: str, headlines_json: str = "[]") -> str:
    """Call Claude API to generate a cross-market regime interpretation. Disk-cached by signals (not headlines)."""
    # Cache keyed on signals only — headlines change frequently but shouldn't trigger re-analysis
    cached = disk_cache_get("regime", signals_json)
    if cached:
        return cached
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except (KeyError, AttributeError):
            return "**API key not configured.** Set ANTHROPIC_API_KEY in secrets or .env file."
    client = anthropic.Anthropic(api_key=api_key)

    headlines_section = ""
    if headlines_json and headlines_json != "[]":
        headlines_section = f"""

Additionally, here are recent macro/geopolitical headlines with their publish dates. Use these to CONTEXTUALIZE the quantitative signals — explain WHY signals may be moving the way they are, and whether headlines suggest the current regime is likely to PERSIST, INTENSIFY, or REVERSE.

CRITICAL TIMING RULE: Pay close attention to headline publish dates vs market data dates. Headlines published on weekends or after market close CANNOT have caused the prior trading day's market moves. Do NOT say a weekend headline "coincides with" or "caused" Friday's price action. Instead, frame weekend/after-hours headlines as FORWARD-LOOKING catalysts — e.g., "This weekend's development may impact Monday's session" rather than attributing them to prior moves.

Recent Headlines:
{headlines_json}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""You are a senior systematic trading strategist. Below is a 5-day trailing window of cross-market signals (oldest to newest). The last entry is the most recent day.

Analyze both the CURRENT state and the TRAJECTORY over the past 5 days. Specifically:

1. **Regime classification** — What is the current market regime (risk-on, risk-off, transitional)? Has the regime SHIFTED over the past 5 days, and in which direction? If headlines provide context for WHY, explain the catalyst.
2. **Strategy implications** — Are breakout or mean-reversion strategies favored? Has this changed from earlier in the window? Do current headlines suggest this will persist or shift?
3. **Trajectory & momentum** — Are conditions improving, deteriorating, or stable? Highlight any signals that are trending in a clear direction. Connect moves to macro catalysts where applicable.
4. **Divergences & risks** — Any indicators moving in opposite directions? Any headline risks not yet reflected in the quantitative data?

Keep it concise (4-5 paragraphs). Use **bold** for key terms. Be direct and actionable. When discussing shifts, reference specific day-over-day changes and connect to catalysts.

5-Day Trailing Market Signals (oldest → newest):
{signals_json}{headlines_section}"""
        }]
    )
    result = message.content[0].text
    disk_cache_set("regime", result, signals_json)
    return result


# --- Page Navigation ---
page = st.sidebar.radio("Dashboard", ["Market Overview", "Sectors & Commodities", "VIX", "Put/Call Ratio", "Advance/Decline", "Treasury Yields", "Pivot Breadth", "Order Flow", "Gamma (GEX)", "SPY Technicals"])

# --- Shared date filter helper ---
def date_filter(df, date_col):
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()
    default_start = max(min_date, max_date - pd.Timedelta(days=365))
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if len(date_range) == 2:
        start, end = date_range
    else:
        start, end = default_start, max_date
    mask = (df[date_col].dt.date >= start) & (df[date_col].dt.date <= end)
    return df.loc[mask].copy()


# =====================================================================
# MARKET OVERVIEW PAGE (Regime + Strategies + Headlines)
# =====================================================================
if page == "Market Overview":
    # Override portrait CSS for this page — needs full width for 2-column layout
    st.markdown("""
    <style>
        [data-testid="stAppViewBlockContainer"] {
            max-width: 95% !important;
        }
        .headline-text { color: #888888; font-size: 0.9em; }
        .headline-text strong { color: #aaaaaa; }
        .headline-text hr { border-color: #333333; }
        .left-content p, .left-content li, .left-content span { color: #888888 !important; }
        .left-content strong { color: #aaaaaa !important; }
        .left-content h2 { color: #aaaaaa !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Market Overview")

    # Load all datasets
    import json
    from collections import OrderedDict

    vix_df = load_vix()
    pcr_df = load_pcr()
    ad_df = load_ad()
    ty_df = load_ty()
    pivot_df = load_pivot()
    ofi_df = load_ofi()
    gex_df = load_gex()
    spy_df = load_spy()
    etf_df = load_etf()
    ema_breadth_df = load_ema_breadth()

    # Latest values
    vix_latest = vix_df.iloc[-1]
    pcr_latest = pcr_df.iloc[-1]
    ad_latest = ad_df.iloc[-1]
    ty_latest = ty_df.iloc[-1]
    pivot_latest = pivot_df.iloc[-1]
    ofi_latest = ofi_df.iloc[-1]
    gex_latest = gex_df.iloc[-1]
    spy_latest = spy_df.iloc[-1]

    # Key metrics row
    st.caption(
        f"VIX: {vix_df['Date'].max().strftime('%Y-%m-%d')} | "
        f"PCR: {pcr_df['date'].max().strftime('%Y-%m-%d')} | "
        f"GEX: {gex_df['Date'].max().strftime('%Y-%m-%d')} | "
        f"Pivots: {pivot_df['date'].max().strftime('%Y-%m-%d')} | "
        f"OFI: {ofi_df['date'].max().strftime('%Y-%m-%d')}"
    )
    mcols = st.columns(8)
    mcols[0].metric("VIX", f"{vix_latest['VIX_Close']:.1f}")
    mcols[1].metric("PCR", f"{pcr_latest['PCR']:.2f}")
    mcols[2].metric("Gamma", f"{gex_latest['Agg_Gamma_norm']:+.4f}" if pd.notna(gex_latest['Agg_Gamma_norm']) else "N/A")
    mcols[3].metric("R1 Fail", f"{pivot_latest['pct_failed_RR1']:.0%}" if pd.notna(pivot_latest['pct_failed_RR1']) else "N/A")
    mcols[4].metric("OFI", f"{ofi_latest['day_ofi_ratio']:+.4f}")
    mcols[5].metric("Adv %", f"{ad_latest['Advance_pct']:.0%}" if pd.notna(ad_latest['Advance_pct']) else "N/A")
    mcols[6].metric("RSI", f"{spy_latest['SPY_RSI_14']:.0f}" if pd.notna(spy_latest['SPY_RSI_14']) else "N/A")
    mcols[7].metric("Spread", f"{ty_latest['TY_Diff_2_30']:.2f}%")

    st.divider()

    # --- Build 5-day trailing signals (comprehensive, used by both regime + strategies) ---
    LOOKBACK = 5

    def safe_round(val, decimals):
        return round(val, decimals) if pd.notna(val) else None

    def get_last_n(df, date_col, n):
        return [(row[date_col].strftime("%Y-%m-%d"), row) for _, row in df.tail(n).iterrows()]

    vix_rows = get_last_n(vix_df, "Date", LOOKBACK)
    pcr_rows = get_last_n(pcr_df, "date", LOOKBACK)
    ad_rows = get_last_n(ad_df, "date", LOOKBACK)
    ty_rows = get_last_n(ty_df, "Date", LOOKBACK)
    piv_rows = get_last_n(pivot_df, "date", LOOKBACK)
    ofi_rows = get_last_n(ofi_df, "date", LOOKBACK)
    gex_rows = get_last_n(gex_df, "Date", LOOKBACK)
    spy_rows = get_last_n(spy_df, "date", LOOKBACK)

    all_dates = sorted(set(
        [d for d, _ in vix_rows] + [d for d, _ in pcr_rows] + [d for d, _ in ad_rows] +
        [d for d, _ in ty_rows] + [d for d, _ in piv_rows] + [d for d, _ in ofi_rows] +
        [d for d, _ in gex_rows] + [d for d, _ in spy_rows]
    ))[-LOOKBACK:]

    vix_by_date = {d: r for d, r in vix_rows}
    pcr_by_date = {d: r for d, r in pcr_rows}
    ad_by_date = {d: r for d, r in ad_rows}
    ty_by_date = {d: r for d, r in ty_rows}
    piv_by_date = {d: r for d, r in piv_rows}
    ofi_by_date = {d: r for d, r in ofi_rows}
    gex_by_date = {d: r for d, r in gex_rows}
    spy_by_date = {d: r for d, r in spy_rows}

    trailing = OrderedDict()
    for d in all_dates:
        s = {}
        v = vix_by_date.get(d)
        if v is not None:
            s["VIX"] = safe_round(v["VIX_Close"], 2)
            s["VIX_z10"] = safe_round(v["VIX_z_score_10"], 2)
            s["VIX_z20"] = safe_round(v["VIX_z_score_20"], 2)
            s["VIX_60d_ratio"] = safe_round(v["VIX_60d_ratio"], 3)
            s["VIX_pct_chg"] = safe_round(v["VIX_Pct_Chg"], 4)
        p = pcr_by_date.get(d)
        if p is not None:
            s["PCR"] = safe_round(p["PCR"], 3)
            s["PCR_z"] = safe_round(p["PCR_z_score"], 2)
        a = ad_by_date.get(d)
        if a is not None:
            s["Adv_pct"] = safe_round(a["Advance_pct"], 3)
            s["AD_z5"] = safe_round(a["AD_z_score_5"], 2)
        t = ty_by_date.get(d)
        if t is not None:
            s["TY_2Y"] = safe_round(t["DGS2_1"], 2)
            s["TY_30Y"] = safe_round(t["DGS30_1"], 2)
            s["TY_spread"] = safe_round(t["TY_Diff_2_30"], 2)
            s["TY_spread_5d_chg"] = safe_round(t.get("TY_Diff_Chg_5"), 4)
        pv = piv_by_date.get(d)
        if pv is not None:
            s["Above_PP"] = safe_round(pv["pct_closed_above_PP"], 3)
            s["R1_fail"] = safe_round(pv["pct_failed_RR1"], 3)
            s["R1_close"] = safe_round(pv["pct_closed_above_RR1"], 3)
            s["S1_fail"] = safe_round(pv["pct_failed_bd_SS1"], 3)
            s["S1_close"] = safe_round(pv["pct_closed_below_SS1"], 3)
        o = ofi_by_date.get(d)
        if o is not None:
            s["OFI_ratio"] = safe_round(o["day_ofi_ratio"], 4)
            s["OFI_30m_ratio"] = safe_round(o.get("ofi_ratio_30m"), 4)
            s["VWAP_dev"] = safe_round(o["final_vwap_dev_pct"], 3)
            s["intra_return"] = safe_round(o.get("intra_return_total"), 3)
            s["OR_range"] = safe_round(o.get("or_range"), 2)
        g = gex_by_date.get(d)
        if g is not None:
            s["Agg_Gamma"] = safe_round(g["Agg_Gamma_norm"], 4)
            s["Spot_Gamma"] = safe_round(g["Spot_Gamma_norm"], 4)
            s["Hedge_Wall"] = safe_round(g["Hedge_wall_1"], 0)
        sp = spy_by_date.get(d)
        if sp is not None:
            s["SPY_close"] = safe_round(sp["close"], 2)
            s["RSI_14"] = safe_round(sp["SPY_RSI_14"], 1)
            s["EMA_8_20"] = safe_round(sp["SPY_EMA_8_20_var"], 4)
            s["EMA_20_200"] = safe_round(sp["SPY_EMA_20_200_var"], 4)
            s["CMF"] = safe_round(sp["SPY_Daily_CMF"], 3)
        # SP500 EMA breadth for this date
        ema_row = ema_breadth_df[ema_breadth_df["date"].dt.strftime("%Y-%m-%d") == d]
        if not ema_row.empty:
            r = ema_row.iloc[0]
            s["pct_above_EMA5"] = safe_round(r.get("pct_above_EMA_5"), 3)
            s["pct_above_EMA8"] = safe_round(r.get("pct_above_EMA_8"), 3)
            s["pct_above_EMA20"] = safe_round(r.get("pct_above_EMA_20"), 3)
            s["pct_above_EMA50"] = safe_round(r.get("pct_above_EMA_50"), 3)
            s["pct_above_EMA200"] = safe_round(r.get("pct_above_EMA_200"), 3)

        # Sector/commodity daily returns for this date
        etf_day = etf_df[etf_df["date"].dt.strftime("%Y-%m-%d") == d]
        if not etf_day.empty:
            cyclical_syms = ["XLK", "XLY", "XLF", "XLI", "XLB"]
            defensive_syms = ["XLU", "XLP", "XLV", "XLRE"]
            # Compute 1-day returns relative to previous close
            etf_prev = etf_df[etf_df["date"] < pd.Timestamp(d)]
            if not etf_prev.empty:
                prev_closes = etf_prev.groupby("symbol")["close"].last()
                day_closes = etf_day.set_index("symbol")["close"]
                day_returns = ((day_closes / prev_closes) - 1) * 100
                day_returns = day_returns.dropna()
                cyc_rets = [day_returns.get(sym, np.nan) for sym in cyclical_syms if sym in day_returns]
                def_rets = [day_returns.get(sym, np.nan) for sym in defensive_syms if sym in day_returns]
                s["Cyclical_avg_ret"] = safe_round(np.nanmean(cyc_rets), 2) if cyc_rets else None
                s["Defensive_avg_ret"] = safe_round(np.nanmean(def_rets), 2) if def_rets else None
                # Individual sector returns
                for sym in SECTOR_ETFS:
                    if sym in day_returns:
                        s[f"{sym}_ret"] = safe_round(day_returns[sym], 2)
                s["GLD_ret"] = safe_round(day_returns.get("GLD", np.nan), 2)
                s["USO_ret"] = safe_round(day_returns.get("USO", np.nan), 2)
                s["SLV_ret"] = safe_round(day_returns.get("SLV", np.nan), 2)
                s["UNG_ret"] = safe_round(day_returns.get("UNG", np.nan), 2)
                # Top/bottom sector
                sector_rets = {sym: day_returns.get(sym, np.nan) for sym in SECTOR_ETFS if sym in day_returns}
                if sector_rets:
                    best = max(sector_rets, key=sector_rets.get)
                    worst = min(sector_rets, key=sector_rets.get)
                    s["Best_sector"] = f"{best} ({sector_rets[best]:+.1f}%)"
                    s["Worst_sector"] = f"{worst} ({sector_rets[worst]:+.1f}%)"

        trailing[d] = s

    # Compute recent RSI divergences (last 30 days) for AI context
    spy_recent = spy_df.tail(60).copy().reset_index(drop=True)
    try:
        bull_divs, bear_divs = detect_rsi_divergences(spy_recent)
        cutoff = spy_recent["date"].max() - pd.Timedelta(days=30)
        recent_bull = [d for d in bull_divs if d["date"] >= cutoff]
        recent_bear = [d for d in bear_divs if d["date"] >= cutoff]
        divergence_summary = {
            "recent_bullish_divergences": [
                {"date": d["date"].strftime("%Y-%m-%d"), "rsi": round(d["rsi"], 1)}
                for d in recent_bull
            ],
            "recent_bearish_divergences": [
                {"date": d["date"].strftime("%Y-%m-%d"), "rsi": round(d["rsi"], 1)}
                for d in recent_bear
            ],
        }
    except Exception:
        divergence_summary = {"recent_bullish_divergences": [], "recent_bearish_divergences": []}

    # Combine trailing signals + divergence info
    full_signals = {"trailing_5d": trailing, "rsi_divergences_30d": divergence_summary}
    signals_json = json.dumps(full_signals, indent=2)

    # Fetch headlines
    raw_headlines = fetch_rss_headlines()
    headline_with_dates = [{"title": h["title"], "published": h["date"]} for h in raw_headlines[:40]]
    regime_headlines_json = json.dumps(headline_with_dates, indent=1) if headline_with_dates else "[]"

    # Headlines for curation
    headlines_for_ai = [{"source": h["source"], "title": h["title"], "description": h["description"]}
                       for h in raw_headlines[:60]]
    curate_headlines_json = json.dumps(headlines_for_ai, indent=1)
    context_signals = json.dumps({
        "VIX": round(vix_latest["VIX_Close"], 1),
        "VIX_z10": round(vix_latest["VIX_z_score_10"], 1),
        "PCR": round(pcr_latest["PCR"], 2),
    })

    # --- Two-column layout: Left (Regime + Strategies) | Right (Headlines) ---
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown('<div class="left-content">', unsafe_allow_html=True)

        # === REGIME SUMMARY ===
        regime_ts = disk_cache_timestamp("regime")
        st.subheader("Regime Summary")
        if regime_ts:
            st.caption(f"Last updated: {regime_ts}")
        with st.spinner("Generating regime analysis..."):
            try:
                interpretation = get_ai_regime_summary(signals_json, regime_headlines_json)
                st.markdown(interpretation)
            except Exception as e:
                interpretation = ""
                st.error(f"Regime analysis failed: {e}")

        st.divider()

        # === ACTIONABLE STRATEGIES ===
        strat_ts = disk_cache_timestamp("strategies")
        st.subheader("Actionable Strategies")
        if strat_ts:
            st.caption(f"Last updated: {strat_ts}")
        with st.spinner("Generating strategy recommendations..."):
            try:
                recommendations = get_ai_strategy_recommendations(signals_json, interpretation)
                st.markdown(recommendations)
            except Exception as e:
                st.error(f"Strategy recommendations failed: {e}")

        st.divider()
        with st.expander("Raw Signal Data (5-day trailing)"):
            st.json(trailing)

        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        # === HEADLINES (grayish font) ===
        st.subheader("Headlines")
        if raw_headlines:
            with st.spinner("Curating headlines..."):
                try:
                    curated = get_ai_curated_headlines(curate_headlines_json, context_signals)
                    st.markdown(f'<div class="headline-text">{curated}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Headline curation failed: {e}")
                    for h in raw_headlines[:10]:
                        st.markdown(f'<div class="headline-text"><strong>{h["title"]}</strong> ({h["source"]})</div>', unsafe_allow_html=True)

            st.divider()
            with st.expander(f"All Raw Headlines ({len(raw_headlines)})"):
                for h in raw_headlines:
                    st.markdown(f'<span class="headline-text">- **{h["title"]}** — *{h["source"]}*</span>', unsafe_allow_html=True)
        else:
            st.warning("Could not fetch headlines.")


# =====================================================================
# SECTORS & COMMODITIES PAGE
# =====================================================================
elif page == "Sectors & Commodities":
    st.title("Sectors & Commodities")
    df = load_etf()
    st.caption(f"Last data: **{df['date'].max().strftime('%Y-%m-%d')}**")

    # Lookback period toggle — inline on the page
    lookback_options = {
        "1W": 7, "2W": 14, "1M": 30, "3M": 90,
        "6M": 180, "12M": 365, "All": None,
    }
    lookback = st.radio("Lookback", list(lookback_options.keys()), index=3, horizontal=True)
    max_dt = df["date"].max()
    days = lookback_options[lookback]
    if days is not None:
        start_dt = max_dt - pd.Timedelta(days=days)
    else:
        start_dt = df["date"].min()

    mask = (df["date"] >= start_dt) & (df["date"] <= max_dt)
    filtered = df.loc[mask].copy()

    # Compute cumulative returns per symbol from start of window
    def calc_cum_returns(data, symbols_dict):
        cum_ret = {}
        for sym, label in symbols_dict.items():
            sub = data[data["symbol"] == sym].sort_values("date")
            if len(sub) > 1:
                base = sub["close"].iloc[0]
                cum_ret[sym] = {
                    "label": label,
                    "dates": sub["date"].values,
                    "cum_return": ((sub["close"] / base) - 1).values * 100,
                    "last_return": ((sub["close"].iloc[-1] / base) - 1) * 100,
                }
        return cum_ret

    sector_returns = calc_cum_returns(filtered, SECTOR_ETFS)
    commodity_returns = calc_cum_returns(filtered, COMMODITY_ETFS)

    # Key metrics — top/bottom sectors and commodities
    if sector_returns and commodity_returns:
        sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1]["last_return"], reverse=True)
        sorted_commodities = sorted(commodity_returns.items(), key=lambda x: x[1]["last_return"], reverse=True)

        cols = st.columns(4)
        cols[0].metric(f"Best Sector", f"{sorted_sectors[0][1]['label']}",
                       delta=f"{sorted_sectors[0][1]['last_return']:+.1f}%")
        cols[1].metric(f"Worst Sector", f"{sorted_sectors[-1][1]['label']}",
                       delta=f"{sorted_sectors[-1][1]['last_return']:+.1f}%")
        cols[2].metric(f"Best Commodity", f"{sorted_commodities[0][1]['label']}",
                       delta=f"{sorted_commodities[0][1]['last_return']:+.1f}%")
        cols[3].metric(f"Worst Commodity", f"{sorted_commodities[-1][1]['label']}",
                       delta=f"{sorted_commodities[-1][1]['last_return']:+.1f}%")

        # Interpretation
        # Defensive vs cyclical check
        cyclical = ["XLK", "XLY", "XLF", "XLI", "XLB"]
        defensive = ["XLU", "XLP", "XLV", "XLRE"]
        cyc_avg = np.mean([sector_returns[s]["last_return"] for s in cyclical if s in sector_returns])
        def_avg = np.mean([sector_returns[s]["last_return"] for s in defensive if s in sector_returns])
        energy_ret = sector_returns.get("XLE", {}).get("last_return", 0)
        gold_ret = commodity_returns.get("GLD", {}).get("last_return", 0)

        lines = ["**Sector & Commodity Interpretation**\n"]
        if cyc_avg > def_avg + 2:
            lines.append(f"Cyclicals ({cyc_avg:+.1f}%) are **outperforming** defensives ({def_avg:+.1f}%) — a **risk-on** signal. Capital is flowing into growth-sensitive sectors, consistent with economic expansion or easing financial conditions.")
        elif def_avg > cyc_avg + 2:
            lines.append(f"Defensives ({def_avg:+.1f}%) are **outperforming** cyclicals ({cyc_avg:+.1f}%) — a **risk-off** signal. Flight to safety into utilities, staples, and healthcare suggests investors are positioning for economic slowdown.")
        else:
            lines.append(f"Cyclicals ({cyc_avg:+.1f}%) and defensives ({def_avg:+.1f}%) are **roughly in line** — no strong sector rotation signal. Market is not clearly favoring risk-on or risk-off sectors.")

        if gold_ret > 5:
            lines.append(f"Gold at **{gold_ret:+.1f}%** signals **safe-haven demand**. Strong gold performance alongside equity weakness confirms risk-off positioning. If gold is rising with equities, it may reflect inflation hedging.")
        elif gold_ret < -5:
            lines.append(f"Gold at **{gold_ret:+.1f}%** suggests **risk appetite** — investors are selling safe havens in favor of growth assets.")

        if energy_ret > 10:
            lines.append(f"Energy at **{energy_ret:+.1f}%** is a standout — driven by oil/geopolitical factors. Strong energy outperformance can be inflationary and may pressure the Fed toward tightening.")

        st.info("\n\n".join(lines))

    # Chart 1: Sector Cumulative Returns
    st.subheader("Sector ETF Cumulative Returns")
    fig_sec = go.Figure()
    sector_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8"]
    for i, (sym, data) in enumerate(sorted(sector_returns.items(), key=lambda x: x[1]["last_return"], reverse=True)):
        fig_sec.add_trace(go.Scatter(
            x=data["dates"], y=data["cum_return"],
            name=f"{sym} ({data['label']})",
            line=dict(width=1.3, color=sector_colors[i % len(sector_colors)]),
        ))
    fig_sec.add_hline(y=0, line_color="gray", line_width=0.5)
    fig_sec.update_layout(height=500, xaxis_title="Date", yaxis_title="Cumulative Return (%)",
                          hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_sec, df["date"].max())
    st.plotly_chart(fig_sec, use_container_width=True)

    # Chart 2: Commodity Cumulative Returns
    st.subheader("Commodity ETF Cumulative Returns")
    fig_com = go.Figure()
    commodity_colors = ["#FFD700", "#C0C0C0", "#2ca02c", "#d62728", "#ff7f0e",
                        "#B87333", "#DAA520", "#8B4513", "#F5DEB3", "#9467bd", "#17becf"]
    for i, (sym, data) in enumerate(sorted(commodity_returns.items(), key=lambda x: x[1]["last_return"], reverse=True)):
        fig_com.add_trace(go.Scatter(
            x=data["dates"], y=data["cum_return"],
            name=f"{sym} ({data['label']})",
            line=dict(width=1.3, color=commodity_colors[i % len(commodity_colors)]),
        ))
    fig_com.add_hline(y=0, line_color="gray", line_width=0.5)
    fig_com.update_layout(height=500, xaxis_title="Date", yaxis_title="Cumulative Return (%)",
                          hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_com, df["date"].max())
    st.plotly_chart(fig_com, use_container_width=True)

    # Table: Period Returns Summary
    with st.expander("Period Return Rankings"):
        rankings = []
        for sym, data in {**sector_returns, **commodity_returns}.items():
            cat = "Sector" if sym in SECTOR_ETFS else "Commodity"
            rankings.append({"Symbol": sym, "Name": data["label"], "Type": cat,
                            "Return (%)": round(data["last_return"], 2)})
        rank_df = pd.DataFrame(rankings).sort_values("Return (%)", ascending=False)
        st.dataframe(rank_df, use_container_width=True, hide_index=True)


# =====================================================================
# VIX PAGE
# =====================================================================
elif page == "VIX":
    st.title("VIX Volatility Dashboard")
    df = load_vix()
    filtered = date_filter(df, "Date")
    st.caption(f"Last data: **{df['Date'].max().strftime('%Y-%m-%d')}**")

    st.sidebar.subheader("Moving Averages")
    show_ma10 = st.sidebar.checkbox("10-day MA", value=True)
    show_ma20 = st.sidebar.checkbox("20-day MA", value=True)
    show_ma60 = st.sidebar.checkbox("60-day MA", value=False)
    show_bands = st.sidebar.checkbox("Bollinger Bands (MA20 +/- 1 std)", value=False)

    # Key Metrics
    if not filtered.empty:
        latest = filtered.iloc[-1]
        vix_now = latest["VIX_Close"]
        if vix_now < 15:
            regime = "Low Vol"
        elif vix_now < 20:
            regime = "Normal"
        elif vix_now < 30:
            regime = "Elevated"
        else:
            regime = "High Vol"

        cols = st.columns(5)
        cols[0].metric("VIX Level", f"{vix_now:.2f}")
        cols[1].metric("Regime", regime)
        cols[2].metric("10d Z-Score", f"{latest['VIX_z_score_10']:.2f}")
        cols[3].metric("20d Z-Score", f"{latest['VIX_z_score_20']:.2f}")
        cols[4].metric("60d Ratio", f"{latest['VIX_60d_ratio']:.3f}")

        z10 = latest["VIX_z_score_10"]
        z20 = latest["VIX_z_score_20"]
        ratio60 = latest["VIX_60d_ratio"]
        lines = ["**Regime Interpretation**\n"]

        # VIX level context
        if vix_now >= 30:
            lines.append("VIX above 30 signals **crisis-level fear**. Markets are pricing in large moves. Historically associated with capitulation bottoms or sustained drawdowns. Option premiums are extremely elevated.")
        elif vix_now >= 20:
            lines.append(f"VIX at {vix_now:.1f} indicates **elevated uncertainty**. The market is pricing in above-average risk. Hedging costs are elevated and directional bets carry wider stop distances.")
        elif vix_now >= 15:
            lines.append(f"VIX at {vix_now:.1f} reflects **normal market conditions**. Implied vol is in a typical range. Trend-following strategies tend to perform well in this regime.")
        else:
            lines.append(f"VIX below 15 signals **complacency**. Low vol regimes can persist but often end abruptly. Mean-reversion strategies on VIX tend to underperform here; watch for vol expansion catalysts.")

        # Z-score context
        if z10 > 2 or z20 > 2:
            lines.append(f"Z-scores ({z10:+.1f} / {z20:+.1f}) show a **sharp vol spike** relative to recent history. This often precedes short-term mean reversion in VIX, but can also mark the start of a sustained vol regime if macro catalysts persist.")
        elif z10 < -1.5 or z20 < -1.5:
            lines.append(f"Negative z-scores ({z10:+.1f} / {z20:+.1f}) indicate VIX is **well below recent averages** -- vol compression. This is a setup for potential vol expansion; consider tail hedges.")
        elif abs(z10) < 0.5 and abs(z20) < 0.5:
            lines.append(f"Z-scores near zero ({z10:+.1f} / {z20:+.1f}) mean VIX is **in line with recent norms**. No extreme positioning signal from vol.")

        # 60d ratio context
        if ratio60 > 1.3:
            lines.append(f"60d ratio at {ratio60:.2f} confirms VIX is **running hot** vs its longer-term average. Sustained readings above 1.3 are rare and typically resolve with either a VIX drop or the 60d MA catching up.")
        elif ratio60 < 0.8:
            lines.append(f"60d ratio at {ratio60:.2f} shows VIX is **depressed** vs its 60d average. This often occurs in late-stage rallies before volatility re-emerges.")

        st.info("\n\n".join(lines))

    # Chart 1: VIX Level + MAs
    st.subheader("VIX Level & Moving Averages")
    fig1 = go.Figure()
    for y0, y1, color, label in [
        (0, 15, "rgba(76,175,80,0.08)", "Low Vol"),
        (15, 20, "rgba(255,193,7,0.08)", "Normal"),
        (20, 30, "rgba(255,87,34,0.08)", "Elevated"),
        (30, 90, "rgba(183,28,28,0.08)", "High Vol"),
    ]:
        fig1.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0,
                       annotation_text=label, annotation_position="top left",
                       annotation_font_size=10, annotation_font_color="gray")
    fig1.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["VIX_Close"],
        name="VIX Close", line=dict(color="#1f77b4", width=1.5),
    ))
    if show_ma10:
        fig1.add_trace(go.Scatter(
            x=filtered["Date"], y=filtered["VIX_MA_10"],
            name="MA 10", line=dict(color="#ff7f0e", width=1, dash="dot"),
        ))
    if show_ma20:
        fig1.add_trace(go.Scatter(
            x=filtered["Date"], y=filtered["VIX_MA_20"],
            name="MA 20", line=dict(color="#2ca02c", width=1, dash="dot"),
        ))
    if show_ma60:
        fig1.add_trace(go.Scatter(
            x=filtered["Date"], y=filtered["VIX_MA_60"],
            name="MA 60", line=dict(color="#d62728", width=1, dash="dash"),
        ))
    if show_bands:
        upper = filtered["VIX_MA_20"] + filtered["VIX_st_dev_20"]
        lower = filtered["VIX_MA_20"] - filtered["VIX_st_dev_20"]
        fig1.add_trace(go.Scatter(
            x=filtered["Date"], y=upper, name="Upper Band",
            line=dict(width=0), showlegend=False,
        ))
        fig1.add_trace(go.Scatter(
            x=filtered["Date"], y=lower, name="Lower Band",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(44,160,44,0.12)", showlegend=False,
        ))
    fig1.update_layout(height=450, xaxis_title="Date", yaxis_title="VIX",
                       hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig1, df["Date"].max())
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Z-Scores
    st.subheader("VIX Z-Scores (10d & 20d)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=filtered["Date"], y=filtered["VIX_z_score_10"],
                              name="Z-Score 10d", line=dict(color="#1f77b4", width=1.2)))
    fig2.add_trace(go.Scatter(x=filtered["Date"], y=filtered["VIX_z_score_20"],
                              name="Z-Score 20d", line=dict(color="#ff7f0e", width=1.2)))
    for level in [-2, -1, 1, 2]:
        fig2.add_hline(y=level, line_dash="dash", line_color="gray", line_width=0.5,
                       annotation_text=f"{level:+d}", annotation_position="bottom right")
    fig2.add_hline(y=0, line_color="black", line_width=0.5)
    fig2.update_layout(height=350, xaxis_title="Date", yaxis_title="Z-Score",
                       hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig2, df["Date"].max())
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Daily % Change
    st.subheader("VIX Daily % Change")
    fig3 = go.Figure()
    colors = [
        "rgba(183,28,28,0.7)" if v > 0.1 else
        "rgba(255,87,34,0.6)" if v > 0 else
        "rgba(76,175,80,0.6)" if v > -0.1 else
        "rgba(27,94,32,0.7)"
        for v in filtered["VIX_Pct_Chg"]
    ]
    fig3.add_trace(go.Bar(x=filtered["Date"], y=filtered["VIX_Pct_Chg"],
                          marker_color=colors, name="Daily Chg"))
    fig3.add_hline(y=0, line_color="black", line_width=0.5)
    fig3.update_layout(height=300, xaxis_title="Date", yaxis_title="% Change",
                       yaxis_tickformat=".0%", hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig3, df["Date"].max())
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: 60d Ratio
    st.subheader("VIX / 60d MA Ratio")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=filtered["Date"], y=filtered["VIX_60d_ratio"],
                              name="60d Ratio", line=dict(color="#9467bd", width=1.2),
                              fill="tozeroy", fillcolor="rgba(148,103,189,0.1)"))
    fig4.add_hline(y=1.0, line_dash="dash", line_color="black", line_width=1,
                   annotation_text="1.0 (neutral)", annotation_position="bottom right")
    fig4.update_layout(height=300, xaxis_title="Date", yaxis_title="Ratio",
                       hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig4, df["Date"].max())
    st.plotly_chart(fig4, use_container_width=True)


# =====================================================================
# PUT/CALL RATIO PAGE
# =====================================================================
elif page == "Put/Call Ratio":
    st.title("SPY Put/Call Ratio Dashboard")
    df = load_pcr()
    filtered = date_filter(df, "date")
    st.caption(f"Last data: **{df['date'].max().strftime('%Y-%m-%d')}**")

    st.sidebar.subheader("Options")
    show_pcr_ma = st.sidebar.checkbox("20-day MA", value=True)
    show_pcr_bands = st.sidebar.checkbox("Bollinger Bands (MA20 +/- 1 std)", value=False)

    # Key Metrics
    if not filtered.empty:
        latest = filtered.iloc[-1]
        pcr_now = latest["PCR"]
        if pcr_now < 0.7:
            sentiment = "Bullish"
        elif pcr_now < 1.0:
            sentiment = "Neutral"
        elif pcr_now < 1.3:
            sentiment = "Cautious"
        else:
            sentiment = "Fearful"

        cols = st.columns(5)
        cols[0].metric("PCR", f"{pcr_now:.3f}")
        cols[1].metric("Sentiment", sentiment)
        cols[2].metric("20d MA", f"{latest['PCR_ma_20']:.3f}")
        cols[3].metric("Z-Score", f"{latest['PCR_z_score']:.2f}")
        cols[4].metric("Total Volume", f"{latest['total_volume']:,.0f}")

        pcr_z = latest["PCR_z_score"]
        pcr_ma = latest["PCR_ma_20"]
        lines = ["**Sentiment Interpretation**\n"]

        # PCR level
        if pcr_now >= 1.3:
            lines.append(f"PCR at {pcr_now:.2f} shows **heavy put buying** relative to calls. This level of fear often acts as a **contrarian bullish signal** -- when everyone is hedged, sell-offs tend to be cushioned and snapback rallies are common.")
        elif pcr_now >= 1.0:
            lines.append(f"PCR at {pcr_now:.2f} indicates **cautious positioning** -- more puts than calls being traded. Participants are hedging, which can either reflect rational risk management ahead of a known event or growing anxiety about downside.")
        elif pcr_now >= 0.7:
            lines.append(f"PCR at {pcr_now:.2f} is in the **neutral zone**. Options flow shows balanced sentiment with no extreme skew. This is typical of range-bound or trending markets with no panic.")
        else:
            lines.append(f"PCR below 0.7 signals **aggressive call buying** and speculative bullishness. Historically, extreme low PCR readings are a **contrarian bearish signal** -- when protection is cheap, few are buying it, and markets are vulnerable to shocks.")

        # Z-score context
        if pcr_z > 2:
            lines.append(f"Z-score at {pcr_z:+.1f} is **2+ standard deviations above the 20d mean** -- an extreme fear reading. Peak fear readings historically cluster near market bottoms. Watch for z-score reversal as a buy signal.")
        elif pcr_z < -2:
            lines.append(f"Z-score at {pcr_z:+.1f} is **2+ standard deviations below the 20d mean** -- extreme complacency. Markets are under-hedged. A catalyst could trigger outsized downside moves.")
        elif pcr_z > 1:
            lines.append(f"Z-score at {pcr_z:+.1f} shows **above-average put activity**. Hedging demand is elevated but not at extreme levels yet.")
        elif pcr_z < -1:
            lines.append(f"Z-score at {pcr_z:+.1f} shows **below-average put protection**. Participants are leaning bullish, reducing hedges.")

        # Trend context
        if pcr_now > pcr_ma * 1.1:
            lines.append(f"PCR is trading **above its 20d MA** ({pcr_ma:.2f}), suggesting a recent shift toward defensiveness. If this persists, it may indicate a transition to a risk-off regime.")
        elif pcr_now < pcr_ma * 0.9:
            lines.append(f"PCR is trading **below its 20d MA** ({pcr_ma:.2f}), reflecting a recent shift toward risk appetite. Call-heavy flow favors continuation of bullish momentum.")

        st.info("\n\n".join(lines))

    # Chart 1: PCR with MA and Bands
    st.subheader("Put/Call Ratio")
    fig_pcr = go.Figure()

    # Sentiment shading
    for y0, y1, color, label in [
        (0, 0.7, "rgba(76,175,80,0.08)", "Bullish"),
        (0.7, 1.0, "rgba(255,193,7,0.06)", "Neutral"),
        (1.0, 1.3, "rgba(255,152,0,0.08)", "Cautious"),
        (1.3, 3.0, "rgba(183,28,28,0.08)", "Fearful"),
    ]:
        fig_pcr.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0,
                          annotation_text=label, annotation_position="top left",
                          annotation_font_size=10, annotation_font_color="gray")

    fig_pcr.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["PCR"],
        name="PCR", line=dict(color="#1f77b4", width=1.2),
    ))
    if show_pcr_ma:
        fig_pcr.add_trace(go.Scatter(
            x=filtered["date"], y=filtered["PCR_ma_20"],
            name="MA 20", line=dict(color="#ff7f0e", width=1, dash="dot"),
        ))
    if show_pcr_bands:
        upper = filtered["PCR_ma_20"] + filtered["PCR_st_dev_20"]
        lower = filtered["PCR_ma_20"] - filtered["PCR_st_dev_20"]
        fig_pcr.add_trace(go.Scatter(
            x=filtered["date"], y=upper, name="Upper Band",
            line=dict(width=0), showlegend=False,
        ))
        fig_pcr.add_trace(go.Scatter(
            x=filtered["date"], y=lower, name="Lower Band",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(44,160,44,0.12)", showlegend=False,
        ))
    fig_pcr.add_hline(y=1.0, line_dash="dash", line_color="gray", line_width=0.5,
                      annotation_text="1.0", annotation_position="bottom right")
    fig_pcr.update_layout(height=450, xaxis_title="Date", yaxis_title="Put/Call Ratio",
                          hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_pcr, df["date"].max())
    st.plotly_chart(fig_pcr, use_container_width=True)

    # Chart 2: PCR Z-Score
    st.subheader("PCR Z-Score (20d)")
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["PCR_z_score"],
        name="Z-Score", line=dict(color="#d62728", width=1.2),
    ))
    for level in [-2, -1, 1, 2]:
        fig_z.add_hline(y=level, line_dash="dash", line_color="gray", line_width=0.5,
                        annotation_text=f"{level:+d}", annotation_position="bottom right")
    fig_z.add_hline(y=0, line_color="black", line_width=0.5)
    fig_z.update_layout(height=350, xaxis_title="Date", yaxis_title="Z-Score",
                        hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_z, df["date"].max())
    st.plotly_chart(fig_z, use_container_width=True)

    # Chart 3: Call vs Put Volume
    st.subheader("Call vs Put Volume")
    fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
    fig_vol.add_trace(go.Bar(
        x=filtered["date"], y=filtered["call_vol"],
        name="Call Volume", marker_color="rgba(76,175,80,0.5)",
    ), secondary_y=False)
    fig_vol.add_trace(go.Bar(
        x=filtered["date"], y=filtered["put_vol"],
        name="Put Volume", marker_color="rgba(183,28,28,0.5)",
    ), secondary_y=False)
    fig_vol.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["PCR"],
        name="PCR", line=dict(color="#1f77b4", width=1.5),
    ), secondary_y=True)
    fig_vol.update_layout(height=400, barmode="group", hovermode="x unified",
                          margin=CHART_MARGIN)
    fig_vol.update_yaxes(title_text="Volume", secondary_y=False)
    fig_vol.update_yaxes(title_text="PCR", secondary_y=True)
    stamp_last_date(fig_vol, df["date"].max())
    st.plotly_chart(fig_vol, use_container_width=True)

    # Chart 4: Total Options Volume
    st.subheader("Total Options Volume")
    fig_tv = go.Figure()
    fig_tv.add_trace(go.Bar(
        x=filtered["date"], y=filtered["total_volume"],
        name="Total Volume", marker_color="rgba(100,149,237,0.5)",
    ))
    fig_tv.update_layout(height=300, xaxis_title="Date", yaxis_title="Volume",
                         hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_tv, df["date"].max())
    st.plotly_chart(fig_tv, use_container_width=True)


# =====================================================================
# ADVANCE / DECLINE PAGE
# =====================================================================
elif page == "Advance/Decline":
    st.title("Market Breadth — Advance/Decline")
    df = load_ad()
    filtered = date_filter(df, "date")
    st.caption(f"Last data: **{df['date'].max().strftime('%Y-%m-%d')}**")

    st.sidebar.subheader("Options")
    show_ad_ma5 = st.sidebar.checkbox("5-day MA", value=True)
    show_ad_ma20 = st.sidebar.checkbox("20-day MA", value=True)

    # Key Metrics
    if not filtered.empty:
        latest = filtered.iloc[-1]
        ad_now = latest["AD"]
        adv_pct = latest["Advance_pct"]
        if adv_pct > 0.65:
            breadth = "Strong"
        elif adv_pct > 0.5:
            breadth = "Positive"
        elif adv_pct > 0.35:
            breadth = "Negative"
        else:
            breadth = "Weak"

        cols = st.columns(5)
        cols[0].metric("A-D", f"{ad_now:+.0f}")
        cols[1].metric("Breadth", breadth)
        cols[2].metric("Advance %", f"{adv_pct:.1%}" if pd.notna(adv_pct) else "N/A")
        cols[3].metric("5d Z-Score", f"{latest['AD_z_score_5']:.2f}" if pd.notna(latest["AD_z_score_5"]) else "N/A")
        cols[4].metric("20d Z-Score", f"{latest['AD_z_score_20']:.2f}" if pd.notna(latest["AD_z_score_20"]) else "N/A")

        z5 = latest["AD_z_score_5"] if pd.notna(latest["AD_z_score_5"]) else 0
        z20 = latest["AD_z_score_20"] if pd.notna(latest["AD_z_score_20"]) else 0
        ad_line = latest["AD_Line"]
        # Check A/D line trend (compare to 20d MA)
        ad_ma20 = latest["AD_ma_20"] if pd.notna(latest["AD_ma_20"]) else ad_line
        lines = ["**Breadth Interpretation**\n"]

        # Advance % context
        if pd.notna(adv_pct):
            if adv_pct > 0.65:
                lines.append(f"Advance % at {adv_pct:.0%} shows **broad-based participation** -- a strong majority of issues are rising. This confirms uptrend health. Rallies with wide breadth are more sustainable than narrow, cap-weighted moves.")
            elif adv_pct > 0.5:
                lines.append(f"Advance % at {adv_pct:.0%} is **modestly positive** -- more stocks advancing than declining, but not overwhelmingly. Consistent with a healthy but not euphoric market.")
            elif adv_pct > 0.35:
                lines.append(f"Advance % at {adv_pct:.0%} shows **more decliners than advancers**. If the index is flat or up, this is a **breadth divergence** -- a warning that the rally may be driven by a few large-caps while the broader market weakens.")
            else:
                lines.append(f"Advance % at {adv_pct:.0%} signals **broad-based selling**. A washout reading like this can be a precursor to a short-term bounce (oversold breadth), but in trending bear markets, it confirms downside momentum.")

        # A/D Line trend
        if ad_line > ad_ma20 * 1.01:
            lines.append("The A/D line is **above its 20d MA**, confirming healthy breadth momentum. Rising A/D line alongside rising index prices is the strongest confirmation of a sustainable uptrend.")
        elif ad_line < ad_ma20 * 0.99:
            lines.append("The A/D line is **below its 20d MA**, signaling deteriorating breadth. If the index continues higher while the A/D line falls, this is a classic **bearish divergence** -- often a leading indicator of a market top.")

        # Z-score extremes
        if z5 > 2 or z20 > 2:
            lines.append(f"Z-scores ({z5:+.1f} / {z20:+.1f}) indicate a **breadth thrust** -- an unusually strong advance day. Breadth thrusts at the start of a rally often signal the beginning of a new uptrend leg.")
        elif z5 < -2 or z20 < -2:
            lines.append(f"Z-scores ({z5:+.1f} / {z20:+.1f}) show an **extreme breadth washout**. Historically, deeply negative breadth z-scores often coincide with short-term bottoms as selling becomes exhausted.")

        st.info("\n\n".join(lines))

    # Chart 1: A/D Line with MAs
    st.subheader("Cumulative A/D Line (rebased to start of window)")
    fig_ad1 = go.Figure()
    # Rebase so the first visible value = 0
    ad_base = filtered["AD_Line"].iloc[0] if not filtered.empty else 0
    fig_ad1.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["AD_Line"] - ad_base,
        name="A/D Line", line=dict(color="#1f77b4", width=1.5),
    ))
    if show_ad_ma5:
        fig_ad1.add_trace(go.Scatter(
            x=filtered["date"], y=filtered["AD_ma_5"] - ad_base,
            name="MA 5", line=dict(color="#ff7f0e", width=1, dash="dot"),
        ))
    if show_ad_ma20:
        fig_ad1.add_trace(go.Scatter(
            x=filtered["date"], y=filtered["AD_ma_20"] - ad_base,
            name="MA 20", line=dict(color="#2ca02c", width=1, dash="dot"),
        ))
    fig_ad1.add_hline(y=0, line_color="gray", line_width=0.5)
    fig_ad1.update_layout(height=450, xaxis_title="Date", yaxis_title="A/D Line (rebased)",
                          hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ad1, df["date"].max())
    st.plotly_chart(fig_ad1, use_container_width=True)

    # Chart 2: Daily A-D (bar)
    st.subheader("Daily Advances - Declines")
    fig_ad2 = go.Figure()
    colors_ad = [
        "rgba(76,175,80,0.6)" if v >= 0 else "rgba(183,28,28,0.6)"
        for v in filtered["AD"]
    ]
    fig_ad2.add_trace(go.Bar(
        x=filtered["date"], y=filtered["AD"],
        marker_color=colors_ad, name="A-D",
    ))
    fig_ad2.add_hline(y=0, line_color="black", line_width=0.5)
    fig_ad2.update_layout(height=350, xaxis_title="Date", yaxis_title="Advances - Declines",
                          hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ad2, df["date"].max())
    st.plotly_chart(fig_ad2, use_container_width=True)

    # Chart 3: Advance % with rolling average
    st.subheader("Advance % (with 20d MA)")
    fig_ad3 = go.Figure()
    fig_ad3.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["Advance_pct"],
        name="Advance %", line=dict(color="#1f77b4", width=1.3),
    ))
    # Compute 20d MA of Advance_pct on the fly
    adv_ma20 = filtered["Advance_pct"].rolling(20, min_periods=1).mean()
    fig_ad3.add_trace(go.Scatter(
        x=filtered["date"], y=adv_ma20,
        name="20d MA", line=dict(color="#ff7f0e", width=1, dash="dot"),
    ))
    fig_ad3.add_hline(y=0.5, line_dash="dash", line_color="gray", line_width=0.5,
                      annotation_text="50%", annotation_position="bottom right")
    fig_ad3.update_layout(height=350, xaxis_title="Date", yaxis_title="Advance %",
                          yaxis_tickformat=".0%", hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ad3, df["date"].max())
    st.plotly_chart(fig_ad3, use_container_width=True)

    # Chart 4: Z-Scores (5d & 20d)
    st.subheader("A/D Z-Scores (5d & 20d)")
    fig_ad4 = go.Figure()
    fig_ad4.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["AD_z_score_5"],
        name="Z-Score 5d", line=dict(color="#1f77b4", width=1.2),
    ))
    fig_ad4.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["AD_z_score_20"],
        name="Z-Score 20d", line=dict(color="#ff7f0e", width=1.2),
    ))
    for level in [-2, -1, 1, 2]:
        fig_ad4.add_hline(y=level, line_dash="dash", line_color="gray", line_width=0.5,
                          annotation_text=f"{level:+d}", annotation_position="bottom right")
    fig_ad4.add_hline(y=0, line_color="black", line_width=0.5)
    fig_ad4.update_layout(height=350, xaxis_title="Date", yaxis_title="Z-Score",
                          hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ad4, df["date"].max())
    st.plotly_chart(fig_ad4, use_container_width=True)


# =====================================================================
# TREASURY YIELDS PAGE
# =====================================================================
elif page == "Treasury Yields":
    st.title("Treasury Yields & Yield Curve")
    df = load_ty()
    filtered = date_filter(df, "Date")
    st.caption(f"Last data: **{df['Date'].max().strftime('%Y-%m-%d')}**")

    # Key Metrics
    if not filtered.empty:
        latest = filtered.iloc[-1]
        spread = latest["TY_Diff_2_30"]
        if spread < 0:
            curve_state = "Inverted"
        elif spread < 0.5:
            curve_state = "Flat"
        elif spread < 1.5:
            curve_state = "Normal"
        else:
            curve_state = "Steep"

        cols = st.columns(5)
        cols[0].metric("2Y Yield", f"{latest['DGS2_1']:.2f}%")
        cols[1].metric("30Y Yield", f"{latest['DGS30_1']:.2f}%")
        cols[2].metric("2-30 Spread", f"{spread:.2f}%")
        cols[3].metric("Curve", curve_state)
        cols[4].metric("5d Spread Chg", f"{latest['TY_Diff_Chg_5']:+.4f}" if pd.notna(latest["TY_Diff_Chg_5"]) else "N/A")

        chg5 = latest["TY_Diff_Chg_5"] if pd.notna(latest["TY_Diff_Chg_5"]) else 0
        chg20 = latest["TY_Diff_Chg_20"] if pd.notna(latest["TY_Diff_Chg_20"]) else 0
        y2 = latest["DGS2_1"]
        y30 = latest["DGS30_1"]
        lines = ["**Yield Curve Interpretation**\n"]

        # Curve shape
        if spread < -0.5:
            lines.append(f"The 2-30 spread at {spread:+.2f}% is **deeply inverted**. This is one of the most reliable recession indicators in macro. An inverted curve signals that the bond market expects rate cuts ahead due to economic weakness. Equities tend to face headwinds 6-18 months after sustained inversion.")
        elif spread < 0:
            lines.append(f"The 2-30 spread at {spread:+.2f}% is **inverted**. Short rates exceeding long rates indicates the bond market is pricing in tighter monetary conditions than the economy can sustain. Historically, inversions precede recessions, though the lag can be long.")
        elif spread < 0.5:
            lines.append(f"The 2-30 spread at {spread:.2f}% is **flat**. A flat curve suggests the market sees limited growth differentiation between the short and long term. This is often a transitional state -- either steepening toward growth optimism or flattening toward inversion risk.")
        elif spread < 1.5:
            lines.append(f"The 2-30 spread at {spread:.2f}% is **normally sloped**. This reflects healthy expectations: the term premium compensates for duration risk, and the market expects stable to improving growth. This is the most favorable yield curve environment for equities and risk assets.")
        else:
            lines.append(f"The 2-30 spread at {spread:.2f}% is **steep**. A steep curve often occurs after aggressive Fed easing (low short rates) while long rates price in recovery or inflation. This is typically bullish for cyclicals, financials, and value stocks that benefit from higher long rates.")

        # Rate of change
        if chg5 > 0.05:
            lines.append(f"The spread is **steepening rapidly** (5d chg: {chg5:+.3f}). Fast steepening can signal a flight from long-duration bonds (inflation fear) or expectations of Fed easing at the front end. Watch whether the steepening is bear (both rising, 30Y faster) or bull (both falling, 2Y faster).")
        elif chg5 < -0.05:
            lines.append(f"The spread is **compressing rapidly** (5d chg: {chg5:+.3f}). Fast flattening suggests the market is pricing in tighter policy or slower growth. If driven by rising 2Y yields, this reflects hawkish Fed expectations.")

        # Absolute level context
        if y2 > 5:
            lines.append(f"The 2Y yield at {y2:.2f}% is **historically elevated**. High front-end rates create competition for equities (cash and short-term bonds offer attractive returns) and increase financing costs for leveraged positions.")
        elif y2 < 1:
            lines.append(f"The 2Y yield at {y2:.2f}% signals **extreme monetary accommodation**. Near-zero front-end rates push capital toward risk assets (TINA -- there is no alternative). This is the most favorable rate environment for equity multiples.")

        st.info("\n\n".join(lines))

    # Chart 1: 2Y and 30Y Yields
    st.subheader("2-Year & 30-Year Treasury Yields")
    fig_ty1 = go.Figure()
    fig_ty1.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["DGS2_1"],
        name="2Y Yield", line=dict(color="#1f77b4", width=1.5),
    ))
    fig_ty1.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["DGS30_1"],
        name="30Y Yield", line=dict(color="#d62728", width=1.5),
    ))
    fig_ty1.update_layout(height=450, xaxis_title="Date", yaxis_title="Yield (%)",
                          hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ty1, df["Date"].max())
    st.plotly_chart(fig_ty1, use_container_width=True)

    # Chart 2: 2-30 Spread
    st.subheader("2Y–30Y Spread")
    fig_ty2 = go.Figure()
    colors_spread = [
        "rgba(183,28,28,0.3)" if v < 0 else "rgba(76,175,80,0.3)"
        for v in filtered["TY_Diff_2_30"]
    ]
    fig_ty2.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["TY_Diff_2_30"],
        name="2-30 Spread", line=dict(color="#ff7f0e", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,127,14,0.1)",
    ))
    fig_ty2.add_hline(y=0, line_dash="dash", line_color="red", line_width=1,
                      annotation_text="Inversion", annotation_position="bottom right")
    fig_ty2.update_layout(height=400, xaxis_title="Date", yaxis_title="Spread (%)",
                          hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ty2, df["Date"].max())
    st.plotly_chart(fig_ty2, use_container_width=True)

    # Chart 3: Spread Change (5d & 20d)
    st.subheader("Spread Rate of Change (5d & 20d)")
    fig_ty3 = go.Figure()
    fig_ty3.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["TY_Diff_Chg_5"],
        name="5d Change", line=dict(color="#1f77b4", width=1.2),
    ))
    fig_ty3.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["TY_Diff_Chg_20"],
        name="20d Change", line=dict(color="#2ca02c", width=1.2),
    ))
    fig_ty3.add_hline(y=0, line_color="black", line_width=0.5)
    fig_ty3.update_layout(height=350, xaxis_title="Date", yaxis_title="Rate of Change",
                          hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ty3, df["Date"].max())
    st.plotly_chart(fig_ty3, use_container_width=True)

    # Chart 4: Fractionally Differenced Spread
    st.subheader("Fractionally Differenced 2-30 Spread")
    fig_ty4 = go.Figure()
    fig_ty4.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["TY_2_30_frac_diff"],
        name="Frac Diff Spread", line=dict(color="#9467bd", width=1.2),
    ))
    fig_ty4.add_hline(y=0, line_color="gray", line_width=0.5)
    fig_ty4.update_layout(height=350, xaxis_title="Date", yaxis_title="Frac Diff Value",
                          hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ty4, df["Date"].max())
    st.plotly_chart(fig_ty4, use_container_width=True)


# =====================================================================
# PIVOT BREADTH PAGE
# =====================================================================
elif page == "Pivot Breadth":
    st.title("S&P 500 Pivot Point Breadth")
    df = load_pivot()
    filtered = date_filter(df, "date")
    st.caption(f"Last data: **{df['date'].max().strftime('%Y-%m-%d')}**")

    st.sidebar.subheader("Options")
    show_ma = st.sidebar.selectbox("Moving Average", ["Raw", "5d MA", "10d MA", "20d MA"], index=2)
    view_mode = st.sidebar.radio("View", ["Resistance (Breakouts)", "Support (Breakdowns)", "Both"])

    # Determine MA suffix
    ma_suffix = {"Raw": "", "5d MA": "_ma5", "10d MA": "_ma10", "20d MA": "_ma20"}[show_ma]

    # Key Metrics
    if not filtered.empty:
        latest = filtered.iloc[-1]
        # Use raw values for metrics
        pp_close = latest["pct_closed_above_PP"]
        rr1_close = latest["pct_closed_above_RR1"]
        rr1_fail = latest["pct_failed_RR1"]
        ss1_close = latest["pct_closed_below_SS1"]
        ss1_fail = latest["pct_failed_bd_SS1"]

        cols = st.columns(5)
        cols[0].metric("Close > PP", f"{pp_close:.0%}" if pd.notna(pp_close) else "N/A")
        cols[1].metric("Close > R1", f"{rr1_close:.0%}" if pd.notna(rr1_close) else "N/A")
        cols[2].metric("Failed R1", f"{rr1_fail:.0%}" if pd.notna(rr1_fail) else "N/A")
        cols[3].metric("Close < S1", f"{ss1_close:.0%}" if pd.notna(ss1_close) else "N/A")
        cols[4].metric("Failed S1", f"{ss1_fail:.0%}" if pd.notna(ss1_fail) else "N/A")

        lines = ["**Breakout Regime Interpretation**\n"]

        # PP breadth
        if pd.notna(pp_close):
            if pp_close > 0.65:
                lines.append(f"**{pp_close:.0%}** of S&P 500 stocks closing above their pivot point signals a **strong bullish breadth day**. When a supermajority of stocks are clearing pivots, it confirms broad-based buying pressure -- breakout strategies have a favorable hit rate in this environment.")
            elif pp_close > 0.5:
                lines.append(f"**{pp_close:.0%}** closing above pivot is **modestly bullish** -- more stocks clearing pivots than failing, consistent with a constructive but not exceptional environment for breakouts.")
            elif pp_close > 0.35:
                lines.append(f"**{pp_close:.0%}** closing above pivot is **below average** -- more stocks failing at their pivot than clearing it. This signals a choppy, rotational market where breakouts are unreliable.")
            else:
                lines.append(f"Only **{pp_close:.0%}** closing above pivot -- a **broad failure day**. Very few breakouts are holding, indicating sellers are dominant. This is a regime where fade and mean-reversion strategies outperform breakout strategies.")

        # R1 failure rate context
        if pd.notna(rr1_fail):
            if rr1_fail > 0.6:
                lines.append(f"R1 failure rate at **{rr1_fail:.0%}** is high -- most stocks that touch first resistance are getting rejected. This is a **fade-friendly regime**. Breakout traders should tighten stops or reduce size; mean-reversion setups at resistance levels have higher edge.")
            elif rr1_fail < 0.4:
                lines.append(f"R1 failure rate at **{rr1_fail:.0%}** is low -- breakouts through first resistance are **sticking**. This is the ideal regime for momentum and breakout strategies. Stocks clearing R1 have follow-through, suggesting strong conviction buying.")

        # S1 breakdown context
        if pd.notna(ss1_fail) and pd.notna(ss1_close):
            if ss1_close > 0.2 and ss1_fail < 0.4:
                lines.append(f"S1 breakdowns are holding ({ss1_fail:.0%} failure rate) -- **support is breaking**. Downside momentum is real; consider short setups or tighter stop-losses on longs.")
            elif ss1_fail > 0.6:
                lines.append(f"S1 breakdown failure rate at **{ss1_fail:.0%}** means stocks are **bouncing off support**. Buyers are defending key levels, which is constructive for the bull case.")

        st.info("\n\n".join(lines))

    # Chart 1: % Closing Above Resistance Levels (PP, R1, R2, R3)
    if view_mode in ["Resistance (Breakouts)", "Both"]:
        st.subheader("% Closing Above Resistance Levels")
        fig_r = go.Figure()
        res_cols = [
            (f"pct_closed_above_PP{ma_suffix}", "Above PP", "#1f77b4"),
            (f"pct_closed_above_RR1{ma_suffix}", "Above R1", "#ff7f0e"),
            (f"pct_closed_above_RR2{ma_suffix}", "Above R2", "#2ca02c"),
            (f"pct_closed_above_RR3{ma_suffix}", "Above R3", "#d62728"),
        ]
        for col, name, color in res_cols:
            if col in filtered.columns:
                fig_r.add_trace(go.Scatter(
                    x=filtered["date"], y=filtered[col],
                    name=name, line=dict(color=color, width=1.3),
                ))
        fig_r.add_hline(y=0.5, line_dash="dash", line_color="gray", line_width=0.5,
                        annotation_text="50%", annotation_position="bottom right")
        fig_r.update_layout(height=400, xaxis_title="Date", yaxis_title="% of S&P 500",
                            yaxis_tickformat=".0%", hovermode="x unified", margin=CHART_MARGIN)
        stamp_last_date(fig_r, df["date"].max())
        st.plotly_chart(fig_r, use_container_width=True)

        # Chart 2: Failed Breakout Rates at Resistance
        st.subheader("Failed Breakout Rate at Resistance")
        fig_rf = go.Figure()
        fail_cols = [
            (f"pct_failed_PP{ma_suffix}", "Failed PP", "#1f77b4"),
            (f"pct_failed_RR1{ma_suffix}", "Failed R1", "#ff7f0e"),
            (f"pct_failed_RR2{ma_suffix}", "Failed R2", "#2ca02c"),
            (f"pct_failed_RR3{ma_suffix}", "Failed R3", "#d62728"),
        ]
        for col, name, color in fail_cols:
            if col in filtered.columns:
                fig_rf.add_trace(go.Scatter(
                    x=filtered["date"], y=filtered[col],
                    name=name, line=dict(color=color, width=1.3),
                ))
        fig_rf.add_hline(y=0.5, line_dash="dash", line_color="gray", line_width=0.5,
                         annotation_text="50%", annotation_position="bottom right")
        fig_rf.update_layout(height=400, xaxis_title="Date", yaxis_title="Failure Rate",
                             yaxis_tickformat=".0%", hovermode="x unified", margin=CHART_MARGIN)
        stamp_last_date(fig_rf, df["date"].max())
        st.plotly_chart(fig_rf, use_container_width=True)

    # Chart 3: % Closing Below Support Levels (S1, S2, S3)
    if view_mode in ["Support (Breakdowns)", "Both"]:
        st.subheader("% Closing Below Support Levels")
        fig_s = go.Figure()
        sup_cols = [
            (f"pct_closed_below_SS1{ma_suffix}", "Below S1", "#e377c2"),
            (f"pct_closed_below_SS2{ma_suffix}", "Below S2", "#9467bd"),
            (f"pct_closed_below_SS3{ma_suffix}", "Below S3", "#8c564b"),
        ]
        for col, name, color in sup_cols:
            if col in filtered.columns:
                fig_s.add_trace(go.Scatter(
                    x=filtered["date"], y=filtered[col],
                    name=name, line=dict(color=color, width=1.3),
                ))
        fig_s.update_layout(height=400, xaxis_title="Date", yaxis_title="% of S&P 500",
                            yaxis_tickformat=".0%", hovermode="x unified", margin=CHART_MARGIN)
        stamp_last_date(fig_s, df["date"].max())
        st.plotly_chart(fig_s, use_container_width=True)

        # Chart 4: Failed Breakdown Rates at Support
        st.subheader("Failed Breakdown Rate at Support")
        fig_sf = go.Figure()
        fail_s_cols = [
            (f"pct_failed_bd_SS1{ma_suffix}", "Failed S1", "#e377c2"),
            (f"pct_failed_bd_SS2{ma_suffix}", "Failed S2", "#9467bd"),
            (f"pct_failed_bd_SS3{ma_suffix}", "Failed S3", "#8c564b"),
        ]
        for col, name, color in fail_s_cols:
            if col in filtered.columns:
                fig_sf.add_trace(go.Scatter(
                    x=filtered["date"], y=filtered[col],
                    name=name, line=dict(color=color, width=1.3),
                ))
        fig_sf.add_hline(y=0.5, line_dash="dash", line_color="gray", line_width=0.5,
                         annotation_text="50%", annotation_position="bottom right")
        fig_sf.update_layout(height=400, xaxis_title="Date", yaxis_title="Failure Rate",
                             yaxis_tickformat=".0%", hovermode="x unified", margin=CHART_MARGIN)
        stamp_last_date(fig_sf, df["date"].max())
        st.plotly_chart(fig_sf, use_container_width=True)

    # Chart 5: Breakout vs Fade Regime (R1 success vs failure)
    st.subheader("Breakout vs Fade Regime (R1)")
    fig_regime = go.Figure()
    r1_success_col = f"pct_closed_above_RR1{ma_suffix}" if f"pct_closed_above_RR1{ma_suffix}" in filtered.columns else "pct_closed_above_RR1"
    r1_fail_col = f"pct_failed_RR1{ma_suffix}" if f"pct_failed_RR1{ma_suffix}" in filtered.columns else "pct_failed_RR1"
    fig_regime.add_trace(go.Scatter(
        x=filtered["date"], y=filtered[r1_success_col],
        name="R1 Breakout Hold %", line=dict(color="#2ca02c", width=1.5),
        fill="tozeroy", fillcolor="rgba(44,160,44,0.08)",
    ))
    fig_regime.add_trace(go.Scatter(
        x=filtered["date"], y=filtered[r1_fail_col],
        name="R1 Failure Rate", line=dict(color="#d62728", width=1.5),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.08)",
    ))
    fig_regime.add_hline(y=0.5, line_dash="dash", line_color="black", line_width=0.5,
                         annotation_text="50%", annotation_position="bottom right")
    fig_regime.update_layout(height=400, xaxis_title="Date", yaxis_title="Rate",
                             yaxis_tickformat=".0%", hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_regime, df["date"].max())
    st.plotly_chart(fig_regime, use_container_width=True)

    # Chart 6: % of S&P 500 closing above EMAs (trend breadth)
    st.subheader("% of S&P 500 Closing Above EMAs")
    ema_df = load_ema_breadth()
    ema_mask = (ema_df["date"].dt.date >= filtered["date"].min().date()) & \
               (ema_df["date"].dt.date <= filtered["date"].max().date())
    ema_filt = ema_df.loc[ema_mask]
    fig_ema_b = go.Figure()
    ema_specs = [
        ("pct_above_EMA_5", "Above EMA 5", "#1f77b4"),
        ("pct_above_EMA_8", "Above EMA 8", "#17becf"),
        ("pct_above_EMA_20", "Above EMA 20", "#2ca02c"),
        ("pct_above_EMA_50", "Above EMA 50", "#ff7f0e"),
        ("pct_above_EMA_200", "Above EMA 200", "#d62728"),
    ]
    for col, name, color in ema_specs:
        if col in ema_filt.columns:
            fig_ema_b.add_trace(go.Scatter(
                x=ema_filt["date"], y=ema_filt[col],
                name=name, line=dict(color=color, width=1.3),
            ))
    fig_ema_b.add_hline(y=0.5, line_dash="dash", line_color="gray", line_width=0.5,
                        annotation_text="50%", annotation_position="bottom right")
    fig_ema_b.update_layout(height=450, xaxis_title="Date", yaxis_title="% of S&P 500",
                            yaxis_tickformat=".0%", hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ema_b, ema_df["date"].max())
    st.plotly_chart(fig_ema_b, use_container_width=True)


# =====================================================================
# ORDER FLOW PAGE
# =====================================================================
elif page == "Order Flow":
    st.title("SPY Daily Order Flow")
    df = load_ofi()
    filtered = date_filter(df, "date")
    st.caption(f"Last data: **{df['date'].max().strftime('%Y-%m-%d')}**")

    # Key Metrics
    if not filtered.empty:
        latest = filtered.iloc[-1]
        ofi_total = latest["day_ofi_total"]
        ofi_ratio = latest["day_ofi_ratio"]
        vwap_dev = latest["final_vwap_dev_pct"]
        intra_ret = latest["intra_return_total"]
        or_range = latest["or_range"]

        if ofi_ratio > 0.05:
            flow_bias = "Strong Buy"
        elif ofi_ratio > 0.01:
            flow_bias = "Buy"
        elif ofi_ratio > -0.01:
            flow_bias = "Neutral"
        elif ofi_ratio > -0.05:
            flow_bias = "Sell"
        else:
            flow_bias = "Strong Sell"

        cols = st.columns(5)
        cols[0].metric("OFI Ratio", f"{ofi_ratio:+.4f}")
        cols[1].metric("Flow Bias", flow_bias)
        cols[2].metric("VWAP Dev", f"{vwap_dev:+.3f}%")
        cols[3].metric("Intra Return", f"{intra_ret:+.3f}%")
        cols[4].metric("OR Range", f"${or_range:.2f}")

        lines = ["**Order Flow Interpretation**\n"]

        # OFI ratio context
        if ofi_ratio > 0.05:
            lines.append(f"OFI ratio at {ofi_ratio:+.4f} shows **aggressive net buying** across the full session. This level of imbalance indicates strong institutional demand. When sustained, it often marks the beginning of multi-day directional moves higher.")
        elif ofi_ratio > 0.01:
            lines.append(f"OFI ratio at {ofi_ratio:+.4f} shows **modest net buying pressure**. Buyers are in control but not overwhelmingly so. This is consistent with a constructive tape that favors continuation of intraday trends.")
        elif ofi_ratio > -0.01:
            lines.append(f"OFI ratio at {ofi_ratio:+.4f} is **near neutral** -- balanced buying and selling. No strong directional conviction from order flow. This environment favors mean-reversion over momentum strategies.")
        elif ofi_ratio > -0.05:
            lines.append(f"OFI ratio at {ofi_ratio:+.4f} shows **modest selling pressure**. Sellers are dominating but not panicking. Longs should be cautious; shorting into strength may offer edge.")
        else:
            lines.append(f"OFI ratio at {ofi_ratio:+.4f} signals **aggressive net selling**. Heavy institutional distribution. This level of selling pressure often coincides with trend days down. Breakout longs are high-risk in this flow environment.")

        # VWAP deviation
        if abs(vwap_dev) > 0.3:
            direction = "above" if vwap_dev > 0 else "below"
            lines.append(f"Final price closed **{abs(vwap_dev):.2f}% {direction} VWAP** -- a significant deviation. When price closes far from VWAP, it signals strong directional conviction. Mean-reversion toward VWAP often occurs the following session.")
        elif abs(vwap_dev) < 0.05:
            lines.append(f"Price closed **near VWAP** ({vwap_dev:+.3f}%) -- a balanced session. Neither buyers nor sellers established a clear edge. This often precedes range-bound action the next day.")

        # Opening range
        if or_range > 3:
            lines.append(f"Opening range of **${or_range:.2f}** is wide, signaling high volatility at the open. Wide opening ranges often set the directional tone for the day -- breakouts from the OR tend to follow through.")
        elif or_range < 1:
            lines.append(f"Opening range of **${or_range:.2f}** is tight, suggesting indecision at the open. Tight ORs often lead to breakout moves later in the session once a direction is established.")

        st.info("\n\n".join(lines))

    # Chart 1: Daily OFI Ratio
    st.subheader("Daily OFI Ratio (Net Imbalance)")
    fig_ofi1 = go.Figure()
    colors_ofi = [
        "rgba(76,175,80,0.7)" if v > 0 else "rgba(183,28,28,0.7)"
        for v in filtered["day_ofi_ratio"]
    ]
    fig_ofi1.add_trace(go.Bar(
        x=filtered["date"], y=filtered["day_ofi_ratio"],
        marker_color=colors_ofi, name="OFI Ratio",
    ))
    fig_ofi1.add_hline(y=0, line_color="black", line_width=0.5)
    fig_ofi1.update_layout(height=400, xaxis_title="Date", yaxis_title="OFI Ratio",
                           hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ofi1, df["date"].max())
    st.plotly_chart(fig_ofi1, use_container_width=True)

    # Chart 2: OFI Breakdown (30m, 60m, last 30m)
    st.subheader("Intraday OFI Breakdown")
    fig_ofi2 = go.Figure()
    fig_ofi2.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["ofi_ratio_30m"],
        name="First 30m", line=dict(color="#1f77b4", width=1.2),
    ))
    fig_ofi2.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["day_ofi_ratio"],
        name="Full Day", line=dict(color="#ff7f0e", width=1.5),
    ))
    fig_ofi2.add_hline(y=0, line_color="black", line_width=0.5)
    fig_ofi2.update_layout(height=350, xaxis_title="Date", yaxis_title="OFI Ratio",
                           hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ofi2, df["date"].max())
    st.plotly_chart(fig_ofi2, use_container_width=True)

    # Chart 3: VWAP Deviation
    st.subheader("Final VWAP Deviation %")
    fig_ofi3 = go.Figure()
    colors_vwap = [
        "rgba(76,175,80,0.6)" if v > 0 else "rgba(183,28,28,0.6)"
        for v in filtered["final_vwap_dev_pct"]
    ]
    fig_ofi3.add_trace(go.Bar(
        x=filtered["date"], y=filtered["final_vwap_dev_pct"],
        marker_color=colors_vwap, name="VWAP Dev %",
    ))
    fig_ofi3.add_hline(y=0, line_color="black", line_width=0.5)
    fig_ofi3.update_layout(height=350, xaxis_title="Date", yaxis_title="VWAP Deviation %",
                           hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ofi3, df["date"].max())
    st.plotly_chart(fig_ofi3, use_container_width=True)

    # Chart 4: Opening Range
    st.subheader("Opening Range ($)")
    fig_ofi4 = go.Figure()
    fig_ofi4.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["or_range"],
        name="OR Range", line=dict(color="#9467bd", width=1.2),
        fill="tozeroy", fillcolor="rgba(148,103,189,0.1)",
    ))
    fig_ofi4.update_layout(height=300, xaxis_title="Date", yaxis_title="Range ($)",
                           hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ofi4, df["date"].max())
    st.plotly_chart(fig_ofi4, use_container_width=True)

    # Chart 5: Total Volume
    st.subheader("SPY Daily Volume")
    fig_ofi5 = go.Figure()
    fig_ofi5.add_trace(go.Bar(
        x=filtered["date"], y=filtered["total_volume"],
        name="Volume", marker_color="rgba(100,149,237,0.5)",
    ))
    fig_ofi5.update_layout(height=300, xaxis_title="Date", yaxis_title="Volume",
                           hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ofi5, df["date"].max())
    st.plotly_chart(fig_ofi5, use_container_width=True)


# =====================================================================
# GAMMA (GEX) PAGE
# =====================================================================
elif page == "Gamma (GEX)":
    st.title("SPY Gamma Exposure (GEX)")
    df = load_gex()
    filtered = date_filter(df, "Date")
    st.caption(f"Last data: **{df['Date'].max().strftime('%Y-%m-%d')}**")

    # Key Metrics
    if not filtered.empty:
        latest = filtered.iloc[-1]
        agg_gamma = latest["Agg_Gamma_norm"]
        spot_gamma = latest["Spot_Gamma_norm"]
        hedge_wall = latest["Hedge_wall_1"]
        spy_close = latest["SPY_prev_close"]

        if pd.notna(agg_gamma):
            if agg_gamma > 0.002:
                gamma_env = "Strong +Gamma"
            elif agg_gamma > 0:
                gamma_env = "Positive Gamma"
            elif agg_gamma > -0.002:
                gamma_env = "Negative Gamma"
            else:
                gamma_env = "Deep -Gamma"
        else:
            gamma_env = "N/A"

        cols = st.columns(5)
        cols[0].metric("Agg Gamma", f"{agg_gamma:+.4f}" if pd.notna(agg_gamma) else "N/A")
        cols[1].metric("Environment", gamma_env)
        cols[2].metric("Spot Gamma", f"{spot_gamma:+.4f}" if pd.notna(spot_gamma) else "N/A")
        cols[3].metric("Hedge Wall", f"${hedge_wall:.0f}" if pd.notna(hedge_wall) else "N/A")
        cols[4].metric("SPY Close", f"${spy_close:.2f}" if pd.notna(spy_close) else "N/A")

        lines = ["**Gamma Regime Interpretation**\n"]

        if pd.notna(agg_gamma):
            if agg_gamma > 0.002:
                lines.append(f"Aggregate gamma at **{agg_gamma:+.4f}** is **strongly positive**. Dealers are long gamma and will hedge by selling rallies and buying dips -- this **suppresses volatility** and pins price action near the hedge wall. Expect low realized vol, tight ranges, and mean-reversion setups. Breakouts are unlikely to sustain.")
            elif agg_gamma > 0:
                lines.append(f"Aggregate gamma at **{agg_gamma:+.4f}** is **modestly positive**. Dealers are still net long gamma, providing a stabilizing force, but not overwhelmingly so. Volatility is contained but directional moves are possible with strong enough catalysts.")
            elif agg_gamma > -0.002:
                lines.append(f"Aggregate gamma at **{agg_gamma:+.4f}** is **negative**. Dealers are short gamma and must hedge in the same direction as the move -- buying into rallies and selling into declines. This **amplifies volatility** and creates momentum-friendly conditions. Breakouts are more likely to follow through.")
            else:
                lines.append(f"Aggregate gamma at **{agg_gamma:+.4f}** is **deeply negative**. Dealers are heavily short gamma, creating a **volatility accelerator**. Moves in either direction will be amplified by dealer hedging flows. This is the most dangerous regime for mean-reversion and the most favorable for trend-following. Expect outsized moves and potential gap risk.")

        if pd.notna(hedge_wall) and pd.notna(spy_close):
            wall_dist = ((hedge_wall - spy_close) / spy_close) * 100
            if abs(wall_dist) < 0.5:
                lines.append(f"SPY (${spy_close:.0f}) is **at the hedge wall** (${hedge_wall:.0f}). This level acts as a magnet -- dealer hedging flows concentrate here, making it a strong support/resistance level. Price tends to gravitate toward the wall in positive gamma environments.")
            elif wall_dist > 0:
                lines.append(f"Hedge wall at **${hedge_wall:.0f}** is {wall_dist:.1f}% above SPY (${spy_close:.0f}). The wall acts as overhead resistance where dealer gamma is most concentrated. In positive gamma, expect price to be drawn toward this level.")
            else:
                lines.append(f"Hedge wall at **${hedge_wall:.0f}** is {abs(wall_dist):.1f}% below SPY (${spy_close:.0f}). Price has broken above the wall, which may reduce the stabilizing effect of dealer hedging. If gamma flips negative, this can accelerate moves away from the wall.")

        st.info("\n\n".join(lines))

    # Chart 1: Call Gamma vs Put Gamma
    st.subheader("Call vs Put Gamma")
    fig_g1 = go.Figure()
    fig_g1.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["Call_gamma_1"],
        name="Call Gamma", line=dict(color="#2ca02c", width=1.2),
        fill="tozeroy", fillcolor="rgba(44,160,44,0.1)",
    ))
    fig_g1.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["Put_gamma_1"],
        name="Put Gamma", line=dict(color="#d62728", width=1.2),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.1)",
    ))
    fig_g1.add_hline(y=0, line_color="black", line_width=0.5)
    fig_g1.update_layout(height=400, xaxis_title="Date", yaxis_title="Gamma",
                         hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_g1, df["Date"].max())
    st.plotly_chart(fig_g1, use_container_width=True)

    # Chart 2: Normalized Aggregate Gamma (positive/negative gamma environment)
    st.subheader("Aggregate Gamma (Normalized)")
    fig_g2 = go.Figure()
    colors_gamma = [
        "rgba(76,175,80,0.7)" if v > 0 else "rgba(183,28,28,0.7)"
        for v in filtered["Agg_Gamma_norm"].fillna(0)
    ]
    fig_g2.add_trace(go.Bar(
        x=filtered["Date"], y=filtered["Agg_Gamma_norm"],
        marker_color=colors_gamma, name="Agg Gamma Norm",
    ))
    fig_g2.add_hline(y=0, line_color="black", line_width=1,
                     annotation_text="Gamma Flip", annotation_position="bottom right")
    fig_g2.update_layout(height=400, xaxis_title="Date", yaxis_title="Normalized Gamma",
                         hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_g2, df["Date"].max())
    st.plotly_chart(fig_g2, use_container_width=True)

    # Chart 3: Spot Gamma (Normalized)
    st.subheader("Spot Gamma at Current Price (Normalized)")
    fig_g3 = go.Figure()
    colors_spot = [
        "rgba(76,175,80,0.6)" if v > 0 else "rgba(183,28,28,0.6)"
        for v in filtered["Spot_Gamma_norm"].fillna(0)
    ]
    fig_g3.add_trace(go.Bar(
        x=filtered["Date"], y=filtered["Spot_Gamma_norm"],
        marker_color=colors_spot, name="Spot Gamma Norm",
    ))
    fig_g3.add_hline(y=0, line_color="black", line_width=0.5)
    fig_g3.update_layout(height=350, xaxis_title="Date", yaxis_title="Spot Gamma (Norm)",
                         hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_g3, df["Date"].max())
    st.plotly_chart(fig_g3, use_container_width=True)

    # Chart 4: Hedge Wall vs SPY Price
    st.subheader("Hedge Wall vs SPY Price")
    fig_g4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig_g4.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["SPY_prev_close"],
        name="SPY Close", line=dict(color="#1f77b4", width=1.5),
    ), secondary_y=False)
    fig_g4.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["Hedge_wall_1"],
        name="Hedge Wall", line=dict(color="#ff7f0e", width=1.5, dash="dot"),
    ), secondary_y=False)
    # Shade the gap between them
    fig_g4.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["SPY_prev_close"],
        line=dict(width=0), showlegend=False,
    ), secondary_y=False)
    fig_g4.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["Hedge_wall_1"],
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(255,127,14,0.08)", showlegend=False,
    ), secondary_y=False)
    # Overlay Agg Gamma on secondary axis for context
    fig_g4.add_trace(go.Bar(
        x=filtered["Date"], y=filtered["Agg_Gamma_norm"],
        name="Agg Gamma", marker_color=[
            "rgba(76,175,80,0.3)" if v > 0 else "rgba(183,28,28,0.3)"
            for v in filtered["Agg_Gamma_norm"].fillna(0)
        ],
    ), secondary_y=True)
    fig_g4.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig_g4.update_yaxes(title_text="Agg Gamma", secondary_y=True)
    fig_g4.update_layout(height=450, xaxis_title="Date",
                         hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_g4, df["Date"].max())
    st.plotly_chart(fig_g4, use_container_width=True)


# =====================================================================
# SPY TECHNICALS PAGE
# =====================================================================
elif page == "SPY Technicals":
    st.title("SPY Technicals")
    df = load_spy()
    filtered = date_filter(df, "date")
    st.caption(f"Last data: **{df['date'].max().strftime('%Y-%m-%d')}**")

    # Key Metrics
    if not filtered.empty:
        latest = filtered.iloc[-1]
        spy_close = latest["close"]
        rsi14 = latest["SPY_RSI_14"]
        rsi9 = latest["SPY_RSI_9"]
        cmf = latest["SPY_Daily_CMF"]
        ema8_20 = latest["SPY_EMA_8_20_var"] if pd.notna(latest.get("SPY_EMA_8_20_var")) else None
        ema20_200 = latest["SPY_EMA_20_200_var"] if pd.notna(latest.get("SPY_EMA_20_200_var")) else None

        # Trend from EMA structure
        if pd.notna(ema8_20) and pd.notna(ema20_200):
            if ema8_20 > 0 and ema20_200 > 0:
                trend = "Uptrend"
            elif ema8_20 < 0 and ema20_200 < 0:
                trend = "Downtrend"
            else:
                trend = "Transitional"
        else:
            trend = "N/A"

        cols = st.columns(5)
        cols[0].metric("SPY Close", f"${spy_close:.2f}" if pd.notna(spy_close) else "N/A")
        cols[1].metric("Trend", trend)
        cols[2].metric("RSI 14", f"{rsi14:.1f}" if pd.notna(rsi14) else "N/A")
        cols[3].metric("RSI 9", f"{rsi9:.1f}" if pd.notna(rsi9) else "N/A")
        cols[4].metric("CMF", f"{cmf:+.3f}" if pd.notna(cmf) else "N/A")

        lines = ["**Technical Regime Interpretation**\n"]

        # EMA structure
        if pd.notna(ema8_20) and pd.notna(ema20_200):
            if ema8_20 > 0 and ema20_200 > 0:
                lines.append(f"EMA structure is **fully bullish** -- fast EMAs above slow EMAs across all timeframes (EMA8>20 spread: {ema8_20:+.4f}, EMA20>200 spread: {ema20_200:+.4f}). This is the strongest trend confirmation. Breakout and trend-following strategies have the highest edge in this configuration.")
            elif ema8_20 < 0 and ema20_200 > 0:
                lines.append(f"Short-term EMAs have crossed below (EMA8<20: {ema8_20:+.4f}) but the long-term structure remains bullish (EMA20>200: {ema20_200:+.4f}). This is a **pullback within an uptrend** -- typically a buying opportunity if breadth and flow confirm. Watch for the short-term crossover to resolve.")
            elif ema8_20 > 0 and ema20_200 < 0:
                lines.append(f"Short-term EMAs have crossed above (EMA8>20: {ema8_20:+.4f}) while the long-term structure is bearish (EMA20<200: {ema20_200:+.4f}). This is a **bear market rally** -- momentum is improving but the primary trend is still down. Be cautious with breakout longs; fading rallies near resistance may offer better edge.")
            else:
                lines.append(f"EMA structure is **fully bearish** -- all timeframes aligned down (EMA8<20: {ema8_20:+.4f}, EMA20<200: {ema20_200:+.4f}). This is the weakest trend configuration. Short setups and fade strategies are favored; avoid breakout longs.")

        # RSI
        if pd.notna(rsi14):
            if rsi14 > 70:
                lines.append(f"RSI 14 at **{rsi14:.0f}** is **overbought**. Price has extended beyond typical bounds. In trending markets this can persist, but in range-bound conditions it signals a high probability of near-term pullback.")
            elif rsi14 < 30:
                lines.append(f"RSI 14 at **{rsi14:.0f}** is **oversold**. Selling pressure is exhausted at this level. Watch for RSI divergence (price making new lows while RSI makes higher lows) as a reversal signal.")
            elif rsi14 > 60:
                lines.append(f"RSI 14 at **{rsi14:.0f}** shows **bullish momentum** -- above the midline but not yet overbought. Consistent with a healthy uptrend.")
            elif rsi14 < 40:
                lines.append(f"RSI 14 at **{rsi14:.0f}** shows **bearish momentum** -- below the midline. Sellers are in control but not at extreme levels yet.")

        # CMF
        if pd.notna(cmf):
            if cmf > 0.1:
                lines.append(f"CMF at **{cmf:+.3f}** signals **strong institutional accumulation**. Money is flowing into SPY aggressively. This is the most bullish money flow reading and typically confirms uptrend sustainability.")
            elif cmf > 0:
                lines.append(f"CMF at **{cmf:+.3f}** shows **modest accumulation** -- more buying pressure than selling on a volume-weighted basis. Constructive for bulls but not extreme conviction.")
            elif cmf > -0.1:
                lines.append(f"CMF at **{cmf:+.3f}** shows **modest distribution** -- selling pressure slightly outweighs buying. Institutional money is quietly exiting. If this persists alongside rising prices, it's a bearish divergence.")
            else:
                lines.append(f"CMF at **{cmf:+.3f}** signals **heavy institutional distribution**. Money is flowing out of SPY aggressively. This often leads price lower and confirms downtrend momentum.")

        st.info("\n\n".join(lines))

    # Chart 1: EMA Structure (spread between fast and slow EMAs)
    st.subheader("EMA Trend Structure")
    fig_ema = go.Figure()

    # EMA8 vs EMA20 spread
    fig_ema.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["SPY_EMA_8_20_var"],
        name="EMA 8-20 Spread", line=dict(color="#1f77b4", width=1.5),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.08)",
    ))
    # EMA20 vs EMA200 spread
    fig_ema.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["SPY_EMA_20_200_var"],
        name="EMA 20-200 Spread", line=dict(color="#d62728", width=1.5),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.08)",
    ))
    fig_ema.add_hline(y=0, line_color="black", line_width=1,
                      annotation_text="Crossover", annotation_position="bottom right")
    fig_ema.update_layout(height=400, xaxis_title="Date", yaxis_title="EMA Spread (% of price)",
                          hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_ema, df["date"].max())
    st.plotly_chart(fig_ema, use_container_width=True)

    # Chart 2: RSI (14 and 9) with divergence detection
    st.subheader("RSI (9d & 14d) with Divergences")
    fig_rsi = go.Figure()

    # Overbought/Oversold shading
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(183,28,28,0.06)", line_width=0,
                      annotation_text="Overbought", annotation_position="top left",
                      annotation_font_size=10, annotation_font_color="gray")
    fig_rsi.add_hrect(y0=0, y1=30, fillcolor="rgba(76,175,80,0.06)", line_width=0,
                      annotation_text="Oversold", annotation_position="bottom left",
                      annotation_font_size=10, annotation_font_color="gray")

    fig_rsi.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["SPY_RSI_14"],
        name="RSI 14", line=dict(color="#1f77b4", width=1.5),
    ))
    fig_rsi.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["SPY_RSI_9"],
        name="RSI 9", line=dict(color="#ff7f0e", width=1.2, dash="dot"),
    ))
    fig_rsi.add_hline(y=50, line_dash="dash", line_color="gray", line_width=0.5)

    # Detect and overlay divergences
    bullish_divs, bearish_divs = detect_rsi_divergences(filtered)

    if bullish_divs:
        fig_rsi.add_trace(go.Scatter(
            x=[d["date"] for d in bullish_divs],
            y=[d["rsi"] for d in bullish_divs],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#2ca02c", line=dict(width=1, color="white")),
            name="Bullish Div",
        ))
    if bearish_divs:
        fig_rsi.add_trace(go.Scatter(
            x=[d["date"] for d in bearish_divs],
            y=[d["rsi"] for d in bearish_divs],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#d62728", line=dict(width=1, color="white")),
            name="Bearish Div",
        ))

    fig_rsi.update_layout(height=400, xaxis_title="Date", yaxis_title="RSI",
                          yaxis_range=[0, 100], hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_rsi, df["date"].max())
    st.plotly_chart(fig_rsi, use_container_width=True)

    # Divergence summary
    recent_bull = [d for d in bullish_divs if (filtered["date"].max() - d["date"]).days <= 30]
    recent_bear = [d for d in bearish_divs if (filtered["date"].max() - d["date"]).days <= 30]
    if recent_bull or recent_bear:
        div_lines = ["**Recent RSI Divergences (last 30 days)**\n"]
        for d in recent_bull:
            div_lines.append(f"**Bullish** on {d['date'].strftime('%Y-%m-%d')} — Price made lower low while RSI made higher low (RSI: {d['rsi']:.1f}). Selling pressure fading; potential upside reversal.")
        for d in recent_bear:
            div_lines.append(f"**Bearish** on {d['date'].strftime('%Y-%m-%d')} — Price made higher high while RSI made lower high (RSI: {d['rsi']:.1f}). Buying pressure fading; potential downside reversal.")
        st.warning("\n\n".join(div_lines))
    else:
        st.caption("No RSI divergences detected in the last 30 days.")

    # Chart 3: Chaikin Money Flow
    st.subheader("Chaikin Money Flow (CMF)")
    fig_cmf = go.Figure()
    colors_cmf = [
        "rgba(76,175,80,0.7)" if v > 0 else "rgba(183,28,28,0.7)"
        for v in filtered["SPY_Daily_CMF"].fillna(0)
    ]
    fig_cmf.add_trace(go.Bar(
        x=filtered["date"], y=filtered["SPY_Daily_CMF"],
        marker_color=colors_cmf, name="CMF",
    ))
    fig_cmf.add_hline(y=0, line_color="black", line_width=0.5)
    fig_cmf.update_layout(height=350, xaxis_title="Date", yaxis_title="CMF",
                          hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_cmf, df["date"].max())
    st.plotly_chart(fig_cmf, use_container_width=True)

    # Chart 4: Bid/Ask Aggressor Ratio
    st.subheader("Bid/Ask Aggressor Ratio")
    fig_agg = go.Figure()
    agg_ratio = filtered["askvol"] / (filtered["askvol"] + filtered["bidvol"])
    fig_agg.add_trace(go.Scatter(
        x=filtered["date"], y=agg_ratio,
        name="Ask Aggressor %", line=dict(color="#9467bd", width=1.2),
    ))
    fig_agg.add_hline(y=0.5, line_dash="dash", line_color="black", line_width=0.5,
                      annotation_text="50% (neutral)", annotation_position="bottom right")
    fig_agg.update_layout(height=350, xaxis_title="Date", yaxis_title="Ask / (Ask+Bid) Ratio",
                          yaxis_tickformat=".0%", hovermode="x unified", margin=CHART_MARGIN)
    stamp_last_date(fig_agg, df["date"].max())
    st.plotly_chart(fig_agg, use_container_width=True)
