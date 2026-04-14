"""
SMB Market Dashboard — Streamlit app
Run: streamlit run market_dashboard.py
Data files expected in the same directory as this script.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from dropbox_loader import load_csv, load_parquet

st.set_page_config(
    page_title="SMB Market Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ────────────────────────────────────────────────────────────────

def load_vix():
    df = load_csv("VIX_data.csv", parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_spy():
    df = load_csv("SPY_historical_data.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_ad():
    df = load_csv("Advance_Decline.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_ty():
    df = load_csv("Treasury_Yields.csv", parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_pcr():
    df = load_csv("Historical_put_call_ratio.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_gex():
    df = load_csv("SPY_historical_gamma_2005_2025.csv", parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_pivot():
    df = load_parquet("sp500_pivot_breadth.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def filter_dates(df, date_col, start, end):
    mask = (df[date_col] >= pd.Timestamp(start)) & (
        df[date_col] <= pd.Timestamp(end))
    return df.loc[mask].copy()


def metric_delta(current, prior, fmt=".2f"):
    delta = current - prior
    return f"{current:{fmt}}", f"{delta:+{fmt}}"


def styled_metric(col, label, value, delta=None, suffix=""):
    col.metric(label, f"{value}{suffix}", delta=delta)


# ── sidebar ────────────────────────────────────────────────────────────────

st.sidebar.title("SMB Market Dashboard")
st.sidebar.markdown("---")

lookback_options = {
    "1 Month": 21,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year": 252,
    "2 Years": 504,
    "5 Years": 1260,
    "All": None}
lookback_label = st.sidebar.selectbox(
    "Lookback", list(lookback_options.keys()), index=3)
lookback_days = lookback_options[lookback_label]

spy_all = load_spy()
max_date = spy_all["date"].max()
if lookback_days:
    # calendar days approx
    min_date = max_date - timedelta(days=lookback_days * 1.5)
else:
    min_date = spy_all["date"].min()

st.sidebar.markdown("---")
st.sidebar.caption(f"Data through: **{max_date.strftime('%Y-%m-%d')}**")

# ── tabs ────────────────────────────────────────────────────────────────────

tabs = st.tabs(["VIX", "SPY EMAs", "A/D Breadth",
               "Yields", "PCR", "GEX", "Pivot Breadth"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — VIX
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("VIX — Volatility Index")
    vix_all = load_vix()
    vix = filter_dates(vix_all, "Date", min_date, max_date)

    if not vix.empty:
        last = vix.iloc[-1]
        prev = vix.iloc[-2] if len(vix) > 1 else last

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "VIX Close", f"{
                last['VIX_Close']:.2f}", f"{
                last['VIX_Close'] - prev['VIX_Close']:+.2f}")
        c2.metric("MA 20", f"{last['VIX_MA_20']:.2f}")
        if "VIX_z_score_20" in vix.columns:
            c3.metric("Z-Score (20d)", f"{last['VIX_z_score_20']:.2f}")
        if "VIX_60d_ratio" in vix.columns:
            c4.metric("60d Ratio", f"{last['VIX_60d_ratio']:.2f}")

        # Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_trace(
            go.Scatter(
                x=vix["Date"],
                y=vix["VIX_Close"],
                name="VIX",
                line=dict(
                    color="#ef5350",
                    width=1.5)),
            row=1,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=vix["Date"],
                y=vix["VIX_MA_10"],
                name="MA10",
                line=dict(
                    color="#ff9800",
                    width=1,
                    dash="dot")),
            row=1,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=vix["Date"],
                y=vix["VIX_MA_20"],
                name="MA20",
                line=dict(
                    color="#42a5f5",
                    width=1,
                    dash="dot")),
            row=1,
            col=1)
        # Add reference lines
        for level, color in [(15, "green"), (20, "orange"),
                             (30, "red"), (40, "darkred")]:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color=color,
                line_width=0.8,
                opacity=0.5,
                row=1,
                col=1)

        if "VIX_z_score_20" in vix.columns:
            fig.add_trace(go.Bar(x=vix["Date"], y=vix["VIX_z_score_20"], name="Z-Score 20d",
                                 marker_color=vix["VIX_z_score_20"].apply(lambda v: "#ef5350" if v > 0 else "#26a69a")), row=2, col=1)
            fig.add_hline(
                y=2,
                line_dash="dash",
                line_color="red",
                line_width=0.8,
                opacity=0.5,
                row=2,
                col=1)
            fig.add_hline(
                y=-2,
                line_dash="dash",
                line_color="green",
                line_width=0.8,
                opacity=0.5,
                row=2,
                col=1)

        fig.update_layout(height=500, template="plotly_dark", showlegend=True,
                          legend=dict(orientation="h", y=1.02, x=0),
                          margin=dict(l=0, r=0, t=30, b=0))
        fig.update_yaxes(title_text="VIX", row=1, col=1)
        fig.update_yaxes(title_text="Z-Score", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — SPY EMAs
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("SPY — Price & EMAs")
    spy = filter_dates(spy_all, "date", min_date, max_date)

    if not spy.empty:
        last = spy.iloc[-1]
        prev = spy.iloc[-2] if len(spy) > 1 else last

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "SPY Close", f"{
                last['close']:.2f}", f"{
                last['close'] - prev['close']:+.2f}")
        if "SPY_RSI_14" in spy.columns:
            c2.metric("RSI 14", f"{last['SPY_RSI_14']:.1f}")
        if "SPY_ATR" in spy.columns:
            c3.metric("ATR 14", f"{last['SPY_ATR']:.2f}")
        if "SPY_EMA_200" in spy.columns:
            gap = (last["close"] - last["SPY_EMA_200"]) / \
                last["SPY_EMA_200"] * 100
            c4.metric("vs EMA200", f"{gap:+.1f}%")

        ema_options = st.multiselect("EMAs to display", ["EMA_8", "EMA_20", "EMA_50", "EMA_100", "EMA_200"],
                                     default=["EMA_20", "EMA_50", "EMA_200"])

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.65, 0.35], vertical_spacing=0.05)

        # Candlestick / line price
        fig.add_trace(
            go.Scatter(
                x=spy["date"],
                y=spy["close"],
                name="SPY",
                line=dict(
                    color="white",
                    width=1.5)),
            row=1,
            col=1)

        ema_colors = {"EMA_8": "#ff9800", "EMA_20": "#42a5f5", "EMA_50": "#ab47bc",
                      "EMA_100": "#26a69a", "EMA_200": "#ef5350"}
        for ema in ema_options:
            col = f"SPY_{ema}"
            if col in spy.columns:
                fig.add_trace(go.Scatter(x=spy["date"], y=spy[col], name=ema,
                                         line=dict(color=ema_colors.get(ema, "gray"), width=1, dash="dot")), row=1, col=1)

        # RSI
        if "SPY_RSI_14" in spy.columns:
            fig.add_trace(go.Scatter(x=spy["date"], y=spy["SPY_RSI_14"], name="RSI 14",
                                     line=dict(color="#ff9800", width=1.5)), row=2, col=1)
            fig.add_hline(
                y=70,
                line_dash="dash",
                line_color="red",
                line_width=0.8,
                opacity=0.6,
                row=2,
                col=1)
            fig.add_hline(
                y=50,
                line_dash="dash",
                line_color="gray",
                line_width=0.8,
                opacity=0.4,
                row=2,
                col=1)
            fig.add_hline(
                y=30,
                line_dash="dash",
                line_color="green",
                line_width=0.8,
                opacity=0.6,
                row=2,
                col=1)

        fig.update_layout(height=500, template="plotly_dark", showlegend=True,
                          legend=dict(orientation="h", y=1.02, x=0),
                          margin=dict(l=0, r=0, t=30, b=0))
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        # EMA deviation table
        if st.checkbox("Show EMA deviation (ATR units)"):
            var_cols = [
                c for c in spy.columns if "EMA" in c and "_var" in c and "_20" not in c][:6]
            if var_cols:
                latest_vars = spy[var_cols].iloc[-1].rename(
                    index=lambda x: x.replace("SPY_", "").replace("_var", " dev"))
                st.dataframe(
                    latest_vars.to_frame("ATR Units").T.style.background_gradient(
                        cmap="RdYlGn", axis=1))


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — A/D Breadth
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("S&P 500 Advance/Decline Breadth")
    ad = filter_dates(load_ad(), "date", min_date, max_date)

    if not ad.empty:
        last = ad.iloc[-1]
        prev = ad.iloc[-2] if len(ad) > 1 else last

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Advances", f"{int(last['advances'])}", f"{
                  int(last['advances'] - prev['advances']):+d}")
        c2.metric("Declines", f"{int(last['declines'])}", f"{
                  int(last['declines'] - prev['declines']):+d}")
        if "Advance_pct" in ad.columns:
            c3.metric("Advance %", f"{last['Advance_pct']:.1%}")
        if "AD_z_score_20" in ad.columns:
            c4.metric("A/D Z-Score (20d)", f"{last['AD_z_score_20']:.2f}")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.45, 0.3, 0.25], vertical_spacing=0.04,
                            subplot_titles=("A/D Line + MA5/20", "Net A/D (daily)", "Advance %"))

        # A/D Line
        if "AD_Line" in ad.columns:
            fig.add_trace(go.Scatter(x=ad["date"], y=ad["AD_Line"], name="A/D Line",
                                     line=dict(color="#42a5f5", width=1.5)), row=1, col=1)
            if "AD_ma_5" in ad.columns:
                fig.add_trace(go.Scatter(x=ad["date"], y=ad["AD_ma_5"], name="MA5",
                                         line=dict(color="#ff9800", width=1, dash="dot")), row=1, col=1)
            if "AD_ma_20" in ad.columns:
                fig.add_trace(go.Scatter(x=ad["date"], y=ad["AD_ma_20"], name="MA20",
                                         line=dict(color="#ef5350", width=1, dash="dot")), row=1, col=1)

        # Net A/D bars
        if "AD" in ad.columns:
            fig.add_trace(go.Bar(x=ad["date"], y=ad["AD"], name="Net A/D",
                                 marker_color=ad["AD"].apply(lambda v: "#26a69a" if v >= 0 else "#ef5350")), row=2, col=1)

        # Advance %
        if "Advance_pct" in ad.columns:
            fig.add_trace(go.Scatter(x=ad["date"], y=ad["Advance_pct"] * 100, name="Advance %",
                                     line=dict(color="#ab47bc", width=1.5)), row=3, col=1)
            fig.add_hline(
                y=50,
                line_dash="dash",
                line_color="gray",
                line_width=0.8,
                opacity=0.5,
                row=3,
                col=1)
            fig.add_hline(
                y=70,
                line_dash="dash",
                line_color="green",
                line_width=0.8,
                opacity=0.5,
                row=3,
                col=1)
            fig.add_hline(
                y=30,
                line_dash="dash",
                line_color="red",
                line_width=0.8,
                opacity=0.5,
                row=3,
                col=1)

        fig.update_layout(height=580, template="plotly_dark", showlegend=True,
                          legend=dict(orientation="h", y=1.02, x=0),
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Treasury Yields
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Treasury Yields — 2Y / 30Y / Spread")
    ty = filter_dates(load_ty(), "Date", min_date, max_date)

    if not ty.empty:
        last = ty.iloc[-1]
        prev = ty.iloc[-2] if len(ty) > 1 else last

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("2Y Yield", f"{last['DGS2_1']:.3f}%", f"{
                  last['DGS2_1'] - prev['DGS2_1']:+.3f}")
        c2.metric("30Y Yield", f"{last['DGS30_1']:.3f}%", f"{
                  last['DGS30_1'] - prev['DGS30_1']:+.3f}")
        c3.metric("Spread (30Y-2Y)",
                  f"{last['TY_Diff_2_30']:.3f}%",
                  f"{last['TY_Diff_2_30'] - prev['TY_Diff_2_30']:+.3f}")
        if "TY_Diff_Chg_20" in ty.columns:
            c4.metric("Spread 20d Chg", f"{last['TY_Diff_Chg_20']:.3f}%")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.35, 0.35, 0.3], vertical_spacing=0.05,
                            subplot_titles=("2Y & 30Y Yields", "30Y–2Y Spread", "Frac-Diff Spread"))

        fig.add_trace(go.Scatter(x=ty["Date"], y=ty["DGS2_1"], name="2Y",
                                 line=dict(color="#42a5f5", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ty["Date"], y=ty["DGS30_1"], name="30Y",
                                 line=dict(color="#ef5350", width=1.5)), row=1, col=1)

        colors_spread = ty["TY_Diff_2_30"].apply(
            lambda v: "#26a69a" if v >= 0 else "#ef5350")
        fig.add_trace(go.Bar(x=ty["Date"], y=ty["TY_Diff_2_30"], name="Spread",
                             marker_color=colors_spread), row=2, col=1)
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="white",
            line_width=0.8,
            opacity=0.5,
            row=2,
            col=1)

        if "TY_2_30_frac_diff" in ty.columns:
            fig.add_trace(go.Scatter(x=ty["Date"], y=ty["TY_2_30_frac_diff"], name="Frac-Diff",
                                     line=dict(color="#ff9800", width=1.5)), row=3, col=1)
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="white",
                line_width=0.8,
                opacity=0.5,
                row=3,
                col=1)

        fig.update_layout(height=560, template="plotly_dark", showlegend=True,
                          legend=dict(orientation="h", y=1.02, x=0),
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — PCR
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("SPY Put/Call Ratio")
    pcr = filter_dates(load_pcr(), "date", min_date, max_date)

    if not pcr.empty:
        last = pcr.iloc[-1]
        prev = pcr.iloc[-2] if len(pcr) > 1 else last

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "PCR", f"{
                last['PCR']:.3f}", f"{
                last['PCR'] - prev['PCR']:+.3f}")
        c2.metric("PCR MA20", f"{last['PCR_ma_20']:.3f}")
        if "PCR_z_score" in pcr.columns:
            c3.metric("PCR Z-Score", f"{last['PCR_z_score']:.2f}")
        c4.metric("Total Volume", f"{last['total_volume'] / 1e6:.1f}M")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.65, 0.35], vertical_spacing=0.05)

        fig.add_trace(go.Scatter(x=pcr["date"], y=pcr["PCR"], name="PCR",
                                 line=dict(color="#ab47bc", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=pcr["date"], y=pcr["PCR_ma_20"], name="MA20",
                                 line=dict(color="#ff9800", width=1, dash="dot")), row=1, col=1)
        # Reference bands
        for lvl, color in [(0.7, "green"), (1.0, "orange"), (1.3, "red")]:
            fig.add_hline(
                y=lvl,
                line_dash="dash",
                line_color=color,
                line_width=0.8,
                opacity=0.5,
                row=1,
                col=1)

        if "PCR_z_score" in pcr.columns:
            fig.add_trace(go.Bar(x=pcr["date"], y=pcr["PCR_z_score"], name="Z-Score",
                                 marker_color=pcr["PCR_z_score"].apply(lambda v: "#ef5350" if v > 0 else "#26a69a")), row=2, col=1)
            fig.add_hline(
                y=2,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                row=2,
                col=1)
            fig.add_hline(
                y=-2,
                line_dash="dash",
                line_color="green",
                opacity=0.5,
                row=2,
                col=1)

        fig.update_layout(height=480, template="plotly_dark", showlegend=True,
                          legend=dict(orientation="h", y=1.02, x=0),
                          margin=dict(l=0, r=0, t=30, b=0))
        fig.update_yaxes(title_text="PCR", row=1, col=1)
        fig.update_yaxes(title_text="Z-Score", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — GEX
# ══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("SPY Gamma Exposure (GEX)")
    gex = filter_dates(load_gex(), "Date", min_date, max_date)

    if not gex.empty:
        last = gex.iloc[-1]
        prev = gex.iloc[-2] if len(gex) > 1 else last

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Agg Gamma ($M)", f"{
                last['Agg_Gamma_1']:.0f}", f"{
                last['Agg_Gamma_1'] - prev['Agg_Gamma_1']:+.0f}")
        c2.metric(
            "Spot Gamma ($M)", f"{
                last['Spot_Gamma_1']:.0f}", f"{
                last['Spot_Gamma_1'] - prev['Spot_Gamma_1']:+.0f}")
        c3.metric("Hedge Wall", f"{last['Hedge_wall_1']:.0f}")
        c4.metric("Agg Gamma Norm", f"{last['Agg_Gamma_norm']:.4f}")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.35, 0.35, 0.3], vertical_spacing=0.05,
                            subplot_titles=("Aggregate GEX ($M)", "Spot GEX ($M)", "Normalized GEX"))

        def gamma_colors(series):
            return series.apply(lambda v: "#26a69a" if v >= 0 else "#ef5350")

        fig.add_trace(go.Bar(x=gex["Date"], y=gex["Agg_Gamma_1"], name="Agg GEX",
                             marker_color=gamma_colors(gex["Agg_Gamma_1"])), row=1, col=1)
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="white",
            opacity=0.4,
            row=1,
            col=1)

        fig.add_trace(go.Bar(x=gex["Date"], y=gex["Spot_Gamma_1"], name="Spot GEX",
                             marker_color=gamma_colors(gex["Spot_Gamma_1"])), row=2, col=1)
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="white",
            opacity=0.4,
            row=2,
            col=1)

        if "Agg_Gamma_norm" in gex.columns:
            fig.add_trace(go.Scatter(x=gex["Date"], y=gex["Agg_Gamma_norm"], name="GEX Norm",
                                     line=dict(color="#ff9800", width=1.5)), row=3, col=1)
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="white",
                opacity=0.4,
                row=3,
                col=1)

        fig.update_layout(height=560, template="plotly_dark", showlegend=True,
                          legend=dict(orientation="h", y=1.02, x=0),
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 7 — Pivot Breadth
# ══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("S&P 500 Pivot Breadth")
    piv = filter_dates(load_pivot(), "date", min_date, max_date)

    if not piv.empty:
        last = piv.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stocks in universe", f"{int(last['n_total'])}")
        if "pct_closed_above_PP" in piv.columns:
            c2.metric("% Above PP", f"{last['pct_closed_above_PP']:.1f}%")
        if "pct_closed_above_RR1" in piv.columns:
            c3.metric("% Above RR1", f"{last['pct_closed_above_RR1']:.1f}%")
        if "pct_closed_below_SS1" in piv.columns:
            c4.metric("% Below SS1", f"{last['pct_closed_below_SS1']:.1f}%")

        level_choice = st.selectbox(
            "Pivot Level",
            ["PP", "RR1", "RR2", "RR3", "SS1", "SS2", "SS3"],
            index=0,
        )

        is_resistance = level_choice in ("PP", "RR1", "RR2", "RR3")

        if is_resistance:
            touch_col = f"pct_touched_{level_choice}"
            close_col = f"pct_closed_above_{level_choice}"
            fail_col = f"pct_failed_{level_choice}"
            ma_col = f"pct_closed_above_{level_choice}_ma20"
            close_label = f"% Closed Above {level_choice}"
            fail_label = f"% Failed Breakout"
        else:
            touch_col = f"pct_touched_{level_choice}"
            close_col = f"pct_closed_below_{level_choice}"
            fail_col = f"pct_failed_bd_{level_choice}"
            ma_col = f"pct_closed_below_{level_choice}_ma20"
            close_label = f"% Closed Below {level_choice}"
            fail_label = f"% Failed Breakdown"

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.4], vertical_spacing=0.05,
                            subplot_titles=(f"{level_choice} Breakout/Breakdown %", f"Failure Rate %"))

        if close_col in piv.columns:
            fig.add_trace(go.Scatter(x=piv["date"], y=piv[close_col], name=close_label,
                                     line=dict(color="#26a69a" if is_resistance else "#ef5350", width=1.5)), row=1, col=1)
        if ma_col in piv.columns:
            fig.add_trace(go.Scatter(x=piv["date"], y=piv[ma_col], name="MA20",
                                     line=dict(color="#ff9800", width=1, dash="dot")), row=1, col=1)
        if touch_col in piv.columns:
            fig.add_trace(go.Scatter(x=piv["date"], y=piv[touch_col], name=f"% Touched {level_choice}",
                                     line=dict(color="#42a5f5", width=1, dash="dot")), row=1, col=1)

        if fail_col in piv.columns:
            fig.add_trace(go.Bar(x=piv["date"], y=piv[fail_col], name=fail_label,
                                 marker_color="#ef5350" if is_resistance else "#26a69a"), row=2, col=1)

        fig.update_layout(height=500, template="plotly_dark", showlegend=True,
                          legend=dict(orientation="h", y=1.02, x=0),
                          margin=dict(l=0, r=0, t=40, b=0))
        fig.update_yaxes(title_text="%", row=1, col=1)
        fig.update_yaxes(title_text="Failure %", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Snapshot table — latest values for all levels
        if st.checkbox("Show full pivot snapshot (latest day)"):
            levels = ["PP", "RR1", "RR2", "RR3", "SS1", "SS2", "SS3"]
            rows = []
            for lvl in levels:
                is_res = lvl in ("PP", "RR1", "RR2", "RR3")
                row = {"Level": lvl}
                row["Touched %"] = f"{
                    last.get(
                        f'pct_touched_{lvl}',
                        float('nan')):.1f}%"
                if is_res:
                    row["Closed Above %"] = f"{
                        last.get(
                            f'pct_closed_above_{lvl}',
                            float('nan')):.1f}%"
                    row["Failed BO %"] = f"{
                        last.get(
                            f'pct_failed_{lvl}',
                            float('nan')):.1f}%"
                else:
                    row["Closed Below %"] = f"{
                        last.get(
                            f'pct_closed_below_{lvl}',
                            float('nan')):.1f}%"
                    row["Failed BD %"] = f"{
                        last.get(
                            f'pct_failed_bd_{lvl}',
                            float('nan')):.1f}%"
                rows.append(row)
            st.dataframe(
                pd.DataFrame(rows).set_index("Level"),
                use_container_width=True)
