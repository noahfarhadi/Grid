"""
Investment Grid: risk and return of the assets in an uploaded price file between a start and an end date.
Each asset is placed in one of four classes by the median of all assets.
Educational use only. Not investment advice.

Run locally:   streamlit run app.py
Files needed:  app.py, requirements.txt, .streamlit/config.toml (optional, light theme)
Input file:    CSV or Excel, first column = date, one column per asset (header = asset name),
               closing prices. Example: "Visa Inc. (V)" shows "V" in the price chart legend.
"""
import io
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------
# Step 0: all settings live here (nothing is hard-coded below)
# ---------------------------------------------------------------
CONFIG = {
    "page_title": "Investment Grid",
    "min_assets": 4,                       # fewer assets give no meaningful median classes
    "min_observations": 20,                # fewer returns than this: no reliable risk
    "days_per_year": 365,                  # years = (last date - first date) / 365, as in the Excel
    # periods per year from the date spacing: (largest median gap in days, periods per year)
    "freq_map": [(4, 252), (10, 52), (40, 12), (120, 4)],
    "fallback_periods_per_year": 1,
    # fixed page: sizes in pixels, the page never stretches with the window
    "page_width_px": 1320,
    "table_width_px": 600,
    "table_row_px": 35,                    # row height of the table (header counts as one row)
    "chart_width_px": 640,
    "scatter_height_px": 480,
    "price_height_px": 330,
    # price development
    "default_price_assets": [1, 2, 3],     # asset numbers preselected
    "max_price_assets": 8,
    "show_median_lines": False,            # True draws the two median lines in the grid chart
    # class rules, identical to the Excel formula (risk above median = high, return above median = high)
    # key = (high risk, high return): (class name, marker symbol, color)
    "classes": {
        (True, True): ("Problem children", "circle", "#eb6834"),
        (True, False): ("Eagles", "diamond", "#4a3aa7"),
        (False, True): ("Trouble makers", "square", "#2a78d6"),
        (False, False): ("Turtles", "triangle-up", "#1baf7a"),
    },
    "line_colors": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                    "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "ink": "#0b0b0b", "ink_soft": "#52514e", "grid": "#e6e5e1", "surface": "#ffffff",
}

st.set_page_config(page_title=CONFIG["page_title"], layout="wide")

# Step 1: fixed page. The container has one fixed width, so nothing stretches or moves
# when the browser window is made larger (a narrow window scrolls sideways instead).
st.markdown(
    f"""
    <style>
    .block-container {{
        width: {CONFIG['page_width_px']}px; max-width: {CONFIG['page_width_px']}px;
        min-width: {CONFIG['page_width_px']}px; margin: 0 auto; padding: 1.6rem 1rem 3rem;
    }}
    [data-testid="stAppViewContainer"] {{ overflow-x: auto; }}
    h1 {{ font-size: 1.9rem !important; padding-bottom: 0.2rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Step 2: read the uploaded file (CSV or Excel) into a price table: dates down, assets across
@st.cache_data(show_spinner=False)
def read_prices(raw: bytes, filename: str):
    if filename.lower().endswith((".xlsx", ".xls")):
        table = pd.read_excel(io.BytesIO(raw), index_col=0)
    else:
        # sep=None lets pandas detect comma or semicolon separators
        table = pd.read_csv(io.BytesIO(raw), index_col=0, sep=None, engine="python")

    # dates: try the default reading first, then day-first (for example 31.12.2022)
    dates = pd.to_datetime(table.index, errors="coerce")
    if dates.isna().mean() > 0.5:
        dates = pd.to_datetime(table.index, errors="coerce", dayfirst=True)
    table.index = dates
    table = table[~table.index.isna()].sort_index()
    table = table[~table.index.duplicated(keep="last")]

    # numbers: accept a decimal comma, turn everything else into empty cells
    for col in table.columns:
        if table[col].dtype == object:
            table[col] = table[col].astype(str).str.replace(",", ".", regex=False)
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table.columns = [str(c).strip() for c in table.columns]
    table = table.loc[:, ~pd.Index(table.columns).duplicated()]
    table = table.dropna(axis=1, how="all").dropna(how="all")
    return table


# Step 3: periods per year from the spacing of the dates (252 daily, 52 weekly, 12 monthly ...)
def detect_periods_per_year(index):
    gaps = pd.Series(index).diff().dt.days.dropna()
    if gaps.empty:
        return CONFIG["fallback_periods_per_year"]
    for max_gap, periods in CONFIG["freq_map"]:
        if gaps.median() <= max_gap:
            return periods
    return CONFIG["fallback_periods_per_year"]


# Step 4: short label for the price chart: the text in the last brackets, else the full name
def short_label(name):
    found = re.search(r"\(([^)]+)\)\s*$", name)
    return found.group(1) if found else name


# Step 5: risk, return and class for every asset between start and end
@st.cache_data(show_spinner=False)
def compute_grid(prices, start, end, periods_per_year):
    window = prices.loc[pd.Timestamp(start):pd.Timestamp(end)]

    # simple returns, only between two real consecutive prices (no returns from empty cells)
    changes = window.pct_change(fill_method=None)
    risk = changes.std(ddof=1) * np.sqrt(periods_per_year)

    # annualized return = (last price / first price) ^ (1 / years) - 1
    first_date = window.apply(lambda s: s.first_valid_index())
    last_date = window.apply(lambda s: s.last_valid_index())
    first = window.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    last = window.apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    years = (pd.to_datetime(last_date) - pd.to_datetime(first_date)).dt.days / CONFIG["days_per_year"]
    ret = (last / first) ** (1.0 / years) - 1.0

    grid = pd.DataFrame({"No": range(1, len(window.columns) + 1), "Asset": list(window.columns),
                         "Risk": risk.values, "Return": ret.values})
    grid = grid.dropna(subset=["Risk", "Return"]).reset_index(drop=True)
    n_obs = int(changes.count().max()) if len(changes) else 0

    # class by the median of all assets (same rule as the Excel formula)
    med_risk, med_ret = float(grid["Risk"].median()), float(grid["Return"].median())
    grid["Class"] = [CONFIG["classes"][(r > med_risk, x > med_ret)][0]
                     for r, x in zip(grid["Risk"], grid["Return"])]
    index100 = window / first * 100.0
    return grid, med_risk, med_ret, n_obs, index100


# Step 6: draw a plotly chart at a fixed pixel size
def show_chart(fig, height):
    fig.update_layout(
        width=CONFIG["chart_width_px"], height=height, autosize=False,
        paper_bgcolor=CONFIG["surface"], plot_bgcolor=CONFIG["surface"],
        font=dict(family="Arial, sans-serif", size=12, color=CONFIG["ink_soft"]),
        margin=dict(l=60, r=15, t=10, b=60),
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0),
    )
    fig.update_xaxes(gridcolor=CONFIG["grid"], zeroline=False, linecolor=CONFIG["grid"])
    fig.update_yaxes(gridcolor=CONFIG["grid"], zeroline=True, zerolinecolor=CONFIG["ink_soft"],
                     zerolinewidth=1, linecolor=CONFIG["grid"])
    plot_config = {"displayModeBar": False, "scrollZoom": False}
    try:
        st.plotly_chart(fig, width="content", config=plot_config)
    except TypeError:  # older Streamlit versions
        st.plotly_chart(fig, use_container_width=False, config=plot_config)


# ---------------------------------------------------------------
# Header and upload (everything below appears after the file is read)
# ---------------------------------------------------------------
st.title("Investment Grid")
st.caption("Risk and return of the assets in your file. Each asset is assigned to a class by the median of all assets.")

uploaded = st.file_uploader(
    "Upload price file (CSV or Excel): first column = date, one column per asset, closing prices",
    type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.info("Upload a price file to start. The start and end date appear after the upload.")
    st.stop()

try:
    prices = read_prices(uploaded.getvalue(), uploaded.name)
except Exception as exc:
    st.error(f"The file could not be read ({type(exc).__name__}). Check the format: first column = date, then one column per asset.")
    st.stop()
if prices.shape[1] < CONFIG["min_assets"] or len(prices) < CONFIG["min_observations"]:
    st.error(f"The file needs a date column, at least {CONFIG['min_assets']} asset columns and "
             f"at least {CONFIG['min_observations']} rows with valid dates and prices.")
    st.stop()

periods = detect_periods_per_year(prices.index)
data_first, data_last = prices.index[0].date(), prices.index[-1].date()
st.caption(f"{prices.shape[1]} assets, {len(prices)} rows, {data_first} to {data_last}, "
           f"{periods} periods per year (detected from the dates).")

# ---------------------------------------------------------------
# The two dates (everything below recomputes when a date changes)
# ---------------------------------------------------------------
d1, d2, _ = st.columns([2, 2, 5])
start = d1.date_input("Start date", data_first, min_value=data_first, max_value=data_last)
end = d2.date_input("End date", data_last, min_value=data_first, max_value=data_last)
if start >= end:
    st.error("The start date must be before the end date.")
    st.stop()

grid, med_risk, med_ret, n_obs, index100 = compute_grid(prices, start, end, periods)
if n_obs < CONFIG["min_observations"] or len(grid) < CONFIG["min_assets"]:
    st.warning(f"Only {n_obs} observations in this range. Please choose a longer period.")
    st.stop()

# ---------------------------------------------------------------
# Left: the grid table. Right: grid chart, zone legend, price development
# ---------------------------------------------------------------
left, right = st.columns([CONFIG["table_width_px"], CONFIG["chart_width_px"]], gap="medium")

with left:
    # Step 7: table with the same columns as the Excel dashboard
    table = pd.DataFrame({
        "#": grid["No"],
        "Assets": grid["Asset"],
        "Median Risk": grid["Risk"] * 100,
        "Median Return": grid["Return"] * 100,
        "Asset Class": grid["Class"],
    })
    st.dataframe(
        table, hide_index=True, width=CONFIG["table_width_px"],
        height=CONFIG["table_row_px"] * (len(table) + 1) + 5,
        column_config={
            "#": st.column_config.NumberColumn("#", width=36, format="%d"),
            "Assets": st.column_config.TextColumn("Assets", width=245),
            "Median Risk": st.column_config.NumberColumn("Median Risk", width=90, format="%.1f%%"),
            "Median Return": st.column_config.NumberColumn("Median Return", width=100, format="%.1f%%"),
            "Asset Class": st.column_config.TextColumn("Asset Class", width=120),
        },
    )
    st.markdown(f"**Median** &nbsp;&nbsp; risk **{med_risk:.1%}** &nbsp;&nbsp; return **{med_ret:.1%}**")

with right:
    # Step 8: grid chart (x = risk, y = return), one marker shape and color per class
    fig = go.Figure()
    for (high_risk, high_ret), (name, symbol, color) in CONFIG["classes"].items():
        part = grid[grid["Class"] == name]
        fig.add_trace(go.Scatter(
            x=part["Risk"], y=part["Return"], mode="markers+text", name=name,
            text=part["No"].astype(str), textposition="top right",
            textfont=dict(size=10, color=CONFIG["ink_soft"]),
            marker=dict(symbol=symbol, size=10, color=color,
                        line=dict(color=CONFIG["surface"], width=1.5)),
            customdata=np.stack([part["Asset"], part["Class"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>Risk %{x:.1%}<br>Return %{y:.1%}"
                          "<br>%{customdata[1]}<extra></extra>",
        ))
    if CONFIG["show_median_lines"]:
        fig.add_vline(x=med_risk, line=dict(color=CONFIG["ink_soft"], width=1, dash="dot"))
        fig.add_hline(y=med_ret, line=dict(color=CONFIG["ink_soft"], width=1, dash="dot"))
    fig.update_xaxes(title_text="Risk (annualized)", tickformat=".0%")
    fig.update_yaxes(title_text="Return (annualized)", tickformat=".0%")
    show_chart(fig, CONFIG["scatter_height_px"])

    # Step 9: zone legend, built from the same rule that sets the classes
    rows = ["| Zone | Risk | Return | Abstract |", "|---|---|---|---|"]
    for k, ((high_risk, high_ret), (name, _, _)) in enumerate(CONFIG["classes"].items(), start=1):
        rows.append(f"| Zone {k} | {'High' if high_risk else 'Low'} | {'High' if high_ret else 'Low'} | {name} |")
    st.markdown("\n".join(rows))

    # Step 10: price development of the chosen assets (start of the range = 100)
    st.markdown("**Price development** (start of the range = 100)")
    label = {int(r.No): f"{int(r.No)}  {r.Asset}" for r in grid.itertuples()}
    defaults = [n for n in CONFIG["default_price_assets"] if n in label]
    picked = st.multiselect("Assets", list(label), default=defaults,
                            format_func=lambda n: label[n], max_selections=CONFIG["max_price_assets"],
                            label_visibility="collapsed")
    if picked:
        fig2 = go.Figure()
        for slot, no in enumerate(picked):
            asset = grid.loc[grid["No"] == no, "Asset"].iloc[0]
            tag = f"{no} {short_label(asset)}"
            fig2.add_trace(go.Scatter(
                x=index100.index, y=index100[asset], mode="lines", name=tag,
                line=dict(width=2, color=CONFIG["line_colors"][slot % len(CONFIG["line_colors"])]),
                hovertemplate="%{y:.1f}<extra>" + tag + "</extra>",
            ))
        fig2.update_layout(hovermode="x unified")
        fig2.update_yaxes(title_text="Index")
        show_chart(fig2, CONFIG["price_height_px"])
    else:
        st.info("Choose one or more assets to see their price development.")

# ---------------------------------------------------------------
# Version log
# v1.0 - Investment Grid app built from InvestmentGridR2.xlsx:
#        grid table (risk, return, class), grid chart, zone legend, start and end date,
#        price development (index = 100), fixed page width, all settings in CONFIG.
#        Data come from an uploaded CSV or Excel file (first column = date, one column per asset);
#        no data is stored in the app. Periods per year are detected from the dates.
#        Risk is calculated from consecutive real prices only. The Excel risk column
#        includes a false -100% return in the last row of 39 assets, so risk values differ.
#        Years for the return use the first and last price date in the chosen range.
# ---------------------------------------------------------------
