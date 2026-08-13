"""Care Transition Efficiency Dashboard.

This app intentionally reproduces the cleaning and KPI formulas in
caretransaction.ipynb. It does not generate or substitute any data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Care Transition Efficiency Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path(__file__).with_name("HHS_Unaccompanied_Alien_Children_Program.csv")
APPREHENDED = "Children apprehended and placed in CBP custody"
CBP_CUSTODY = "Children in CBP custody"
TRANSFERRED = "Children transferred out of CBP custody"
HHS_CARE = "Children in HHS Care"
DISCHARGED = "Children discharged from HHS Care"


def apply_theme() -> None:
    """Apply a restrained, public-sector healthcare visual system."""
    st.markdown(
        """
        <style>
        .stApp { background: #f5f8fb; }
        section[data-testid="stSidebar"] { background: linear-gradient(180deg, #073b5c 0%, #0b4d70 52%, #06334e 100%); }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: .7rem .85rem 2rem; }
        section[data-testid="stSidebar"] { color: #ffffff; }
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #ffffff !important; }
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #dceef6; font-size: .80rem; font-weight: 650; }
        section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div, section[data-testid="stSidebar"] [data-testid="stDateInput"] input, section[data-testid="stSidebar"] [data-testid="stNumberInput"] input { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.27); border-radius: 8px; }
        section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); margin: 1rem 0; }
        /* Keep selected values legible on the light input controls. */
        section[data-testid="stSidebar"] [data-baseweb="select"] *,
        section[data-testid="stSidebar"] [data-baseweb="input"] input,
        section[data-testid="stSidebar"] [data-testid="stDateInput"] input { color: #123247 !important; }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] [data-testid="stDateInput"] input { background: #ffffff !important; }
        [data-baseweb="popover"] [role="option"],
        [data-baseweb="popover"] [role="option"] * { color: #123247 !important; }
        .sidebar-header { border-left: 3px solid #67d3dd; padding: .25rem 0 .25rem .7rem; margin: .2rem 0 1.1rem; }
        .sidebar-eyebrow { color: #9dd7eb; font-size: .69rem; letter-spacing: .13em; font-weight: 700; }
        .sidebar-title { color: white; font-size: 1.25rem; font-weight: 750; line-height: 1.15; margin-top: .16rem; }
        .sidebar-section { color: #9dd7eb; font-size: .70rem; letter-spacing: .12em; font-weight: 750; margin: .7rem 0 .35rem; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #d9e4ec; border-radius: 12px;
            padding: 1rem; box-shadow: 0 2px 8px rgba(11, 53, 80, .05);
        }
        [data-testid="stMetricLabel"] { color: #406072; }
        h1, h2, h3 { color: #093b5a; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading care-transition data...")
def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Run the notebook's cleaning, features, and calculation cells unchanged."""
    df = pd.read_csv(file_path)
    df = df.dropna(how="all")
    df["Date"] = pd.to_datetime(df["Date"])
    df[HHS_CARE] = df[HHS_CARE].str.replace(",", "", regex=False).astype(int)
    df.rename(
        columns={
            "Children apprehended and placed in CBP custody*": APPREHENDED,
        },
        inplace=True,
    )

    # Date features from the notebook.
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()
    df["Month_Num"] = df["Date"].dt.month
    df["Weekday"] = df["Date"].dt.day_name()

    # KPI formulas from the notebook. The zero guard is the notebook's later
    # refinement for Pipeline Throughput.
    df["Transfer Efficiency Ratio"] = df[TRANSFERRED] / df[CBP_CUSTODY]
    df["Discharge Effectiveness"] = df[DISCHARGED] / df[HHS_CARE]
    df["Pipeline Throughput"] = np.where(
        df[APPREHENDED] == 0, np.nan, df[DISCHARGED] / df[APPREHENDED]
    )
    df["Backlog"] = df[APPREHENDED] - df[DISCHARGED]

    df["Day"] = df["Date"].dt.day
    df["Day_Type"] = df["Weekday"].apply(
        lambda value: "Weekend" if value in ["Saturday", "Sunday"] else "Weekday"
    )
    df["Rolling Mean"] = df[DISCHARGED].rolling(window=7).mean()
    df["Rolling Std"] = df[DISCHARGED].rolling(window=7).std()
    df["Discharge Change (%)"] = df[DISCHARGED].pct_change() * 100
    df["Discharge Change (%)"] = df["Discharge Change (%)"].replace(
        [np.inf, -np.inf], np.nan
    )
    df["Previous Discharge"] = df[DISCHARGED].shift(1)
    return df.sort_values("Date").reset_index(drop=True)


def line_chart(data: pd.DataFrame, y: str, title: str, color: str = "#0b6e8a") -> go.Figure:
    """Create a consistent Plotly time-series chart."""
    fig = px.line(data, x="Date", y=y, title=title, markers=False)
    fig.update_traces(line=dict(color=color, width=2.5))
    fig.update_layout(template="plotly_white", hovermode="x unified", margin=dict(l=15, r=15, t=50, b=15))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None, gridcolor="#e5edf2")
    return fig


def metric_value(metric: str, data: pd.DataFrame) -> tuple[str, str | None]:
    """Format values for KPI cards without altering notebook aggregations."""
    if metric == "Outcome Stability Score":
        return f"{data[DISCHARGED].std():.2f}", "Std. dev. of daily discharges"

    value = data[metric].mean()
    if metric in {"Transfer Efficiency Ratio", "Discharge Effectiveness", "Pipeline Throughput"}:
        return f"{value:.1%}", "Average for selected period"
    return f"{value:,.0f}", "Average daily backlog"


def main() -> None:
    apply_theme()
    st.title("🏥 Care Transition Efficiency Dashboard")
    st.caption("Operational monitoring for the CBP-to-HHS care pipeline")

    if not DATA_FILE.exists():
        st.error(f"Data file not found: `{DATA_FILE.name}`")
        st.info("Place the same CSV used in `caretransaction.ipynb` beside `app.py`, then reload this page.")
        return

    try:
        df = load_and_prepare_data(str(DATA_FILE))
    except (KeyError, TypeError, ValueError, pd.errors.ParserError) as error:
        st.error("The source file could not be prepared using the notebook logic.")
        st.exception(error)
        return

    st.sidebar.markdown("""<div class="sidebar-header"><div class="sidebar-eyebrow">CARE OPERATIONS</div><div class="sidebar-title">Dashboard Controls</div></div>""", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-section'>FILTER SETTINGS</div>", unsafe_allow_html=True)
    min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        format="DD/MM/YYYY",
        key="date_range_calendar_v2",
    )
    selected_month = st.sidebar.selectbox("Month", ["All months"] + list(range(1, 13)), index=0, key="month_dropdown_v2", format_func=lambda month: month if isinstance(month, str) else pd.Timestamp(2000, month, 1).month_name())
    day_type = st.sidebar.radio("Weekday / Weekend", ["All", "Weekday", "Weekend"], horizontal=True)
    selected_kpi = st.sidebar.selectbox("KPI metric toggle", ["Transfer Efficiency Ratio", "Discharge Effectiveness", "Pipeline Throughput", "Backlog"], index=0, key="kpi_metric_toggle_v2")
    st.sidebar.divider()
    with st.sidebar.expander("Alert thresholds", expanded=False):
        transfer_threshold = st.number_input("Low transfer efficiency", min_value=0.0, max_value=1.0, value=0.50, step=0.01, format="%.2f")
        discharge_threshold = st.number_input("Low discharge effectiveness", min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.2f")
        backlog_threshold = st.number_input("High backlog", value=float(df["Backlog"].mean()), step=1.0)

    if len(date_range) != 2:
        st.warning("Select both a start and end date to view the dashboard.")
        return
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    if selected_month != "All months":
        filtered = filtered[filtered["Month_Num"] == selected_month]
    if day_type != "All":
        filtered = filtered[filtered["Day_Type"] == day_type]
    if filtered.empty:
        st.warning("No records match the selected filters.")
        return

    trend_export_columns = [
        "Date", APPREHENDED, CBP_CUSTODY, TRANSFERRED, HHS_CARE, DISCHARGED,
        "Transfer Efficiency Ratio", "Discharge Effectiveness", "Pipeline Throughput", "Backlog",
    ]
    st.download_button(
        "Download filtered trend data (CSV)",
        data=filtered[trend_export_columns].to_csv(index=False).encode("utf-8"),
        file_name="care_transition_filtered_trends.csv",
        mime="text/csv",
        help="Downloads the data used by the dashboard charts for the current filters.",
    )

    cards = ["Transfer Efficiency Ratio", "Discharge Effectiveness", "Pipeline Throughput", "Backlog", "Outcome Stability Score"]
    for column, metric in zip(st.columns(5), cards):
        display, help_text = metric_value(metric, filtered)
        column.metric(metric, display, help=help_text)

    st.divider()
    st.subheader("Selected KPI Trend")
    kpi_colors = {
        "Transfer Efficiency Ratio": "#007c91",
        "Discharge Effectiveness": "#3c7d45",
        "Pipeline Throughput": "#5b5bd6",
        "Backlog": "#c65102",
    }
    selected_kpi_fig = line_chart(filtered, selected_kpi, f"{selected_kpi} Trend", kpi_colors[selected_kpi])
    if selected_kpi != "Backlog":
        selected_kpi_fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(selected_kpi_fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": True})

    st.subheader("Daily Trends")
    st.caption("Hover over a chart and use the camera icon in the toolbar to download it as a PNG.")
    trend_specs = [(APPREHENDED, "Daily Apprehension Trend"), (CBP_CUSTODY, "CBP Custody Trend"), (HHS_CARE, "HHS Care Trend"), (DISCHARGED, "Daily Discharge Trend")]
    for left, right in zip(trend_specs[::2], trend_specs[1::2]):
        col1, col2 = st.columns(2)
        chart_config = {"displaylogo": False, "displayModeBar": True, "toImageButtonOptions": {"format": "png", "scale": 2}}
        col1.plotly_chart(line_chart(filtered, *left), use_container_width=True, config=chart_config)
        col2.plotly_chart(line_chart(filtered, *right), use_container_width=True, config=chart_config)

    st.subheader("Care Pipeline Flow")
    pipeline = pd.DataFrame({"Stage": ["Children Apprehended", "Children in CBP Custody", "Transferred to HHS", "Children in HHS Care", "Children Discharged"], "Average Count": [filtered[APPREHENDED].mean(), filtered[CBP_CUSTODY].mean(), filtered[TRANSFERRED].mean(), filtered[HHS_CARE].mean(), filtered[DISCHARGED].mean()]})
    flow = px.funnel(pipeline, x="Average Count", y="Stage", title="Average Care Pipeline Flow")
    flow.update_traces(marker=dict(color="#0b6e8a"))
    flow.update_layout(template="plotly_white", margin=dict(l=15, r=15, t=50, b=15), coloraxis_showscale=False)
    st.plotly_chart(flow, use_container_width=True)

    st.subheader("Transfer & Discharge Efficiency")
    col1, col2 = st.columns(2)
    col1.plotly_chart(line_chart(filtered, "Transfer Efficiency Ratio", "Transfer Efficiency Trend", "#007c91"), use_container_width=True)
    col2.plotly_chart(line_chart(filtered, "Discharge Effectiveness", "Discharge Effectiveness Trend", "#3c7d45"), use_container_width=True)

    st.subheader("Bottleneck Detection")
    col1, col2 = st.columns(2)
    col1.plotly_chart(line_chart(filtered, "Pipeline Throughput", "Pipeline Throughput Trend", "#5b5bd6"), use_container_width=True)
    backlog_fig = line_chart(filtered, "Backlog", "Backlog Trend", "#c65102")
    backlog_fig.add_hline(y=filtered["Backlog"].mean(), line_dash="dash", line_color="#c62828", annotation_text="Selected-period average")
    col2.plotly_chart(backlog_fig, use_container_width=True)

    st.subheader("Outcome Trend Analysis")
    rolling_fig = go.Figure()
    rolling_fig.add_trace(go.Scatter(x=filtered["Date"], y=filtered[DISCHARGED], name="Daily Discharge", line=dict(color="#9bb9c8")))
    rolling_fig.add_trace(go.Scatter(x=filtered["Date"], y=filtered["Rolling Mean"], name="7-Day Rolling Mean", line=dict(color="#0b6e8a", width=3)))
    rolling_fig.update_layout(title="Rolling Mean", template="plotly_white", hovermode="x unified", margin=dict(l=15, r=15, t=50, b=15))
    col1, col2 = st.columns(2)
    col1.plotly_chart(rolling_fig, use_container_width=True)
    col2.plotly_chart(line_chart(filtered, "Rolling Std", "Rolling Standard Deviation", "#8e4d92"), use_container_width=True)

    st.subheader("Threshold-based Alerts")
    high_backlog = filtered["Backlog"] > backlog_threshold
    low_transfer = filtered["Transfer Efficiency Ratio"] < transfer_threshold
    low_discharge = filtered["Discharge Effectiveness"] < discharge_threshold
    a, b, c = st.columns(3)
    a.warning(f"High Backlog: {high_backlog.sum()} day(s) above {backlog_threshold:,.0f}.")
    b.warning(f"Low Transfer Efficiency: {low_transfer.sum()} day(s) below {transfer_threshold:.0%}.")
    c.warning(f"Low Discharge Effectiveness: {low_discharge.sum()} day(s) below {discharge_threshold:.0%}.")

    st.subheader("Exception Tables")
    increase = filtered[filtered["Discharge Change (%)"] > 20][["Date", "Previous Discharge", DISCHARGED, "Discharge Change (%)"]]
    drop = filtered[filtered["Discharge Change (%)"] < -20][["Date", DISCHARGED, "Discharge Change (%)"]]
    highest_backlog = filtered.sort_values("Backlog", ascending=False)[["Date", "Backlog"]].head(10)
    col1, col2, col3 = st.columns(3)
    col1.markdown("**Sudden Increase Days**")
    col1.dataframe(increase, use_container_width=True, hide_index=True)
    col2.markdown("**Sudden Drop Days**")
    col2.dataframe(drop, use_container_width=True, hide_index=True)
    col3.markdown("**Highest Backlog Days**")
    col3.dataframe(highest_backlog, use_container_width=True, hide_index=True)

    st.caption(f"KPI metric toggle selection: {selected_kpi}. Source calculations are reproduced from the supplied notebook.")


if __name__ == "__main__":
    main()
