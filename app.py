import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

#  Page Config 
st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0d0f14;
    color: #e8eaf0;
}

section[data-testid="stSidebar"] {
    background-color: #13161e;
    border-right: 1px solid #1f2330;
}

.metric-card {
    background: linear-gradient(135deg, #13161e 0%, #1a1d28 100%);
    border: 1px solid #1f2330;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent, #4ade80);
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #e8eaf0;
    line-height: 1;
}

.metric-delta {
    font-size: 12px;
    margin-top: 6px;
    color: #6b7280;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4ade80;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1f2330;
}

.risk-high { color: #f87171; }
.risk-med  { color: #fbbf24; }
.risk-low  { color: #4ade80; }

div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6b7280 !important;
}

.stPlotlyChart { border-radius: 12px; overflow: hidden; }

h1 { font-family: 'Space Mono', monospace !important; color: #e8eaf0 !important; }
h2, h3 { font-family: 'DM Sans', sans-serif !important; color: #c9cdd8 !important; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

#  Load & Preprocess Data 
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Binary encode
    df["ChurnBinary"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Tenure group
    bins = [0, 12, 24, 48, 72]
    labels = ["012 mo", "1324 mo", "2548 mo", "4972 mo"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # Risk score (simple)
    risk = (
        (df["Contract"] == "Month-to-month").astype(int) * 3 +
        (df["InternetService"] == "Fiber optic").astype(int) * 2 +
        (df["tenure"] < 12).astype(int) * 2 +
        (df["MonthlyCharges"] > 70).astype(int) * 1 +
        (df["TechSupport"] == "No").astype(int) * 1 +
        (df["OnlineSecurity"] == "No").astype(int) * 1
    )
    df["RiskScore"] = risk
    df["RiskSegment"] = pd.cut(risk, bins=[-1, 2, 5, 10],
                                labels=["Low Risk", "Medium Risk", "High Risk"])
    return df

df = load_data()

#  Sidebar Filters 
with st.sidebar:
    st.markdown("##  CHURN ANALYTICS")
    st.markdown("<div style='height:1px;background:#1f2330;margin:8px 0 20px'></div>", unsafe_allow_html=True)

    st.markdown("**FILTERS**")

    contract_filter = st.multiselect(
        "Contract Type",
        options=df["Contract"].unique(),
        default=list(df["Contract"].unique())
    )

    internet_filter = st.multiselect(
        "Internet Service",
        options=df["InternetService"].unique(),
        default=list(df["InternetService"].unique())
    )

    tenure_filter = st.multiselect(
        "Tenure Group",
        options=list(df["TenureGroup"].cat.categories),
        default=list(df["TenureGroup"].cat.categories)
    )

    senior_filter = st.multiselect(
        "Senior Citizen",
        options=["Yes", "No"],
        default=["Yes", "No"]
    )

    st.markdown("<div style='height:1px;background:#1f2330;margin:20px 0'></div>", unsafe_allow_html=True)
    st.caption(f"Dataset: {len(df):,} customers  Telco Churn")

#  Apply Filters 
filtered = df[
    df["Contract"].isin(contract_filter) &
    df["InternetService"].isin(internet_filter) &
    df["TenureGroup"].isin(tenure_filter) &
    df["SeniorCitizen"].isin(senior_filter)
]

#  Plot theme 
PLOT_BG   = "#0d0f14"
PAPER_BG  = "#13161e"
GRID_COL  = "#1f2330"
TEXT_COL  = "#9ca3af"
GREEN     = "#4ade80"
RED       = "#f87171"
AMBER     = "#fbbf24"
BLUE      = "#60a5fa"
PURPLE    = "#a78bfa"

def base_layout(**kwargs):
    defaults = dict(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="DM Sans", color=TEXT_COL, size=12),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    defaults.update(kwargs)
    return defaults

#  Header 
st.markdown("# Customer Churn Analytics")
st.markdown(f"<p style='color:#6b7280;font-size:14px;margin-top:-12px'>Showing <b style='color:#e8eaf0'>{len(filtered):,}</b> of {len(df):,} customers after filters</p>", unsafe_allow_html=True)

#  KPI Row 
total      = len(filtered)
churned    = filtered["ChurnBinary"].sum()
churn_rate = churned / total * 100 if total else 0
avg_tenure = filtered["tenure"].mean()
avg_charge = filtered["MonthlyCharges"].mean()
rev_at_risk = filtered[filtered["Churn"] == "Yes"]["MonthlyCharges"].sum()
high_risk  = (filtered["RiskSegment"] == "High Risk").sum()

col1, col2, col3, col4, col5 = st.columns(5)

def kpi(col, label, value, delta="", accent="#4ade80"):
    col.markdown(f"""
    <div class="metric-card" style="--accent:{accent}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

with col1: kpi(col1, "Total Customers", f"{total:,}", "in filtered view", BLUE)
with col2: kpi(col2, "Churned", f"{churned:,}", f"{churn_rate:.1f}% churn rate", RED)
with col3: kpi(col3, "Avg Tenure", f"{avg_tenure:.0f} mo", "months retained", GREEN)
with col4: kpi(col4, "Avg Monthly $", f"${avg_charge:.0f}", "per customer", AMBER)
with col5: kpi(col5, "Revenue at Risk", f"${rev_at_risk:,.0f}", "monthly from churned", RED)

#  Row 1: Churn by Contract + Churn Rate by Tenure 
st.markdown("<div class='section-header'>CHURN DRIVERS</div>", unsafe_allow_html=True)

r1c1, r1c2 = st.columns(2)

with r1c1:
    grp = filtered.groupby("Contract").agg(
        Total=("ChurnBinary", "count"),
        Churned=("ChurnBinary", "sum")
    ).reset_index()
    grp["Retained"] = grp["Total"] - grp["Churned"]
    grp["ChurnRate"] = (grp["Churned"] / grp["Total"] * 100).round(1)

    fig = go.Figure()
    fig.add_bar(name="Retained", x=grp["Contract"], y=grp["Retained"],
                marker_color="#1f2d3d", marker_line_width=0)
    fig.add_bar(name="Churned", x=grp["Contract"], y=grp["Churned"],
                marker_color=RED, marker_line_width=0,
                text=grp["ChurnRate"].apply(lambda x: f"{x}%"),
                textposition="outside", textfont=dict(color=RED, size=13, family="Space Mono"))
    fig.update_layout(**base_layout(title="Churn by Contract Type", barmode="stack",
                                     legend=dict(orientation="h", y=-0.15),
                                     xaxis=dict(gridcolor=GRID_COL),
                                     yaxis=dict(gridcolor=GRID_COL)))
    st.plotly_chart(fig, width='stretch')

with r1c2:
    tenure_grp = filtered.groupby("TenureGroup", observed=True).agg(
        ChurnRate=("ChurnBinary", "mean")
    ).reset_index()
    tenure_grp["ChurnRate"] *= 100

    fig2 = go.Figure()
    fig2.add_scatter(
        x=tenure_grp["TenureGroup"].astype(str),
        y=tenure_grp["ChurnRate"],
        mode="lines+markers",
        line=dict(color=AMBER, width=3),
        marker=dict(size=10, color=AMBER, line=dict(color=PLOT_BG, width=2)),
        fill="tozeroy",
        fillcolor="rgba(251,191,36,0.08)"
    )
    fig2.update_layout(**base_layout(title="Churn Rate by Tenure Group",
                                      xaxis=dict(gridcolor=GRID_COL),
                                      yaxis=dict(gridcolor=GRID_COL, ticksuffix="%")))
    st.plotly_chart(fig2, width="stretch")

#  Row 2: Monthly Charges Distribution + Internet Service 
r2c1, r2c2 = st.columns(2)

with r2c1:
    fig3 = go.Figure()
    for label, color in [("No", GREEN), ("Yes", RED)]:
        subset = filtered[filtered["Churn"] == label]["MonthlyCharges"]
        fig3.add_trace(go.Histogram(
            x=subset, name=f"Churn: {label}",
            marker_color=color, opacity=0.7,
            nbinsx=30, histnorm="probability density"
        ))
    fig3.update_layout(**base_layout(title="Monthly Charges Distribution",
                                      barmode="overlay",
                                      xaxis=dict(gridcolor=GRID_COL, tickprefix="$"),
                                      yaxis=dict(gridcolor=GRID_COL),
                                      legend=dict(orientation="h", y=-0.15)))
    st.plotly_chart(fig3, width="stretch")

with r2c2:
    inet = filtered.groupby(["InternetService", "Churn"]).size().reset_index(name="Count")
    fig4 = px.bar(inet, x="InternetService", y="Count", color="Churn",
                  color_discrete_map={"Yes": RED, "No": "#1f3a2d"},
                  barmode="group", title="Churn by Internet Service")
    fig4.update_layout(**base_layout(legend=dict(orientation="h", y=-0.15),
                                      xaxis=dict(gridcolor=GRID_COL),
                                      yaxis=dict(gridcolor=GRID_COL)))
    st.plotly_chart(fig4, width="stretch")

#  Row 3: Risk Segmentation 
st.markdown("<div class='section-header'>RISK SEGMENTATION</div>", unsafe_allow_html=True)

r3c1, r3c2, r3c3 = st.columns([1, 1, 2])

with r3c1:
    risk_counts = filtered["RiskSegment"].value_counts().reset_index()
    risk_counts.columns = ["Segment", "Count"]
    colors_map = {"High Risk": RED, "Medium Risk": AMBER, "Low Risk": GREEN}
    fig5 = go.Figure(go.Pie(
        labels=risk_counts["Segment"],
        values=risk_counts["Count"],
        hole=0.65,
        marker_colors=[colors_map.get(s, BLUE) for s in risk_counts["Segment"]],
        textinfo="percent+label",
        textfont=dict(size=11)
    ))
    fig5.update_layout(**base_layout(title="Risk Distribution",
                                      showlegend=False,
                                      margin=dict(l=10, r=10, t=40, b=10)))
    st.plotly_chart(fig5, width="stretch")

with r3c2:
    risk_churn = filtered.groupby("RiskSegment", observed=True).agg(
        ChurnRate=("ChurnBinary", "mean"),
        Count=("ChurnBinary", "count")
    ).reset_index()
    risk_churn["ChurnRate"] = (risk_churn["ChurnRate"] * 100).round(1)

    fig6 = go.Figure(go.Bar(
        x=risk_churn["RiskSegment"].astype(str),
        y=risk_churn["ChurnRate"],
        marker_color=[colors_map.get(s, BLUE) for s in risk_churn["RiskSegment"]],
        text=risk_churn["ChurnRate"].apply(lambda x: f"{x}%"),
        textposition="outside",
        textfont=dict(family="Space Mono", size=12),
        marker_line_width=0
    ))
    fig6.update_layout(**base_layout(title="Churn Rate per Risk Tier",
                                      xaxis=dict(gridcolor=GRID_COL),
                                      yaxis=dict(gridcolor=GRID_COL, ticksuffix="%")))
    st.plotly_chart(fig6, width="stretch")

with r3c3:
    fig7 = px.scatter(
        filtered.sample(min(800, len(filtered)), random_state=42),
        x="tenure", y="MonthlyCharges",
        color="RiskSegment",
        color_discrete_map={"High Risk": RED, "Medium Risk": AMBER, "Low Risk": GREEN},
        title="Customer Landscape: Tenure vs Monthly Charges",
        labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges ($)"},
        opacity=0.7
    )
    fig7.update_traces(marker=dict(size=6))
    fig7.update_layout(**base_layout(legend=dict(orientation="h", y=-0.15),
                                      xaxis=dict(gridcolor=GRID_COL),
                                      yaxis=dict(gridcolor=GRID_COL)))
    st.plotly_chart(fig7, width="stretch")

#  Row 4: Retention Metrics 
st.markdown("<div class='section-header'>RETENTION INSIGHTS</div>", unsafe_allow_html=True)

r4c1, r4c2 = st.columns(2)

with r4c1:
    features = ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]
    churn_rates = []
    for f in features:
        rate = filtered[filtered[f] == "Yes"]["ChurnBinary"].mean() * 100
        no_rate = filtered[filtered[f] == "No"]["ChurnBinary"].mean() * 100
        churn_rates.append({"Feature": f, "With Service": round(rate, 1), "Without Service": round(no_rate, 1)})

    feat_df = pd.DataFrame(churn_rates)
    fig8 = go.Figure()
    fig8.add_bar(name="With Service", x=feat_df["Feature"], y=feat_df["With Service"],
                 marker_color=GREEN, marker_line_width=0)
    fig8.add_bar(name="Without Service", x=feat_df["Feature"], y=feat_df["Without Service"],
                 marker_color=RED, marker_line_width=0)
    fig8.update_layout(**base_layout(title="Churn Rate: With vs Without Add-on Services",
                                      barmode="group",
                                      xaxis=dict(gridcolor=GRID_COL, tickangle=-20),
                                      yaxis=dict(gridcolor=GRID_COL, ticksuffix="%"),
                                      legend=dict(orientation="h", y=-0.2)))
    st.plotly_chart(fig8, width="stretch")

with r4c2:
    pay_grp = filtered.groupby("PaymentMethod").agg(
        ChurnRate=("ChurnBinary", "mean"),
        Count=("ChurnBinary", "count")
    ).reset_index()
    pay_grp["ChurnRate"] = (pay_grp["ChurnRate"] * 100).round(1)
    pay_grp = pay_grp.sort_values("ChurnRate", ascending=True)

    fig9 = go.Figure(go.Bar(
        x=pay_grp["ChurnRate"],
        y=pay_grp["PaymentMethod"],
        orientation="h",
        marker=dict(
            color=pay_grp["ChurnRate"],
            colorscale=[[0, GREEN], [0.5, AMBER], [1, RED]],
            line=dict(width=0)
        ),
        text=pay_grp["ChurnRate"].apply(lambda x: f"{x}%"),
        textposition="outside",
        textfont=dict(family="Space Mono", size=11)
    ))
    fig9.update_layout(**base_layout(title="Churn Rate by Payment Method",
                                      xaxis=dict(gridcolor=GRID_COL, ticksuffix="%"),
                                      yaxis=dict(gridcolor=GRID_COL)))
    st.plotly_chart(fig9, width="stretch")

#  High Risk Customer Table 
st.markdown("<div class='section-header'>HIGH RISK CUSTOMERS</div>", unsafe_allow_html=True)

high_risk_df = filtered[filtered["RiskSegment"] == "High Risk"][[
    "customerID", "Contract", "tenure", "MonthlyCharges",
    "InternetService", "TechSupport", "OnlineSecurity", "RiskScore", "Churn"
]].sort_values("RiskScore", ascending=False).head(15).reset_index(drop=True)

high_risk_df.columns = ["Customer ID", "Contract", "Tenure (mo)", "Monthly $",
                         "Internet", "Tech Support", "Security", "Risk Score", "Churned"]

st.dataframe(
    high_risk_df.style
    .map(lambda v: "color: #f87171; font-weight: bold" if v == "Yes" else "", subset=["Churned"])
    .format({"Monthly $": "${:.2f}", "Risk Score": "{:.0f}"}),
    width="stretch",
    height=400
)

#  Footer 
st.markdown("""
<div style='text-align:center;padding:32px 0 16px;color:#374151;font-family:Space Mono,monospace;font-size:11px;letter-spacing:2px'>
TELCO CUSTOMER CHURN ANALYTICS  BUILT WITH STREAMLIT + PLOTLY
</div>
""", unsafe_allow_html=True)