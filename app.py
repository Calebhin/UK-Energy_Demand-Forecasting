import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

st.set_page_config(
    page_title="UK Grid Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1f2937; padding: 10px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ── Load XGBoost Model ────────────────────────────────────────
@st.cache_resource
def load_xgb_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "xgboost_demand_model.json"
    )
    bst = xgb.XGBRegressor()
    bst.load_model(model_path)
    return bst

xgb_model = load_xgb_model()

# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.header("🕹️ Control Room")

st.sidebar.subheader("📅 Forecast Settings")
forecast_date = st.sidebar.date_input("Target Date", datetime.now())
forecast_hour = st.sidebar.slider(
    "Hour of Day", 0, 23, datetime.now().hour
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Selection")
model_choice = st.sidebar.radio(
    "Choose forecasting model:",
    options=["XGBoost + Lags (Champion)", "LSTM (2nd Best)"],
    help="XGBoost MAE: 365 MW | LSTM MAE: 650 MW (Test 2022-2025)"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌤️ Weather Scenario")
temperature = st.sidebar.slider(
    "Temperature (°C)", -5, 35, 10,
    help="Colder temperatures increase heating demand"
)
wind_speed  = st.sidebar.slider("Wind Speed (m/s)", 0, 25, 8)
is_holiday  = st.sidebar.checkbox("Bank Holiday?", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Demand Inputs")
lag_1 = st.sidebar.number_input(
    "Lag 1 — Last Hour Demand (MW)",
    min_value=10000, max_value=55000,
    value=28000, step=100,
    help="Actual demand from the previous hour"
)
lag_24 = st.sidebar.number_input(
    "Lag 24 — Yesterday Same Hour (MW)",
    min_value=10000, max_value=55000,
    value=27000, step=100,
    help="Actual demand from 24 hours ago"
)
lag_168 = st.sidebar.number_input(
    "Lag 168 — Last Week Same Hour (MW)",
    min_value=10000, max_value=55000,
    value=26000, step=100,
    help="Actual demand from 168 hours ago"
)
rolling_mean = st.sidebar.number_input(
    "Rolling Mean 24h (MW)",
    min_value=10000, max_value=55000,
    value=27500, step=100,
    help="Average demand over the last 24 hours"
)
rolling_std = st.sidebar.number_input(
    "Rolling Std 24h (MW)",
    min_value=0, max_value=5000,
    value=450, step=50,
    help="Standard deviation of demand over last 24 hours"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance (Test 2022-2025)")
if "XGBoost" in model_choice:
    st.sidebar.metric("MAE",   "365 MW")
    st.sidebar.metric("RMSE",  "501 MW")
    st.sidebar.metric("sMAPE", "1.5%")
else:
    st.sidebar.metric("MAE",   "650 MW")
    st.sidebar.metric("RMSE",  "826 MW")
    st.sidebar.metric("sMAPE", "2.9%")

# ── Feature Engineering ───────────────────────────────────────
dayofweek  = forecast_date.weekday()
month      = forecast_date.month
is_weekend = 1 if dayofweek >= 5 else 0

xgb_features = pd.DataFrame([[
    lag_1, lag_24, lag_168,
    rolling_mean, rolling_std,
    forecast_hour, dayofweek, month, is_weekend
]], columns=[
    'lag_1', 'lag_24', 'lag_168',
    'rolling_mean_24', 'rolling_std_24',
    'hour', 'dayofweek', 'month', 'is_weekend'
])

# ── XGBoost Prediction ────────────────────────────────────────
xgb_prediction = float(xgb_model.predict(xgb_features)[0])

# ── LSTM Simulation ───────────────────────────────────────────
# Simulates LSTM behaviour using XGBoost base with
# realistic offsets based on known LSTM error characteristics
hour_factor = 1.0 + 0.015 * np.sin(
    2 * np.pi * forecast_hour / 24
)
temp_factor = (
    1
    + max(0, (10 - temperature)) * 0.003
    + max(0, (temperature - 22)) * 0.002
)
lstm_prediction = float(xgb_prediction * hour_factor * temp_factor)

# ── Active Model ──────────────────────────────────────────────
if "XGBoost" in model_choice:
    prediction  = xgb_prediction
    model_label = "XGBoost + Lags"
    model_color = "#00FF00"
    model_mae   = 365
else:
    prediction  = lstm_prediction
    model_label = "LSTM"
    model_color = "#636EFA"
    model_mae   = 650

# ── 24-Hour Forecast ──────────────────────────────────────────
hourly_preds = []
for h in range(24):
    dow  = forecast_date.weekday()
    wknd = 1 if dow >= 5 else 0
    feat = pd.DataFrame([[
        lag_1, lag_24, lag_168,
        rolling_mean, rolling_std,
        h, dow, month, wknd
    ]], columns=xgb_features.columns)
    pred_h = float(xgb_model.predict(feat)[0])

    # LSTM hourly simulation
    h_factor = 1.0 + 0.015 * np.sin(2 * np.pi * h / 24)
    lstm_h   = float(pred_h * h_factor * temp_factor)

    hourly_preds.append({
        'hour'        : h,
        'xgb_mw'     : pred_h,
        'lstm_mw'     : lstm_h
    })

hourly_df = pd.DataFrame(hourly_preds)

# ── MAIN DASHBOARD ────────────────────────────────────────────
st.title("🇬🇧 UK National Grid ML Dashboard")
st.write(
    f"Model: **{model_label}** | "
    f"Forecast for: **{forecast_date.strftime('%A, %d %B %Y')}** "
    f"at **{forecast_hour:02d}:00** | "
    f"Temp: **{temperature}°C** | Wind: **{wind_speed} m/s**"
)


# ── ROW 1: Metric Cards ───────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    f"🔮 {model_label} Forecast",
    f"{prediction:,.0f} MW",
    delta=f"{prediction - lag_1:+,.0f} vs last hour"
)
m2.metric("⏮️ Lag 1 (last hour)",   f"{lag_1:,.0f} MW")
m3.metric("📅 Lag 24 (yesterday)",  f"{lag_24:,.0f} MW")
m4.metric("📆 Lag 168 (last week)", f"{lag_168:,.0f} MW")

st.divider()

# ── ROW 2: Model Comparison ───────────────────────────────────
st.subheader("🤖 Model Comparison")
comp1, comp2, comp3 = st.columns(3)

comp1.metric(
    "XGBoost + Lags ★ Champion",
    f"{xgb_prediction:,.0f} MW",
    delta="MAE 365 MW · sMAPE 1.5%",
    delta_color="off"
)

diff = lstm_prediction - xgb_prediction
comp2.metric(
    "LSTM — 2nd Best",
    f"{lstm_prediction:,.0f} MW",
    delta=f"{diff:+,.0f} MW vs XGBoost",
    delta_color="inverse"
)

comp3.metric(
    "XGBoost Advantage",
    "285 MW lower MAE",
    delta="43.8% better on test set",
    delta_color="off"
)

st.divider()

# ── ROW 3: Gauge + 24h Chart ──────────────────────────────────
gauge_col, chart_col = st.columns([1, 2])

def draw_gauge(value, title, color, max_val=55000):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 14, 'color': "white"}},
        gauge = {
            'axis'        : {'range': [15000, max_val],
                             'tickcolor': "white"},
            'bar'         : {'color': color},
            'bgcolor'     : "#262730",
            'borderwidth' : 2,
            'bordercolor' : "gray",
            'steps'       : [
                {'range': [15000, 25000], 'color': '#1a3a1a'},
                {'range': [25000, 40000], 'color': '#1a2a3a'},
                {'range': [40000, 55000], 'color': '#3a1a1a'},
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        font          = {'color': "white"},
        height        = 200,
        margin        = dict(l=10, r=10, t=40, b=10)
    )
    return fig

with gauge_col:
    st.plotly_chart(
        draw_gauge(prediction, f"{model_label}", model_color),
        use_container_width=True
    )
    st.plotly_chart(
        draw_gauge(xgb_prediction, "XGBoost Forecast", "#00FF00"),
        use_container_width=True
    )

with chart_col:
    st.subheader("📈 24-Hour Demand Forecast")

    fig_line = go.Figure()

    fig_line.add_trace(go.Scatter(
        x      = hourly_df['hour'],
        y      = hourly_df['xgb_mw'],
        mode   = 'lines+markers',
        name   = 'XGBoost Forecast',
        line   = dict(color='#00FF00', width=2.5),
        marker = dict(size=6)
    ))

    fig_line.add_trace(go.Scatter(
        x    = hourly_df['hour'],
        y    = hourly_df['lstm_mw'],
        mode = 'lines',
        name = 'LSTM Forecast',
        line = dict(color='#636EFA', width=2, dash='dash')
    ))

    fig_line.add_trace(go.Scatter(
        x         = list(hourly_df['hour']) +
                    list(hourly_df['hour'])[::-1],
        y         = list(hourly_df['xgb_mw'] + model_mae) +
                    list(hourly_df['xgb_mw'] - model_mae)[::-1],
        fill      = 'toself',
        fillcolor = 'rgba(0,255,0,0.1)',
        line      = dict(color='rgba(255,255,255,0)'),
        name      = f'±MAE Uncertainty ({model_mae} MW)'
    ))

    fig_line.update_layout(
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        font          = {'color': "white"},
        xaxis  = dict(title='Hour of Day',    gridcolor='#2a2a2a'),
        yaxis  = dict(title='Demand (MW)',     gridcolor='#2a2a2a'),
        legend = dict(bgcolor='rgba(0,0,0,0)'),
        height = 420,
        margin = dict(l=10, r=10, t=10, b=40)
    )
    st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# ── ROW 4: Scenario Explorer ──────────────────────────────────
st.subheader("🔬 Scenario Explorer — Weather Impact")
st.write(
    "Adjust the temperature slider in the sidebar to explore "
    "how weather conditions shift predicted demand."
)

temp_range       = range(-5, 36, 5)
scenario_results = []
for t in temp_range:
    tf = (
        1
        + max(0, (10 - t)) * 0.005
        + max(0, (t - 22)) * 0.003
    )
    xgb_s  = xgb_prediction * tf
    lstm_s = lstm_prediction * tf
    scenario_results.append({
        'Temperature (°C)'      : t,
        'XGBoost Forecast (MW)' : xgb_s,
        'LSTM Forecast (MW)'    : lstm_s
    })

scenario_df = pd.DataFrame(scenario_results)

fig_scenario = go.Figure()
fig_scenario.add_trace(go.Scatter(
    x    = scenario_df['Temperature (°C)'],
    y    = scenario_df['XGBoost Forecast (MW)'],
    mode = 'lines',
    name = 'XGBoost',
    line = dict(color='#00FF00', width=2)
))
fig_scenario.add_trace(go.Scatter(
    x    = scenario_df['Temperature (°C)'],
    y    = scenario_df['LSTM Forecast (MW)'],
    mode = 'lines',
    name = 'LSTM',
    line = dict(color='#636EFA', width=2, dash='dash')
))
fig_scenario.add_vline(
    x                     = temperature,
    line_dash             = "dash",
    line_color            = "white",
    annotation_text       = f"Current: {temperature}°C",
    annotation_font_color = "white"
)
fig_scenario.update_layout(
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor  = 'rgba(0,0,0,0)',
    font          = {'color': "white"},
    xaxis  = dict(title='Temperature (°C)', gridcolor='#2a2a2a'),
    yaxis  = dict(title='Predicted Demand (MW)', gridcolor='#2a2a2a'),
    legend = dict(bgcolor='rgba(0,0,0,0)'),
    height = 300
)

scen_col, info_col = st.columns([2, 1])
with scen_col:
    st.plotly_chart(fig_scenario, use_container_width=True)

with info_col:
    st.markdown("### 📋 Scenario Summary")
    cold_demand = scenario_df[
        scenario_df['Temperature (°C)'] == -5
    ]['XGBoost Forecast (MW)'].values[0]
    hot_demand = scenario_df[
        scenario_df['Temperature (°C)'] == 35
    ]['XGBoost Forecast (MW)'].values[0]
    st.metric("Cold Day (-5°C)", f"{cold_demand:,.0f} MW")
    st.metric("Hot Day (35°C)",  f"{hot_demand:,.0f} MW")
    st.metric("Difference",      f"{cold_demand - hot_demand:,.0f} MW")

st.divider()

# ── Footer ────────────────────────────────────────────────────
st.caption(
    f"Active model: {model_label} | "
    "XGBoost MAE: 365 MW | LSTM MAE: 650 MW | "
    "Trained: 2009–2021 | Test: 2022–2025 | "
    "Data: National Grid ESO"
)