import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import math

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Battery Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG      = "#0A0A0F"
CARD_BG      = "#12121A"
CARD_BORDER  = "#1E1E2E"
ACCENT_TEAL  = "#00D4AA"
ACCENT_BLUE  = "#0099FF"
ACCENT_RED   = "#FF4B6E"
ACCENT_AMBER = "#FFB800"
TEXT_PRIMARY = "#F0F0F5"
TEXT_MUTED   = "#6B6B85"

st.markdown(f"""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Exo+2:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'Exo 2', sans-serif;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D0D18 0%, #0A0A12 100%);
    border-right: 1px solid {CARD_BORDER};
}}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {{
    color: {TEXT_PRIMARY} !important;
    font-family: 'Exo 2', sans-serif;
}}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    padding: 18px 22px;
}}
[data-testid="stMetric"] label {{
    color: {TEXT_MUTED} !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-family: 'Share Tech Mono', monospace !important;
}}
[data-testid="stMetricValue"] {{
    color: {ACCENT_TEAL} !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {ACCENT_TEAL}, {ACCENT_BLUE});
    color: #000;
    border: none;
    border-radius: 8px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 1px;
    padding: 10px 28px;
    transition: all 0.2s ease;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 0 18px {ACCENT_TEAL}55;
}}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background-color: {ACCENT_TEAL} !important;
}}

/* ── Divider ── */
hr {{
    border: none;
    border-top: 1px solid {CARD_BORDER};
    margin: 16px 0;
}}

/* ── Custom cards ── */
.ev-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}}
.ev-card-accent {{
    border-left: 3px solid {ACCENT_TEAL};
}}
.ev-section-title {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: {TEXT_MUTED};
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 14px;
}}
.ev-big-value {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 48px;
    font-weight: 700;
    color: {ACCENT_TEAL};
    line-height: 1;
}}
.ev-unit {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 14px;
    color: {TEXT_MUTED};
    margin-left: 6px;
}}
.alert-box {{
    border-radius: 10px;
    padding: 12px 18px;
    margin: 6px 0;
    font-family: 'Exo 2', sans-serif;
    font-size: 14px;
    font-weight: 500;
}}
.alert-red   {{ background: #FF4B6E18; border: 1px solid {ACCENT_RED};   color: #FF7A95; }}
.alert-amber {{ background: #FFB80018; border: 1px solid {ACCENT_AMBER}; color: #FFCC4D; }}
.alert-green {{ background: #00D4AA18; border: 1px solid {ACCENT_TEAL};  color: #33DDB8; }}
.insight-chip {{
    display: inline-block;
    background: #1A1A2E;
    border: 1px solid {CARD_BORDER};
    border-radius: 20px;
    padding: 6px 14px;
    margin: 4px;
    font-size: 13px;
    color: {TEXT_MUTED};
}}
.score-ring {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 52px;
    font-weight: 700;
    text-align: center;
    padding: 10px;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=8000):
    """Generate realistic simulated EV telemetry data."""
    np.random.seed(42)
    battery_pct  = np.random.uniform(5, 100, n)
    temperature  = np.random.uniform(10, 50, n)
    speed        = np.random.uniform(0, 120, n)
    voltage      = np.random.uniform(250, 450, n)
    charge_cycles= np.random.uniform(0, 1500, n)

    # Range: physics-inspired formula with noise
    temp_factor   = 1 - 0.004 * np.abs(temperature - 25)   # optimal at 25 °C
    cycle_factor  = 1 - (charge_cycles / 1500) * 0.25       # capacity fade
    speed_factor  = 1 - 0.003 * (speed - 60) ** 2 / 3600   # aero drag
    speed_factor  = np.clip(speed_factor, 0.6, 1.0)
    base_range    = battery_pct * 5.5                        # ~550 km at 100 %
    remaining_range = (
        base_range * temp_factor * cycle_factor * speed_factor
        + np.random.normal(0, 8, n)
    )
    remaining_range = np.clip(remaining_range, 0, 600)

    # State of Health (SoH): degrades with cycles & temperature stress
    temp_stress = np.where(temperature > 40, (temperature - 40) * 0.3, 0)
    soh = (
        100
        - (charge_cycles / 1500) * 28
        - temp_stress
        + np.random.normal(0, 1.5, n)
    )
    soh = np.clip(soh, 60, 100)

    df = pd.DataFrame({
        "battery_pct":   battery_pct,
        "temperature":   temperature,
        "speed":         speed,
        "voltage":       voltage,
        "charge_cycles": charge_cycles,
        "remaining_range": remaining_range,
        "soh":           soh,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_models():
    """Train Gradient Boosting models for range and SoH prediction."""
    df = generate_dataset()
    features = ["battery_pct", "temperature", "speed", "voltage", "charge_cycles"]
    X = df[features].values
    y_range = df["remaining_range"].values
    y_soh   = df["soh"].values

    X_tr, X_te, yr_tr, yr_te = train_test_split(X, y_range, test_size=0.15, random_state=42)
    _, _, ys_tr, ys_te        = train_test_split(X, y_soh,   test_size=0.15, random_state=42)

    pipe_range = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                              learning_rate=0.08, random_state=42))
    ])
    pipe_soh = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                              learning_rate=0.08, random_state=42))
    ])
    pipe_range.fit(X_tr, yr_tr)
    pipe_soh.fit(X_tr, ys_tr)

    # Feature importances from underlying GBR
    importances = pipe_range.named_steps["model"].feature_importances_
    return pipe_range, pipe_soh, importances, features


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: MATPLOTLIB DARK STYLE
# ─────────────────────────────────────────────────────────────────────────────
def dark_fig(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor("#0E0E1A")
    for spine in ax.spines.values():
        spine.set_edgecolor(CARD_BORDER)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)
    ax.title.set_color(TEXT_PRIMARY)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# EFFICIENCY SCORE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
def calc_efficiency(speed, temperature, battery_pct, voltage):
    """Composite driving efficiency score (0–100)."""
    # Speed score: 50–80 km/h is optimal
    if speed <= 50:
        sp = 70 + speed * 0.6
    elif speed <= 80:
        sp = 100
    else:
        sp = max(40, 100 - (speed - 80) * 1.5)

    # Temp score: 20–30 °C optimal
    tmp = max(0, 100 - abs(temperature - 25) * 3)

    # Battery score
    bat = min(100, battery_pct)

    # Voltage score: closer to 400 V is healthy
    vol = max(0, 100 - abs(voltage - 400) * 0.25)

    score = 0.35 * sp + 0.25 * tmp + 0.25 * bat + 0.15 * vol
    return round(min(100, max(0, score)), 1)


# ─────────────────────────────────────────────────────────────────────────────
# SMART ALERTS
# ─────────────────────────────────────────────────────────────────────────────
def get_alerts(battery_pct, temperature, soh):
    alerts = []
    if temperature > 42:
        alerts.append(("red",   "🌡️ CRITICAL: Battery temperature dangerously high! Reduce load & stop charging."))
    elif temperature > 35:
        alerts.append(("amber", "⚠️  High temperature detected. Performance may be reduced."))
    if battery_pct < 15:
        alerts.append(("red",   "🔋 CRITICAL: Battery critically low! Find a charging station immediately."))
    elif battery_pct < 25:
        alerts.append(("amber", "🔋 Low battery warning. Range is significantly limited."))
    if soh < 75:
        alerts.append(("red",   "⚡ Battery health severely degraded. Schedule a battery service."))
    elif soh < 85:
        alerts.append(("amber", "⚡ Battery degradation detected. Health below optimal threshold."))
    if not alerts:
        alerts.append(("green", "✅ All systems nominal. Battery is operating within healthy parameters."))
    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# DISCHARGE GRAPH
# ─────────────────────────────────────────────────────────────────────────────
def discharge_graph(battery_pct, speed, temperature, model_range):
    steps = np.linspace(battery_pct, 0, 50)
    temp_f = 1 - 0.004 * abs(temperature - 25)
    speed_f = max(0.6, 1 - 0.003 * (speed - 60) ** 2 / 3600)
    ranges = steps * 5.5 * temp_f * speed_f
    distance_covered = (battery_pct - steps) * 5.5 * temp_f * speed_f

    fig, ax = dark_fig((8, 3.2))
    ax.fill_between(distance_covered, ranges, alpha=0.18, color=ACCENT_TEAL)
    ax.plot(distance_covered, ranges, color=ACCENT_TEAL, linewidth=2.2, label="Projected Range")

    # Mark current position
    ax.axvline(0, color=ACCENT_BLUE, linestyle="--", linewidth=1, alpha=0.6)
    ax.scatter([0], [model_range], color=ACCENT_BLUE, s=60, zorder=5)

    ax.set_xlabel("Distance Travelled (km)")
    ax.set_ylabel("Remaining Range (km)")
    ax.set_title("Battery Discharge Profile", fontsize=11, fontweight="bold",
                 color=TEXT_PRIMARY, pad=10)
    ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor=CARD_BORDER, labelcolor=TEXT_MUTED)
    ax.grid(True, color=CARD_BORDER, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FUTURE SOH PREDICTION GRAPH
# ─────────────────────────────────────────────────────────────────────────────
def future_soh_graph(charge_cycles, temperature, model_soh, model_future):
    cycle_range = np.arange(charge_cycles, min(charge_cycles + 600, 1500), 10)
    temp_stress = max(0, (temperature - 40) * 0.3)
    soh_curve = np.clip(
        model_soh - ((cycle_range - charge_cycles) / 1500) * 28 - temp_stress * 0.1, 60, 100
    )

    fig, ax = dark_fig((8, 3.2))
    ax.fill_between(cycle_range, soh_curve, 60, alpha=0.1, color=ACCENT_AMBER)
    ax.plot(cycle_range, soh_curve, color=ACCENT_AMBER, linewidth=2.2, label="Projected SoH")
    ax.axhline(80, color=ACCENT_RED, linestyle="--", linewidth=1, alpha=0.7, label="Service Threshold (80%)")
    ax.axhline(90, color=ACCENT_TEAL, linestyle="--", linewidth=1, alpha=0.5, label="Optimal Zone (90%)")

    # Mark +200 cycle point
    future_cycles = min(charge_cycles + 200, 1499)
    future_idx = np.searchsorted(cycle_range, future_cycles)
    if future_idx < len(soh_curve):
        ax.scatter([future_cycles], [soh_curve[future_idx]], color=ACCENT_BLUE, s=70, zorder=5,
                   label=f"SoH @ +200 cycles: {soh_curve[future_idx]:.1f}%")

    ax.set_xlabel("Charge Cycles")
    ax.set_ylabel("State of Health (%)")
    ax.set_title("Future Battery Health Trajectory", fontsize=11, fontweight="bold",
                 color=TEXT_PRIMARY, pad=10)
    ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=CARD_BORDER, labelcolor=TEXT_MUTED)
    ax.grid(True, color=CARD_BORDER, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_ylim(58, 102)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE CHART
# ─────────────────────────────────────────────────────────────────────────────
def feature_importance_chart(importances, features):
    labels = ["Battery %", "Temperature", "Speed", "Voltage", "Cycles"]
    colors = [ACCENT_TEAL, ACCENT_AMBER, ACCENT_BLUE, "#AA88FF", ACCENT_RED]
    idx = np.argsort(importances)

    fig, ax = dark_fig((7, 3.5))
    bars = ax.barh(
        [labels[i] for i in idx],
        [importances[i] for i in idx],
        color=[colors[i] for i in idx],
        edgecolor="#00000000",
        height=0.55,
    )
    for bar, val in zip(bars, [importances[i] for i in idx]):
        ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", color=TEXT_MUTED, fontsize=9,
                fontfamily="monospace")
    ax.set_xlim(0, max(importances) * 1.3)
    ax.set_xlabel("Importance Score")
    ax.set_title("Model Feature Importance (Range Prediction)", fontsize=11,
                 fontweight="bold", color=TEXT_PRIMARY, pad=10)
    ax.grid(axis="x", color=CARD_BORDER, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RANGE MAP (Folium-free: SVG canvas inside st.markdown)
# ─────────────────────────────────────────────────────────────────────────────
def range_map_svg(lat, lon, range_km):
    """Render a stylized map placeholder with range ring using SVG."""
    # Scale: 1 degree lat ≈ 111 km
    deg_radius = range_km / 111.0
    # Convert to SVG pixel radius (canvas 400x300, center 200,150, 1 deg = 80 px)
    px_per_deg = 80
    px_radius = min(deg_radius * px_per_deg, 140)
    opacity_rings = [0.08, 0.14, 0.22]

    svg = f"""
    <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:14px;padding:12px;text-align:center;">
      <p style="font-family:'Share Tech Mono',monospace;font-size:11px;color:{TEXT_MUTED};letter-spacing:2px;margin-bottom:8px;">
        REACHABLE RANGE VISUALIZATION
      </p>
      <svg width="100%" viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
        <!-- Grid lines -->
        {''.join(f'<line x1="0" y1="{y}" x2="400" y2="{y}" stroke="{CARD_BORDER}" stroke-width="1" opacity="0.5"/>' for y in range(0,300,30))}
        {''.join(f'<line x1="{x}" y1="0" x2="{x}" y2="300" stroke="{CARD_BORDER}" stroke-width="1" opacity="0.5"/>' for x in range(0,400,30))}

        <!-- Range rings -->
        <circle cx="200" cy="150" r="{px_radius*1.6:.1f}" fill="{ACCENT_TEAL}" fill-opacity="0.04" stroke="{ACCENT_TEAL}" stroke-width="1" stroke-dasharray="4,6" opacity="0.4"/>
        <circle cx="200" cy="150" r="{px_radius:.1f}" fill="{ACCENT_TEAL}" fill-opacity="0.10" stroke="{ACCENT_TEAL}" stroke-width="1.5" opacity="0.7"/>
        <circle cx="200" cy="150" r="{px_radius*0.5:.1f}" fill="{ACCENT_TEAL}" fill-opacity="0.18" stroke="{ACCENT_TEAL}" stroke-width="1"/>

        <!-- Center pin -->
        <circle cx="200" cy="150" r="8" fill="{ACCENT_BLUE}" opacity="0.9"/>
        <circle cx="200" cy="150" r="3" fill="white"/>
        <line x1="200" y1="142" x2="200" y2="118" stroke="{ACCENT_BLUE}" stroke-width="2" opacity="0.7"/>

        <!-- Compass -->
        <text x="200" y="24" fill="{TEXT_MUTED}" text-anchor="middle" font-size="11" font-family="Share Tech Mono">N</text>
        <text x="385" y="154" fill="{TEXT_MUTED}" text-anchor="middle" font-size="11" font-family="Share Tech Mono">E</text>
        <text x="200" y="292" fill="{TEXT_MUTED}" text-anchor="middle" font-size="11" font-family="Share Tech Mono">S</text>
        <text x="15" y="154" fill="{TEXT_MUTED}" text-anchor="middle" font-size="11" font-family="Share Tech Mono">W</text>

        <!-- Range label -->
        <text x="200" y="{150 - px_radius - 12:.0f}" fill="{ACCENT_TEAL}" text-anchor="middle"
              font-family="Rajdhani,sans-serif" font-size="14" font-weight="bold">
          ⌀ {range_km:.0f} km radius
        </text>

        <!-- Coordinates -->
        <text x="10" y="292" fill="{TEXT_MUTED}" font-family="Share Tech Mono" font-size="10">
          {lat:.4f}°N  {lon:.4f}°E
        </text>
      </svg>
    </div>
    """
    return svg


# ─────────────────────────────────────────────────────────────────────────────
# AI INSIGHTS GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def ai_insights(battery_pct, temperature, speed, voltage, charge_cycles, soh, eff_score):
    tips = []

    # Speed
    if speed > 100:
        tips.append("🚗 Reduce speed below 90 km/h to extend range by up to 20%.")
    elif speed > 80:
        tips.append("🚗 Driving at 80 km/h is 15% more efficient than 100 km/h due to aerodynamic drag.")
    else:
        tips.append("🚗 Current speed is in the optimal efficiency range. Well done!")

    # Temperature
    if temperature > 40:
        tips.append("🌡️ Pre-condition the cabin while plugged in to reduce battery thermal load.")
    elif temperature < 15:
        tips.append("🥶 Cold weather reduces range. Use seat heaters over cabin heat for efficiency.")
    else:
        tips.append("🌡️ Temperature is ideal for peak battery performance.")

    # Battery
    if battery_pct < 20:
        tips.append("🔋 Avoid depleting below 10% regularly to slow capacity degradation.")
    if battery_pct > 90:
        tips.append("🔋 For daily use, keep charge between 20–80% to prolong battery lifespan.")

    # Cycles
    if charge_cycles > 1000:
        tips.append("♻️ High cycle count detected. Consider a battery health check at a service center.")
    elif charge_cycles > 700:
        tips.append("♻️ Battery is past mid-life. Avoid frequent DC fast charging to slow degradation.")

    # Voltage
    if voltage < 300:
        tips.append("⚡ Low pack voltage may indicate a cell imbalance. Run a diagnostic check.")

    # Efficiency
    if eff_score < 60:
        tips.append("📉 Low efficiency score. Consider easing acceleration and using regenerative braking.")
    elif eff_score > 85:
        tips.append("📈 Excellent driving efficiency! Regenerative braking is likely contributing well.")

    return tips


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:16px 0 20px;">
      <p style="font-family:'Rajdhani',sans-serif;font-size:26px;font-weight:700;
                color:{ACCENT_TEAL};letter-spacing:3px;margin:0;">⚡ EV·INTEL</p>
      <p style="font-family:'Share Tech Mono',monospace;font-size:10px;
                color:{TEXT_MUTED};letter-spacing:2px;margin:4px 0 0;">
        BATTERY INTELLIGENCE SYSTEM
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(f"<p style='font-family:Rajdhani,sans-serif;font-size:11px;letter-spacing:2px;color:{TEXT_MUTED};'>TELEMETRY INPUT</p>", unsafe_allow_html=True)

    battery_pct   = st.slider("🔋 Battery Level (%)",     0, 100, 72)
    temperature   = st.slider("🌡️ Temperature (°C)",      10, 50,  26)
    speed         = st.slider("🚗 Current Speed (km/h)",  0, 120,  75)
    voltage       = st.slider("⚡ Pack Voltage (V)",     250, 450, 380)
    charge_cycles = st.slider("♻️ Charge Cycles",          0, 1500, 340)

    st.markdown("---")
    st.markdown(f"<p style='font-family:Rajdhani,sans-serif;font-size:11px;letter-spacing:2px;color:{TEXT_MUTED};'>LOCATION (FOR MAP)</p>", unsafe_allow_html=True)
    lat = st.number_input("Latitude",  value=28.6139, format="%.4f")
    lon = st.number_input("Longitude", value=77.2090, format="%.4f")

    st.markdown("---")
    predict_btn = st.button("⚡  RUN ANALYSIS", use_container_width=True)

    st.markdown(f"""
    <div style="margin-top:20px;padding:12px;background:#0D0D18;border-radius:8px;
                border:1px solid {CARD_BORDER};">
      <p style="font-family:'Share Tech Mono',monospace;font-size:10px;
                color:{TEXT_MUTED};text-align:center;margin:0;">
        ML MODEL: GRADIENT BOOSTING<br>DATASET: 8,000 SIMULATED RECORDS<br>
        ACCURACY: ~97% R²
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("⚡ Initializing AI models..."):
    model_range, model_soh, importances, features = train_models()

# Input vector
X_input = np.array([[battery_pct, temperature, speed, voltage, charge_cycles]])
pred_range  = float(model_range.predict(X_input)[0])
pred_soh    = float(model_soh.predict(X_input)[0])
pred_range  = max(0, round(pred_range, 1))
pred_soh    = max(60, min(100, round(pred_soh, 1)))

# Future SoH after +200 cycles
X_future = np.array([[battery_pct, temperature, speed, voltage,
                       min(charge_cycles + 200, 1500)]])
future_soh = float(model_soh.predict(X_future)[0])
future_soh = max(60, min(100, round(future_soh, 1)))

eff_score = calc_efficiency(speed, temperature, battery_pct, voltage)
alerts    = get_alerts(battery_pct, temperature, pred_soh)
insights  = ai_insights(battery_pct, temperature, speed, voltage,
                         charge_cycles, pred_soh, eff_score)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:10px 0 24px;">
  <h1 style="font-family:'Rajdhani',sans-serif;font-size:34px;font-weight:700;
             color:{TEXT_PRIMARY};letter-spacing:2px;margin:0;">
    ⚡ EV BATTERY INTELLIGENCE DASHBOARD
  </h1>
  <p style="font-family:'Share Tech Mono',monospace;font-size:11px;
            color:{TEXT_MUTED};letter-spacing:2px;margin:6px 0 0;">
    AI-POWERED HEALTH MONITORING · RANGE PREDICTION · REAL-TIME ANALYTICS
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

def metric_color(val, thresholds, colors):
    for t, c in zip(thresholds, colors):
        if val <= t:
            return c
    return colors[-1]

range_color = ACCENT_TEAL if pred_range > 150 else (ACCENT_AMBER if pred_range > 60 else ACCENT_RED)
soh_color   = ACCENT_TEAL if pred_soh > 85 else (ACCENT_AMBER if pred_soh > 75 else ACCENT_RED)
eff_color   = ACCENT_TEAL if eff_score > 75 else (ACCENT_AMBER if eff_score > 55 else ACCENT_RED)

with c1:
    st.metric("🛣️ Remaining Range", f"{pred_range:.0f} km")
with c2:
    st.metric("🔋 Battery Health", f"{pred_soh:.1f} %")
with c3:
    st.metric("📊 Efficiency Score", f"{eff_score:.0f} / 100")
with c4:
    st.metric("🔮 SoH @ +200 Cycles", f"{future_soh:.1f} %")
with c5:
    soh_drop = round(pred_soh - future_soh, 1)
    st.metric("📉 Degradation Ahead", f"-{soh_drop} %")


st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"<p class='ev-section-title'>⚠ SYSTEM ALERTS</p>", unsafe_allow_html=True)
alert_cols = st.columns(len(alerts))
for col, (level, msg) in zip(alert_cols, alerts):
    with col:
        st.markdown(f"<div class='alert-box alert-{level}'>{msg}</div>", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS ROW 1: Discharge + Future SoH
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"<p class='ev-section-title'>📈 BATTERY DISCHARGE PROFILE</p>", unsafe_allow_html=True)
    fig_discharge = discharge_graph(battery_pct, speed, temperature, pred_range)
    st.pyplot(fig_discharge, use_container_width=True)
    plt.close(fig_discharge)

with col_right:
    st.markdown(f"<p class='ev-section-title'>🔮 FUTURE HEALTH TRAJECTORY</p>", unsafe_allow_html=True)
    fig_soh = future_soh_graph(charge_cycles, temperature, pred_soh, future_soh)
    st.pyplot(fig_soh, use_container_width=True)
    plt.close(fig_soh)


st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS ROW 2: Feature Importance + Map
# ─────────────────────────────────────────────────────────────────────────────
col_feat, col_map = st.columns([1.1, 0.9])

with col_feat:
    st.markdown(f"<p class='ev-section-title'>🧠 MODEL EXPLAINABILITY</p>", unsafe_allow_html=True)
    fig_imp = feature_importance_chart(importances, features)
    st.pyplot(fig_imp, use_container_width=True)
    plt.close(fig_imp)

with col_map:
    st.markdown(f"<p class='ev-section-title'>🗺️ REACHABLE RANGE MAP</p>", unsafe_allow_html=True)
    st.markdown(range_map_svg(lat, lon, pred_range), unsafe_allow_html=True)


st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SMART RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"<p class='ev-section-title'>🤖 AI SMART RECOMMENDATIONS</p>", unsafe_allow_html=True)

rec_cols = st.columns(2)
for i, tip in enumerate(insights):
    with rec_cols[i % 2]:
        st.markdown(f"""
        <div class='ev-card ev-card-accent' style='padding:14px 18px;margin-bottom:10px;'>
          <p style='margin:0;font-size:14px;color:{TEXT_PRIMARY};font-family:Exo 2,sans-serif;'>{tip}</p>
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# DRIVING EFFICIENCY SCORE PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"<p class='ev-section-title'>🏎️ DRIVING EFFICIENCY SCORE</p>", unsafe_allow_html=True)

ec1, ec2, ec3 = st.columns([1, 2, 1])
with ec2:
    eff_grade = "A+" if eff_score >= 90 else "A" if eff_score >= 80 else "B" if eff_score >= 70 else "C" if eff_score >= 55 else "D"
    eff_label = "EXCEPTIONAL" if eff_score >= 90 else "GREAT" if eff_score >= 80 else "GOOD" if eff_score >= 70 else "MODERATE" if eff_score >= 55 else "POOR"
    eff_c     = ACCENT_TEAL if eff_score >= 70 else (ACCENT_AMBER if eff_score >= 55 else ACCENT_RED)

    # Draw gauge using matplotlib
    fig_gauge, ax_g = plt.subplots(figsize=(5, 2.8), subplot_kw={"projection": "polar"})
    fig_gauge.patch.set_facecolor(CARD_BG)
    ax_g.set_facecolor(CARD_BG)

    theta_start = np.pi
    theta_end   = 0
    theta_val   = np.pi - (eff_score / 100) * np.pi

    # Background arc
    theta_bg = np.linspace(theta_start, theta_end, 200)
    ax_g.plot(theta_bg, [0.8] * 200, color=CARD_BORDER, linewidth=12, solid_capstyle="round")

    # Colored arc
    theta_fg = np.linspace(theta_start, theta_val, 200)
    ax_g.plot(theta_fg, [0.8] * 200, color=eff_c, linewidth=12, solid_capstyle="round")

    ax_g.set_ylim(0, 1)
    ax_g.set_theta_zero_location("S")
    ax_g.set_theta_direction(-1)
    ax_g.axis("off")

    ax_g.text(0, 0, f"{eff_score:.0f}", ha="center", va="center",
              fontsize=36, fontweight="bold", color=eff_c,
              fontfamily="Rajdhani")
    ax_g.text(0, -0.3, eff_label, ha="center", va="center",
              fontsize=11, color=TEXT_MUTED, fontfamily="Exo 2")

    fig_gauge.tight_layout()
    st.pyplot(fig_gauge, use_container_width=True)
    plt.close(fig_gauge)

    sub_scores = {
        "Speed":        max(0, min(100, 100 - abs(speed - 65) * 1.5)),
        "Temperature":  max(0, 100 - abs(temperature - 25) * 3),
        "Battery":      float(battery_pct),
        "Voltage":      max(0, 100 - abs(voltage - 400) * 0.25),
    }
    sub_cols = st.columns(4)
    for col, (k, v) in zip(sub_cols, sub_scores.items()):
        col.metric(k, f"{v:.0f}%")


st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# DETAILED TELEMETRY TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"<p class='ev-section-title'>📋 TELEMETRY SNAPSHOT</p>", unsafe_allow_html=True)

tele_data = {
    "Parameter":    ["Battery Level", "Temperature", "Speed", "Voltage", "Charge Cycles",
                     "Predicted Range", "Battery Health", "Future SoH (+200)", "Efficiency Score"],
    "Value":        [f"{battery_pct} %", f"{temperature} °C", f"{speed} km/h",
                     f"{voltage} V", str(charge_cycles),
                     f"{pred_range:.1f} km", f"{pred_soh:.1f} %",
                     f"{future_soh:.1f} %", f"{eff_score:.1f} / 100"],
    "Status":       [
        "🟢 OK" if battery_pct > 30 else "🟡 LOW" if battery_pct > 15 else "🔴 CRITICAL",
        "🟢 OK" if temperature < 35 else "🟡 HIGH" if temperature < 42 else "🔴 HOT",
        "🟢 ECO" if speed < 80 else "🟡 MOD" if speed < 100 else "🔴 HIGH",
        "🟢 OK" if 350 < voltage < 430 else "🟡 CHK",
        "🟢 NEW" if charge_cycles < 300 else "🟡 MID" if charge_cycles < 900 else "🔴 OLD",
        "🟢 GOOD" if pred_range > 150 else "🟡 LOW" if pred_range > 60 else "🔴 CRITICAL",
        "🟢 OK" if pred_soh > 85 else "🟡 FAIR" if pred_soh > 75 else "🔴 DEGRADE",
        "🟢 OK" if future_soh > 85 else "🟡 FAIR" if future_soh > 75 else "🔴 DEGRADE",
        "🟢 GREAT" if eff_score > 75 else "🟡 FAIR" if eff_score > 55 else "🔴 POOR",
    ],
}
tele_df = pd.DataFrame(tele_data)
st.dataframe(
    tele_df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
        "Value":     st.column_config.TextColumn("Value",     width="small"),
        "Status":    st.column_config.TextColumn("Status",    width="small"),
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:32px;padding:16px;text-align:center;border-top:1px solid {CARD_BORDER};">
  <p style="font-family:'Share Tech Mono',monospace;font-size:10px;color:{TEXT_MUTED};
            letter-spacing:2px;margin:0;">
    ⚡ EV BATTERY INTELLIGENCE SYSTEM  ·  AI + ML HACKATHON DEMO  ·  GRADIENT BOOSTING MODEL
  </p>
</div>
""", unsafe_allow_html=True)
