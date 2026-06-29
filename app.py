import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="WECC Power Price Risk Model", layout="wide")

st.title("Palo Verde Peak — Power Price Risk Model")
st.markdown("Monte Carlo simulation of WECC wholesale electricity prices using Geometric Brownian Motion.")

# ── SIDEBAR INPUTS ───────────────────────────────────────
st.sidebar.header("Model Inputs")

S = st.sidebar.slider("Current Price ($/MWh)", 10.0, 150.0, 35.0, 0.50)
K = st.sidebar.slider("Strike Price ($/MWh)", 10.0, 200.0, 42.0, 0.50)
sigma = st.sidebar.slider("Annualized Volatility", 0.10, 10.0, 0.65, 0.05)
days = st.sidebar.slider("Time Horizon (days)", 7, 90, 30, 1)
n_simulations = st.sidebar.selectbox("Simulations", [1000, 5000, 10000], index=2)

T = days / 365
r = 0.043
n_steps = days

# ── RUN SIMULATION ───────────────────────────────────────
np.random.seed(42)
dt = T / n_steps
Z = np.random.standard_normal((n_simulations, n_steps))
drift = (r - 0.5 * sigma**2) * dt
diffusion = sigma * np.sqrt(dt)

price_paths = np.zeros((n_simulations, n_steps + 1))
price_paths[:, 0] = S
for t in range(1, n_steps + 1):
    price_paths[:, t] = price_paths[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])

final_prices = price_paths[:, -1]
payoffs = np.maximum(final_prices - K, 0)
option_price = np.exp(-r * T) * np.mean(payoffs)
prob_above = np.mean(final_prices > K)

# ── METRICS ROW ──────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Option Price", f"${option_price:.2f}/MWh")
col2.metric("Prob Above Strike", f"{prob_above*100:.1f}%")
col3.metric("Mean Final Price", f"${np.mean(final_prices):.2f}/MWh")
col4.metric("Max Final Price", f"${np.min([np.max(final_prices), 9999]):.0f}/MWh")

# ── CHARTS ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
time_axis = np.linspace(0, days, n_steps + 1)

# Price paths
for i in range(200):
    axes[0].plot(time_axis, price_paths[i], alpha=0.1, linewidth=0.5, color="steelblue")
axes[0].axhline(y=K, color="red", linestyle="--", linewidth=1.5, label=f"Strike ${K:.0f}")
axes[0].axhline(y=S, color="green", linestyle="--", linewidth=1.5, label=f"Current ${S:.0f}")
axes[0].set_title("Simulated Price Paths")
axes[0].set_xlabel("Days")
axes[0].set_ylabel("Price ($/MWh)")
axes[0].set_ylim(0, S * 8)
axes[0].legend()

# Final price distribution
capped = np.clip(final_prices, 0, S * 8)
axes[1].hist(capped, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
axes[1].axvline(x=K, color="red", linestyle="--", linewidth=1.5, label=f"Strike ${K:.0f}")
axes[1].axvline(x=S, color="green", linestyle="--", linewidth=1.5, label=f"Current ${S:.0f}")
axes[1].set_title("Final Price Distribution")
axes[1].set_xlabel("Price ($/MWh)")
axes[1].legend()

# Payoff distribution
nonzero = payoffs[payoffs > 0]
if len(nonzero) > 0:
    axes[2].hist(nonzero, bins=60, color="darkorange", edgecolor="white", alpha=0.8)
    axes[2].axvline(x=option_price, color="red", linestyle="--", linewidth=1.5, label=f"Option Price ${option_price:.2f}")
    axes[2].set_title(f"Payoff Distribution ({prob_above*100:.1f}% profitable)")
    axes[2].set_xlabel("Payoff ($/MWh)")
    axes[2].legend()
else:
    axes[2].text(0.5, 0.5, "No paths above strike", ha="center", va="center")
    axes[2].set_title("Payoff Distribution")

plt.tight_layout()
st.pyplot(fig)

# ── 3D SURFACE PLOT ──────────────────────────────────────
st.subheader("Option Price Surface — Volatility vs Stock Price")

import plotly.graph_objects as go

price_range = np.linspace(S * 0.5, S * 2.0, 40)
vol_range = np.linspace(0.10, 5.0, 40)
price_grid, vol_grid = np.meshgrid(price_range, vol_range)

def bs_call(s, k, t, r, sig):
    d1 = (np.log(s / k) + (r + 0.5 * sig**2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    return s * stats.norm.cdf(d1) - k * np.exp(-r * t) * stats.norm.cdf(d2)

price_surface = bs_call(price_grid, K, T, r, vol_grid)

fig3d = go.Figure(data=[go.Surface(
    x=price_range,
    y=vol_range,
    z=price_surface,
    colorscale="Viridis",
    colorbar=dict(title="Option Price ($/MWh)")
)])

fig3d.update_layout(
    scene=dict(
        xaxis_title="Stock Price ($/MWh)",
        yaxis_title="Volatility",
        zaxis_title="Option Price ($/MWh)",
        bgcolor="rgba(0,0,0,0)"
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    height=600,
    margin=dict(l=0, r=0, t=30, b=0)
)

st.plotly_chart(fig3d, use_container_width=True)
