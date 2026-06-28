# Palo Verde Peak Power Price — Monte Carlo Simulation

Monte Carlo simulation of wholesale electricity prices at the Palo Verde 
hub on the WECC grid, built using real 2026 ICE trading data.

## What This Does

Simulates 10,000 possible price paths over a 30-day horizon using Geometric 
Brownian Motion to price a call option on power and estimate the probability 
of prices exceeding a target threshold. Volatility is derived from actual 
market data rather than assumptions.

## Why Power Markets Are Different

Power can't be stored. When demand spikes, price has to clear the market 
instantly with no inventory buffer to absorb it. That's why Palo Verde 
shows 739% annualized volatility while a stock like AAPL sits around 28%. 
This model captures that behavior directly from the data.

Results from current market conditions:
- Current Palo Verde price: $35.00/MWh
- Annualized volatility: 739%
- Probability of exceeding $42/MWh in 30 days: 12.3%
- Monte Carlo option price: $25.37/MWh

## Stack

- Python 3
- numpy, pandas, matplotlib, scipy

## Files

- `palo_verde_monte_carlo.ipynb` — full simulation notebook
- `palo_verde_monte_carlo.png` — output charts
