# VaR-risk-metric-ideas
<h3>VaR: A risk indicator proposal</h3>
Value-at-Risk (VaR), and especially Conditional-VaR/Expected shortfall (CVaR/ES) are two widely applicable, interpretable and theoretically grounded risk measures. The code included in this repository displays how VaR might be used for monitoring position risk for multiple time horizons, as well as a proof-of-concept to show the strength of CVaR, not only as a risk measure but also a signal for allocation purposes. By a simple CVaR constraint, we produce a signal investing in only BTC and a risk free asset with a dynamical weight w' in bitcoin (thus (1-w') in cash) that achieves a 1958% return from 2019-01-01 to 2025-11-13 with a max drawdown of -37.26% and Sharpe Ratio 1.243. 

## Instructions
Download all three files and put them in the same folder (pip install necessary packages). *risk_metrics_fns.py* contains functions to be imported to the notebook *btc_risk_monitor.ipynb* and does not require to be interacted with. Open the notebook and run all the cells. The notebook *cvar_alloc_poc.ipynb* shows the proof-of-concept portfolio, with relevant plots and summary. 

## Explanation of the CVaR allocation rule
As a risk-allocator, we are aware that BTC returns have fat left tails and negative skew; thus from a compounding perspective avoiding especially severe one-day (or shorter period for that matter) drawdowns is of great importance. Thus we come up with a simple rule: we calculate rolling (365d window) next-day CVaR at 5% level for BTC returns, and using this, determine the weight w' to allocate in BTC such that the next-day ***portfolio CVaR*** is no greater than 2%. The signal then rebalances daily with a new value w'.
