# VaR-risk-metric-ideas
<h3>VaR: A risk indicator proposal</h3>
Value-at-Risk (VaR), and especially Conditional-VaR/Expected shortfall (CVaR/ES) are two widely applicable, interpretable and theoretically grounded risk measures. The code included in this repository displays how VaR might be used for monitoring position risk for multiple time horizons, as well as a proof-of-concept to show the strength of CVaR, not only as a risk measure but also a signal for allocation purposes. By a simple CVaR constraint, we produce a signal investing in only BTC and a risk free asset with a dynamical weight w' in bitcoin (thus (1-w') in cash) that achieves a 1958% return from 2019-01-01 to 2025-11-13 with a max drawdown of -37.26% and Sharpe Ratio 1.243. 

## Instructions
Download all three files and put them in the same folder (pip install necessary packages). *risk_metrics_fns.py* contains functions to be imported to the notebook *btc_risk_monitor.ipynb* and does not require to be interacted with. Open the notebook and run all the cells. The notebook *cvar_alloc_poc.ipynb* shows the proof-of-concept portfolio, with relevant plots and summary. 

## Explanation of risk monitor
First, we fetch daily OHLCV BTCUSD data using yfinance. The timespan should be seen as sample size, as we use the returns to create the parametric distribution of returns. Default values are end = today, start = today-700d (approx 2y lookback). 

Then we QQ-plot the daily returns against the fitted composite distribution, to validate that it is a good fit. MAE, RMSE and R^2 also helps explain this. 

<img width="514" height="765" alt="comp_fit_ex" src="https://github.com/user-attachments/assets/2b3c87b6-2eac-4e02-a1cf-1e4049b78472" />

Below that, we impose a portfolio CVaR target (1d: -2%, 7d: -5%, 30d: 10% - just an example, horizon and CVaR values easily changed in the code) and obtain BTC return CVaR values from Monte-Carlo simulations to calculate the optimal weights w' for BTC ((1-w') in cash) s.t. the portfolio CVaR target is satisfied for each horizon and CVaR target respectively. 

<img width="929" height="506" alt="opt_alloc_sum_ex" src="https://github.com/user-attachments/assets/5525fe7f-1ec6-4914-94e0-134ec0a49dc7" />


## Explanation of the CVaR allocation rule
As a risk-allocator, we are aware that BTC returns have fat left tails and negative skew; thus from a compounding perspective avoiding especially severe one-day (or shorter period for that matter) drawdowns is of great importance. We then use the same idea as above to find our BTC and cash allocations: we calculate *rolling* (365d window) next-day CVaR at 5% level for BTC returns, and using this, determine the weight w' to allocate in BTC such that the next-day ***portfolio CVaR*** is no greater than 2%. The signal then rebalances daily with a new value w'.

<img width="1789" height="1376" alt="a4596e3c-d868-42d0-b882-8778d7b583bf" src="https://github.com/user-attachments/assets/d2bd8f2e-c878-42f2-b792-e573d78f2e4d" />
