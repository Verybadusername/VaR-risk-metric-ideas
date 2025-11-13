# risk_metrics_fns.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import genpareto, t as t_dist
from statsmodels.tsa.ar_model import AutoReg

__all__ = [
    "remove_autocorrelation",
    "fit_gpd_tails",
    "evt_student_t_composite_cdf",
    "evt_student_t_composite_ppf",
    "qq_statssum_composite",
    "complete_qq_plot",
    "evt_student_t_composite_rvs",
    "simulate_price_paths",
    "optim_weights",
]

def remove_autocorrelation(returns, lag=1):
    returns = returns.dropna()
    model = AutoReg(returns, lags=lag, old_names=False)
    fitted = model.fit()
    const = fitted.params[0]
    ar_coef = fitted.params[1]
    r_lagged = returns.shift(1)
    unsmoothed = (1 / (1 - ar_coef)) * (returns - ar_coef * r_lagged)
    unsmoothed = unsmoothed.dropna()
    return unsmoothed, fitted, ar_coef, const

def fit_gpd_tails(data, threshold_quantile=0.95):
    upper_threshold = np.quantile(data, threshold_quantile)
    lower_threshold = np.quantile(data, 1 - threshold_quantile)

    upper_exceedances = data[data > upper_threshold] - upper_threshold
    lower_exceedances = np.abs(data[data < lower_threshold] - lower_threshold)

    upper_params = genpareto.fit(upper_exceedances, floc=0)
    lower_params = genpareto.fit(lower_exceedances, floc=0)

    return {
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
        "upper_shape": upper_params[0],
        "upper_scale": upper_params[2],
        "lower_shape": lower_params[0],
        "lower_scale": lower_params[2],
        "n_upper_exceedances": len(upper_exceedances),
        "n_lower_exceedances": len(lower_exceedances),
        "n_total": len(data),
    }

def evt_student_t_composite_cdf(x, evt_results, df, loc, scale):
    x = np.atleast_1d(x)
    cdf_vals = np.zeros_like(x, dtype=float)
    upper_thresh = evt_results["upper_threshold"]
    lower_thresh = evt_results["lower_threshold"]
    F_lower = t_dist.cdf(lower_thresh, df, loc=loc, scale=scale)
    F_upper = t_dist.cdf(upper_thresh, df, loc=loc, scale=scale)

    for i, val in enumerate(x):
        if val <= lower_thresh:
            exceedance = lower_thresh - val
            gpd_survival = genpareto.sf(
                exceedance, evt_results["lower_shape"], loc=0, scale=evt_results["lower_scale"]
            )
            cdf_vals[i] = F_lower * gpd_survival
        elif val >= upper_thresh:
            exceedance = val - upper_thresh
            gpd_cdf = genpareto.cdf(
                exceedance, evt_results["upper_shape"], loc=0, scale=evt_results["upper_scale"]
            )
            cdf_vals[i] = F_upper + (1 - F_upper) * gpd_cdf
        else:
            cdf_vals[i] = t_dist.cdf(val, df, loc=loc, scale=scale)

    return cdf_vals if len(x) > 1 else cdf_vals[0]

def evt_student_t_composite_ppf(p, evt_results, df, loc, scale):
    p = np.atleast_1d(p)
    quantiles = np.zeros_like(p, dtype=float)
    upper_thresh = evt_results["upper_threshold"]
    lower_thresh = evt_results["lower_threshold"]
    F_lower = t_dist.cdf(lower_thresh, df, loc=loc, scale=scale)
    F_upper = t_dist.cdf(upper_thresh, df, loc=loc, scale=scale)

    for i, prob in enumerate(p):
        if prob <= F_lower:
            gpd_survival = prob / F_lower
            exceedance = genpareto.ppf(
                1 - gpd_survival, evt_results["lower_shape"], loc=0, scale=evt_results["lower_scale"]
            )
            quantiles[i] = lower_thresh - exceedance
        elif prob >= F_upper:
            gpd_cdf = (prob - F_upper) / (1 - F_upper)
            exceedance = genpareto.ppf(
                gpd_cdf, evt_results["upper_shape"], loc=0, scale=evt_results["upper_scale"]
            )
            quantiles[i] = upper_thresh + exceedance
        else:
            quantiles[i] = t_dist.ppf(prob, df, loc=loc, scale=scale)

    return quantiles if len(p) > 1 else quantiles[0]

def qq_statssum_composite(return_data, evt_results):
    sorted_data = np.sort(return_data.dropna().values)
    n = len(sorted_data)
    empirical_probs = (np.arange(1, n + 1) - 0.5) / n
    df_t, loc_t, scale_t = stats.t.fit(return_data.dropna())

    theoretical_composite_quantiles = evt_student_t_composite_ppf(
        empirical_probs, evt_results, df_t, loc_t, scale_t
    )

    prob_grid = np.linspace(0.01, 0.99, 99)
    sample_q = return_data.quantile(prob_grid)
    theoretical_q = evt_student_t_composite_ppf(prob_grid, evt_results, df_t, loc_t, scale_t)

    mae = np.mean(np.abs(sample_q - theoretical_q))
    rmse = np.sqrt(np.mean((sample_q - theoretical_q) ** 2))
    ss_res = np.sum((sample_q - theoretical_q) ** 2)
    ss_tot = np.sum((sample_q - sample_q.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print("\nSemi-Parametric Distribution:")
    print("=" * 60)
    print(f"Body (5%-95%):    Student-t (df={df_t:.2f})")
    print(f"Lower tail (<5%): GPD (ξ={evt_results['lower_shape']:.4f}, β={evt_results['lower_scale']:.4f})")
    print(f"Upper tail (>95%): GPD (ξ={evt_results['upper_shape']:.4f}, β={evt_results['upper_scale']:.4f})")
    print("\nThresholds:")
    print(f"  Lower: {evt_results['lower_threshold']:.6f}")
    print(f"  Upper: {evt_results['upper_threshold']:.6f}")
    print("=" * 60)

    print("\nComposite Distribution Goodness of Fit:")
    print("=" * 60)
    print(f"Mean Absolute Error:      {mae:.6f}")
    print(f"Root Mean Square Error:   {rmse:.6f}")
    print(f"R-squared:                {r_squared:.6f}")
    print("=" * 60)
    print("\nModel Specification:")
    print(f"  Body (5%-95%):    Student-t (df={df_t:.2f})")
    print(f"  Lower tail (<5%): GPD (ξ={evt_results['lower_shape']:.4f})")
    print(f"  Upper tail (>95%): GPD (ξ={evt_results['upper_shape']:.4f})")
    print("=" * 60)

def complete_qq_plot(return_series, evt_results):
    sorted_data = np.sort(return_series.dropna().values)
    n = len(sorted_data)
    empirical_probs = (np.arange(1, n + 1) - 0.5) / n
    df_t, loc_t, scale_t = stats.t.fit(return_series.dropna())

    theoretical_composite = evt_student_t_composite_ppf(
        empirical_probs, evt_results, df_t, loc_t, scale_t
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(theoretical_composite, sorted_data, alpha=0.5, s=20, edgecolors="none")
    diag = [sorted_data.min(), sorted_data.max()]
    ax.plot(diag, diag, "r--", linewidth=2, label="45° line")
    lower_thresh = evt_results["lower_threshold"]
    upper_thresh = evt_results["upper_threshold"]
    ax.axhline(lower_thresh, color="orange", linestyle=":", linewidth=1.5, alpha=0.7, label="GPD tail thresholds")
    ax.axhline(upper_thresh, color="orange", linestyle=":", linewidth=1.5, alpha=0.7)

    ax.set_title(
        f"Student-t Body + GPD Tails Q-Q Plot\nξ_lower={evt_results['lower_shape']:.3f}, ξ_upper={evt_results['upper_shape']:.3f}",
        fontweight="bold",
        fontsize=12,
    )
    ax.set_xlabel("Theoretical Quantiles (Composite)", fontweight="bold")
    ax.set_ylabel("Sample Quantiles", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

def evt_student_t_composite_rvs(size, evt_results, df, loc, scale, random_state=None):
    rng = np.random.default_rng(random_state)
    u = rng.uniform(0.0, 1.0, size)
    flat = u.ravel()
    draws = evt_student_t_composite_ppf(flat, evt_results, df, loc, scale)
    return draws.reshape(u.shape)

def simulate_price_paths(generating_fn, S0, w, evt_results, df_t_unsm, loc_t_unsm, scale_t_unsm, n_paths, horizon):
    """
    Simulate portfolio paths with weight w on risky asset and (1-w) in cash.
    
    Parameters:
    -----------
    S0 : float
        Initial capital
    w : float
        Weight on risky asset (0 <= w <= 1)
    """
    shocks = generating_fn(
        size=(n_paths, horizon),
        evt_results=evt_results,
        df=df_t_unsm,
        loc=loc_t_unsm,
        scale=scale_t_unsm,
        random_state=123,
    )
    # Portfolio return: w * risky_return + (1-w) * 0
    
    

def optim_weights(max_loss_pct, generating_fn, S0, evt_results, df_t_unsm, loc_t_unsm, scale_t_unsm, n_paths, horizon):
    """
    Find optimal weight w such that portfolio CVaR_5% ≈ max_loss_pct * S0.
    
    Approach:
    1. Simulate 100% risky (w=1) to get CVaR_100
    2. Calculate loss from 100%: loss_100 = S0 - CVaR_100
    3. Approximate w = (max_loss_pct * S0) / loss_100
    4. (Optional) Verify by simulating at w and adjusting if needed
    """
    # Simulate 100% risky
    shocks = generating_fn(
        size=(n_paths, horizon),
        evt_results=evt_results,
        df=df_t_unsm,
        loc=loc_t_unsm,
        scale=scale_t_unsm,
        random_state=123,
    )
    terminal_values_100 = S0 * np.exp(np.cumsum(shocks, axis=1))[:, -1]
    var_05_100 = np.percentile(terminal_values_100, 5)
    cvar_05_100 = terminal_values_100[terminal_values_100 <= var_05_100].mean()
    
    # Calculate loss at 100% risky
    loss_100 = S0 - cvar_05_100
    
    # Target loss (dollars)
    target_loss = max_loss_pct * S0
    
    # Optimal weight (linear approximation)
    if loss_100 > 0:
        w_opt = target_loss / loss_100
    else:
        w_opt = 1.0  # If no expected loss, full allocation
    
    # Clamp to [0, 1]
    w_opt = np.clip(w_opt, 0.0, 1.0)
    
    print("\n" + "="*70)
    print("OPTIMAL ALLOCATION SUMMARY")
    print("="*70)
    print(f"Horizon:              {horizon} days")
    print(f"Initial Capital:      ${S0:,.2f}")
    print(f"Portfolio CVaR Target (max loss):      {max_loss_pct*100:.1f}% (${target_loss:,.2f})")
    print(f"HODL CVaR Loss: ${loss_100:,.2f}")
    print(f"Optimal Weight w' (portfolio = w'*BTC + (1-w')*cash):       {w_opt*100:.2f}% risky / {(1-w_opt)*100:.2f}% cash")
    print("="*70)
    
    portfolio_shocks = w_opt * shocks
    paths = S0 * np.exp(np.cumsum(portfolio_shocks, axis=1))
    terminal_values = paths[:, -1]
    
    var_05 = np.percentile(terminal_values, 5)
    mc_cvar_05 = terminal_values[terminal_values <= var_05].mean()
    var_01 = np.percentile(terminal_values, 1)
    mc_cvar_01 = terminal_values[terminal_values <= var_01].mean()
    
    fig, (ax_var, ax_cvar) = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    
    # --- VaR view -----------------------------------------------------------
    ax_var.scatter([0], [S0], color="steelblue", s=100, zorder=5, label="Initial Capital")
    ax_var.plot([0, horizon], [S0, var_05], color="orange", linewidth=2, linestyle="-")
    ax_var.plot([0, horizon], [S0, var_01], color="crimson", linewidth=2, linestyle="-")
    ax_var.scatter([horizon], [var_05], color="orange", s=80, label="5% VaR", zorder=5)
    ax_var.scatter([horizon], [var_01], color="crimson", s=80, label="1% VaR", zorder=5)
    
    # Shaded drawdown regions
    x = np.array([0, horizon])
    ax_var.fill_between(x, [S0, var_05], [S0, S0], color="orange", alpha=0.20)
    ax_var.fill_between(x, [S0, var_01], [S0, S0], color="crimson", alpha=0.15)
    
    ax_var.set_title(f"{horizon}-day VaR (w={w_opt:.1%})")
    ax_var.set_xticks([0, horizon])
    ax_var.set_xticklabels(["Today", f"Day {horizon}"])
    ax_var.set_ylabel("Portfolio Value")
    ax_var.grid(True, alpha=0.3)
    ax_var.legend(loc="best")

    # --- CVaR view ----------------------------------------------------------
    ax_cvar.scatter([0], [S0], color="steelblue", s=100, zorder=5, label="Initial Capital")
    ax_cvar.plot([0, horizon], [S0, mc_cvar_05], color="orange", linewidth=2, linestyle="-")
    ax_cvar.plot([0, horizon], [S0, mc_cvar_01], color="crimson", linewidth=2, linestyle="-")
    ax_cvar.scatter([horizon], [mc_cvar_05], color="orange", s=80, label="5% CVaR", zorder=5)
    ax_cvar.scatter([horizon], [mc_cvar_01], color="crimson", s=80, label="1% CVaR", zorder=5)
    
    # Shaded drawdown regions
    ax_cvar.fill_between(x, [S0, mc_cvar_05], [S0, S0], color="orange", alpha=0.20)
    ax_cvar.fill_between(x, [S0, mc_cvar_01], [S0, S0], color="crimson", alpha=0.15)
    
    ax_cvar.set_title(f"{horizon}-day CVaR (w={w_opt:.1%})")
    ax_cvar.set_xticks([0, horizon])
    ax_cvar.set_xticklabels(["Today", f"Day {horizon}"])
    ax_cvar.grid(True, alpha=0.3)
    ax_cvar.legend(loc="best")

    plt.tight_layout()
    plt.show()