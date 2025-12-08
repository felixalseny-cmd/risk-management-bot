"""
risk_engine.py - Optimized Monte Carlo engine with fallbacks
- Numba-accelerated if available
- Memory efficient path storage
- Maximum drawdown calculations
- Plotting with optional logo
"""

import math
from io import BytesIO
from typing import Optional
import numpy as np

# Try to import numba
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    print("Numba not available, using fallback functions")

# Core MC engine
if _NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _mc_simulator_numba(S0, mu, sigma, leverage, steps, N, store_flags):
        final_vals = np.empty(N, dtype=np.float64)
        stored_count = 0
        for i in range(N):
            if store_flags[i] == 1:
                stored_count += 1
        sampled_paths = np.empty((stored_count, steps + 1), dtype=np.float64)
        store_idx = 0
        for i in prange(N):
            s = S0
            state = 123456789 ^ (i + N)
            if store_flags[i] == 1:
                sampled_paths[store_idx, 0] = s
            for t in range(1, steps + 1):
                state = (1103515245 * state + 12345) & 0xffffffff
                u1 = state / 4294967296.0
                state = (1103515245 * state + 12345) & 0xffffffff
                u2 = state / 4294967296.0
                z = math.sqrt(-2.0 * math.log(max(u1, 1e-12))) * math.cos(2.0 * math.pi * u2)
                ret = (mu * leverage) + (sigma * math.sqrt(max(leverage, 0.0)) * z)
                s = s * math.exp(ret)
                if store_flags[i] == 1:
                    sampled_paths[store_idx, t] = s
            final_vals[i] = s
            if store_flags[i] == 1:
                store_idx += 1
        return final_vals, sampled_paths
else:
    def _mc_simulator_fallback(S0, mu, sigma, leverage, steps, N, store_flags):
        final_vals = np.empty(N, dtype=np.float64)
        stored_count = int(store_flags.sum())
        sampled_paths = np.empty((stored_count, steps + 1), dtype=np.float64)
        store_idx = 0
        for i in range(N):
            s = S0
            if store_flags[i] == 1:
                sampled_paths[store_idx, 0] = s
            for t in range(1, steps + 1):
                z = np.random.randn()
                ret = (mu * leverage) + (sigma * math.sqrt(max(leverage, 0.0)) * z)
                s = s * math.exp(ret)
                if store_flags[i] == 1:
                    sampled_paths[store_idx, t] = s
            final_vals[i] = s
            if store_flags[i] == 1:
                store_idx += 1
        return final_vals, sampled_paths

def monte_carlo_numba(capital: float, mu: float, sigma: float, leverage: float = 1.0,
                      days: int = 365, simulations: int = 10000, sample_paths: int = 200):
    """Run Monte Carlo simulation"""
    N = int(simulations)
    store_count = min(int(sample_paths), N)
    store_flags = np.zeros(N, dtype=np.int8)
    
    if store_count > 0:
        idxs = np.linspace(0, N-1, store_count, dtype=np.int64)
        for i in idxs:
            store_flags[int(i)] = 1
    
    if _NUMBA_AVAILABLE:
        return _mc_simulator_numba(capital, mu, sigma, leverage, days, N, store_flags)
    else:
        return _mc_simulator_fallback(capital, mu, sigma, leverage, days, N, store_flags)

def max_drawdown_from_path(path: np.ndarray) -> float:
    """Calculate maximum drawdown from a price path"""
    peak = np.maximum.accumulate(path)
    dd = (path - peak) / peak
    return float(np.min(dd))

def compute_metrics_from_simulation(capital: float, final_values: np.ndarray, sampled_paths: np.ndarray) -> dict:
    """Compute risk metrics from simulation results"""
    final = final_values
    prob_double = float((final >= capital * 2).mean() * 100.0)
    prob_half = float((final <= capital * 0.5).mean() * 100.0)
    var5 = float(np.percentile(final, 5))
    exp_return = float(final.mean() / capital - 1.0)
    
    if sampled_paths.size == 0:
        mdd95 = 0.0
    else:
        mdds = np.array([max_drawdown_from_path(sampled_paths[i]) for i in range(sampled_paths.shape[0])], dtype=np.float64)
        mdd95 = float(np.percentile(mdds, 5))
    
    return {
        'prob_double_pct': prob_double,
        'prob_half_pct': prob_half,
        'var5': var5,
        'exp_return': exp_return,
        'mdd95': mdd95,
    }

# Plotting
try:
    from PIL import Image
    import matplotlib.pyplot as plt
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

def make_risk_plot(capital: float, sampled_paths: np.ndarray, final_values: np.ndarray,
                   symbol: str, leverage: float, simulations: int, days: int,
                   logo_path: Optional[str] = None) -> BytesIO:
    """Generate risk plot as PNG"""
    if not _PLOTTING_AVAILABLE:
        raise ImportError("Matplotlib/PIL not available for plotting")
    
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if sampled_paths.size == 0:
        ax.hist(final_values, bins=60, alpha=0.7)
    else:
        n_plot = min(sampled_paths.shape[0], 200)
        for i in range(n_plot):
            ax.plot(sampled_paths[i].T, linewidth=0.9, alpha=0.08, color='blue')
        
        p50 = np.percentile(sampled_paths, 50, axis=0)
        ax.plot(p50, linewidth=2.0, color='red', label='Median Path')
    
    ax.axhline(capital, linewidth=2.0, color='green', linestyle='--', label='Initial Capital')
    ax.set_title(f"{symbol} ×{leverage} • {simulations:,} simulations • {days} days")
    ax.set_xlabel("Days")
    ax.set_ylabel("Capital ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if logo_path is not None:
        try:
            logo = Image.open(logo_path)
            fig.figimage(logo, xo=fig.bbox.xmax - logo.width - 10,
                         yo=fig.bbox.ymax - logo.height - 10, zorder=10)
        except Exception:
            pass
    
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
