# demo_risk.py
import math
from pathlib import Path
import importlib, sys
sys.path.insert(0, '/mnt/data')  # если используешь /mnt/data

import risk_engine as re
import numpy as np

capital = 10000.0
mu = 0.4 / 365.0
sigma = 0.8 / math.sqrt(365.0)
final, sampled = re.monte_carlo_numba(capital, mu, sigma, leverage=3.0, days=365, simulations=3000, sample_paths=150)
metrics = re.compute_metrics_from_simulation(capital, final, sampled)
print('metrics:', metrics)

logo = '/mnt/data/fxwave_logo.bmp'  # путь к твоему логотипу
buf = re.make_risk_plot(capital, sampled, final, 'BTC/USDT', 3.0, 3000, 365, logo_path=logo)
out = Path('/mnt/data/risk_plot_example.png')
with open(out, 'wb') as f:
    f.write(buf.getbuffer())
print('Saved plot to', out)
