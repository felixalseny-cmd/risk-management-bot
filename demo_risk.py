#!/usr/bin/env python3
"""
demo_risk.py - Demonstration script for risk engine
"""

import math
import numpy as np
from io import BytesIO

# Import risk engine
import risk_engine as re

def demo_monte_carlo():
    """Demonstrate Monte Carlo simulation"""
    print("🎲 Running Monte Carlo Simulation Demo")
    print("-" * 40)
    
    capital = 10000.0
    mu = 0.4 / 365.0  # Annual return 40%
    sigma = 0.8 / math.sqrt(365.0)  # Annual volatility 80%
    
    print(f"Initial Capital: ${capital:,.2f}")
    print(f"Daily Return (mu): {mu:.6f}")
    print(f"Daily Volatility (sigma): {sigma:.6f}")
    
    # Run simulation
    final_values, sampled_paths = re.monte_carlo_numba(
        capital=capital,
        mu=mu,
        sigma=sigma,
        leverage=3.0,
        days=365,
        simulations=3000,
        sample_paths=150
    )
    
    # Compute metrics
    metrics = re.compute_metrics_from_simulation(capital, final_values, sampled_paths)
    
    print("\n📊 Simulation Results:")
    print(f"• Probability of Doubling: {metrics['prob_double_pct']:.2f}%")
    print(f"• Probability of Halving: {metrics['prob_half_pct']:.2f}%")
    print(f"• 5% VaR (Value at Risk): ${metrics['var5']:.2f}")
    print(f"• Expected Return: {metrics['exp_return']*100:.2f}%")
    print(f"• 5th Percentile Max Drawdown: {metrics['mdd95']*100:.2f}%")
    
    # Generate plot
    try:
        buf = re.make_risk_plot(
            capital=capital,
            sampled_paths=sampled_paths,
            final_values=final_values,
            symbol='BTC/USDT',
            leverage=3.0,
            simulations=3000,
            days=365,
            logo_path=None  # Optional: path to logo
        )
        
        # Save plot
        with open('risk_plot_demo.png', 'wb') as f:
            f.write(buf.getbuffer())
        print(f"\n✅ Plot saved as 'risk_plot_demo.png'")
        
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")
    
    return metrics

def demo_data_provider():
    """Demonstrate data provider"""
    print("\n📈 Data Provider Demo")
    print("-" * 40)
    
    try:
        import data_provider as dp
        
        # Test getting prices
        print("Testing price data retrieval...")
        
        # Note: This would require actual API access
        # For demo, we'll show the structure
        print("Data provider ready with:")
        print("• Async HTTP requests with aiohttp")
        print("• TTL caching (10 minutes)")
        print("• Efficient numpy operations")
        print("• Circuit breaker pattern")
        
    except ImportError as e:
        print(f"⚠️ Data provider not available: {e}")

if __name__ == "__main__":
    print("🚀 Risk Management Bot - Demo Script")
    print("=" * 50)
    
    # Run demos
    metrics = demo_monte_carlo()
    demo_data_provider()
    
    print("\n" + "=" * 50)
    print("✅ Demo completed successfully!")
    print("Next steps:")
    print("1. Run the bot: python bot_mvp.py")
    print("2. Connect via Telegram")
    print("3. Use /help to see available commands")
