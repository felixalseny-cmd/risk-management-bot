# bot_mvp.py — PRO Risk Calculator MVP Phase 1
import os
import logging
import asyncio
import time
import functools
import json
import re
import html
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict

# --- Load .env ---
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found!")

PORT = int(os.getenv("PORT", 10000))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").rstrip("/")
WEBHOOK_PATH = f"/webhook/{TOKEN}"

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY") 
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
EXCHANGERATE_API_KEY = os.getenv("EXCHANGERATE_API_KEY", "d8f8278cf29f8fe18445e8b7")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "972d1359cbf04ff68dd0feba7e32cc8d")
FMP_API_KEY = os.getenv("FMP_API_KEY", "nZm3b15R1rJvjnUO67wPb0eaJHPXarK2")
METALPRICE_API_KEY = os.getenv("METALPRICE_API_KEY", "e6e8aa0b29f4e612751cde3985a7b8ec")

# Donation Wallets
USDT_WALLET_ADDRESS = os.getenv("USDT_WALLET_ADDRESS", "TVRGFPKVs1nN3fUXBTQfu5syTcmYGgADre")
TON_WALLET_ADDRESS = os.getenv("TON_WALLET_ADDRESS", "UQD2GekkF3W-ZTUkRobEfSgnVM5nymzuiWtTOe4T5fog07Vi")

# --- Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("risk_calculator_mvp")

# ==================== PHASE 1: CORE STABILITY ====================
# 1. Fixed imports and startup
# 2. Basic data provider with caching
# 3. Core risk calculations (VaR, CVaR, MDD)
# 4. Simple Telegram commands

# ---------------------------
# Performance Monitoring Decorator
# ---------------------------
def monitor_performance(func):
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            if elapsed > 1.0:  # Log slow operations
                logger.warning(f"Slow operation: {func.__name__} took {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in {func.__name__} after {elapsed:.2f}s: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            if elapsed > 1.0:  # Log slow operations
                logger.warning(f"Slow operation: {func.__name__} took {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in {func.__name__} after {elapsed:.2f}s: {e}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# ---------------------------
# Memory Guardian for Render Free (512MB)
# ---------------------------
class MemoryGuardian:
    """Memory management for Render Free constraints"""
    
    MEMORY_THRESHOLD = 400  # MB
    CACHE_CLEAR_PERCENT = 0.5
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0  # Fallback if psutil not available
    
    @staticmethod
    def clear_oldest_cache(cache_obj, percent=0.5):
        """Clear oldest entries from cache"""
        try:
            if hasattr(cache_obj, 'cache'):
                cache = cache_obj.cache
                if hasattr(cache, '__len__'):
                    items_to_remove = int(len(cache) * percent)
                    if items_to_remove > 0:
                        # Simple strategy: clear based on insertion order
                        keys = list(cache.keys())[:items_to_remove]
                        for key in keys:
                            cache.pop(key, None)
                        logger.info(f"Cleared {items_to_remove} cache entries")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    @staticmethod
    def check_and_clear(cache_objects=None):
        """Check memory and clear if above threshold"""
        try:
            memory_usage = MemoryGuardian.get_memory_usage()
            if memory_usage > MemoryGuardian.MEMORY_THRESHOLD:
                logger.warning(f"Memory usage high: {memory_usage:.1f}MB, clearing cache")
                
                # Clear global caches
                if cache_objects:
                    for cache_obj in cache_objects:
                        MemoryGuardian.clear_oldest_cache(cache_obj, MemoryGuardian.CACHE_CLEAR_PERCENT)
                
                # Force garbage collection
                gc.collect()
                
                new_usage = MemoryGuardian.get_memory_usage()
                logger.info(f"Memory after cleanup: {new_usage:.1f}MB")
                return True
            return False
        except Exception as e:
            logger.error(f"Memory check error: {e}")
            return False

# ---------------------------
# Circuit Breaker for API Calls
# ---------------------------
class CircuitBreaker:
    """Circuit breaker pattern for API resilience"""
    
    def __init__(self, max_failures=3, reset_timeout=60, half_open_timeout=30):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def record_failure(self):
        """Record an API failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.max_failures:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPEN for API after {self.failure_count} failures")
    
    def record_success(self):
        """Record an API success"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def can_execute(self):
        """Check if API call can be executed"""
        if self.state == "CLOSED":
            return True
        
        elif self.state == "OPEN":
            # Check if reset timeout has passed
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moving to HALF_OPEN state")
                return True
            return False
        
        elif self.state == "HALF_OPEN":
            # Allow one trial call
            return True
        
        return False

# ---------------------------
# MVP Data Provider (Optimized for Render Free)
# ---------------------------
class MVPDataProvider:
    """Optimized data provider with circuit breakers and caching"""
    
    def __init__(self):
        import cachetools
        self.cache = cachetools.TTLCache(maxsize=200, ttl=600)  # 10 minute cache
        self.session = None
        self.circuit_breakers = defaultdict(lambda: CircuitBreaker())
        self.api_priority = [
            (self._get_binance_price, "Binance"),
            (self._get_fmp_price, "FMP"),
            (self._get_frankfurter_price, "Frankfurter"),
            (self._get_fallback_price, "Static")
        ]
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            import aiohttp
            self.session = aiohttp.ClientSession()
        return self.session
    
    @monitor_performance
    async def get_price(self, symbol: str) -> float:
        """Get price with circuit breakers and caching"""
        # Check cache first
        cache_key = f"price_{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Check circuit breaker for this symbol
        if not self.circuit_breakers[symbol].can_execute():
            logger.warning(f"Circuit breaker blocked for {symbol}, using cache/fallback")
            return self._get_fallback_price(symbol)
        
        # Try APIs in priority order
        price = None
        source = None
        
        for api_func, api_name in self.api_priority:
            try:
                price = await api_func(symbol)
                if price and price > 0:
                    source = api_name
                    self.circuit_breakers[symbol].record_success()
                    break
            except Exception as e:
                logger.warning(f"{api_name} failed for {symbol}: {e}")
                self.circuit_breakers[symbol].record_failure()
        
        if price is None or price <= 0:
            price = self._get_fallback_price(symbol)
            source = "Fallback"
        
        # Cache the result
        if price:
            self.cache[cache_key] = price
        
        logger.info(f"Price for {symbol}: ${price:.2f} ({source})")
        return price
    
    @monitor_performance
    async def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get multiple prices in parallel"""
        tasks = [self.get_price(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        prices = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error getting price for {symbol}: {result}")
                prices[symbol] = self._get_fallback_price(symbol)
            else:
                prices[symbol] = result
        
        return prices
    
    async def get_returns(self, symbol: str, period: int = 30) -> List[float]:
        """Get historical returns (simplified - using mock data for MVP)"""
        # For MVP, we'll generate simulated returns
        # In PRO phase, this will use actual historical data
        import numpy as np
        np.random.seed(hash(symbol) % 10000)
        
        # Generate realistic returns based on asset type
        if any(c in symbol for c in ['BTC', 'ETH', 'XRP']):
            # Crypto: high volatility
            returns = np.random.normal(0.001, 0.04, period).tolist()  # 4% daily volatility
        elif any(c in symbol for c in ['AAPL', 'TSLA', 'GOOG']):
            # Stocks: medium volatility
            returns = np.random.normal(0.0005, 0.02, period).tolist()  # 2% daily volatility
        elif any(c in symbol for c in ['EUR', 'GBP', 'JPY']):
            # Forex: low volatility
            returns = np.random.normal(0.0001, 0.008, period).tolist()  # 0.8% daily volatility
        else:
            # Default
            returns = np.random.normal(0.0003, 0.015, period).tolist()
        
        return returns
    
    def get_volatility(self, returns: List[float], annualize: bool = True) -> float:
        """Calculate volatility from returns"""
        import numpy as np
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        daily_vol = np.std(returns_array)
        
        if annualize:
            return daily_vol * np.sqrt(252)  # Trading days
        return daily_vol
    
    # API implementations
    async def _get_binance_price(self, symbol: str) -> Optional[float]:
        """Get price from Binance"""
        if not BINANCE_API_KEY or 'USDT' not in symbol:
            return None
        
        try:
            session = await self.get_session()
            # Format: BTCUSDT -> BTCUSDT, ETHUSDT -> ETHUSDT
            binance_symbol = symbol.replace('USD', '') + 'USDT' if 'USD' in symbol else symbol
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"
            
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data['price'])
        except Exception as e:
            logger.debug(f"Binance API error for {symbol}: {e}")
        return None
    
    async def _get_fmp_price(self, symbol: str) -> Optional[float]:
        """Get price from Financial Modeling Prep"""
        if not FMP_API_KEY:
            return None
        
        try:
            session = await self.get_session()
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
            
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and isinstance(data, list) and len(data) > 0:
                        return data[0]['price']
        except Exception as e:
            logger.debug(f"FMP API error for {symbol}: {e}")
        return None
    
    async def _get_frankfurter_price(self, symbol: str) -> Optional[float]:
        """Get Forex prices from Frankfurter"""
        try:
            # Check if it's a Forex pair (6 characters like EURUSD)
            if len(symbol) == 6 and symbol.isalpha():
                from_curr = symbol[:3]
                to_curr = symbol[3:]
                url = f"https://api.frankfurter.app/latest?from={from_curr}&to={to_curr}"
                
                session = await self.get_session()
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['rates'][to_curr]
        except Exception as e:
            logger.debug(f"Frankfurter API error for {symbol}: {e}")
        return None
    
    def _get_fallback_price(self, symbol: str) -> float:
        """Static fallback prices"""
        fallback_prices = {
            # Forex
            'EURUSD': 1.08, 'GBPUSD': 1.26, 'USDJPY': 151.0, 'USDCHF': 0.88,
            'AUDUSD': 0.66, 'USDCAD': 1.36, 'NZDUSD': 0.61,
            # Crypto
            'BTCUSDT': 105000.0, 'ETHUSDT': 5500.0, 'XRPUSDT': 1.20,
            # Stocks
            'AAPL': 210.0, 'TSLA': 310.0, 'GOOGL': 155.0, 'MSFT': 410.0,
            # Indices
            'SPX': 5500.0, 'NAS100': 21000.0, 'DJI': 40000.0,
            # Metals
            'XAUUSD': 2550.0, 'XAGUSD': 32.0,
            # Energy
            'OIL': 82.0, 'NATGAS': 3.20
        }
        return fallback_prices.get(symbol, 100.0)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()

# ---------------------------
# MVP Risk Engine (NumPy + Basic Calculations)
# ---------------------------
class MVPRiskEngine:
    """Core risk calculations without pandas"""
    
    @staticmethod
    @monitor_performance
    def historical_var(returns: List[float], confidence: float = 0.95) -> float:
        """Historical Value at Risk"""
        if not returns:
            return 0.0
        
        import numpy as np
        returns_array = np.array(returns)
        var_percentile = np.percentile(returns_array, (1 - confidence) * 100)
        return abs(var_percentile)  # Return as positive loss
    
    @staticmethod
    @monitor_performance
    def parametric_var(mean_return: float, volatility: float, confidence: float = 0.95) -> float:
        """Parametric (Normal) Value at Risk"""
        import numpy as np
        from scipy.stats import norm
        
        z_score = norm.ppf(confidence)
        var = mean_return - z_score * volatility
        return abs(var)  # Return as positive loss
    
    @staticmethod
    @monitor_performance
    def conditional_var(returns: List[float], confidence: float = 0.95) -> float:
        """Conditional Value at Risk (Expected Shortfall)"""
        if not returns:
            return 0.0
        
        import numpy as np
        returns_array = np.array(returns)
        var_threshold = np.percentile(returns_array, (1 - confidence) * 100)
        
        # Average of losses beyond VaR
        losses_beyond_var = returns_array[returns_array <= var_threshold]
        if len(losses_beyond_var) == 0:
            return abs(var_threshold)
        
        cvar = np.mean(losses_beyond_var)
        return abs(cvar)
    
    @staticmethod
    @monitor_performance
    def max_drawdown(prices: List[float]) -> float:
        """Maximum Drawdown calculation"""
        if len(prices) < 2:
            return 0.0
        
        import numpy as np
        prices_array = np.array(prices)
        
        # Calculate cumulative max
        cumulative_max = np.maximum.accumulate(prices_array)
        
        # Calculate drawdown
        drawdown = (prices_array - cumulative_max) / cumulative_max
        
        # Maximum drawdown (most negative)
        max_dd = np.min(drawdown)
        return abs(max_dd)  # Return as positive percentage
    
    @staticmethod
    @monitor_performance
    def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Sharpe Ratio (annualized)"""
        if len(returns) < 2:
            return 0.0
        
        import numpy as np
        returns_array = np.array(returns)
        
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess == 0:
            return 0.0
        
        # Annualize
        sharpe = (mean_excess / std_excess) * np.sqrt(252)
        return sharpe
    
    @staticmethod
    @monitor_performance
    def sortino_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Sortino Ratio (annualized)"""
        if len(returns) < 2:
            return 0.0
        
        import numpy as np
        returns_array = np.array(returns)
        
        excess_returns = returns_array - (risk_free_rate / 252)
        mean_excess = np.mean(excess_returns)
        
        # Downside deviation (only negative returns)
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) == 0:
            return 0.0
        
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
        
        # Annualize
        sortino = (mean_excess / downside_std) * np.sqrt(252)
        return sortino
    
    @staticmethod
    @monitor_performance
    def monte_carlo_simulation(
        initial_price: float,
        mean_return: float,
        volatility: float,
        days: int = 30,
        simulations: int = 1000
    ) -> Tuple[List[float], List[List[float]]]:
        """Monte Carlo simulation with path sampling"""
        import numpy as np
        
        # For MVP, we limit simulations to save memory
        simulations = min(simulations, 1000)
        
        dt = 1/252  # Daily steps
        prices = np.zeros((simulations, days))
        
        # Generate all random numbers at once for efficiency
        random_shocks = np.random.normal(0, 1, (simulations, days))
        
        for t in range(days):
            if t == 0:
                prices[:, t] = initial_price
            else:
                drift = (mean_return - 0.5 * volatility**2) * dt
                diffusion = volatility * np.sqrt(dt) * random_shocks[:, t]
                prices[:, t] = prices[:, t-1] * np.exp(drift + diffusion)
        
        # Final prices
        final_prices = prices[:, -1].tolist()
        
        # Sample only 5% of paths for memory efficiency
        sample_size = max(1, int(simulations * 0.05))
        sample_indices = np.random.choice(simulations, sample_size, replace=False)
        sample_paths = prices[sample_indices, :].tolist()
        
        return final_prices, sample_paths

# ---------------------------
# MVP Stress Tester (Basic Scenarios)
# ---------------------------
class MVPStressTester:
    """Basic stress testing scenarios"""
    
    @staticmethod
    def crisis_2008(portfolio_value: float, asset_allocation: Dict[str, float]) -> Dict[str, Any]:
        """2008 Financial Crisis scenario"""
        # Historical crisis impacts
        crisis_impacts = {
            'stocks': -0.50,      # -50% for stocks
            'indices': -0.45,     # -45% for indices
            'crypto': -0.60,      # -60% for crypto (though crypto wasn't big in 2008)
            'forex': -0.15,       # -15% for forex (USD safe haven)
            'metals': 0.25,       # +25% for gold (safe haven)
            'energy': -0.40       # -40% for oil
        }
        
        total_impact = 0.0
        for asset_type, allocation in asset_allocation.items():
            impact = crisis_impacts.get(asset_type, -0.30)  # -30% default
            total_impact += impact * allocation
        
        stressed_value = portfolio_value * (1 + total_impact)
        drawdown = abs(total_impact) * 100
        
        return {
            'scenario': '2008 Financial Crisis',
            'stressed_value': round(stressed_value, 2),
            'loss_percent': round(abs(total_impact) * 100, 1),
            'drawdown': round(drawdown, 1),
            'recovery_months': 24
        }
    
    @staticmethod
    def covid_2020(portfolio_value: float, asset_allocation: Dict[str, float]) -> Dict[str, Any]:
        """COVID-19 Crash scenario"""
        covid_impacts = {
            'stocks': -0.35,      # -35% for stocks
            'indices': -0.30,     # -30% for indices
            'crypto': -0.50,      # -50% for crypto
            'forex': -0.10,       # -10% for forex
            'metals': 0.15,       # +15% for gold
            'energy': -0.60       # -60% for oil
        }
        
        total_impact = 0.0
        for asset_type, allocation in asset_allocation.items():
            impact = covid_impacts.get(asset_type, -0.25)
            total_impact += impact * allocation
        
        stressed_value = portfolio_value * (1 + total_impact)
        
        return {
            'scenario': 'COVID-19 Crash (2020)',
            'stressed_value': round(stressed_value, 2),
            'loss_percent': round(abs(total_impact) * 100, 1),
            'drawdown': round(abs(total_impact) * 100, 1),
            'recovery_months': 6
        }
    
    @staticmethod
    def crypto_winter_2022(portfolio_value: float, crypto_allocation: float) -> Dict[str, Any]:
        """Crypto Winter 2022 scenario"""
        crypto_impact = -0.75  # -75% for crypto assets
        non_crypto_impact = -0.20  # -20% for everything else
        
        # Simple: if portfolio has crypto, apply crypto winter
        total_impact = (crypto_impact * crypto_allocation) + (non_crypto_impact * (1 - crypto_allocation))
        stressed_value = portfolio_value * (1 + total_impact)
        
        return {
            'scenario': 'Crypto Winter 2022',
            'stressed_value': round(stressed_value, 2),
            'loss_percent': round(abs(total_impact) * 100, 1),
            'drawdown': round(abs(total_impact) * 100, 1),
            'recovery_months': 18
        }

# ---------------------------
# MVP Alert Manager (Basic)
# ---------------------------
class MVPAlertManager:
    """Basic alert system for critical events"""
    
    def __init__(self):
        self.alerts = {}
        self.alert_id_counter = 0
    
    def add_price_alert(self, user_id: int, symbol: str, target_price: float, condition: str):
        """Add price alert (above/below)"""
        alert_id = self.alert_id_counter
        self.alert_id_counter += 1
        
        alert = {
            'id': alert_id,
            'user_id': user_id,
            'type': 'price',
            'symbol': symbol,
            'target_price': target_price,
            'condition': condition,  # 'above' or 'below'
            'triggered': False,
            'created_at': datetime.now().isoformat()
        }
        
        if user_id not in self.alerts:
            self.alerts[user_id] = []
        
        self.alerts[user_id].append(alert)
        return alert_id
    
    def add_margin_alert(self, user_id: int, threshold_percent: float):
        """Add margin alert"""
        alert_id = self.alert_id_counter
        self.alert_id_counter += 1
        
        alert = {
            'id': alert_id,
            'user_id': user_id,
            'type': 'margin',
            'threshold_percent': threshold_percent,
            'triggered': False,
            'created_at': datetime.now().isoformat()
        }
        
        if user_id not in self.alerts:
            self.alerts[user_id] = []
        
        self.alerts[user_id].append(alert)
        return alert_id
    
    def check_price_alerts(self, user_id: int, symbol: str, current_price: float):
        """Check and trigger price alerts"""
        if user_id not in self.alerts:
            return []
        
        triggered = []
        for alert in self.alerts[user_id]:
            if alert['type'] == 'price' and alert['symbol'] == symbol and not alert['triggered']:
                if alert['condition'] == 'above' and current_price >= alert['target_price']:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now().isoformat()
                    triggered.append(alert)
                elif alert['condition'] == 'below' and current_price <= alert['target_price']:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now().isoformat()
                    triggered.append(alert)
        
        return triggered
    
    def check_margin_alert(self, user_id: int, margin_level: float):
        """Check margin alert"""
        if user_id not in self.alerts:
            return []
        
        triggered = []
        for alert in self.alerts[user_id]:
            if alert['type'] == 'margin' and not alert['triggered']:
                if margin_level <= alert['threshold_percent']:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now().isoformat()
                    triggered.append(alert)
        
        return triggered
    
    def get_user_alerts(self, user_id: int):
        """Get all alerts for user"""
        return self.alerts.get(user_id, [])
    
    def delete_alert(self, user_id: int, alert_id: int):
        """Delete specific alert"""
        if user_id in self.alerts:
            self.alerts[user_id] = [a for a in self.alerts[user_id] if a['id'] != alert_id]
            return True
        return False
    
    def save_to_file(self, filename: str = 'alerts.json'):
        """Save alerts to file"""
        try:
            with open(filename, 'w') as f:
                # Convert datetime objects to strings
                serializable = {}
                for user_id, user_alerts in self.alerts.items():
                    serializable[str(user_id)] = user_alerts
                json.dump(serializable, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")
    
    def load_from_file(self, filename: str = 'alerts.json'):
        """Load alerts from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.alerts = {}
                    for user_id_str, user_alerts in data.items():
                        self.alerts[int(user_id_str)] = user_alerts
        except Exception as e:
            logger.error(f"Error loading alerts: {e}")

# ---------------------------
# Telegram Bot Setup
# ---------------------------
# Note: We'll implement the Telegram bot in the next phase
# For now, we create the core components

# Global instances
data_provider = MVPDataProvider()
risk_engine = MVPRiskEngine()
stress_tester = MVPStressTester()
alert_manager = MVPAlertManager()

# Load saved alerts on startup
alert_manager.load_from_file()

# ---------------------------
# Utility Functions
# ---------------------------
async def calculate_asset_risk(symbol: str, period: int = 30) -> Dict[str, Any]:
    """Calculate comprehensive risk metrics for an asset"""
    # Get current price
    current_price = await data_provider.get_price(symbol)
    
    # Get historical returns
    returns = await data_provider.get_returns(symbol, period)
    
    if not returns or len(returns) < 5:
        return {
            'symbol': symbol,
            'current_price': current_price,
            'error': 'Insufficient data for risk calculation',
            'period': period
        }
    
    # Calculate metrics
    import numpy as np
    returns_array = np.array(returns)
    
    mean_return = np.mean(returns_array)
    volatility = data_provider.get_volatility(returns)
    annual_volatility = data_provider.get_volatility(returns, annualize=True)
    
    # Risk metrics
    historical_var_95 = risk_engine.historical_var(returns, 0.95)
    historical_var_99 = risk_engine.historical_var(returns, 0.99)
    parametric_var_95 = risk_engine.parametric_var(mean_return, volatility, 0.95)
    conditional_var_95 = risk_engine.conditional_var(returns, 0.95)
    
    # Sharpe and Sortino
    sharpe = risk_engine.sharpe_ratio(returns)
    sortino = risk_engine.sortino_ratio(returns)
    
    # Generate price history for drawdown
    # Simulate prices from returns
    prices = [current_price]
    for ret in reversed(returns[-30:]):  # Last 30 days
        prices.insert(0, prices[0] / (1 + ret))
    
    max_dd = risk_engine.max_drawdown(prices)
    
    # Monte Carlo simulation
    final_prices, sample_paths = risk_engine.monte_carlo_simulation(
        current_price, mean_return, volatility, days=30, simulations=500
    )
    
    # Calculate VaR from Monte Carlo
    mc_returns = [(fp - current_price) / current_price for fp in final_prices]
    mc_var_95 = risk_engine.historical_var(mc_returns, 0.95)
    
    return {
        'symbol': symbol,
        'current_price': round(current_price, 2),
        'period': period,
        'mean_return': round(mean_return * 100, 3),  # Percentage
        'daily_volatility': round(volatility * 100, 3),
        'annual_volatility': round(annual_volatility * 100, 2),
        'historical_var_95': round(historical_var_95 * 100, 3),
        'historical_var_99': round(historical_var_99 * 100, 3),
        'parametric_var_95': round(parametric_var_95 * 100, 3),
        'conditional_var_95': round(conditional_var_95 * 100, 3),
        'max_drawdown': round(max_dd * 100, 2),
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(sortino, 2),
        'monte_carlo_var_95': round(mc_var_95 * 100, 3),
        'monte_carlo_metrics': {
            'simulations': len(final_prices),
            'average_final_price': round(np.mean(final_prices), 2),
            'worst_final_price': round(np.min(final_prices), 2),
            'best_final_price': round(np.max(final_prices), 2),
            'probability_loss': round(sum(1 for p in final_prices if p < current_price) / len(final_prices) * 100, 1)
        }
    }

def format_risk_report(risk_data: Dict[str, Any]) -> str:
    """Format risk metrics for display"""
    if 'error' in risk_data:
        return f"❌ Error for {risk_data['symbol']}: {risk_data['error']}"
    
    report = [
        f"📊 **Risk Analysis: {risk_data['symbol']}**",
        f"Current Price: ${risk_data['current_price']:.2f}",
        f"Analysis Period: {risk_data['period']} days",
        "",
        "📈 **Returns & Volatility:**",
        f"• Mean Daily Return: {risk_data['mean_return']}%",
        f"• Daily Volatility: {risk_data['daily_volatility']}%",
        f"• Annual Volatility: {risk_data['annual_volatility']}%",
        "",
        "⚠️ **Value at Risk (VaR):**",
        f"• Historical 95% VaR: {risk_data['historical_var_95']}%",
        f"• Historical 99% VaR: {risk_data['historical_var_99']}%",
        f"• Parametric 95% VaR: {risk_data['parametric_var_95']}%",
        f"• Conditional VaR (95%): {risk_data['conditional_var_95']}%",
        f"• Monte Carlo 95% VaR: {risk_data['monte_carlo_var_95']}%",
        "",
        "📉 **Drawdown & Ratios:**",
        f"• Maximum Drawdown: {risk_data['max_drawdown']}%",
        f"• Sharpe Ratio: {risk_data['sharpe_ratio']:.2f}",
        f"• Sortino Ratio: {risk_data['sortino_ratio']:.2f}",
        "",
        "🎲 **Monte Carlo Simulation:**",
        f"• Simulations: {risk_data['monte_carlo_metrics']['simulations']}",
        f"• Average 30-day Price: ${risk_data['monte_carlo_metrics']['average_final_price']:.2f}",
        f"• Probability of Loss: {risk_data['monte_carlo_metrics']['probability_loss']}%",
    ]
    
    return "\n".join(report)

# ---------------------------
# Health Check Endpoint
# ---------------------------
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check"""
    import sys
    
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "MVP 1.0",
        "python_version": sys.version.split()[0],
        "services": {}
    }
    
    # Check data provider
    try:
        btc_price = await data_provider.get_price("BTCUSDT")
        health["services"]["data_provider"] = {
            "status": "operational",
            "sample_price": btc_price
        }
    except Exception as e:
        health["services"]["data_provider"] = {
            "status": "error",
            "error": str(e)
        }
        health["status"] = "degraded"
    
    # Check memory
    try:
        memory_usage = MemoryGuardian.get_memory_usage()
        health["memory_usage_mb"] = round(memory_usage, 1)
        health["services"]["memory"] = {
            "status": "good" if memory_usage < 300 else "warning",
            "usage_mb": round(memory_usage, 1)
        }
        
        if memory_usage > 350:
            health["status"] = "warning"
    except Exception as e:
        health["services"]["memory"] = {
            "status": "error",
            "error": str(e)
        }
    
    return health

# ---------------------------
# Main Application
# ---------------------------
async def main():
    """Main application entry point"""
    logger.info("🚀 Launching MVP Risk Calculator - Phase 1")
    logger.info("✅ Core stability components initialized")
    
    # Test the system
    try:
        # Test data provider
        symbols = ["BTCUSDT", "AAPL", "EURUSD"]
        logger.info("Testing data provider...")
        
        for symbol in symbols:
            price = await data_provider.get_price(symbol)
            logger.info(f"  {symbol}: ${price:.2f}")
        
        # Test risk engine
        logger.info("Testing risk engine...")
        returns = await data_provider.get_returns("BTCUSDT", 30)
        if returns:
            var = risk_engine.historical_var(returns, 0.95)
            logger.info(f"  BTC 95% VaR: {var*100:.2f}%")
        
        # Health check
        health = await health_check()
        logger.info(f"Health status: {health['status']}")
        logger.info(f"Memory usage: {health.get('memory_usage_mb', 0):.1f}MB")
        
        # Save alerts
        alert_manager.save_to_file()
        
        logger.info("✅ MVP Phase 1 components are working!")
        
        # Keep running for webhook mode
        if WEBHOOK_URL:
            logger.info("🌐 Webhook mode would start here")
            # In Phase 2, we'll add the actual web server
        else:
            logger.info("🔄 Polling mode would start here")
            # In Phase 2, we'll add the actual bot polling
            
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
    
    finally:
        # Cleanup
        await data_provider.cleanup()
        logger.info("🧹 Cleanup completed")

# ---------------------------
# Run the application
# ---------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹ Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
