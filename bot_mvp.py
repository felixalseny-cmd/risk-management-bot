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

# bot_mvp_phase2.py — PRO Risk Calculator MVP Phase 2
import os
import logging
import asyncio
import time
import functools
import json
import re
import html
import gc
import io
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

# --- Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("risk_calculator")

# ==================== ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ ИЗ PHASE 1 ====================
# (В реальном коде здесь были бы импорты, но для целостности включаем ключевые части)

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
            if elapsed > 1.0:
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
            if elapsed > 1.0:
                logger.warning(f"Slow operation: {func.__name__} took {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in {func.__name__} after {elapsed:.2f}s: {e}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# ---------------------------
# Memory Guardian
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
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0
    
    @staticmethod
    def clear_oldest_cache(cache_obj, percent=0.5):
        """Clear oldest entries from cache"""
        try:
            if hasattr(cache_obj, 'cache'):
                cache = cache_obj.cache
                if hasattr(cache, '__len__'):
                    items_to_remove = int(len(cache) * percent)
                    if items_to_remove > 0:
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
                
                if cache_objects:
                    for cache_obj in cache_objects:
                        MemoryGuardian.clear_oldest_cache(cache_obj, MemoryGuardian.CACHE_CLEAR_PERCENT)
                
                gc.collect()
                
                new_usage = MemoryGuardian.get_memory_usage()
                logger.info(f"Memory after cleanup: {new_usage:.1f}MB")
                return True
            return False
        except Exception as e:
            logger.error(f"Memory check error: {e}")
            return False

# ==================== OPTIMIZED RISK ENGINE WITH NUMBA ====================
# Используем Numba для JIT компиляции критических функций

class OptimizedRiskEngine:
    """Optimized risk calculations with Numba acceleration"""
    
    @staticmethod
    @monitor_performance
    def calculate_var(returns: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
        """Calculate Historical, Parametric and Conditional VaR in one pass"""
        import numpy as np
        from scipy.stats import norm
        
        returns_array = np.array(returns)
        
        if len(returns_array) < 10:
            return 0.0, 0.0, 0.0
        
        # Historical VaR
        historical_var = np.percentile(returns_array, (1 - confidence) * 100)
        
        # Parametric VaR
        mean_return = np.mean(returns_array)
        volatility = np.std(returns_array)
        z_score = norm.ppf(confidence)
        parametric_var = mean_return - z_score * volatility
        
        # Conditional VaR (Expected Shortfall)
        losses_beyond_var = returns_array[returns_array <= historical_var]
        conditional_var = np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else historical_var
        
        return (
            abs(historical_var) * 100,  # Convert to percentage
            abs(parametric_var) * 100,
            abs(conditional_var) * 100
        )
    
    @staticmethod
    @monitor_performance
    def monte_carlo_numba(S0: float, mu: float, sigma: float, 
                         days: int = 30, simulations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo simulation with Numba acceleration
        Returns: (final_prices, sample_paths)
        """
        import numpy as np
        
        try:
            # Try to use Numba for JIT compilation
            from numba import jit, prange
            
            @jit(nopython=True, parallel=True, cache=True)
            def run_simulation(S0, mu, sigma, days, simulations):
                dt = 1/252
                drift = (mu - 0.5 * sigma**2) * dt
                diffusion = sigma * np.sqrt(dt)
                
                # Pre-allocate arrays
                prices = np.zeros((simulations, days))
                random_shocks = np.random.normal(0, 1, (simulations, days))
                
                for i in prange(simulations):
                    prices[i, 0] = S0
                    for t in range(1, days):
                        prices[i, t] = prices[i, t-1] * np.exp(drift + diffusion * random_shocks[i, t])
                
                return prices
            
            # Run simulation
            prices = run_simulation(S0, mu, sigma, days, simulations)
            final_prices = prices[:, -1]
            
            # Sample 5% of paths
            sample_size = max(1, int(simulations * 0.05))
            sample_indices = np.random.choice(simulations, sample_size, replace=False)
            sample_paths = prices[sample_indices, :]
            
            return final_prices, sample_paths
            
        except ImportError:
            # Fallback to vectorized NumPy if Numba not available
            logger.info("Numba not available, using vectorized NumPy")
            return OptimizedRiskEngine.monte_carlo_vectorized(S0, mu, sigma, days, simulations)
    
    @staticmethod
    @monitor_performance
    def monte_carlo_vectorized(S0: float, mu: float, sigma: float, 
                              days: int = 30, simulations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized Monte Carlo without Numba"""
        import numpy as np
        
        dt = 1/252
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate all random numbers at once
        random_shocks = np.random.normal(0, 1, (simulations, days))
        
        # Vectorized calculation
        increments = np.exp(drift + diffusion * random_shocks)
        # Cumprod along time axis
        price_paths = S0 * np.cumprod(increments, axis=1)
        
        final_prices = price_paths[:, -1]
        
        # Sample 5% of paths
        sample_size = max(1, int(simulations * 0.05))
        sample_indices = np.random.choice(simulations, sample_size, replace=False)
        sample_paths = price_paths[sample_indices, :]
        
        return final_prices, sample_paths
    
    @staticmethod
    @monitor_performance
    def calculate_portfolio_risk(weights: List[float], returns_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        import numpy as np
        
        weights_array = np.array(weights)
        
        if len(weights_array) != returns_matrix.shape[1]:
            raise ValueError("Weights must match number of assets")
        
        # Portfolio returns
        portfolio_returns = np.dot(returns_matrix, weights_array)
        
        # Basic metrics
        mean_return = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns)
        annual_volatility = volatility * np.sqrt(252)
        
        # VaR metrics
        historical_var_95 = np.percentile(portfolio_returns, 5)
        conditional_var_95 = np.mean(portfolio_returns[portfolio_returns <= historical_var_95])
        
        # Sharpe ratio (annualized)
        risk_free_rate = 0.02 / 252  # Daily
        excess_returns = portfolio_returns - risk_free_rate
        sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Sortino ratio
        negative_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino = (np.mean(excess_returns) / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        return {
            'portfolio_mean_return': mean_return * 100,
            'portfolio_volatility': volatility * 100,
            'portfolio_annual_volatility': annual_volatility,
            'portfolio_var_95': abs(historical_var_95) * 100,
            'portfolio_cvar_95': abs(conditional_var_95) * 100,
            'portfolio_sharpe': sharpe,
            'portfolio_sortino': sortino,
            'portfolio_skewness': float(np.cov(portfolio_returns)[0, 0]) if len(portfolio_returns) > 1 else 0
        }
    
    @staticmethod
    @monitor_performance
    def stress_test_portfolio(portfolio_value: float, allocations: Dict[str, float]) -> List[Dict[str, Any]]:
        """Run multiple stress test scenarios"""
        scenarios = []
        
        # 2008 Financial Crisis
        crisis_2008 = {
            'stocks': -0.50,
            'indices': -0.45,
            'crypto': -0.60,
            'forex': -0.15,
            'metals': 0.25,
            'energy': -0.40
        }
        
        # COVID-19 Crash
        covid_2020 = {
            'stocks': -0.35,
            'indices': -0.30,
            'crypto': -0.50,
            'forex': -0.10,
            'metals': 0.15,
            'energy': -0.60
        }
        
        # Inflation Shock 2022
        inflation_2022 = {
            'stocks': -0.25,
            'indices': -0.20,
            'crypto': -0.65,
            'forex': -0.05,
            'metals': 0.35,
            'energy': 0.20
        }
        
        # Run each scenario
        for scenario_name, impacts in [('2008 Crisis', crisis_2008), 
                                       ('COVID Crash', covid_2020),
                                       ('Inflation Shock', inflation_2022)]:
            total_impact = 0
            for asset_type, allocation in allocations.items():
                impact = impacts.get(asset_type, -0.30)
                total_impact += impact * allocation
            
            stressed_value = portfolio_value * (1 + total_impact)
            loss_percent = abs(total_impact) * 100
            
            scenarios.append({
                'scenario': scenario_name,
                'stressed_value': round(stressed_value, 2),
                'loss_percent': round(loss_percent, 1),
                'drawdown': round(loss_percent, 1),
                'recovery_months': 12 if '2008' in scenario_name else 6
            })
        
        return scenarios

# ==================== TELEGRAM BOT IMPLEMENTATION ====================
# Теперь создаем полноценного Telegram бота с inline keyboard

import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    CallbackQueryHandler,
    ConversationHandler
)

# States for conversation
class BotStates(Enum):
    MAIN_MENU = 1
    SELECT_ASSET = 2
    ENTER_PORTFOLIO = 3
    SET_ALERT = 4
    VIEW_REPORT = 5

# Global instances
risk_engine = OptimizedRiskEngine()

# User sessions storage (in production use Redis/DB)
user_sessions = {}
user_portfolios = {}
user_alerts = {}

class TelegramRiskBot:
    """Main Telegram bot class"""
    
    def __init__(self, token: str):
        self.token = token
        self.application = None
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        
        welcome_text = (
            "🚀 *Добро пожаловать в PRO Risk Calculator MVP!*\n\n"
            "Профессиональный инструмент для расчета рисков с:\n"
            "✅ Value at Risk (VaR)\n"
            "✅ Monte Carlo симуляции\n"
            "✅ Стресс-тестирование\n"
            "✅ Уведомления о рисках\n\n"
            "Выберите действие:"
        )
        
        keyboard = [
            [InlineKeyboardButton("📊 Анализ актива", callback_data="analyze_asset")],
            [InlineKeyboardButton("💰 Управление портфелем", callback_data="manage_portfolio")],
            [InlineKeyboardButton("⚠️ Стресс-тест", callback_data="stress_test")],
            [InlineKeyboardButton("🔔 Уведомления", callback_data="alerts")],
            [InlineKeyboardButton("📈 Быстрый анализ", callback_data="quick_analysis")],
            [InlineKeyboardButton("ℹ️ Помощь", callback_data="help")]
        ]
        
        await update.message.reply_text(
            welcome_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def analyze_asset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start asset analysis flow"""
        query = update.callback_query
        await query.answer()
        
        text = (
            "📊 *Анализ риска актива*\n\n"
            "Введите тикер актива (например: BTCUSDT, AAPL, EURUSD):"
        )
        
        await query.edit_message_text(
            text=text,
            parse_mode='Markdown'
        )
        
        # Set state for next message
        context.user_data['state'] = 'awaiting_asset'
        
        return BotStates.SELECT_ASSET.value
    
    async def handle_asset_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle asset symbol input"""
        if context.user_data.get('state') != 'awaiting_asset':
            return
        
        symbol = update.message.text.strip().upper()
        
        # Validate symbol
        if not re.match(r'^[A-Z0-9]{2,20}$', symbol):
            await update.message.reply_text(
                "❌ Неверный формат тикера. Попробуйте снова (например: BTCUSDT):"
            )
            return BotStates.SELECT_ASSET.value
        
        # Show processing message
        processing_msg = await update.message.reply_text(
            f"🔄 Анализирую {symbol}... Это займет несколько секунд."
        )
        
        try:
            # Get data and calculate risk
            from data_provider import MVPDataProvider  # Импортируем из Phase 1
            
            data_provider = MVPDataProvider()
            
            # Get current price
            current_price = await data_provider.get_price(symbol)
            
            # Get historical returns (for MVP, using simulated data)
            import numpy as np
            np.random.seed(hash(symbol) % 10000)
            returns = np.random.normal(0.001, 0.02, 100).tolist()  # Simulated returns
            
            # Calculate risk metrics
            var_95, var_99, cvar_95 = risk_engine.calculate_var(returns)
            
            # Monte Carlo simulation
            mu = np.mean(returns)
            sigma = np.std(returns)
            final_prices, sample_paths = risk_engine.monte_carlo_numba(
                current_price, mu, sigma, days=30, simulations=1000
            )
            
            # Calculate probabilities
            prob_loss = (final_prices < current_price).mean() * 100
            expected_return = (final_prices.mean() - current_price) / current_price * 100
            
            # Format results
            result_text = (
                f"📈 *Анализ риска: {symbol}*\n\n"
                f"💰 Текущая цена: ${current_price:.2f}\n\n"
                f"⚠️ *Метрики риска:*\n"
                f"• VaR 95% (1 день): {var_95:.2f}%\n"
                f"• VaR 99% (1 день): {var_99:.2f}%\n"
                f"• Conditional VaR 95%: {cvar_95:.2f}%\n\n"
                f"🎲 *Monte Carlo (30 дней):*\n"
                f"• Ожидаемая доходность: {expected_return:.1f}%\n"
                f"• Вероятность убытка: {prob_loss:.1f}%\n"
                f"• Лучший сценарий: ${final_prices.max():.2f}\n"
                f"• Худший сценарий: ${final_prices.min():.2f}\n\n"
                f"📊 *Волатильность:*\n"
                f"• Дневная: {sigma*100:.2f}%\n"
                f"• Годовая: {sigma*np.sqrt(252)*100:.1f}%"
            )
            
            keyboard = [
                [InlineKeyboardButton("📉 График симуляции", callback_data=f"chart_{symbol}")],
                [InlineKeyboardButton("🔄 Новый анализ", callback_data="analyze_asset")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
            
            await processing_msg.delete()
            await update.message.reply_text(
                result_text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing asset {symbol}: {e}")
            await processing_msg.delete()
            await update.message.reply_text(
                f"❌ Ошибка при анализе {symbol}. Попробуйте другой актив."
            )
        
        context.user_data.clear()
        return ConversationHandler.END
    
    async def manage_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Portfolio management interface"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        if user_id not in user_portfolios:
            text = (
                "💰 *Управление портфелем*\n\n"
                "У вас пока нет портфеля. Создать новый?"
            )
            
            keyboard = [
                [InlineKeyboardButton("➕ Создать портфель", callback_data="create_portfolio")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
        else:
            portfolio = user_portfolios[user_id]
            
            # Calculate portfolio value
            total_value = sum(asset['value'] for asset in portfolio.values())
            
            text = (
                f"💰 *Ваш портфель*\n\n"
                f"📊 Общая стоимость: ${total_value:,.2f}\n"
                f"📈 Активов: {len(portfolio)}\n\n"
                f"*Состав портфеля:*\n"
            )
            
            for symbol, data in portfolio.items():
                text += f"• {symbol}: ${data['value']:,.2f} ({data['weight']*100:.1f}%)\n"
            
            keyboard = [
                [InlineKeyboardButton("➕ Добавить актив", callback_data="add_asset")],
                [InlineKeyboardButton("📊 Анализ риска", callback_data="portfolio_risk")],
                [InlineKeyboardButton("🗑 Очистить портфель", callback_data="clear_portfolio")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
        
        await query.edit_message_text(
            text=text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def stress_test(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stress testing interface"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        if user_id not in user_portfolios or not user_portfolios[user_id]:
            text = (
                "⚠️ *Стресс-тестирование*\n\n"
                "Для стресс-теста нужен портфель. Сначала создайте портфель."
            )
            
            keyboard = [
                [InlineKeyboardButton("💰 Создать портфель", callback_data="create_portfolio")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
        else:
            portfolio = user_portfolios[user_id]
            total_value = sum(asset['value'] for asset in portfolio.values())
            
            # Estimate allocations by asset type (simplified)
            allocations = {
                'stocks': 0.4,      # 40% stocks
                'crypto': 0.3,      # 30% crypto
                'forex': 0.2,       # 20% forex
                'metals': 0.1       # 10% metals
            }
            
            # Run stress tests
            scenarios = risk_engine.stress_test_portfolio(total_value, allocations)
            
            text = "⚠️ *Результаты стресс-тестирования*\n\n"
            text += f"💰 Текущая стоимость портфеля: ${total_value:,.2f}\n\n"
            
            for scenario in scenarios:
                text += (
                    f"📉 *{scenario['scenario']}:*\n"
                    f"• Стоимость после шока: ${scenario['stressed_value']:,.2f}\n"
                    f"• Потери: {scenario['loss_percent']}%\n"
                    f"• Восстановление: ~{scenario['recovery_months']} мес.\n\n"
                )
            
            keyboard = [
                [InlineKeyboardButton("🔄 Обновить тесты", callback_data="stress_test")],
                [InlineKeyboardButton("📊 Детальный анализ", callback_data="detailed_stress")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
        
        await query.edit_message_text(
            text=text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def alerts_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Alerts management"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        if user_id not in user_alerts or not user_alerts[user_id]:
            text = (
                "🔔 *Управление уведомлениями*\n\n"
                "У вас нет активных уведомлений.\n\n"
                "Вы можете настроить уведомления о:\n"
                "• Достижении цены актива\n"
                "• Превышении уровня риска\n"
                "• Резких изменениях волатильности"
            )
        else:
            alerts = user_alerts[user_id]
            text = "🔔 *Ваши уведомления:*\n\n"
            
            for i, alert in enumerate(alerts, 1):
                if alert['type'] == 'price':
                    text += (
                        f"{i}. Цена {alert['symbol']} "
                        f"{'выше' if alert['condition'] == 'above' else 'ниже'} "
                        f"${alert['target_price']}\n"
                    )
                elif alert['type'] == 'risk':
                    text += f"{i}. Риск портфеля > {alert['threshold']}%\n"
        
        keyboard = [
            [InlineKeyboardButton("➕ Ценовое уведомление", callback_data="add_price_alert")],
            [InlineKeyboardButton("⚠️ Уведомление о риске", callback_data="add_risk_alert")],
            [InlineKeyboardButton("🗑 Удалить все", callback_data="clear_alerts")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            text=text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def quick_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Quick analysis of popular assets"""
        query = update.callback_query
        await query.answer()
        
        popular_assets = [
            ("BTCUSDT", "Bitcoin"),
            ("ETHUSDT", "Ethereum"),
            ("AAPL", "Apple"),
            ("SPY", "S&P 500 ETF"),
            ("EURUSD", "Euro/USD"),
            ("XAUUSD", "Gold")
        ]
        
        text = "📈 *Быстрый анализ популярных активов*\n\n"
        
        keyboard = []
        for symbol, name in popular_assets:
            text += f"• {name} ({symbol})\n"
            keyboard.append([InlineKeyboardButton(
                f"📊 {symbol}", 
                callback_data=f"quick_{symbol}"
            )])
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
        
        await query.edit_message_text(
            text=text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command"""
        query = update.callback_query
        await query.answer()
        
        help_text = (
            "ℹ️ *Помощь по Risk Calculator*\n\n"
            
            "*Основные команды:*\n"
            "• /start - Главное меню\n"
            "• /analyze [тикер] - Быстрый анализ актива\n"
            "• /portfolio - Управление портфелем\n"
            "• /stress - Стресс-тест портфеля\n"
            "• /alerts - Управление уведомлениями\n\n"
            
            "*Метрики риска:*\n"
            "• *VaR (Value at Risk)* - Максимальные потери с заданной вероятностью\n"
            "• *CVaR (Conditional VaR)* - Средние потери в худших сценариях\n"
            "• *Monte Carlo* - Симуляция будущих ценовых траекторий\n"
            "• *Стресс-тесты* - Моделирование кризисных сценариев\n\n"
            
            "*Примеры тикеров:*\n"
            "• Крипто: BTCUSDT, ETHUSDT\n"
            "• Акции: AAPL, TSLA, GOOGL\n"
            "• Форекс: EURUSD, GBPJPY\n"
            "• Индексы: SPY, QQQ\n\n"
            
            "💡 *Совет:* Начните с анализа отдельных активов, затем создайте портфель."
        )
        
        keyboard = [
            [InlineKeyboardButton("📚 Детальная документация", callback_data="docs")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            text=help_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle all callback queries"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "main_menu":
            await self.show_main_menu(query)
        
        elif data == "analyze_asset":
            await self.analyze_asset(update, context)
        
        elif data == "manage_portfolio":
            await self.manage_portfolio(update, context)
        
        elif data == "stress_test":
            await self.stress_test(update, context)
        
        elif data == "alerts":
            await self.alerts_menu(update, context)
        
        elif data == "quick_analysis":
            await self.quick_analysis(update, context)
        
        elif data == "help":
            await self.help_command(update, context)
        
        elif data.startswith("quick_"):
            symbol = data.replace("quick_", "")
            # Quick analysis for popular asset
            await self.perform_quick_analysis(query, symbol)
        
        elif data == "create_portfolio":
            await self.create_portfolio(update, context)
        
        elif data == "portfolio_risk":
            await self.analyze_portfolio_risk(update, context)
    
    async def show_main_menu(self, query):
        """Show main menu"""
        welcome_text = (
            "🚀 *PRO Risk Calculator MVP*\n\n"
            "Выберите действие:"
        )
        
        keyboard = [
            [InlineKeyboardButton("📊 Анализ актива", callback_data="analyze_asset")],
            [InlineKeyboardButton("💰 Управление портфелем", callback_data="manage_portfolio")],
            [InlineKeyboardButton("⚠️ Стресс-тест", callback_data="stress_test")],
            [InlineKeyboardButton("🔔 Уведомления", callback_data="alerts")],
            [InlineKeyboardButton("📈 Быстрый анализ", callback_data="quick_analysis")],
            [InlineKeyboardButton("ℹ️ Помощь", callback_data="help")]
        ]
        
        await query.edit_message_text(
            text=welcome_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def perform_quick_analysis(self, query, symbol: str):
        """Perform quick analysis for a symbol"""
        await query.edit_message_text(
            text=f"🔄 Анализирую {symbol}...",
            parse_mode='Markdown'
        )
        
        try:
            # Simulate analysis (in real app, this would call data provider)
            import numpy as np
            import random
            
            # Simulated data
            current_price = random.uniform(100, 10000)
            var_95 = random.uniform(1, 5)
            var_99 = random.uniform(3, 8)
            prob_loss = random.uniform(30, 60)
            volatility = random.uniform(10, 50)
            
            result_text = (
                f"📊 *Быстрый анализ: {symbol}*\n\n"
                f"💰 Примерная цена: ${current_price:,.2f}\n\n"
                f"⚠️ *Оценка риска:*\n"
                f"• VaR 95%: ~{var_95:.1f}%\n"
                f"• VaR 99%: ~{var_99:.1f}%\n"
                f"• Вероятность убытка: ~{prob_loss:.1f}%\n"
                f"• Годовая волатильность: ~{volatility:.1f}%\n\n"
                f"💡 *Рекомендация:*\n"
                f"{'Высокий риск' if prob_loss > 50 else 'Умеренный риск' if prob_loss > 30 else 'Низкий риск'}"
            )
            
            keyboard = [
                [InlineKeyboardButton("📊 Полный анализ", callback_data=f"analyze_{symbol}")],
                [InlineKeyboardButton("📈 Быстрый анализ", callback_data="quick_analysis")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
            
            await query.edit_message_text(
                text=result_text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error in quick analysis: {e}")
            await query.edit_message_text(
                text="❌ Ошибка при анализе. Попробуйте снова.",
                parse_mode='Markdown'
            )
    
    async def create_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Create portfolio interface"""
        query = update.callback_query
        await query.answer()
        
        text = (
            "💰 *Создание портфеля*\n\n"
            "Введите активы в формате:\n"
            "ТИКЕР1 СУММА1, ТИКЕР2 СУММА2, ...\n\n"
            "*Пример:*\n"
            "BTCUSDT 5000, AAPL 3000, EURUSD 2000"
        )
        
        await query.edit_message_text(
            text=text,
            parse_mode='Markdown'
        )
        
        context.user_data['state'] = 'awaiting_portfolio'
        
        return BotStates.ENTER_PORTFOLIO.value
    
    async def analyze_portfolio_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Analyze portfolio risk"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        if user_id not in user_portfolios:
            await query.edit_message_text(
                text="❌ У вас нет портфеля для анализа.",
                parse_mode='Markdown'
            )
            return
        
        portfolio = user_portfolios[user_id]
        
        await query.edit_message_text(
            text="🔄 Анализирую риски портфеля...",
            parse_mode='Markdown'
        )
        
        try:
            # Simulate portfolio analysis
            total_value = sum(asset['value'] for asset in portfolio.values())
            
            # Generate simulated returns for portfolio
            import numpy as np
            
            # For MVP, use simplified correlation matrix
            num_assets = len(portfolio)
            returns_matrix = np.random.multivariate_normal(
                mean=[0.001] * num_assets,
                cov=np.eye(num_assets) * 0.02 + 0.01,  # Some correlation
                size=100
            )
            
            # Calculate portfolio risk
            weights = [asset['weight'] for asset in portfolio.values()]
            risk_metrics = risk_engine.calculate_portfolio_risk(weights, returns_matrix)
            
            # Run stress tests
            allocations = {'stocks': 0.5, 'crypto': 0.3, 'forex': 0.2}
            stress_scenarios = risk_engine.stress_test_portfolio(total_value, allocations)
            
            # Format results
            result_text = (
                f"📊 *Анализ риска портфеля*\n\n"
                f"💰 Общая стоимость: ${total_value:,.2f}\n"
                f"📈 Активов: {num_assets}\n\n"
                f"⚠️ *Метрики риска:*\n"
                f"• VaR 95% портфеля: {risk_metrics['portfolio_var_95']:.1f}%\n"
                f"• CVaR 95% портфеля: {risk_metrics['portfolio_cvar_95']:.1f}%\n"
                f"• Годовая волатильность: {risk_metrics['portfolio_annual_volatility']:.1f}%\n"
                f"• Sharpe Ratio: {risk_metrics['portfolio_sharpe']:.2f}\n"
                f"• Sortino Ratio: {risk_metrics['portfolio_sortino']:.2f}\n\n"
                f"📉 *Стресс-тесты:*\n"
            )
            
            for scenario in stress_scenarios[:2]:  # Show first 2 scenarios
                result_text += (
                    f"• {scenario['scenario']}: -{scenario['loss_percent']}%\n"
                )
            
            result_text += "\n💡 *Рекомендации:*\n"
            if risk_metrics['portfolio_var_95'] > 5:
                result_text += "• Высокий риск. Рассмотрите диверсификацию.\n"
            elif risk_metrics['portfolio_sharpe'] < 1:
                result_text += "• Низкая доходность на единицу риска.\n"
            else:
                result_text += "• Портфель сбалансирован.\n"
            
            keyboard = [
                [InlineKeyboardButton("🔄 Обновить анализ", callback_data="portfolio_risk")],
                [InlineKeyboardButton("⚠️ Детальные стресс-тесты", callback_data="stress_test")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
            
            await query.edit_message_text(
                text=result_text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            await query.edit_message_text(
                text="❌ Ошибка при анализе портфеля.",
                parse_mode='Markdown'
            )
    
    async def handle_portfolio_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle portfolio input"""
        if context.user_data.get('state') != 'awaiting_portfolio':
            return
        
        user_id = update.message.from_user.id
        input_text = update.message.text.strip()
        
        try:
            # Parse portfolio input
            assets = []
            total_value = 0
            
            for item in input_text.split(','):
                item = item.strip()
                if not item:
                    continue
                
                parts = item.split()
                if len(parts) != 2:
                    raise ValueError("Неверный формат")
                
                symbol = parts[0].upper()
                try:
                    value = float(parts[1])
                except ValueError:
                    raise ValueError("Сумма должна быть числом")
                
                assets.append({'symbol': symbol, 'value': value})
                total_value += value
            
            if total_value == 0:
                raise ValueError("Общая стоимость не может быть нулевой")
            
            # Calculate weights and store portfolio
            portfolio = {}
            for asset in assets:
                portfolio[asset['symbol']] = {
                    'value': asset['value'],
                    'weight': asset['value'] / total_value
                }
            
            user_portfolios[user_id] = portfolio
            
            # Confirmation message
            text = f"✅ Портфель создан!\n\n💰 Общая стоимость: ${total_value:,.2f}\n📈 Активов: {len(assets)}\n\n"
            
            for asset in assets:
                text += f"• {asset['symbol']}: ${asset['value']:,.2f}\n"
            
            text += "\nТеперь вы можете проанализировать риски портфеля."
            
            keyboard = [
                [InlineKeyboardButton("📊 Анализ риска", callback_data="portfolio_risk")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
            
            await update.message.reply_text(
                text=text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
            context.user_data.clear()
            
        except ValueError as e:
            await update.message.reply_text(
                f"❌ Ошибка: {str(e)}\n\n"
                "Пожалуйста, введите активы в правильном формате:\n"
                "ТИКЕР1 СУММА1, ТИКЕР2 СУММА2\n\n"
                "Пример: BTCUSDT 5000, AAPL 3000"
            )
            return BotStates.ENTER_PORTFOLIO.value
        
        except Exception as e:
            logger.error(f"Error creating portfolio: {e}")
            await update.message.reply_text(
                "❌ Произошла ошибка при создании портфеля. Попробуйте снова."
            )
            return BotStates.ENTER_PORTFOLIO.value
        
        return ConversationHandler.END

# ==================== WEB SERVER FOR WEBHOOK MODE ====================
from aiohttp import web

class WebhookServer:
    """HTTP server for webhook mode"""
    
    def __init__(self, application: Application, port: int = 10000):
        self.application = application
        self.port = port
        self.runner = None
        self.site = None
    
    async def handle_webhook(self, request):
        """Handle incoming webhook requests"""
        try:
            data = await request.text()
            update = Update.de_json(json.loads(data), self.application.bot)
            await self.application.process_update(update)
            return web.Response(status=200)
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return web.Response(status=400)
    
    async def health_check(self, request):
        """Health check endpoint"""
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "risk-calculator-bot",
            "version": "MVP 2.0"
        }
        return web.json_response(health_data)
    
    async def start(self):
        """Start the web server"""
        app = web.Application()
        
        # Add routes
        app.router.add_post(WEBHOOK_PATH, self.handle_webhook)
        app.router.add_get('/health', self.health_check)
        app.router.add_get('/', self.health_check)  # Root also returns health
        
        # Start server
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await self.site.start()
        
        logger.info(f"Webhook server started on port {self.port}")
        logger.info(f"Webhook URL: {WEBHOOK_URL}{WEBHOOK_PATH}")
    
    async def stop(self):
        """Stop the web server"""
        if self.runner:
            await self.runner.cleanup()
            logger.info("Webhook server stopped")

# ==================== MAIN APPLICATION ====================
async def main():
    """Main application entry point"""
    logger.info("🚀 Запуск PRO Risk Calculator MVP Phase 2")
    
    # Initialize bot
    bot = TelegramRiskBot(TOKEN)
    
    # Create application
    application = Application.builder().token(TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("portfolio", bot.manage_portfolio))
    application.add_handler(CommandHandler("stress", bot.stress_test))
    application.add_handler(CommandHandler("alerts", bot.alerts_menu))
    
    # Add conversation handler for asset analysis
    asset_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(bot.analyze_asset, pattern="^analyze_asset$")],
        states={
            BotStates.SELECT_ASSET.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_asset_input)
            ]
        },
        fallbacks=[CommandHandler("cancel", lambda u, c: ConversationHandler.END)],
        name="asset_analysis"
    )
    application.add_handler(asset_conv)
    
    # Add conversation handler for portfolio creation
    portfolio_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(bot.create_portfolio, pattern="^create_portfolio$")],
        states={
            BotStates.ENTER_PORTFOLIO.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_portfolio_input)
            ]
        },
        fallbacks=[CommandHandler("cancel", lambda u, c: ConversationHandler.END)],
        name="portfolio_creation"
    )
    application.add_handler(portfolio_conv)
    
    # Add callback query handler
    application.add_handler(CallbackQueryHandler(bot.callback_handler))
    
    # Add fallback message handler
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        lambda u, c: u.message.reply_text(
            "Используйте /start для начала работы или выберите команду из меню."
        )
    ))
    
    # Start bot based on mode
    if WEBHOOK_URL and WEBHOOK_URL.strip():
        logger.info("🌐 Запуск в режиме Webhook")
        
        # Set webhook
        webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
        await application.bot.set_webhook(
            url=webhook_url,
            allowed_updates=Update.ALL_TYPES
        )
        logger.info(f"Webhook установлен: {webhook_url}")
        
        # Start web server
        server = WebhookServer(application, PORT)
        await server.start()
        
        # Keep running
        try:
            while True:
                # Periodically check memory
                MemoryGuardian.check_and_clear()
                await asyncio.sleep(300)  # Check every 5 minutes
                
        except KeyboardInterrupt:
            logger.info("⏹ Остановка по команде пользователя")
            await server.stop()
            
    else:
        logger.info("🔄 Запуск в режиме Polling")
        
        # Start polling
        await application.initialize()
        await application.start()
        await application.updater.start_polling(
            poll_interval=1.0,
            timeout=30,
            drop_pending_updates=True
        )
        
        logger.info("✅ Бот запущен в режиме polling")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("⏹ Остановка по команде пользователя")
            await application.stop()

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        raise

# bot_mvp_phase3.py — PRO Risk Calculator MVP Phase 3 (Complete)
import os
import sys
import logging
import asyncio
import time
import functools
import json
import re
import html
import gc
import io
import csv
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
from dataclasses import dataclass, asdict
import base64

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

# API Keys (используем из .env или дефолты)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

# --- Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("risk_calculator_pro")

# ==================== ADVANCED DATA STRUCTURES ====================
@dataclass
class AssetMetrics:
    """Расширенные метрики актива"""
    symbol: str
    current_price: float
    mean_return: float
    volatility: float
    annual_volatility: float
    historical_var_95: float
    historical_var_99: float
    parametric_var_95: float
    conditional_var_95: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    skewness: float
    kurtosis: float
    monte_carlo_var_95: float
    monte_carlo_prob_loss: float
    atr_14: float  # Average True Range
    rsi_14: float  # Relative Strength Index
    last_updated: str

@dataclass
class PortfolioMetrics:
    """Метрики портфеля"""
    total_value: float
    num_assets: int
    portfolio_var_95: float
    portfolio_cvar_95: float
    portfolio_volatility: float
    portfolio_sharpe: float
    portfolio_sortino: float
    portfolio_beta: float
    diversification_score: float  # 0-100
    concentration_risk: float  # Herfindahl-Hirschman Index
    correlation_risk: float
    worst_case_loss: float
    expected_shortfall: float
    stress_test_results: List[Dict]

# ==================== ADVANCED GRAPH GENERATOR ====================
class AdvancedGraphGenerator:
    """Генератор профессиональных графиков для анализа рисков"""
    
    def __init__(self):
        self.figures_created = 0
        self.memory_limit = 50  # Максимум 50 графиков в памяти
        
    def cleanup_old_figures(self):
        """Очистка старых графиков для экономии памяти"""
        if self.figures_created > self.memory_limit:
            gc.collect()
            self.figures_created = 0
            logger.info("Очистка памяти графиков")
    
    @monitor_performance
    def generate_monte_carlo_chart(self, initial_price: float, sample_paths: List[List[float]], 
                                 final_prices: List[float], symbol: str) -> io.BytesIO:
        """Генерация графика Monte Carlo симуляции"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import numpy as np
            
            self.cleanup_old_figures()
            
            plt.figure(figsize=(12, 8), dpi=100)
            
            # Plot sample paths
            days = len(sample_paths[0]) if sample_paths else 30
            x = np.arange(days)
            
            for i, path in enumerate(sample_paths[:50]):  # Limit to 50 paths for clarity
                alpha = 0.1 if i > 10 else 0.3  # Highlight first 10 paths
                plt.plot(x, path, 'b-', alpha=alpha, linewidth=0.5)
            
            # Calculate and plot confidence intervals
            if sample_paths:
                paths_array = np.array(sample_paths)
                mean_path = np.mean(paths_array, axis=0)
                std_path = np.std(paths_array, axis=0)
                
                plt.plot(x, mean_path, 'r-', linewidth=3, label='Средняя траектория')
                plt.fill_between(x, mean_path - std_path, mean_path + std_path, 
                               alpha=0.2, color='red', label='±1σ')
                plt.fill_between(x, mean_path - 2*std_path, mean_path + 2*std_path, 
                               alpha=0.1, color='red', label='±2σ')
            
            # Plot initial price line
            plt.axhline(y=initial_price, color='g', linestyle='--', linewidth=2, 
                       label=f'Начальная цена: ${initial_price:.2f}')
            
            # Formatting
            plt.title(f'Monte Carlo Симуляция: {symbol}\n30-дневный прогноз', fontsize=16, fontweight='bold')
            plt.xlabel('Дни', fontsize=12)
            plt.ylabel('Цена ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            
            # Add statistics box
            if final_prices:
                stats_text = (
                    f'Статистика финальных цен:\n'
                    f'Средняя: ${np.mean(final_prices):.2f}\n'
                    f'Медиана: ${np.median(final_prices):.2f}\n'
                    f'95% VaR: ${initial_price * (1 - np.percentile(final_prices, 5)/100):.2f}\n'
                    f'Вероятность убытка: {100 * sum(1 for p in final_prices if p < initial_price)/len(final_prices):.1f}%'
                )
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            plt.close()
            self.figures_created += 1
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating Monte Carlo chart: {e}")
            raise
    
    @monitor_performance
    def generate_distribution_chart(self, returns: List[float], symbol: str) -> io.BytesIO:
        """Генерация графика распределения доходностей"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
            from scipy.stats import norm
            
            self.cleanup_old_figures()
            
            plt.figure(figsize=(12, 8), dpi=100)
            
            returns_array = np.array(returns)
            mean, std = np.mean(returns_array), np.std(returns_array)
            
            # Histogram of returns
            plt.hist(returns_array, bins=50, density=True, alpha=0.6, color='blue', 
                   label='Исторические доходности')
            
            # Normal distribution fit
            x = np.linspace(min(returns_array), max(returns_array), 100)
            p = norm.pdf(x, mean, std)
            plt.plot(x, p, 'r-', linewidth=2, label=f'Нормальное распределение (μ={mean:.4f}, σ={std:.4f})')
            
            # VaR lines
            var_95 = np.percentile(returns_array, 5)
            var_99 = np.percentile(returns_array, 1)
            
            plt.axvline(x=var_95, color='orange', linestyle='--', linewidth=2, label=f'95% VaR = {var_95:.2%}')
            plt.axvline(x=var_99, color='red', linestyle='--', linewidth=2, label=f'99% VaR = {var_99:.2%}')
            
            # CVaR area
            cvar_returns = returns_array[returns_array <= var_95]
            if len(cvar_returns) > 0:
                plt.axvspan(min(cvar_returns), var_95, alpha=0.2, color='red', label='Conditional VaR область')
            
            # Formatting
            plt.title(f'Распределение доходностей: {symbol}\nДневные доходности', fontsize=16, fontweight='bold')
            plt.xlabel('Доходность', fontsize=12)
            plt.ylabel('Плотность вероятности', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
            # Add statistics
            skew = float(np.cov(returns_array)[0, 0]) if len(returns_array) > 1 else 0
            kurt = float(np.cov(returns_array, rowvar=False)[0, 0]) if len(returns_array) > 1 else 0
            
            stats_text = (
                f'Статистика распределения:\n'
                f'Среднее: {mean:.4f}\n'
                f'Стандартное отклонение: {std:.4f}\n'
                f'Асимметрия: {skew:.2f}\n'
                f'Эксцесс: {kurt:.2f}\n'
                f'Тест на нормальность: {"Отклонено" if abs(skew) > 0.5 or abs(kurt-3) > 1 else "Не отклонено"}'
            )
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            plt.close()
            self.figures_created += 1
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating distribution chart: {e}")
            raise
    
    @monitor_performance
    def generate_correlation_matrix(self, assets: List[str], returns_matrix: np.ndarray) -> io.BytesIO:
        """Генерация матрицы корреляций активов"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
            import seaborn as sns
            
            self.cleanup_old_figures()
            
            plt.figure(figsize=(12, 10), dpi=100)
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(returns_matrix.T)
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn_r',
                       center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
            
            plt.title('Матрица корреляций активов портфеля', fontsize=16, fontweight='bold')
            plt.xticks(ticks=np.arange(len(assets)) + 0.5, labels=assets, rotation=45, ha='right')
            plt.yticks(ticks=np.arange(len(assets)) + 0.5, labels=assets, rotation=0)
            
            # Add portfolio diversification score
            if len(assets) > 1:
                avg_correlation = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
                diversification = 1 - avg_correlation
                
                plt.text(0.02, -0.1, 
                        f'Средняя корреляция: {avg_correlation:.2f} | Оценка диверсификации: {diversification:.2f}/1.0',
                        transform=plt.gca().transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            plt.close()
            self.figures_created += 1
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating correlation matrix: {e}")
            raise
    
    @monitor_performance
    def generate_stress_test_chart(self, stress_results: List[Dict]) -> io.BytesIO:
        """Генерация графика стресс-тестирования"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
            
            self.cleanup_old_figures()
            
            plt.figure(figsize=(14, 8), dpi=100)
            
            scenarios = [r['scenario'] for r in stress_results]
            losses = [r['loss_percent'] for r in stress_results]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            bars = plt.barh(scenarios, losses, color=colors[:len(scenarios)])
            plt.xlabel('Потери (%)', fontsize=12)
            plt.title('Результаты стресс-тестирования портфеля', fontsize=16, fontweight='bold')
            
            # Add value labels on bars
            for bar, loss in zip(bars, losses):
                plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{loss:.1f}%', va='center', fontweight='bold')
            
            # Add recovery time annotations
            for i, result in enumerate(stress_results):
                plt.text(-2, i, f"Восстановление: ~{result['recovery_months']} мес.",
                        va='center', fontsize=9, color='gray')
            
            plt.grid(True, alpha=0.3, axis='x')
            plt.xlim(-5, max(losses) * 1.2)
            
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            plt.close()
            self.figures_created += 1
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating stress test chart: {e}")
            raise
    
    @monitor_performance
    def generate_risk_radar_chart(self, metrics: Dict[str, float]) -> io.BytesIO:
        """Генерация радар-диаграммы рисков"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
            
            self.cleanup_old_figures()
            
            categories = ['Волатильность', 'Риск снижения', 'Концентрация', 
                         'Корреляция', 'Ликвидность', 'Рыночный риск']
            
            values = [
                metrics.get('volatility_score', 50),
                metrics.get('drawdown_risk', 50),
                metrics.get('concentration_risk', 50),
                metrics.get('correlation_risk', 50),
                metrics.get('liquidity_risk', 50),
                metrics.get('market_risk', 50)
            ]
            
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            values += values[:1]
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'), dpi=100)
            
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12)
            ax.set_ylim(0, 100)
            
            # Add risk level annotations
            risk_level = np.mean(values)
            if risk_level < 30:
                risk_text = "НИЗКИЙ РИСК"
                color = 'green'
            elif risk_level < 60:
                risk_text = "УМЕРЕННЫЙ РИСК"
                color = 'orange'
            else:
                risk_text = "ВЫСОКИЙ РИСК"
                color = 'red'
            
            plt.title(f'Профиль риска портфеля\n{risk_text}', fontsize=16, fontweight='bold', color=color)
            
            # Add value labels
            for angle, value, category in zip(angles[:-1], values[:-1], categories):
                ax.text(angle, value + 5, f'{value:.0f}', ha='center', va='center', fontsize=10)
            
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            plt.close()
            self.figures_created += 1
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating radar chart: {e}")
            raise

# ==================== ADVANCED REPORT GENERATOR ====================
class ReportGenerator:
    """Генератор профессиональных отчетов в различных форматах"""
    
    @staticmethod
    def generate_text_report(metrics: AssetMetrics) -> str:
        """Генерация текстового отчета"""
        report = [
            "=" * 60,
            f"АНАЛИТИЧЕСКИЙ ОТЧЕТ: {metrics.symbol}",
            f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            "📊 ОСНОВНЫЕ МЕТРИКИ:",
            f"  • Текущая цена: ${metrics.current_price:.2f}",
            f"  • Дневная волатильность: {metrics.volatility:.2f}%",
            f"  • Годовая волатильность: {metrics.annual_volatility:.2f}%",
            f"  • Средняя доходность: {metrics.mean_return:.4f}",
            "",
            "⚠️ МЕТРИКИ РИСКА:",
            f"  • VaR 95% (1 день): {metrics.historical_var_95:.2f}%",
            f"  • VaR 99% (1 день): {metrics.historical_var_99:.2f}%",
            f"  • Conditional VaR 95%: {metrics.conditional_var_95:.2f}%",
            f"  • Максимальное снижение: {metrics.max_drawdown:.2f}%",
            "",
            "📈 ПОКАЗАТЕЛИ ЭФФЕКТИВНОСТИ:",
            f"  • Коэффициент Шарпа: {metrics.sharpe_ratio:.2f}",
            f"  • Коэффициент Сортино: {metrics.sortino_ratio:.2f}",
            f"  • Асимметрия: {metrics.skewness:.2f}",
            f"  • Эксцесс: {metrics.kurtosis:.2f}",
            "",
            "🎲 MONTE CARLO АНАЛИЗ:",
            f"  • VaR 95% (30 дней): {metrics.monte_carlo_var_95:.2f}%",
            f"  • Вероятность убытка: {metrics.monte_carlo_prob_loss:.1f}%",
            "",
            "📊 ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:",
            f"  • ATR (14): {metrics.atr_14:.2f}",
            f"  • RSI (14): {metrics.rsi_14:.1f}",
            f"    { '⟳ Перепроданность' if metrics.rsi_14 < 30 else '⟳ Перекупленность' if metrics.rsi_14 > 70 else '⟳ Нейтрально'}",
            "",
            "💡 РЕКОМЕНДАЦИИ:",
        ]
        
        # Добавляем рекомендации на основе метрик
        recommendations = []
        
        if metrics.historical_var_95 > 5:
            recommendations.append("  • Высокий риск - рассмотрите уменьшение позиции")
        if metrics.sharpe_ratio < 1:
            recommendations.append("  • Низкая доходность на единицу риска")
        if metrics.rsi_14 > 70:
            recommendations.append("  • Возможная перекупленность")
        elif metrics.rsi_14 < 30:
            recommendations.append("  • Возможная перепроданность")
        if not recommendations:
            recommendations.append("  • Актив выглядит сбалансированно")
        
        report.extend(recommendations)
        report.extend([
            "",
            "=" * 60,
            "Сгенерировано: PRO Risk Calculator MVP v3.0",
            "=" * 60
        ])
        
        return "\n".join(report)
    
    @staticmethod
    def generate_portfolio_report(portfolio_metrics: PortfolioMetrics, 
                                assets: List[AssetMetrics]) -> str:
        """Генерация отчета по портфелю"""
        report = [
            "=" * 60,
            "ОТЧЕТ ПО УПРАВЛЕНИЮ РИСКАМИ ПОРТФЕЛЯ",
            f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            f"💰 ОБЩАЯ СТОИМОСТЬ: ${portfolio_metrics.total_value:,.2f}",
            f"📈 АКТИВОВ: {portfolio_metrics.num_assets}",
            "",
            "📊 СВОДНЫЕ МЕТРИКИ РИСКА:",
            f"  • VaR 95% портфеля: {portfolio_metrics.portfolio_var_95:.1f}%",
            f"  • CVaR 95% портфеля: {portfolio_metrics.portfolio_cvar_95:.1f}%",
            f"  • Волатильность портфеля: {portfolio_metrics.portfolio_volatility:.1f}%",
            f"  • Коэффициент Шарпа: {portfolio_metrics.portfolio_sharpe:.2f}",
            f"  • Коэффициент Сортино: {portfolio_metrics.portfolio_sortino:.2f}",
            f"  • Бета портфеля: {portfolio_metrics.portfolio_beta:.2f}",
            "",
            "🎯 ОЦЕНКА ДИВЕРСИФИКАЦИИ:",
            f"  • Оценка диверсификации: {portfolio_metrics.diversification_score:.0f}/100",
            f"  • Риск концентрации: {portfolio_metrics.concentration_risk:.2f}",
            f"  • Риск корреляции: {portfolio_metrics.correlation_risk:.2f}",
            "",
            "⚠️ ЭКСТРЕМАЛЬНЫЕ СЦЕНАРИИ:",
            f"  • Худшие потери (99%): {portfolio_metrics.worst_case_loss:.1f}%",
            f"  • Ожидаемые потери в кризис: {portfolio_metrics.expected_shortfall:.1f}%",
            "",
            "📉 РЕЗУЛЬТАТЫ СТРЕСС-ТЕСТИРОВАНИЯ:",
        ]
        
        for stress in portfolio_metrics.stress_test_results[:3]:  # Показываем 3 сценария
            report.append(f"  • {stress['scenario']}: -{stress['loss_percent']:.1f}% (восстановление: {stress['recovery_months']} мес.)")
        
        report.extend([
            "",
            "📊 ДЕТАЛИЗАЦИЯ ПО АКТИВАМ:",
        ])
        
        for asset in assets:
            report.extend([
                f"  ─ {asset.symbol}:",
                f"    • Цена: ${asset.current_price:.2f}",
                f"    • VaR 95%: {asset.historical_var_95:.1f}%",
                f"    • Вклад в риск: {asset.volatility/portfolio_metrics.portfolio_volatility*100:.1f}%",
            ])
        
        report.extend([
            "",
            "💡 РЕКОМЕНДАЦИИ ПО УПРАВЛЕНИЮ РИСКАМИ:",
        ])
        
        recommendations = []
        if portfolio_metrics.diversification_score < 60:
            recommendations.append("  • Низкая диверсификация - добавьте активы из разных классов")
        if portfolio_metrics.portfolio_var_95 > 8:
            recommendations.append("  • Высокий совокупный риск - уменьшите позиции в волатильных активах")
        if portfolio_metrics.concentration_risk > 0.3:
            recommendations.append("  • Высокий риск концентрации - распределите капитал более равномерно")
        if not recommendations:
            recommendations.append("  • Портфель хорошо сбалансирован по рискам")
        
        report.extend(recommendations)
        report.extend([
            "",
            "=" * 60,
            "Сгенерировано: PRO Risk Calculator MVP v3.0",
            "=" * 60
        ])
        
        return "\n".join(report)
    
    @staticmethod
    def generate_csv_report(metrics: AssetMetrics) -> str:
        """Генерация CSV отчета"""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Заголовок
        writer.writerow(["PRO Risk Calculator - Аналитический отчет"])
        writer.writerow([f"Актив: {metrics.symbol}"])
        writer.writerow([f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M')}"])
        writer.writerow([])
        
        # Основные метрики
        writer.writerow(["ОСНОВНЫЕ МЕТРИКИ"])
        writer.writerow(["Показатель", "Значение", "Единица измерения"])
        writer.writerow(["Текущая цена", f"{metrics.current_price:.2f}", "USD"])
        writer.writerow(["Средняя доходность", f"{metrics.mean_return:.6f}", "доли"])
        writer.writerow(["Дневная волатильность", f"{metrics.volatility:.4f}", "доли"])
        writer.writerow(["Годовая волатильность", f"{metrics.annual_volatility:.4f}", "доли"])
        writer.writerow([])
        
        # Метрики риска
        writer.writerow(["МЕТРИКИ РИСКА"])
        writer.writerow(["Показатель", "Значение", "Единица измерения"])
        writer.writerow(["VaR 95% (1 день)", f"{metrics.historical_var_95:.4f}", "%"])
        writer.writerow(["VaR 99% (1 день)", f"{metrics.historical_var_99:.4f}", "%"])
        writer.writerow(["Conditional VaR 95%", f"{metrics.conditional_var_95:.4f}", "%"])
        writer.writerow(["Максимальное снижение", f"{metrics.max_drawdown:.4f}", "%"])
        writer.writerow([])
        
        # Показатели эффективности
        writer.writerow(["ПОКАЗАТЕЛИ ЭФФЕКТИВНОСТИ"])
        writer.writerow(["Показатель", "Значение"])
        writer.writerow(["Коэффициент Шарпа", f"{metrics.sharpe_ratio:.4f}"])
        writer.writerow(["Коэффициент Сортино", f"{metrics.sortino_ratio:.4f}"])
        writer.writerow(["Асимметрия", f"{metrics.skewness:.4f}"])
        writer.writerow(["Эксцесс", f"{metrics.kurtosis:.4f}"])
        writer.writerow([])
        
        return output.getvalue()
    
    @staticmethod
    def generate_json_report(metrics: AssetMetrics) -> str:
        """Генерация JSON отчета"""
        report = {
            "metadata": {
                "report_type": "asset_risk_analysis",
                "generated_at": datetime.now().isoformat(),
                "version": "3.0",
                "asset": metrics.symbol
            },
            "price_data": {
                "current_price": metrics.current_price,
                "currency": "USD"
            },
            "risk_metrics": {
                "daily_volatility": metrics.volatility,
                "annual_volatility": metrics.annual_volatility,
                "var_95": metrics.historical_var_95,
                "var_99": metrics.historical_var_99,
                "cvar_95": metrics.conditional_var_95,
                "max_drawdown": metrics.max_drawdown,
                "monte_carlo_var_95": metrics.monte_carlo_var_95
            },
            "performance_metrics": {
                "mean_return": metrics.mean_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "skewness": metrics.skewness,
                "kurtosis": metrics.kurtosis
            },
            "technical_indicators": {
                "atr_14": metrics.atr_14,
                "rsi_14": metrics.rsi_14,
                "rsi_interpretation": "oversold" if metrics.rsi_14 < 30 else "overbought" if metrics.rsi_14 > 70 else "neutral"
            },
            "recommendations": {
                "risk_level": "high" if metrics.historical_var_95 > 5 else "medium" if metrics.historical_var_95 > 2 else "low",
                "action": "reduce" if metrics.historical_var_95 > 5 else "hold" if metrics.sharpe_ratio > 1 else "review"
            }
        }
        
        return json.dumps(report, indent=2, ensure_ascii=False)

# ==================== ENHANCED ANALYTICS ENGINE ====================
class EnhancedAnalyticsEngine:
    """Расширенный аналитический движок с техническими индикаторами"""
    
    @staticmethod
    @monitor_performance
    def calculate_technical_indicators(prices: List[float]) -> Dict[str, float]:
        """Расчет технических индикаторов"""
        import numpy as np
        
        if len(prices) < 15:
            return {"atr_14": 0.0, "rsi_14": 50.0}
        
        prices_array = np.array(prices)
        
        # Calculate RSI (Relative Strength Index)
        def calculate_rsi(prices, period=14):
            deltas = np.diff(prices)
            seed = deltas[:period]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            rs = up / down if down != 0 else 0
            rsi = 100 - 100 / (1 + rs)
            
            for i in range(period, len(deltas)):
                delta = deltas[i]
                if delta > 0:
                    up_val = delta
                    down_val = 0
                else:
                    up_val = 0
                    down_val = -delta
                
                up = (up * (period - 1) + up_val) / period
                down = (down * (period - 1) + down_val) / period
                rs = up / down if down != 0 else 0
                rsi = np.append(rsi, 100 - 100 / (1 + rs))
            
            return rsi[-1] if len(rsi) > 0 else 50
        
        # Calculate ATR (Average True Range)
        def calculate_atr(prices, period=14):
            if len(prices) < period + 1:
                return 0.0
            
            high = prices  # Simplified - using same prices for high/low
            low = prices
            close = prices
            
            tr = np.maximum(
                high[1:] - low[1:],
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
            
            atr = np.zeros_like(prices)
            atr[period] = np.mean(tr[:period])
            
            for i in range(period + 1, len(prices)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
            
            return atr[-1] if len(atr) > period else 0.0
        
        rsi_14 = calculate_rsi(prices_array)
        atr_14 = calculate_atr(prices_array)
        
        return {
            "atr_14": float(atr_14),
            "rsi_14": float(rsi_14),
            "price_trend": "up" if prices_array[-1] > prices_array[0] else "down"
        }
    
    @staticmethod
    @monitor_performance
    def calculate_portfolio_diversification(weights: List[float], 
                                          correlation_matrix: np.ndarray) -> Dict[str, float]:
        """Расчет показателей диверсификации портфеля"""
        import numpy as np
        
        weights_array = np.array(weights)
        
        # Herfindahl-Hirschman Index (HHI) for concentration
        hhi = np.sum(weights_array ** 2)
        
        # Effective number of assets (diversification metric)
        effective_n = 1 / hhi if hhi > 0 else len(weights)
        
        # Average pairwise correlation
        if len(weights) > 1:
            # Get upper triangle of correlation matrix (excluding diagonal)
            upper_tri = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            avg_correlation = np.mean(np.abs(upper_tri)) if len(upper_tri) > 0 else 0
        else:
            avg_correlation = 0
        
        # Diversification score (0-100)
        diversification_score = min(100, effective_n / len(weights) * 100) if len(weights) > 0 else 0
        diversification_score *= (1 - avg_correlation)  # Penalize for high correlations
        
        return {
            "concentration_risk": float(hhi),
            "effective_assets": float(effective_n),
            "avg_correlation": float(avg_correlation),
            "diversification_score": float(diversification_score)
        }
    
    @staticmethod
    @monitor_performance
    def calculate_scenario_analysis(portfolio_value: float, 
                                  asset_allocations: Dict[str, float],
                                  scenarios: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Расширенный анализ сценариев"""
        results = []
        
        for scenario_name, impacts in scenarios.items():
            total_impact = 0
            for asset_class, allocation in asset_allocations.items():
                impact = impacts.get(asset_class, 0)
                total_impact += impact * allocation
            
            stressed_value = portfolio_value * (1 + total_impact)
            
            # Calculate additional metrics
            recovery_time = EnhancedAnalyticsEngine.estimate_recovery_time(total_impact)
            margin_call_risk = EnhancedAnalyticsEngine.calculate_margin_call_risk(total_impact)
            
            results.append({
                "scenario": scenario_name,
                "stressed_value": round(stressed_value, 2),
                "loss_percent": round(abs(total_impact) * 100, 1),
                "drawdown": round(abs(total_impact) * 100, 1),
                "recovery_months": recovery_time,
                "margin_call_risk": margin_call_risk,
                "severity": "high" if abs(total_impact) > 0.4 else "medium" if abs(total_impact) > 0.2 else "low"
            })
        
        return results
    
    @staticmethod
    def estimate_recovery_time(loss_percent: float) -> int:
        """Оценка времени восстановления после потерь"""
        loss_abs = abs(loss_percent)
        
        if loss_abs <= 0.1:  # 10%
            return 3
        elif loss_abs <= 0.2:  # 20%
            return 6
        elif loss_abs <= 0.3:  # 30%
            return 12
        elif loss_abs <= 0.4:  # 40%
            return 18
        else:  # > 40%
            return 24
    
    @staticmethod
    def calculate_margin_call_risk(loss_percent: float) -> str:
        """Расчет риска маржин-колла"""
        loss_abs = abs(loss_percent)
        
        if loss_abs > 0.5:
            return "Критический"
        elif loss_abs > 0.3:
            return "Высокий"
        elif loss_abs > 0.15:
            return "Умеренный"
        else:
            return "Низкий"

# ==================== ENHANCED TELEGRAM BOT ====================
# Расширяем бота из Phase 2 с новыми функциями

class EnhancedTelegramBot(TelegramRiskBot):
    """Расширенный Telegram бот с графиками и отчетами"""
    
    def __init__(self, token: str):
        super().__init__(token)
        self.graph_generator = AdvancedGraphGenerator()
        self.report_generator = ReportGenerator()
        self.analytics_engine = EnhancedAnalyticsEngine()
        
        # История запросов пользователей
        self.user_history = defaultdict(list)
    
    async def advanced_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Расширенный анализ с графиками"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        text = (
            "📈 *РАСШИРЕННЫЙ АНАЛИЗ*\n\n"
            "Выберите тип анализа:\n\n"
            "• 📊 *Полный отчет* - детальный анализ с графиками\n"
            "• 🎲 *Monte Carlo* - симуляция ценовых траекторий\n"
            "• 📉 *Распределение доходностей* - анализ статистики\n"
            "• ⚠️ *Стресс-тест* - анализ устойчивости портфеля\n"
            "• 🎯 *Технические индикаторы* - ATR, RSI, тренды\n"
            "• 📤 *Экспорт отчета* - отчеты в TXT/CSV/JSON"
        )
        
        keyboard = [
            [InlineKeyboardButton("📊 Полный отчет", callback_data="full_report")],
            [InlineKeyboardButton("🎲 Monte Carlo", callback_data="monte_carlo_chart")],
            [InlineKeyboardButton("📉 Распределение", callback_data="distribution_chart")],
            [
                InlineKeyboardButton("⚡ Быстрый анализ", callback_data="quick_analysis"),
                InlineKeyboardButton("📤 Экспорт", callback_data="export_menu")
            ],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            text=text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def generate_full_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Генерация полного отчета с графиками"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        # Проверяем историю пользователя
        if user_id not in self.user_history or not self.user_history[user_id]:
            await query.edit_message_text(
                text="❌ Сначала проанализируйте актив или портфель.",
                parse_mode='Markdown'
            )
            return
        
        # Получаем последний анализ
        last_analysis = self.user_history[user_id][-1]
        
        await query.edit_message_text(
            text="🔄 Генерирую полный отчет... Это займет несколько секунд.",
            parse_mode='Markdown'
        )
        
        try:
            # Генерация текстового отчета
            text_report = self.report_generator.generate_text_report(last_analysis['metrics'])
            
            # Генерация графиков
            if 'monte_carlo_data' in last_analysis:
                mc_chart = self.graph_generator.generate_monte_carlo_chart(
                    last_analysis['metrics'].current_price,
                    last_analysis['monte_carlo_data']['sample_paths'],
                    last_analysis['monte_carlo_data']['final_prices'],
                    last_analysis['metrics'].symbol
                )
                
                # Отправляем график
                await context.bot.send_photo(
                    chat_id=query.message.chat_id,
                    photo=InputFile(mc_chart, filename=f"mc_{last_analysis['metrics'].symbol}.png"),
                    caption=f"📈 Monte Carlo симуляция для {last_analysis['metrics'].symbol}"
                )
            
            if 'returns_data' in last_analysis:
                dist_chart = self.graph_generator.generate_distribution_chart(
                    last_analysis['returns_data'],
                    last_analysis['metrics'].symbol
                )
                
                await context.bot.send_photo(
                    chat_id=query.message.chat_id,
                    photo=InputFile(dist_chart, filename=f"dist_{last_analysis['metrics'].symbol}.png"),
                    caption=f"📊 Распределение доходностей {last_analysis['metrics'].symbol}"
                )
            
            # Отправляем текстовый отчет (разбиваем на части если длинный)
            chunks = [text_report[i:i+4000] for i in range(0, len(text_report), 4000)]
            
            for i, chunk in enumerate(chunks):
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text=f"```\n{chunk}\n```" if i == 0 else f"```\n{chunk}",
                    parse_mode='Markdown'
                )
            
            # Предлагаем экспорт
            keyboard = [
                [
                    InlineKeyboardButton("📝 TXT", callback_data=f"export_txt_{last_analysis['metrics'].symbol}"),
                    InlineKeyboardButton("📊 CSV", callback_data=f"export_csv_{last_analysis['metrics'].symbol}"),
                    InlineKeyboardButton("📋 JSON", callback_data=f"export_json_{last_analysis['metrics'].symbol}")
                ],
                [InlineKeyboardButton("🔄 Новый анализ", callback_data="analyze_asset")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
            
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="✅ Отчет сгенерирован!\n\nВыберите формат для экспорта:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error generating full report: {e}")
            await query.edit_message_text(
                text="❌ Ошибка при генерации отчета. Попробуйте снова.",
                parse_mode='Markdown'
            )
    
    async def export_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Меню экспорта отчетов"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        if user_id not in self.user_history or not self.user_history[user_id]:
            text = "❌ Нет данных для экспорта. Сначала проанализируйте актив."
        else:
            last_analysis = self.user_history[user_id][-1]
            symbol = last_analysis['metrics'].symbol
            
            text = f"📤 *Экспорт отчета для {symbol}*\n\nВыберите формат:"
        
        keyboard = [
            [
                InlineKeyboardButton("📝 TXT", callback_data=f"export_txt_{symbol if 'symbol' in locals() else ''}"),
                InlineKeyboardButton("📊 CSV", callback_data=f"export_csv_{symbol if 'symbol' in locals() else ''}"),
                InlineKeyboardButton("📋 JSON", callback_data=f"export_json_{symbol if 'symbol' in locals() else ''}")
            ],
            [InlineKeyboardButton("📈 PDF (скоро)", callback_data="coming_soon")],
            [InlineKeyboardButton("📊 Полный отчет", callback_data="full_report")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            text=text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def handle_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка экспорта отчетов"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        user_id = query.from_user.id
        
        if user_id not in self.user_history or not self.user_history[user_id]:
            await query.answer("❌ Нет данных для экспорта", show_alert=True)
            return
        
        last_analysis = self.user_history[user_id][-1]
        metrics = last_analysis['metrics']
        symbol = metrics.symbol
        
        await query.edit_message_text(
            text=f"🔄 Генерирую отчет для {symbol}...",
            parse_mode='Markdown'
        )
        
        try:
            if data.startswith("export_txt"):
                # TXT экспорт
                report = self.report_generator.generate_text_report(metrics)
                bio = io.BytesIO(report.encode('utf-8'))
                bio.seek(0)
                
                await context.bot.send_document(
                    chat_id=query.message.chat_id,
                    document=InputFile(bio, filename=f"report_{symbol}_{datetime.now().strftime('%Y%m%d')}.txt"),
                    caption=f"📝 Текстовый отчет: {symbol}"
                )
                
            elif data.startswith("export_csv"):
                # CSV экспорт
                report = self.report_generator.generate_csv_report(metrics)
                bio = io.BytesIO(report.encode('utf-8'))
                bio.seek(0)
                
                await context.bot.send_document(
                    chat_id=query.message.chat_id,
                    document=InputFile(bio, filename=f"report_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"),
                    caption=f"📊 CSV отчет: {symbol}"
                )
                
            elif data.startswith("export_json"):
                # JSON экспорт
                report = self.report_generator.generate_json_report(metrics)
                bio = io.BytesIO(report.encode('utf-8'))
                bio.seek(0)
                
                await context.bot.send_document(
                    chat_id=query.message.chat_id,
                    document=InputFile(bio, filename=f"report_{symbol}_{datetime.now().strftime('%Y%m%d')}.json"),
                    caption=f"📋 JSON отчет: {symbol}"
                )
            
            # Возвращаемся в меню
            keyboard = [
                [InlineKeyboardButton("📤 Еще отчеты", callback_data="export_menu")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
            
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="✅ Отчет успешно экспортирован!",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            await query.edit_message_text(
                text="❌ Ошибка при экспорте отчета.",
                parse_mode='Markdown'
            )
    
    async def portfolio_correlation_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Анализ корреляций в портфеле"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        if user_id not in user_portfolios or not user_portfolios[user_id]:
            await query.edit_message_text(
                text="❌ У вас нет портфеля для анализа.",
                parse_mode='Markdown'
            )
            return
        
        await query.edit_message_text(
            text="🔄 Анализирую корреляции в портфеле...",
            parse_mode='Markdown'
        )
        
        try:
            portfolio = user_portfolios[user_id]
            assets = list(portfolio.keys())
            
            # Генерируем симулированные корреляции для MVP
            import numpy as np
            
            num_assets = len(assets)
            returns_matrix = np.random.randn(100, num_assets)  # Случайные доходности
            
            # Создаем реалистичную матрицу корреляций
            base_corr = 0.3  # Базовая корреляция
            corr_matrix = np.eye(num_assets) + base_corr
            np.fill_diagonal(corr_matrix, 1)
            
            # Генерируем данные с заданной корреляцией
            L = np.linalg.cholesky(corr_matrix)
            correlated_returns = np.dot(returns_matrix, L.T)
            
            # Генерируем график
            chart = self.graph_generator.generate_correlation_matrix(assets, correlated_returns)
            
            # Расчет диверсификации
            weights = [portfolio[asset]['weight'] for asset in assets]
            div_metrics = self.analytics_engine.calculate_portfolio_diversification(weights, corr_matrix)
            
            # Отправляем график
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=InputFile(chart, filename="correlation_matrix.png"),
                caption=(
                    f"📊 *Матрица корреляций портфеля*\n\n"
                    f"• Активов: {num_assets}\n"
                    f"• Средняя корреляция: {div_metrics['avg_correlation']:.2f}\n"
                    f"• Оценка диверсификации: {div_metrics['diversification_score']:.0f}/100\n"
                    f"• Эффективных активов: {div_metrics['effective_assets']:.1f}"
                ),
                parse_mode='Markdown'
            )
            
            # Дополнительные рекомендации
            text = "💡 *Рекомендации по диверсификации:*\n\n"
            
            if div_metrics['avg_correlation'] > 0.7:
                text += "• Высокая корреляция - добавьте активы из других классов\n"
            if div_metrics['diversification_score'] < 60:
                text += "• Низкая диверсификация - распределите капитал\n"
            if div_metrics['concentration_risk'] > 0.3:
                text += "• Высокая концентрация - уменьшите позиции в крупнейших активах\n"
            
            if "Высокая" not in text and "Низкая" not in text and "Высокая" not in text:
                text += "• Портфель хорошо диверсифицирован\n"
            
            keyboard = [
                [InlineKeyboardButton("📈 Risk Radar", callback_data="risk_radar")],
                [InlineKeyboardButton("💰 Портфель", callback_data="manage_portfolio")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
            
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            await query.edit_message_text(
                text="❌ Ошибка при анализе корреляций.",
                parse_mode='Markdown'
            )
    
    async def risk_radar_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Радар-диаграмма рисков портфеля"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        if user_id not in user_portfolios or not user_portfolios[user_id]:
            await query.edit_message_text(
                text="❌ У вас нет портфеля для анализа.",
                parse_mode='Markdown'
            )
            return
        
        await query.edit_message_text(
            text="🔄 Генерирую радар-диаграмму рисков...",
            parse_mode='Markdown'
        )
        
        try:
            # Генерируем метрики риска (для MVP - симулированные)
            risk_metrics = {
                'volatility_score': 65,
                'drawdown_risk': 45,
                'concentration_risk': 70,
                'correlation_risk': 55,
                'liquidity_risk': 30,
                'market_risk': 60
            }
            
            # Генерируем график
            chart = self.graph_generator.generate_risk_radar_chart(risk_metrics)
            
            # Отправляем график
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=InputFile(chart, filename="risk_radar.png"),
                caption=(
                    "🎯 *Профиль риска портфеля*\n\n"
                    "• 📊 Волатильность: Умеренная\n"
                    "• 📉 Риск снижения: Низкий\n"
                    "• 🎯 Концентрация: Высокая\n"
                    "• 🔗 Корреляция: Умеренная\n"
                    "• 💧 Ликвидность: Хорошая\n"
                    "• 📈 Рыночный риск: Умеренный"
                ),
                parse_mode='Markdown'
            )
            
            keyboard = [
                [InlineKeyboardButton("📊 Корреляции", callback_data="portfolio_correlation")],
                [InlineKeyboardButton("⚠️ Стресс-тест", callback_data="stress_test")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
            
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="💡 *Интерпретация:*\nЧем ближе к краю - тем выше риск по этому параметру.",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error generating risk radar: {e}")
            await query.edit_message_text(
                text="❌ Ошибка при генерации радар-диаграммы.",
                parse_mode='Markdown'
            )

# ==================== ENHANCED WEB SERVER ====================
class EnhancedWebhookServer(WebhookServer):
    """Расширенный веб-сервер с мониторингом"""
    
    async def enhanced_health_check(self, request):
        """Расширенная проверка здоровья"""
        import psutil
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "risk-calculator-pro",
            "version": "MVP 3.0",
            "resources": {}
        }
        
        try:
            # Мониторинг ресурсов
            process = psutil.Process()
            memory_info = process.memory_info()
            
            health_data["resources"] = {
                "memory_mb": round(memory_info.rss / 1024 / 1024, 1),
                "memory_percent": round(process.memory_percent(), 1),
                "cpu_percent": round(process.cpu_percent(), 1),
                "threads": process.num_threads(),
                "connections": len(process.connections()) if hasattr(process, 'connections') else 0
            }
            
            # Проверка компонентов
            components = {}
            
            # Проверка графического генератора
            try:
                test_chart = AdvancedGraphGenerator().generate_monte_carlo_chart(
                    100, [[100, 110, 105]], [105], "TEST"
                )
                components["graph_generator"] = "operational"
            except Exception as e:
                components["graph_generator"] = f"error: {str(e)}"
                health_data["status"] = "degraded"
            
            # Проверка аналитического движка
            try:
                test_metrics = EnhancedAnalyticsEngine().calculate_technical_indicators([100, 105, 103, 108])
                components["analytics_engine"] = "operational"
            except Exception as e:
                components["analytics_engine"] = f"error: {str(e)}"
                health_data["status"] = "degraded"
            
            health_data["components"] = components
            
            # Memory guard check
            if health_data["resources"]["memory_mb"] > 400:
                health_data["status"] = "warning"
                health_data["message"] = "High memory usage detected"
                MemoryGuardian.check_and_clear()
            
        except Exception as e:
            health_data["status"] = "error"
            health_data["error"] = str(e)
        
        return web.json_response(health_data)
    
    async def start(self):
        """Запуск расширенного сервера"""
        app = web.Application()
        
        # Add enhanced routes
        app.router.add_post(WEBHOOK_PATH, self.handle_webhook)
        app.router.add_get('/health', self.enhanced_health_check)
        app.router.add_get('/health/simple', self.health_check)
        app.router.add_get('/metrics', self.metrics_endpoint)
        app.router.add_get('/', self.health_check)
        
        # Start server
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await self.site.start()
        
        logger.info(f"Enhanced webhook server started on port {self.port}")
    
    async def metrics_endpoint(self, request):
        """Endpoint для метрик Prometheus"""
        import psutil
        
        metrics = []
        process = psutil.Process()
        
        # Memory metrics
        memory = process.memory_info()
        metrics.append(f"memory_rss_bytes {memory.rss}")
        metrics.append(f"memory_vms_bytes {memory.vms}")
        metrics.append(f"memory_percent {process.memory_percent()}")
        
        # CPU metrics
        metrics.append(f"cpu_percent {process.cpu_percent()}")
        
        # Thread count
        metrics.append(f"threads_total {process.num_threads()}")
        
        # User sessions
        metrics.append(f"user_sessions_total {len(user_sessions)}")
        metrics.append(f"user_portfolios_total {len(user_portfolios)}")
        
        # Graph generator stats
        if hasattr(graph_generator, 'figures_created'):
            metrics.append(f"figures_created_total {graph_generator.figures_created}")
        
        response_text = "\n".join([f"risk_calculator_{m}" for m in metrics])
        return web.Response(text=response_text, content_type='text/plain')

# ==================== MAIN APPLICATION ====================
async def main():
    """Основная функция приложения"""
    logger.info("🚀 Запуск PRO Risk Calculator MVP Phase 3 (Complete)")
    logger.info("✅ Все компоненты загружены")
    
    # Инициализация расширенного бота
    bot = EnhancedTelegramBot(TOKEN)
    
    # Создание приложения
    application = Application.builder().token(TOKEN).build()
    
    # Добавление обработчиков команд из Phase 2
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("portfolio", bot.manage_portfolio))
    application.add_handler(CommandHandler("stress", bot.stress_test))
    application.add_handler(CommandHandler("alerts", bot.alerts_menu))
    application.add_handler(CommandHandler("export", bot.export_menu))
    application.add_handler(CommandHandler("advanced", bot.advanced_analysis))
    
    # Добавление обработчиков Phase 3
    application.add_handler(CallbackQueryHandler(bot.advanced_analysis, pattern="^advanced_analysis$"))
    application.add_handler(CallbackQueryHandler(bot.generate_full_report, pattern="^full_report$"))
    application.add_handler(CallbackQueryHandler(bot.export_menu, pattern="^export_menu$"))
    application.add_handler(CallbackQueryHandler(bot.handle_export, pattern="^export_"))
    application.add_handler(CallbackQueryHandler(bot.portfolio_correlation_analysis, pattern="^portfolio_correlation$"))
    application.add_handler(CallbackQueryHandler(bot.risk_radar_analysis, pattern="^risk_radar$"))
    
    # Остальные обработчики из Phase 2
    application.add_handler(CallbackQueryHandler(bot.callback_handler))
    
    # Fallback обработчик
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        lambda u, c: u.message.reply_text(
            "Используйте /start для начала работы или выберите команду из меню."
        )
    ))
    
    # Запуск в зависимости от режима
    if WEBHOOK_URL and WEBHOOK_URL.strip():
        logger.info("🌐 Запуск в режиме Webhook")
        
        # Установка вебхука
        webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
        await application.bot.set_webhook(
            url=webhook_url,
            allowed_updates=Update.ALL_TYPES
        )
        logger.info(f"Webhook установлен: {webhook_url}")
        
        # Запуск расширенного сервера
        server = EnhancedWebhookServer(application, PORT)
        await server.start()
        
        # Основной цикл с мониторингом
        try:
            while True:
                # Проверка памяти каждые 5 минут
                MemoryGuardian.check_and_clear()
                
                # Логирование статистики каждые 30 минут
                if datetime.now().minute % 30 == 0:
                    logger.info(f"Статистика: Пользователей: {len(user_sessions)}, "
                              f"Портфелей: {len(user_portfolios)}, "
                              f"Графиков создано: {bot.graph_generator.figures_created}")
                
                await asyncio.sleep(300)  # 5 минут
                
        except KeyboardInterrupt:
            logger.info("⏹ Остановка по команде пользователя")
            await server.stop()
            
    else:
        logger.info("🔄 Запуск в режиме Polling")
        
        # Запуск polling
        await application.initialize()
        await application.start()
        await application.updater.start_polling(
            poll_interval=1.0,
            timeout=30,
            drop_pending_updates=True
        )
        
        logger.info("✅ Бот запущен в режиме polling")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("⏹ Остановка по команде пользователя")
            await application.stop()

# ==================== GLOBALS AND INITIALIZATION ====================
# Глобальные экземпляры
graph_generator = AdvancedGraphGenerator()
report_generator = ReportGenerator()
analytics_engine = EnhancedAnalyticsEngine()

# Пользовательские данные
user_sessions = {}
user_portfolios = {}
user_alerts = {}

# Декоратор мониторинга производительности (определен ранее)
def monitor_performance(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                logger.warning(f"Slow operation: {func.__name__} took {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in {func.__name__} after {elapsed:.2f}s: {e}")
            raise
    return async_wrapper

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
