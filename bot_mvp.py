#!/usr/bin/env python3
"""
bot_mvp.py - MVP Risk Calculator Telegram Bot
Optimized for Render Free (512MB RAM)
Phase 1: Core Stability
"""

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
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict

# --- Load .env ---
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables!")

PORT = int(os.getenv("PORT", 10000))
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_URL", RENDER_EXTERNAL_URL + "/webhook")
WEBHOOK_URL = WEBHOOK_BASE_URL + f"/{TOKEN}"  # Полный URL с токеном
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

# Import optimization: Lazy imports for heavy modules
def lazy_import(module_name):
    """Lazy import helper"""
    import importlib
    return importlib.import_module(module_name)

# ==================== PHASE 1: CORE STABILITY ====================
# 1. Fixed imports and startup
# 2. Basic data provider with caching
# 3. Core risk calculations (VaR, CVaR, MDD)
# 4. Simple Telegram commands
# 5. Optimized for 512MB RAM

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
            psutil = lazy_import("psutil")
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
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def record_success(self):
        """Record an API success"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def can_execute(self):
        """Check if API call can be executed"""
        if self.state == "CLOSED":
            return True
        
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moving to HALF_OPEN state")
                return True
            return False
        
        elif self.state == "HALF_OPEN":
            return True
        
        return False

# ---------------------------
# MVP Data Provider (Optimized for Render Free)
# ---------------------------
class MVPDataProvider:
    """Optimized data provider with circuit breakers and caching"""
    
    def __init__(self):
        cachetools = lazy_import("cachetools")
        self.cache = cachetools.TTLCache(maxsize=100, ttl=600)  # 10 minute cache, reduced size
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
            aiohttp = lazy_import("aiohttp")
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
            logger.warning(f"Circuit breaker blocked for {symbol}, using fallback")
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
        np = lazy_import("numpy")
        np.random.seed(hash(symbol) % 10000)
        
        # Generate realistic returns based on asset type
        if any(c in symbol for c in ['BTC', 'ETH', 'XRP']):
            returns = np.random.normal(0.001, 0.04, period).tolist()
        elif any(c in symbol for c in ['AAPL', 'TSLA', 'GOOG']):
            returns = np.random.normal(0.0005, 0.02, period).tolist()
        elif any(c in symbol for c in ['EUR', 'GBP', 'JPY']):
            returns = np.random.normal(0.0001, 0.008, period).tolist()
        else:
            returns = np.random.normal(0.0003, 0.015, period).tolist()
        
        return returns
    
    def get_volatility(self, returns: List[float], annualize: bool = True) -> float:
        """Calculate volatility from returns"""
        if len(returns) < 2:
            return 0.0
        
        np = lazy_import("numpy")
        returns_array = np.array(returns)
        daily_vol = np.std(returns_array)
        
        if annualize:
            return daily_vol * np.sqrt(252)
        return daily_vol
    
    # API implementations
    async def _get_binance_price(self, symbol: str) -> Optional[float]:
        """Get price from Binance"""
        if not BINANCE_API_KEY or 'USDT' not in symbol:
            return None
        
        try:
            session = await self.get_session()
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
        
        np = lazy_import("numpy")
        returns_array = np.array(returns)
        var_percentile = np.percentile(returns_array, (1 - confidence) * 100)
        return abs(var_percentile)
    
    @staticmethod
    @monitor_performance
    def parametric_var(mean_return: float, volatility: float, confidence: float = 0.95) -> float:
        """Parametric (Normal) Value at Risk"""
        from scipy.stats import norm
        
        z_score = norm.ppf(confidence)
        var = mean_return - z_score * volatility
        return abs(var)
    
    @staticmethod
    @monitor_performance
    def conditional_var(returns: List[float], confidence: float = 0.95) -> float:
        """Conditional Value at Risk (Expected Shortfall)"""
        if not returns:
            return 0.0
        
        np = lazy_import("numpy")
        returns_array = np.array(returns)
        var_threshold = np.percentile(returns_array, (1 - confidence) * 100)
        
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
        
        np = lazy_import("numpy")
        prices_array = np.array(prices)
        
        cumulative_max = np.maximum.accumulate(prices_array)
        drawdown = (prices_array - cumulative_max) / cumulative_max
        max_dd = np.min(drawdown)
        return abs(max_dd)
    
    @staticmethod
    @monitor_performance
    def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Sharpe Ratio (annualized)"""
        if len(returns) < 2:
            return 0.0
        
        np = lazy_import("numpy")
        returns_array = np.array(returns)
        
        excess_returns = returns_array - (risk_free_rate / 252)
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess == 0:
            return 0.0
        
        sharpe = (mean_excess / std_excess) * np.sqrt(252)
        return sharpe
    
    @staticmethod
    @monitor_performance
    def sortino_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Sortino Ratio (annualized)"""
        if len(returns) < 2:
            return 0.0
        
        np = lazy_import("numpy")
        returns_array = np.array(returns)
        
        excess_returns = returns_array - (risk_free_rate / 252)
        mean_excess = np.mean(excess_returns)
        
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) == 0:
            return 0.0
        
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
        
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
        np = lazy_import("numpy")
        
        simulations = min(simulations, 1000)  # Limit for memory
        
        dt = 1/252
        prices = np.zeros((simulations, days))
        
        random_shocks = np.random.normal(0, 1, (simulations, days))
        
        for t in range(days):
            if t == 0:
                prices[:, t] = initial_price
            else:
                drift = (mean_return - 0.5 * volatility**2) * dt
                diffusion = volatility * np.sqrt(dt) * random_shocks[:, t]
                prices[:, t] = prices[:, t-1] * np.exp(drift + diffusion)
        
        final_prices = prices[:, -1].tolist()
        
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
        crisis_impacts = {
            'stocks': -0.50,
            'indices': -0.45,
            'crypto': -0.60,
            'forex': -0.15,
            'metals': 0.25,
            'energy': -0.40
        }
        
        total_impact = 0.0
        for asset_type, allocation in asset_allocation.items():
            impact = crisis_impacts.get(asset_type, -0.30)
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
            'stocks': -0.35,
            'indices': -0.30,
            'crypto': -0.50,
            'forex': -0.10,
            'metals': 0.15,
            'energy': -0.60
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
        crypto_impact = -0.75
        non_crypto_impact = -0.20
        
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
            'condition': condition,
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
# Note: Basic bot setup for Phase 1
# In Phase 2, we'll add full Telegram integration

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not available. Telegram features disabled.")

# Global instances
data_provider = MVPDataProvider()
risk_engine = MVPRiskEngine()
stress_tester = MVPStressTester()
alert_manager = MVPAlertManager()

# Load saved alerts on startup
alert_manager.load_from_file()

# ---------------------------
# Telegram Handlers
# ---------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "🚀 **Risk Management Bot MVP**\n\n"
        "Welcome! I can help you with:\n"
        "• Risk analysis for assets\n"
        "• VaR calculations\n"
        "• Monte Carlo simulations\n"
        "• Stress testing\n\n"
        "Commands:\n"
        "/risk [symbol] - Analyze asset risk\n"
        "/price [symbol] - Get current price\n"
        "/health - Check bot status\n"
        "/help - Show all commands"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_text = """
📋 **Available Commands:**

**Analysis:**
/risk [symbol] - Comprehensive risk analysis
/price [symbol] - Get current price
/var [symbol] - Calculate Value at Risk

**Stress Testing:**
/stress [portfolio_value] - Run stress tests

**Alerts:**
/alert price [symbol] [condition] [price] - Set price alert
/alerts list - List your alerts
/alerts delete [id] - Delete alert

**System:**
/health - System health check
/stats - Performance statistics

**Donations:**
/donate - Support development

**Examples:**
/risk BTCUSDT
/price AAPL
/alert price BTCUSDT above 110000
/stress 10000
"""
    await update.message.reply_text(help_text)

async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /risk command"""
    if not context.args:
        await update.message.reply_text("Please provide a symbol. Example: /risk BTCUSDT")
        return
    
    symbol = context.args[0].upper()
    
    # Send processing message
    processing_msg = await update.message.reply_text(f"📊 Analyzing {symbol}...")
    
    try:
        # Calculate risk metrics
        risk_data = await calculate_asset_risk(symbol)
        report = format_risk_report(risk_data)
        
        # Send report
        await update.message.reply_text(report, parse_mode='Markdown')
        await processing_msg.delete()
        
    except Exception as e:
        logger.error(f"Error in risk command: {e}")
        await update.message.reply_text(f"❌ Error analyzing {symbol}: {str(e)}")
        await processing_msg.delete()

async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /price command"""
    if not context.args:
        await update.message.reply_text("Please provide a symbol. Example: /price BTCUSDT")
        return
    
    symbol = context.args[0].upper()
    
    try:
        price = await data_provider.get_price(symbol)
        await update.message.reply_text(f"💰 **{symbol}**: ${price:.2f}")
    except Exception as e:
        logger.error(f"Error in price command: {e}")
        await update.message.reply_text(f"❌ Error getting price for {symbol}: {str(e)}")

async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /health command"""
    health_data = await health_check()
    
    status_emoji = "✅" if health_data["status"] == "healthy" else "⚠️" if health_data["status"] == "warning" else "❌"
    
    message = f"{status_emoji} **Bot Health Status**\n\n"
    message += f"**Status**: {health_data['status']}\n"
    message += f"**Version**: {health_data['version']}\n"
    message += f"**Uptime**: {health_data.get('uptime_seconds', 0):.0f}s\n"
    
    if 'memory_usage_mb' in health_data:
        message += f"**Memory Usage**: {health_data['memory_usage_mb']:.1f}MB\n"
    
    if 'services' in health_data:
        message += "\n**Services**:\n"
        for service, info in health_data['services'].items():
            status = info.get('status', 'unknown')
            emoji = "🟢" if status == 'operational' else "🟡" if status == 'degraded' else "🔴"
            message += f"{emoji} {service}: {status}\n"
    
    await update.message.reply_text(message, parse_mode='Markdown')

async def donate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /donate command"""
    donate_text = """
🙏 **Support Development**

If you find this bot useful, consider supporting its development:

**USDT (TRC20):**
    TVRGFPKVs1nN3fUXBTQfu5syTcmYGgADre
**TON:**
    UQDpCH-pGSzp3zEkpJY1Wc46gaorw9K-7T9FX7gHTrthMWMj
    
**Bitcoin (BTC):** Coming soon

Your support helps maintain and improve the bot!
"""
    await update.message.reply_text(donate_text)

# ---------------------------
# Utility Functions
# ---------------------------
async def calculate_asset_risk(symbol: str, period: int = 30) -> Dict[str, Any]:
    """Calculate comprehensive risk metrics for an asset"""
    current_price = await data_provider.get_price(symbol)
    returns = await data_provider.get_returns(symbol, period)
    
    if not returns or len(returns) < 5:
        return {
            'symbol': symbol,
            'current_price': current_price,
            'error': 'Insufficient data for risk calculation',
            'period': period
        }
    
    np = lazy_import("numpy")
    returns_array = np.array(returns)
    
    mean_return = np.mean(returns_array)
    volatility = data_provider.get_volatility(returns)
    annual_volatility = data_provider.get_volatility(returns, annualize=True)
    
    historical_var_95 = risk_engine.historical_var(returns, 0.95)
    historical_var_99 = risk_engine.historical_var(returns, 0.99)
    parametric_var_95 = risk_engine.parametric_var(mean_return, volatility, 0.95)
    conditional_var_95 = risk_engine.conditional_var(returns, 0.95)
    
    sharpe = risk_engine.sharpe_ratio(returns)
    sortino = risk_engine.sortino_ratio(returns)
    
    prices = [current_price]
    for ret in reversed(returns[-30:]):
        prices.insert(0, prices[0] / (1 + ret))
    
    max_dd = risk_engine.max_drawdown(prices)
    
    final_prices, sample_paths = risk_engine.monte_carlo_simulation(
        current_price, mean_return, volatility, days=30, simulations=500
    )
    
    mc_returns = [(fp - current_price) / current_price for fp in final_prices]
    mc_var_95 = risk_engine.historical_var(mc_returns, 0.95)
    
    return {
        'symbol': symbol,
        'current_price': round(current_price, 2),
        'period': period,
        'mean_return': round(mean_return * 100, 3),
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
        "version": "MVP Phase 1",
        "python_version": sys.version.split()[0],
        "uptime_seconds": time.time() - start_time if 'start_time' in globals() else 0,
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
            "error": str(e)[:100]
        }
        health["status"] = "degraded"
    
    # Check memory
    try:
        memory_usage = MemoryGuardian.get_memory_usage()
        health["memory_usage_mb"] = round(memory_usage, 1)
        health["services"]["memory"] = {
            "status": "good" if memory_usage < 350 else "warning",
            "usage_mb": round(memory_usage, 1)
        }
        
        if memory_usage > 350:
            health["status"] = "warning"
    except Exception as e:
        health["services"]["memory"] = {
            "status": "error",
            "error": str(e)[:100]
        }
    
    # Check Telegram availability
    health["services"]["telegram"] = {
        "status": "available" if TELEGRAM_AVAILABLE else "unavailable"
    }
    
    return health

# ---------------------------
# Webhook Setup (for Render)
# ---------------------------
async def handle_webhook(request):
    """Handle webhook requests from Telegram"""
    if not TELEGRAM_AVAILABLE:
        return
    
    from telegram import Update
    from telegram.ext import ContextTypes
    
    try:
        data = await request.json()
        update = Update.de_json(data, application.bot)
        
        await application.initialize()
        await application.process_update(update)
        
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error", "message": str(e)}

# ---------------------------
# HTTP Server for Health Checks
# ---------------------------
from aiohttp import web

async def http_health_check(request):
    """HTTP health check endpoint"""
    health_data = await health_check()
    return web.json_response(health_data)

async def http_main_page(request):
    """Main HTTP page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Risk Management Bot MVP</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            .status { padding: 10px; border-radius: 5px; margin: 20px 0; }
            .healthy { background: #d4edda; color: #155724; }
            .warning { background: #fff3cd; color: #856404; }
            .error { background: #f8d7da; color: #721c24; }
            .endpoints { margin-top: 30px; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background: #eee; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Risk Management Bot MVP</h1>
            <p>Telegram bot for financial risk analysis and management.</p>
            
            <div class="status healthy">
                <strong>Status:</strong> Running
            </div>
            
            <div class="endpoints">
                <h3>API Endpoints:</h3>
                <div class="endpoint">
                    <strong>GET</strong> <code>/health</code> - Health check
                </div>
                <div class="endpoint">
                    <strong>POST</strong> <code>/webhook/{token}</code> - Telegram webhook
                </div>
            </div>
            
            <p style="margin-top: 30px; font-size: 0.9em; color: #666;">
                Version: MVP Phase 1 | Running on Render Free
            </p>
        </div>
    </body>
    </html>
    """
    return web.Response(text=html_content, content_type='text/html')

# ---------------------------
# Main Application
# ---------------------------
async def setup_telegram_bot():
    """Setup Telegram bot handlers"""
    if not TELEGRAM_AVAILABLE:
        logger.warning("Telegram bot not available. Skipping Telegram setup.")
        return None
    
    try:
        application = Application.builder().token(TOKEN).build()
        
        # Add command handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("risk", risk_command))
        application.add_handler(CommandHandler("price", price_command))
        application.add_handler(CommandHandler("health", health_command))
        application.add_handler(CommandHandler("donate", donate_command))
        
        # Add message handler
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, 
                                             lambda update, context: update.message.reply_text("Use /help to see available commands")))
        
        return application
    except Exception as e:
        logger.error(f"Failed to setup Telegram bot: {e}")
        return None

async def setup_http_server():
    """Setup HTTP server for health checks"""
    app = web.Application()
    app.router.add_get('/', http_main_page)
    app.router.add_get('/health', http_health_check)
    
    if TELEGRAM_AVAILABLE and WEBHOOK_URL:
        app.router.add_post(f'/webhook/{TOKEN}', handle_webhook)
    
    return app

async def main():
    """Main application entry point"""
    global start_time
    start_time = time.time()
    
    logger.info("🚀 Launching MVP Risk Calculator - Phase 1")
    logger.info(f"Telegram Bot Token: {TOKEN[:10]}...")
    logger.info(f"Webhook URL: {WEBHOOK_URL}")
    
    # Initialize components
    logger.info("Initializing components...")
    
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
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        return
    
    # Setup Telegram bot
    telegram_app = await setup_telegram_bot()
    
    # Setup HTTP server
    http_app = await setup_http_server()
    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    
    logger.info(f"Starting HTTP server on port {PORT}...")
    await site.start()
    
    if telegram_app and WEBHOOK_URL:
        # Set webhook for Telegram
        try:
            await telegram_app.bot.set_webhook(WEBHOOK_URL)
            logger.info(f"Webhook set to: {WEBHOOK_URL}")
        except Exception as e:
            logger.error(f"Failed to set webhook: {e}")
    
    # Keep running
    logger.info("✅ Bot is now running!")
    logger.info(f"Health endpoint: http://0.0.0.0:{PORT}/health")
    
    try:
        # Run forever
        while True:
            # Periodic memory check
            MemoryGuardian.check_and_clear([data_provider.cache])
            
            # Sleep for 60 seconds
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("⏹ Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
    finally:
        # Cleanup
        await data_provider.cleanup()
        if telegram_app:
            await telegram_app.shutdown()
        await runner.cleanup()
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
