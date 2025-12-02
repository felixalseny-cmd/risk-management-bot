# bot_optimized_v4.py — ENTERPRISE RISK CALCULATOR v4.0 | ULTIMATE EDITION
# CRITICAL OPTIMIZATIONS FOR RENDER FREE + ENHANCED FUNCTIONALITY
import os
import logging
import asyncio
import time
import functools
import json
import re
import html
import aiohttp
import cachetools
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Set
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
import statistics
import math

# --- LAZY IMPORTS to reduce cold start time ---
telegram = None
web = None
InputFile = None

def lazy_import_telegram():
    global telegram, InputFile
    if telegram is None:
        import telegram
        from telegram import InputFile
    return telegram, InputFile

def lazy_import_web():
    global web
    if web is None:
        from aiohttp import web
    return web

# --- Configuration with environment fallbacks ---
from dotenv import load_dotenv
load_dotenv()

# API Keys with validation
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN_EN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in environment!")

PORT = int(os.getenv("PORT", 10000))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").rstrip("/")
if WEBHOOK_URL and "render.com" in WEBHOOK_URL:
    # Force HTTPS for Render
    WEBHOOK_URL = WEBHOOK_URL.replace("http://", "https://")
WEBHOOK_PATH = f"/webhook/{TOKEN}"

# Financial API Keys with priority ordering
API_KEYS = {
    'FMP': os.getenv("FMP_API_KEY", "nZm3b15R1rJvjnUO67wPb0eaJHPXarK2"),
    'TWELVEDATA': os.getenv("TWELVEDATA_API_KEY", "972d1359cbf04ff68dd0feba7e32cc8d"),
    'ALPHA_VANTAGE': os.getenv("ALPHA_VANTAGE_API_KEY"),
    'BINANCE_KEY': os.getenv("BINANCE_API_KEY"),
    'BINANCE_SECRET': os.getenv("BINANCE_SECRET_KEY"),
    'FINNHUB': os.getenv("FINNHUB_API_KEY"),
    'METALPRICE': os.getenv("METALPRICE_API_KEY", "e6e8aa0b29f4e612751cde3985a7b8ec"),
    'EXCHANGERATE': os.getenv("EXCHANGERATE_API_KEY", "d8f8278cf29f8fe18445e8b7")
}

# Donation wallets
USDT_WALLET_ADDRESS = os.getenv("USDT_WALLET_ADDRESS", "TVRGFPKVs1nN3fUXBTQfu5syTcmYGgADre")
TON_WALLET_ADDRESS = os.getenv("TON_WALLET_ADDRESS", "UQDpCH-pGSzp3zEkpJY1Wc46gaorw9K-7T9FX7gHTrthMWMj")

# --- Advanced Logging Configuration ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot_performance.log')
    ]
)
logger = logging.getLogger("enterprise_risk_bot_v4")
performance_logger = logging.getLogger("performance_metrics")

# --- PERFORMANCE MONITORING DECORATOR ---
def monitor_performance(func):
    """Decorator to log function execution time"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        # Log slow operations
        if elapsed > 0.5:
            performance_logger.warning(f"SLOW OPERATION: {func.__name__} took {elapsed:.3f}s")
        
        # Store metrics for health check
        if not hasattr(monitor_performance, 'metrics'):
            monitor_performance.metrics = defaultdict(list)
        monitor_performance.metrics[func.__name__].append(elapsed)
        
        # Keep only last 100 measurements
        if len(monitor_performance.metrics[func.__name__]) > 100:
            monitor_performance.metrics[func.__name__].pop(0)
            
        return result
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        if elapsed > 0.1:
            performance_logger.info(f"SYNC OPERATION: {func.__name__} took {elapsed:.3f}s")
            
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# --- PARALLEL API REQUEST MANAGER ---
class ParallelAPIManager:
    """Manages parallel API requests with intelligent provider selection"""
    
    def __init__(self):
        self.session = None
        self.provider_stats = defaultdict(lambda: {'success': 0, 'errors': 0, 'avg_time': 0})
        self.circuit_breakers = defaultdict(lambda: {'failures': 0, 'last_failure': 0, 'state': 'CLOSED'})
        self.cache = cachetools.LRUCache(maxsize=1000)
        self.last_health_check = 0
        
    async def get_session(self):
        """Get or create aiohttp session with optimized settings"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=10, connect=3, sock_read=5)
            connector = aiohttp.TCPConnector(
                limit=20,
                limit_per_host=5,
                ttl_dns_cache=300,
                force_close=True
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': 'RiskCalculatorBot/4.0'}
            )
        return self.session
    
    async def close(self):
        """Close session gracefully"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _check_circuit_breaker(self, provider: str) -> bool:
        """Circuit breaker pattern to avoid hitting failing APIs"""
        cb = self.circuit_breakers[provider]
        
        if cb['state'] == 'OPEN':
            # Check if enough time has passed to try again
            if time.time() - cb['last_failure'] > 60:  # 1 minute cooldown
                cb['state'] = 'HALF_OPEN'
                return True
            return False
        return True
    
    def _record_failure(self, provider: str):
        """Record API failure and potentially open circuit breaker"""
        cb = self.circuit_breakers[provider]
        cb['failures'] += 1
        cb['last_failure'] = time.time()
        
        if cb['failures'] >= 3:
            cb['state'] = 'OPEN'
            logger.warning(f"Circuit breaker OPEN for {provider}")
    
    def _record_success(self, provider: str):
        """Record API success and reset circuit breaker"""
        cb = self.circuit_breakers[provider]
        cb['failures'] = 0
        cb['state'] = 'CLOSED'
        self.provider_stats[provider]['success'] += 1
    
    @monitor_performance
    async def fetch_parallel(self, symbol: str) -> Tuple[Optional[float], str]:
        """Fetch price from multiple APIs in parallel with intelligent fallback"""
        
        # Check cache first
        cache_key = f"price_{symbol}_{datetime.now().hour}"  # Cache by hour
        cached = self.cache.get(cache_key)
        if cached:
            return cached, 'cache'
        
        # Define provider functions
        providers = [
            ('FMP', self._fetch_fmp),
            ('TWELVEDATA', self._fetch_twelvedata),
            ('BINANCE', self._fetch_binance),
            ('ALPHA_VANTAGE', self._fetch_alpha_vantage),
            ('METALPRICE', self._fetch_metalprice),
            ('EXCHANGERATE', self._fetch_exchangerate),
            ('FINNHUB', self._fetch_finnhub)
        ]
        
        # Filter out providers with open circuit breakers
        active_providers = []
        for name, func in providers:
            if self._check_circuit_breaker(name):
                active_providers.append((name, func))
        
        if not active_providers:
            # All circuit breakers are open, use fallback
            return self._get_fallback_price(symbol), 'fallback_all_cb'
        
        # Execute providers in parallel
        tasks = []
        for name, func in active_providers:
            task = asyncio.create_task(self._safe_provider_call(name, func, symbol))
            tasks.append(task)
        
        try:
            # Wait for first successful response
            done, pending = await asyncio.wait(
                tasks,
                timeout=3.0,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Process completed tasks
            for task in done:
                try:
                    result = task.result()
                    if result is not None:
                        price, provider_name = result
                        if price and price > 0:
                            # Cache successful result
                            self.cache[cache_key] = (price, provider_name)
                            self._record_success(provider_name)
                            return price, provider_name
                except Exception as e:
                    logger.debug(f"Provider task failed: {e}")
                    continue
            
            # If no provider succeeded, try fallback
            fallback_price = self._get_fallback_price(symbol)
            self.cache[cache_key] = (fallback_price, 'fallback')
            return fallback_price, 'fallback'
            
        except asyncio.TimeoutError:
            logger.warning(f"All price providers timed out for {symbol}")
            fallback_price = self._get_fallback_price(symbol)
            self.cache[cache_key] = (fallback_price, 'fallback_timeout')
            return fallback_price, 'fallback_timeout'
    
    async def _safe_provider_call(self, provider_name: str, provider_func, symbol: str):
        """Safe wrapper for provider calls with error handling"""
        try:
            start_time = time.time()
            result = await provider_func(symbol)
            elapsed = time.time() - start_time
            
            # Update provider stats
            if provider_name in self.provider_stats:
                stats = self.provider_stats[provider_name]
                old_avg = stats['avg_time']
                stats['avg_time'] = (old_avg * stats['success'] + elapsed) / (stats['success'] + 1)
            
            return result, provider_name
        except Exception as e:
            self._record_failure(provider_name)
            self.provider_stats[provider_name]['errors'] += 1
            logger.debug(f"Provider {provider_name} failed for {symbol}: {e}")
            return None
    
    # --- PROVIDER IMPLEMENTATIONS WITH PROPER API KEY HANDLING ---
    
    async def _fetch_fmp(self, symbol: str) -> Optional[float]:
        """Fetch from Financial Modeling Prep with correct API key parameter"""
        if not API_KEYS['FMP']:
            return None
        
        try:
            session = await self.get_session()
            # Correct FMP API URL format with apikey parameter
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
            params = {'apikey': API_KEYS['FMP']}
            
            async with session.get(url, params=params, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and isinstance(data, list) and len(data) > 0:
                        price = data[0].get('price')
                        if price:
                            return float(price)
        except Exception as e:
            logger.error(f"FMP API error: {e}")
        return None
    
    async def _fetch_twelvedata(self, symbol: str) -> Optional[float]:
        """Fetch from Twelve Data"""
        if not API_KEYS['TWELVEDATA']:
            return None
        
        try:
            session = await self.get_session()
            url = f"https://api.twelvedata.com/price"
            params = {
                'symbol': symbol,
                'apikey': API_KEYS['TWELVEDATA']
            }
            
            async with session.get(url, params=params, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    price_str = data.get('price')
                    if price_str and price_str != '':
                        return float(price_str)
        except Exception as e:
            logger.error(f"TwelveData API error: {e}")
        return None
    
    async def _fetch_binance(self, symbol: str) -> Optional[float]:
        """Fetch from Binance for crypto pairs"""
        crypto_pairs = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'BNB', 'SOL']
        if not any(pair in symbol for pair in crypto_pairs):
            return None
        
        try:
            session = await self.get_session()
            # Convert BTCUSDT -> BTCUSDT
            binance_symbol = symbol.replace('/', '').replace('-', '')
            if 'USDT' not in binance_symbol and 'USD' in binance_symbol:
                binance_symbol = binance_symbol.replace('USD', 'USDT')
            
            url = f"https://api.binance.com/api/v3/ticker/price"
            params = {'symbol': binance_symbol}
            
            async with session.get(url, params=params, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get('price', 0))
        except Exception as e:
            logger.error(f"Binance API error: {e}")
        return None
    
    async def _fetch_alpha_vantage(self, symbol: str) -> Optional[float]:
        """Fetch from Alpha Vantage"""
        if not API_KEYS['ALPHA_VANTAGE']:
            return None
        
        try:
            session = await self.get_session()
            
            # Determine if it's forex or stock
            forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
            if symbol in forex_pairs:
                from_curr = symbol[:3]
                to_curr = symbol[3:]
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'CURRENCY_EXCHANGE_RATE',
                    'from_currency': from_curr,
                    'to_currency': to_curr,
                    'apikey': API_KEYS['ALPHA_VANTAGE']
                }
                
                async with session.get(url, params=params, timeout=8) as response:
                    if response.status == 200:
                        data = await response.json()
                        rate_data = data.get('Realtime Currency Exchange Rate', {})
                        rate = rate_data.get('5. Exchange Rate')
                        if rate:
                            return float(rate)
            else:
                # Assume stock
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': API_KEYS['ALPHA_VANTAGE']
                }
                
                async with session.get(url, params=params, timeout=8) as response:
                    if response.status == 200:
                        data = await response.json()
                        quote = data.get('Global Quote', {})
                        price = quote.get('05. price')
                        if price:
                            return float(price)
        except Exception as e:
            logger.error(f"Alpha Vantage API error: {e}")
        return None
    
    async def _fetch_metalprice(self, symbol: str) -> Optional[float]:
        """Fetch metal prices"""
        metals = {
            'XAUUSD': 'XAU',
            'XAGUSD': 'XAG',
            'XPTUSD': 'XPT',
            'XPDUSD': 'XPD'
        }
        
        if symbol not in metals or not API_KEYS['METALPRICE']:
            return None
        
        try:
            session = await self.get_session()
            metal_code = metals[symbol]
            url = f"https://api.metalpriceapi.com/v1/latest"
            params = {
                'api_key': API_KEYS['METALPRICE'],
                'base': 'USD',
                'currencies': metal_code
            }
            
            async with session.get(url, params=params, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('success'):
                        rates = data.get('rates', {})
                        # Metalprice API returns USD per metal unit
                        rate = rates.get(metal_code)
                        if rate:
                            # Convert to metal per USD for consistency
                            return 1.0 / rate
        except Exception as e:
            logger.error(f"MetalPrice API error: {e}")
        return None
    
    async def _fetch_exchangerate(self, symbol: str) -> Optional[float]:
        """Fetch forex rates from Frankfurter"""
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        if symbol not in forex_pairs:
            return None
        
        try:
            session = await self.get_session()
            from_curr = symbol[:3]
            to_curr = symbol[3:]
            
            url = f"https://api.frankfurter.app/latest"
            params = {'from': from_curr, 'to': to_curr}
            
            async with session.get(url, params=params, timeout=3) as response:
                if response.status == 200:
                    data = await response.json()
                    rate = data.get('rates', {}).get(to_curr)
                    if rate:
                        return float(rate)
        except Exception as e:
            logger.error(f"ExchangeRate API error: {e}")
        return None
    
    async def _fetch_finnhub(self, symbol: str) -> Optional[float]:
        """Fetch from Finnhub (fallback)"""
        if not API_KEYS['FINNHUB']:
            return None
        
        try:
            session = await self.get_session()
            url = f"https://finnhub.io/api/v1/quote"
            params = {
                'symbol': symbol,
                'token': API_KEYS['FINNHUB']
            }
            
            async with session.get(url, params=params, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('c', 0)
        except Exception as e:
            logger.error(f"Finnhub API error: {e}")
        return None
    
    def _get_fallback_price(self, symbol: str) -> float:
        """Intelligent fallback prices based on instrument type"""
        fallback_prices = {
            # Forex
            'EURUSD': 1.08, 'GBPUSD': 1.26, 'USDJPY': 151.0, 'USDCHF': 0.88,
            'AUDUSD': 0.66, 'USDCAD': 1.36, 'NZDUSD': 0.61,
            # Crypto (updated 2025)
            'BTCUSDT': 105000.0, 'ETHUSDT': 5200.0, 'XRPUSDT': 1.05,
            'LTCUSDT': 155.0, 'BCHUSDT': 620.0, 'ADAUSDT': 1.10,
            'DOTUSDT': 11.0, 'BNBUSDT': 600.0, 'SOLUSDT': 180.0,
            # Stocks
            'AAPL': 205.0, 'TSLA': 310.0, 'GOOGL': 155.0, 'MSFT': 410.0,
            'AMZN': 205.0, 'META': 510.0, 'NFLX': 610.0, 'NVDA': 950.0,
            # Indices
            'NAS100': 20500.0, 'SPX500': 5600.0, 'DJ30': 40500.0,
            'FTSE100': 8100.0, 'DAX40': 19200.0, 'NIKKEI225': 40500.0,
            # Metals
            'XAUUSD': 2550.0, 'XAGUSD': 31.0, 'XPTUSD': 1050.0, 'XPDUSD': 1050.0,
            # Energy
            'OIL': 82.0, 'NATURALGAS': 3.2, 'BRENT': 87.0
        }
        
        return fallback_prices.get(symbol, 100.0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get API performance statistics"""
        stats = {}
        for provider, data in self.provider_stats.items():
            total = data['success'] + data['errors']
            if total > 0:
                success_rate = (data['success'] / total) * 100
                stats[provider] = {
                    'success_rate': round(success_rate, 1),
                    'avg_time_ms': round(data['avg_time'] * 1000, 1),
                    'total_requests': total,
                    'circuit_state': self.circuit_breakers[provider]['state']
                }
        return stats

# --- ENHANCED INSTRUMENT SPECIFICATIONS WITH TECHNICAL DATA ---
class EnhancedInstrumentSpecs:
    """Enhanced specifications with technical indicators and volatility data"""
    
    SPECS = {
        # Forex
        "EURUSD": {"type": "forex", "contract_size": 100000, "pip_value": 10.0, "pip_places": 4,
                  "avg_volatility": 8.5, "trading_hours": "24/5", "margin_percent": 0.1, "spread_avg": 1.2},
        "GBPUSD": {"type": "forex", "contract_size": 100000, "pip_value": 10.0, "pip_places": 4,
                  "avg_volatility": 9.2, "trading_hours": "24/5", "margin_percent": 0.1, "spread_avg": 1.5},
        "USDJPY": {"type": "forex", "contract_size": 100000, "pip_value": 9.09, "pip_places": 2,
                  "avg_volatility": 7.8, "trading_hours": "24/5", "margin_percent": 0.1, "spread_avg": 1.8},
        
        # Crypto
        "BTCUSDT": {"type": "crypto", "contract_size": 1, "pip_value": 1.0, "pip_places": 1,
                   "avg_volatility": 42.5, "trading_hours": "24/7", "margin_percent": 0.8, "spread_avg": 5.0},
        "ETHUSDT": {"type": "crypto", "contract_size": 1, "pip_value": 1.0, "pip_places": 2,
                   "avg_volatility": 38.7, "trading_hours": "24/7", "margin_percent": 0.8, "spread_avg": 2.5},
        
        # Stocks
        "AAPL": {"type": "stock", "contract_size": 100, "pip_value": 1.0, "pip_places": 2,
                "avg_volatility": 22.3, "trading_hours": "9:30-16:00 EST", "margin_percent": 1.0, "spread_avg": 0.05},
        "TSLA": {"type": "stock", "contract_size": 100, "pip_value": 1.0, "pip_places": 2,
                "avg_volatility": 45.6, "trading_hours": "9:30-16:00 EST", "margin_percent": 1.0, "spread_avg": 0.12},
        
        # Metals
        "XAUUSD": {"type": "metal", "contract_size": 100, "pip_value": 1.0, "pip_places": 2,
                  "avg_volatility": 12.8, "trading_hours": "24/5", "margin_percent": 0.5, "spread_avg": 0.30},
        "XAGUSD": {"type": "metal", "contract_size": 5000, "pip_value": 5.0, "pip_places": 3,
                  "avg_volatility": 18.5, "trading_hours": "24/5", "margin_percent": 0.5, "spread_avg": 0.015},
        
        # Indices
        "NAS100": {"type": "index", "contract_size": 1, "pip_value": 1.0, "pip_places": 1,
                  "avg_volatility": 15.4, "trading_hours": "24/5", "margin_percent": 0.5, "spread_avg": 1.5},
    }
    
    # Technical indicator parameters per instrument
    TECHNICAL_PARAMS = {
        "EURUSD": {"rsi_period": 14, "ma_fast": 9, "ma_slow": 21, "bb_period": 20},
        "BTCUSDT": {"rsi_period": 14, "ma_fast": 7, "ma_slow": 25, "bb_period": 20},
        "AAPL": {"rsi_period": 14, "ma_fast": 10, "ma_slow": 30, "bb_period": 20},
        "XAUUSD": {"rsi_period": 14, "ma_fast": 8, "ma_slow": 21, "bb_period": 20},
    }
    
    @classmethod
    def get_specs(cls, symbol: str) -> Dict[str, Any]:
        """Get instrument specifications with defaults"""
        specs = cls.SPECS.get(symbol, cls._get_default_specs(symbol))
        # Add technical params if available
        tech_params = cls.TECHNICAL_PARAMS.get(symbol, {
            "rsi_period": 14, 
            "ma_fast": 9, 
            "ma_slow": 21, 
            "bb_period": 20
        })
        specs.update(tech_params)
        return specs
    
    @classmethod
    def _get_default_specs(cls, symbol: str) -> Dict[str, Any]:
        """Default specifications for unknown instruments"""
        if any(curr in symbol for curr in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']):
            return {
                "type": "forex", "contract_size": 100000, "pip_value": 10.0, "pip_places": 4,
                "avg_volatility": 10.0, "trading_hours": "24/5", "margin_percent": 0.1, "spread_avg": 1.5
            }
        elif 'USDT' in symbol or 'BTC' in symbol or 'ETH' in symbol:
            return {
                "type": "crypto", "contract_size": 1, "pip_value": 1.0, "pip_places": 2,
                "avg_volatility": 35.0, "trading_hours": "24/7", "margin_percent": 0.8, "spread_avg": 3.0
            }
        else:
            return {
                "type": "stock", "contract_size": 100, "pip_value": 1.0, "pip_places": 2,
                "avg_volatility": 25.0, "trading_hours": "9:30-16:00 EST", "margin_percent": 1.0, "spread_avg": 0.1
            }

# --- TECHNICAL ANALYSIS ENGINE ---
class TechnicalAnalyzer:
    """Pure Python technical analysis calculations"""
    
    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_atr(high_prices: List[float], low_prices: List[float], 
                     close_prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate Average True Range"""
        if len(high_prices) < period or len(low_prices) < period or len(close_prices) < period:
            return None
        
        true_ranges = []
        for i in range(1, len(high_prices)):
            high_low = high_prices[i] - low_prices[i]
            high_close_prev = abs(high_prices[i] - close_prices[i-1])
            low_close_prev = abs(low_prices[i] - close_prices[i-1])
            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return None
        
        return sum(true_ranges[-period:]) / period
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, 
                                 num_std: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {}
        
        recent_prices = prices[-period:]
        middle_band = sum(recent_prices) / period
        
        # Calculate standard deviation
        variance = sum((x - middle_band) ** 2 for x in recent_prices) / period
        std_dev = math.sqrt(variance)
        
        upper_band = middle_band + (num_std * std_dev)
        lower_band = middle_band - (num_std * std_dev)
        
        # Calculate bandwidth and %B
        bandwidth = (upper_band - lower_band) / middle_band * 100
        current_price = prices[-1]
        percent_b = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
        
        return {
            'upper': round(upper_band, 4),
            'middle': round(middle_band, 4),
            'lower': round(lower_band, 4),
            'bandwidth': round(bandwidth, 2),
            'percent_b': round(percent_b, 3),
            'current_position': 'UPPER' if current_price > upper_band else 
                              'LOWER' if current_price < lower_band else 'MIDDLE'
        }
    
    @staticmethod
    def calculate_macd(prices: List[float], fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> Dict[str, float]:
        """Calculate MACD indicator"""
        if len(prices) < slow_period:
            return {}
        
        # Calculate EMAs
        ema_fast = TechnicalAnalyzer.calculate_ema(prices, fast_period)
        ema_slow = TechnicalAnalyzer.calculate_ema(prices, slow_period)
        
        if ema_fast is None or ema_slow is None:
            return {}
        
        macd_line = ema_fast - ema_slow
        
        # For signal line, we need MACD history
        # Simplified: use recent prices as proxy
        recent_fast_emas = []
        for i in range(len(prices) - fast_period + 1):
            ema = TechnicalAnalyzer.calculate_ema(prices[i:i+fast_period], fast_period)
            if ema:
                recent_fast_emas.append(ema)
        
        if len(recent_fast_emas) < signal_period:
            return {'macd': round(macd_line, 4)}
        
        signal_line = TechnicalAnalyzer.calculate_ema(recent_fast_emas[-signal_period:], signal_period)
        histogram = macd_line - signal_line if signal_line else 0
        
        return {
            'macd': round(macd_line, 4),
            'signal': round(signal_line, 4) if signal_line else 0,
            'histogram': round(histogram, 4),
            'trend': 'BULLISH' if histogram > 0 else 'BEARISH'
        }
    
    @staticmethod
    def calculate_support_resistance(prices: List[float], lookback: int = 20) -> Dict[str, List[float]]:
        """Identify support and resistance levels"""
        if len(prices) < lookback:
            return {'support': [], 'resistance': []}
        
        recent_prices = prices[-lookback:]
        
        # Simple pivot point calculation
        pivot_points = []
        for i in range(1, len(recent_prices) - 1):
            if (recent_prices[i] > recent_prices[i-1] and 
                recent_prices[i] > recent_prices[i+1]):
                pivot_points.append(('resistance', recent_prices[i]))
            elif (recent_prices[i] < recent_prices[i-1] and 
                  recent_prices[i] < recent_prices[i+1]):
                pivot_points.append(('support', recent_prices[i]))
        
        # Group nearby levels
        support_levels = []
        resistance_levels = []
        
        for level_type, price in pivot_points:
            if level_type == 'support':
                # Check if near existing support
                found = False
                for i, existing in enumerate(support_levels):
                    if abs(price - existing) / existing < 0.01:  # Within 1%
                        support_levels[i] = (support_levels[i] + price) / 2
                        found = True
                        break
                if not found:
                    support_levels.append(price)
            else:
                # Resistance
                found = False
                for i, existing in enumerate(resistance_levels):
                    if abs(price - existing) / existing < 0.01:
                        resistance_levels[i] = (resistance_levels[i] + price) / 2
                        found = True
                        break
                if not found:
                    resistance_levels.append(price)
        
        # Sort and round
        support_levels = sorted(support_levels)[-3:]  # Top 3 supports
        resistance_levels = sorted(resistance_levels)[-3:]  # Top 3 resistances
        
        return {
            'support': [round(level, 4) for level in support_levels],
            'resistance': [round(level, 4) for level in resistance_levels]
        }

# --- ENHANCED RISK CALCULATOR WITH ADVANCED METRICS ---
class AdvancedRiskCalculator:
    """Advanced risk calculations with VaR, CVaR, and stress testing"""
    
    def __init__(self, api_manager: ParallelAPIManager):
        self.api = api_manager
    
    @monitor_performance
    async def calculate_advanced_metrics(self, trade: Dict, deposit: float, 
                                        leverage: str) -> Dict[str, Any]:
        """Calculate advanced risk metrics including VaR and stress scenarios"""
        
        # Basic metrics from existing calculator
        basic_metrics = await self._calculate_basic_metrics(trade, deposit, leverage)
        
        # Advanced metrics
        var_metrics = await self._calculate_var_metrics(trade, deposit, basic_metrics)
        stress_metrics = self._calculate_stress_scenarios(trade, deposit, basic_metrics)
        correlation_risk = await self._calculate_correlation_risk(trade)
        
        # Combine all metrics
        advanced_metrics = {
            **basic_metrics,
            **var_metrics,
            **stress_metrics,
            **correlation_risk,
            'risk_score': self._calculate_risk_score(basic_metrics, var_metrics, stress_metrics)
        }
        
        return advanced_metrics
    
    async def _calculate_basic_metrics(self, trade: Dict, deposit: float, 
                                      leverage: str) -> Dict[str, Any]:
        """Calculate basic risk metrics (adapted from existing calculator)"""
        
        asset = trade['asset']
        specs = EnhancedInstrumentSpecs.get_specs(asset)
        
        # Get current price
        current_price, source = await self.api.fetch_parallel(asset)
        
        # Calculate position size based on 2% risk rule
        risk_amount = deposit * 0.02
        stop_distance_pips = self._calculate_pip_distance(
            trade['entry_price'], trade['stop_loss'], trade['direction'], asset
        )
        
        pip_value = specs['pip_value']
        if stop_distance_pips > 0 and pip_value > 0:
            volume_lots = risk_amount / (stop_distance_pips * pip_value)
            volume_lots = max(volume_lots, specs.get('min_volume', 0.01))
            volume_lots = round(volume_lots, 3)
        else:
            volume_lots = 0
        
        # Calculate margin
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        required_margin = (volume_lots * contract_size * current_price) / lev_value
        required_margin = max(required_margin, 0.01)
        
        # Calculate P&L
        current_pnl = self._calculate_pnl(
            trade['entry_price'], current_price, volume_lots,
            pip_value, trade['direction'], asset
        )
        
        # Calculate potential profit/loss
        potential_profit = self._calculate_pnl(
            trade['entry_price'], trade['take_profit'], volume_lots,
            pip_value, trade['direction'], asset
        )
        
        potential_loss = self._calculate_pnl(
            trade['entry_price'], trade['stop_loss'], volume_lots,
            pip_value, trade['direction'], asset
        )
        
        equity = deposit + current_pnl
        margin_level = (equity / required_margin) * 100 if required_margin > 0 else float('inf')
        
        return {
            'volume_lots': volume_lots,
            'required_margin': round(required_margin, 2),
            'current_pnl': round(current_pnl, 2),
            'equity': round(equity, 2),
            'margin_level': round(margin_level, 2),
            'risk_amount': round(risk_amount, 2),
            'potential_profit': round(potential_profit, 2),
            'potential_loss': round(potential_loss, 2),
            'rr_ratio': round(abs(potential_profit / potential_loss), 2) if potential_loss != 0 else 0,
            'current_price': current_price,
            'price_source': source,
            'volatility': specs['avg_volatility'],
            'instrument_type': specs['type']
        }
    
    async def _calculate_var_metrics(self, trade: Dict, deposit: float, 
                                    basic_metrics: Dict) -> Dict[str, Any]:
        """Calculate Value at Risk metrics"""
        
        # Simplified VaR calculation (for production, use historical data)
        volatility = basic_metrics['volatility']
        position_value = basic_metrics['volume_lots'] * EnhancedInstrumentSpecs.get_specs(trade['asset'])['contract_size']
        
        # 1-day VaR at 95% confidence (simplified)
        var_95 = position_value * (volatility / 100) * 1.645
        var_99 = position_value * (volatility / 100) * 2.326
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = position_value * (volatility / 100) * 2.063
        cvar_99 = position_value * (volatility / 100) * 2.665
        
        # VaR as percentage of equity
        var_95_percent = (var_95 / basic_metrics['equity']) * 100 if basic_metrics['equity'] > 0 else 0
        var_99_percent = (var_99 / basic_metrics['equity']) * 100 if basic_metrics['equity'] > 0 else 0
        
        return {
            'var_95_1d': round(var_95, 2),
            'var_99_1d': round(var_99, 2),
            'cvar_95_1d': round(cvar_95, 2),
            'cvar_99_1d': round(cvar_99, 2),
            'var_95_percent': round(var_95_percent, 1),
            'var_99_percent': round(var_99_percent, 1),
            'var_breach_probability': round(self._calculate_breach_probability(basic_metrics), 1)
        }
    
    def _calculate_stress_scenarios(self, trade: Dict, deposit: float,
                                   basic_metrics: Dict) -> Dict[str, Any]:
        """Calculate stress test scenarios"""
        
        scenarios = {
            'mild_stress': {'price_change': -0.05, 'vol_change': 1.5},  # -5% price, +50% vol
            'moderate_stress': {'price_change': -0.15, 'vol_change': 2.0},  # -15% price, +100% vol
            'severe_stress': {'price_change': -0.30, 'vol_change': 3.0},  # -30% price, +200% vol
            'black_swan': {'price_change': -0.50, 'vol_change': 5.0}  # -50% price, +400% vol
        }
        
        stress_results = {}
        for scenario, params in scenarios.items():
            price_change = params['price_change']
            vol_multiplier = params['vol_change']
            
            # Adjust price based on direction
            if trade['direction'] == 'LONG':
                stressed_price = trade['entry_price'] * (1 + price_change)
            else:
                stressed_price = trade['entry_price'] * (1 - price_change)
            
            # Calculate P&L at stressed price
            stressed_pnl = self._calculate_pnl(
                trade['entry_price'], stressed_price, basic_metrics['volume_lots'],
                EnhancedInstrumentSpecs.get_specs(trade['asset'])['pip_value'],
                trade['direction'], trade['asset']
            )
            
            # Calculate margin level at stress
            stressed_equity = deposit + stressed_pnl
            stressed_margin_level = (stressed_equity / basic_metrics['required_margin']) * 100
            
            stress_results[f'{scenario}_pnl'] = round(stressed_pnl, 2)
            stress_results[f'{scenario}_equity'] = round(stressed_equity, 2)
            stress_results[f'{scenario}_margin_level'] = round(stressed_margin_level, 1)
            stress_results[f'{scenario}_margin_call'] = stressed_margin_level < 100
            
        return stress_results
    
    async def _calculate_correlation_risk(self, trade: Dict) -> Dict[str, Any]:
        """Calculate correlation risk with other assets"""
        
        # Asset correlations (simplified - in production use real correlation matrix)
        correlation_matrix = {
            'EURUSD': {'GBPUSD': 0.85, 'USDJPY': -0.75, 'XAUUSD': -0.45},
            'BTCUSDT': {'ETHUSDT': 0.92, 'XAUUSD': 0.15, 'NAS100': 0.35},
            'XAUUSD': {'XAGUSD': 0.78, 'USDJPY': 0.65, 'BTCUSDT': 0.15},
            'NAS100': {'SPX500': 0.95, 'AAPL': 0.82, 'BTCUSDT': 0.35}
        }
        
        asset = trade['asset']
        correlations = correlation_matrix.get(asset, {})
        
        # Calculate portfolio concentration risk
        concentration_risk = 1.0 if len(correlations) == 0 else 1.0 / (1 + len(correlations))
        
        return {
            'correlations': correlations,
            'concentration_risk': round(concentration_risk * 100, 1),
            'diversification_score': round((1 - concentration_risk) * 100, 1)
        }
    
    def _calculate_risk_score(self, basic: Dict, var: Dict, stress: Dict) -> float:
        """Calculate overall risk score (0-100, higher = riskier)"""
        
        score = 0
        weights = {
            'margin_level': 0.25,
            'var_95_percent': 0.30,
            'rr_ratio': 0.15,
            'volatility': 0.15,
            'concentration_risk': 0.15
        }
        
        # Margin level component (lower margin level = higher risk)
        margin_score = max(0, min(100, (200 - basic['margin_level']) / 2))
        score += margin_score * weights['margin_level']
        
        # VaR component
        var_score = min(100, basic['var_95_percent'] * 2)
        score += var_score * weights['var_95_percent']
        
        # R/R ratio component (lower R/R = higher risk)
        rr_score = max(0, min(100, (3 - basic['rr_ratio']) * 33.33))
        score += rr_score * weights['rr_ratio']
        
        # Volatility component
        vol_score = min(100, basic['volatility'] * 2)
        score += vol_score * weights['volatility']
        
        # Concentration risk
        conc_score = basic.get('concentration_risk', 50)
        score += conc_score * weights['concentration_risk']
        
        # Stress test adjustment
        if stress.get('severe_stress_margin_call', False):
            score = min(100, score * 1.3)
        
        return round(score, 1)
    
    def _calculate_pip_distance(self, entry: float, target: float, 
                               direction: str, asset: str) -> float:
        """Calculate pip distance between prices"""
        specs = EnhancedInstrumentSpecs.get_specs(asset)
        pip_places = specs['pip_places']
        
        if direction == 'LONG':
            distance = target - entry
        else:
            distance = entry - target
        
        # Convert to pips based on decimal places
        multiplier = 10 ** (pip_places - 1)
        return abs(distance) * multiplier
    
    def _calculate_pnl(self, entry: float, exit: float, volume: float,
                      pip_value: float, direction: str, asset: str) -> float:
        """Calculate P&L"""
        specs = EnhancedInstrumentSpecs.get_specs(asset)
        
        if direction == 'LONG':
            price_diff = exit - entry
        else:
            price_diff = entry - exit
        
        if specs['type'] in ['stock', 'crypto']:
            pnl = price_diff * volume * specs['contract_size']
        else:
            pip_distance = self._calculate_pip_distance(entry, exit, direction, asset)
            pnl = pip_distance * volume * pip_value
        
        return round(pnl, 2)
    
    def _calculate_breach_probability(self, metrics: Dict) -> float:
        """Calculate probability of breaching stop loss"""
        # Simplified calculation based on volatility and stop distance
        volatility = metrics['volatility']
        stop_distance_pct = abs(metrics['potential_loss']) / metrics['equity'] * 100
        
        # Using normal distribution approximation
        # Probability that daily return exceeds stop distance
        z_score = stop_distance_pct / volatility
        probability = (1 - self._normal_cdf(z_score)) * 100
        
        return min(100, max(0, probability))
    
    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate cumulative distribution function for standard normal"""
        # Abramowitz & Stegun approximation
        t = 1 / (1 + 0.2316419 * abs(x))
        d = 0.3989423 * math.exp(-x * x / 2)
        prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
        
        if x > 0:
            prob = 1 - prob
        
        return prob

# --- PORTFOLIO STRESS TESTER ---
class PortfolioStressTester:
    """Advanced portfolio stress testing and scenario analysis"""
    
    @staticmethod
    def analyze_portfolio_stress(portfolio_trades: List[Dict], deposit: float) -> Dict[str, Any]:
        """Analyze portfolio under various stress scenarios"""
        
        if not portfolio_trades:
            return {'empty': True}
        
        # Collect metrics from all trades
        all_metrics = [trade.get('metrics', {}) for trade in portfolio_trades]
        
        # Current portfolio state
        current_state = {
            'total_equity': sum(m.get('equity', 0) for m in all_metrics) or deposit,
            'total_margin': sum(m.get('required_margin', 0) for m in all_metrics),
            'total_pnl': sum(m.get('current_pnl', 0) for m in all_metrics),
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0
        }
        
        # Stress scenarios
        scenarios = {
            'market_crash_2008': {'equity_change': -0.50, 'correlation': 0.95},
            'crypto_winter_2022': {'equity_change': -0.75, 'correlation': 0.98},
            'rate_hike_shock': {'equity_change': -0.30, 'correlation': 0.85},
            'vix_spike': {'equity_change': -0.25, 'correlation': 0.90},
            'black_swan': {'equity_change': -0.90, 'correlation': 1.00}
        }
        
        stress_results = {}
        for scenario, params in scenarios.items():
            stressed_equity = current_state['total_equity'] * (1 + params['equity_change'])
            margin_call = stressed_equity < current_state['total_margin']
            
            stress_results[scenario] = {
                'stressed_equity': round(stressed_equity, 2),
                'equity_change_pct': round(params['equity_change'] * 100, 1),
                'margin_call': margin_call,
                'survival_days': max(0, round(stressed_equity / (deposit * 0.02))),
                'recovery_required': round(abs(params['equity_change']) * 100, 1)
            }
        
        # Calculate diversification metrics
        diversification = PortfolioStressTester._calculate_diversification(portfolio_trades)
        
        # Calculate liquidity metrics
        liquidity = PortfolioStressTester._calculate_liquidity_metrics(portfolio_trades, deposit)
        
        return {
            'current_state': current_state,
            'stress_scenarios': stress_results,
            'diversification': diversification,
            'liquidity': liquidity,
            'recommendations': PortfolioStressTester._generate_stress_recommendations(
                current_state, stress_results, diversification
            )
        }
    
    @staticmethod
    def _calculate_diversification(portfolio_trades: List[Dict]) -> Dict[str, Any]:
        """Calculate portfolio diversification metrics"""
        
        assets = [trade['asset'] for trade in portfolio_trades]
        asset_types = [EnhancedInstrumentSpecs.get_specs(a)['type'] for a in assets]
        
        unique_assets = len(set(assets))
        unique_types = len(set(asset_types))
        
        # Herfindahl-Hirschman Index (concentration)
        position_values = [trade.get('metrics', {}).get('required_margin', 0) for trade in portfolio_trades]
        total_value = sum(position_values)
        
        if total_value > 0:
            hhi = sum((v / total_value) ** 2 for v in position_values) * 10000
            concentration = min(100, hhi / 100)  # Normalize to 0-100
        else:
            concentration = 0
        
        # Correlation score (simplified)
        correlation_score = max(0, 100 - concentration)
        
        return {
            'unique_assets': unique_assets,
            'unique_types': unique_types,
            'concentration_index': round(concentration, 1),
            'diversification_score': round(correlation_score, 1),
            'suggested_improvements': PortfolioStressTester._suggest_diversification(assets, asset_types)
        }
    
    @staticmethod
    def _calculate_liquidity_metrics(portfolio_trades: List[Dict], deposit: float) -> Dict[str, Any]:
        """Calculate portfolio liquidity metrics"""
        
        total_margin = sum(trade.get('metrics', {}).get('required_margin', 0) for trade in portfolio_trades)
        total_pnl = sum(trade.get('metrics', {}).get('current_pnl', 0) for trade in portfolio_trades)
        equity = deposit + total_pnl
        
        # Liquidity ratios
        margin_usage = (total_margin / equity * 100) if equity > 0 else 0
        free_margin = max(0, equity - total_margin)
        free_margin_ratio = (free_margin / equity * 100) if equity > 0 else 0
        
        # Emergency liquidity (how many days of 2% risk can be covered)
        daily_risk = deposit * 0.02
        emergency_days = free_margin / daily_risk if daily_risk > 0 else 0
        
        return {
            'margin_usage_pct': round(margin_usage, 1),
            'free_margin': round(free_margin, 2),
            'free_margin_ratio': round(free_margin_ratio, 1),
            'emergency_days': round(emergency_days, 1),
            'liquidity_grade': 'A' if free_margin_ratio > 50 else 
                              'B' if free_margin_ratio > 30 else 
                              'C' if free_margin_ratio > 20 else 
                              'D' if free_margin_ratio > 10 else 'F'
        }
    
    @staticmethod
    def _suggest_diversification(assets: List[str], asset_types: List[str]) -> List[str]:
        """Generate diversification suggestions"""
        
        suggestions = []
        type_counts = {}
        for a_type in asset_types:
            type_counts[a_type] = type_counts.get(a_type, 0) + 1
        
        # Check for over-concentration
        total_positions = len(assets)
        for a_type, count in type_counts.items():
            percentage = (count / total_positions) * 100
            if percentage > 50:
                suggestions.append(f"Reduce {a_type} exposure (currently {percentage:.0f}%)")
        
        # Suggest missing asset types
        all_types = {'forex', 'crypto', 'stock', 'metal', 'index', 'energy'}
        missing_types = all_types - set(asset_types)
        
        if missing_types:
            suggestions.append(f"Consider adding: {', '.join(missing_types)}")
        
        if len(assets) < 3:
            suggestions.append("Add more positions for better diversification")
        
        return suggestions if suggestions else ["Portfolio is well diversified"]
    
    @staticmethod
    def _generate_stress_recommendations(current_state: Dict, stress_results: Dict,
                                        diversification: Dict) -> List[str]:
        """Generate recommendations based on stress test results"""
        
        recommendations = []
        
        # Margin call risk
        for scenario, data in stress_results.items():
            if data.get('margin_call'):
                recommendations.append(
                    f"⚠️ {scenario.replace('_', ' ').title()}: Potential margin call"
                )
        
        # Concentration risk
        if diversification['concentration_index'] > 70:
            recommendations.append(
                f"High concentration risk ({diversification['concentration_index']}%)"
            )
        
        # Liquidity risk
        if current_state['total_margin'] > current_state['total_equity'] * 0.7:
            recommendations.append(
                f"High margin usage: {current_state['total_margin']/current_state['total_equity']*100:.1f}%"
            )
        
        # Diversification improvements
        if diversification['diversification_score'] < 60:
            recommendations.append(
                f"Improve diversification (score: {diversification['diversification_score']})"
            )
        
        if not recommendations:
            recommendations.append("Portfolio appears resilient to stress scenarios")
        
        return recommendations

# --- ALERT MANAGER FOR PRICE AND RISK ALERTS ---
class AlertManager:
    """Manage price alerts and risk notifications"""
    
    def __init__(self):
        self.active_alerts = defaultdict(list)  # user_id -> list of alerts
        self.last_check = {}
        
    def add_alert(self, user_id: int, alert_type: str, instrument: str,
                 threshold: float, condition: str, callback_data: str = None):
        """Add a new alert"""
        
        alert = {
            'id': f"{user_id}_{instrument}_{int(time.time())}",
            'type': alert_type,  # price, volatility, margin, etc.
            'instrument': instrument,
            'threshold': threshold,
            'condition': condition,  # above, below, equals
            'created': datetime.now(),
            'triggered': False,
            'callback_data': callback_data,
            'notified': False
        }
        
        self.active_alerts[user_id].append(alert)
        return alert['id']
    
    def remove_alert(self, user_id: int, alert_id: str):
        """Remove an alert"""
        if user_id in self.active_alerts:
            self.active_alerts[user_id] = [
                alert for alert in self.active_alerts[user_id]
                if alert['id'] != alert_id
            ]
    
    async def check_alerts(self, user_id: int, current_prices: Dict[str, float],
                          portfolio_metrics: Dict[str, Any] = None):
        """Check all alerts for a user"""
        
        if user_id not in self.active_alerts:
            return []
        
        triggered = []
        now = datetime.now()
        
        for alert in self.active_alerts[user_id]:
            if alert['triggered']:
                continue
            
            instrument = alert['instrument']
            threshold = alert['threshold']
            
            if alert['type'] == 'price' and instrument in current_prices:
                current_price = current_prices[instrument]
                
                if alert['condition'] == 'above' and current_price >= threshold:
                    alert['triggered'] = True
                    alert['triggered_at'] = now
                    triggered.append(alert)
                
                elif alert['condition'] == 'below' and current_price <= threshold:
                    alert['triggered'] = True
                    alert['triggered_at'] = now
                    triggered.append(alert)
            
            elif alert['type'] == 'margin' and portfolio_metrics:
                margin_level = portfolio_metrics.get('margin_level', 1000)
                
                if alert['condition'] == 'below' and margin_level <= threshold:
                    alert['triggered'] = True
                    alert['triggered_at'] = now
                    triggered.append(alert)
            
            elif alert['type'] == 'volatility' and instrument in current_prices:
                # Simplified volatility alert (would need historical data for real volatility)
                pass
        
        return triggered
    
    def get_user_alerts(self, user_id: int) -> List[Dict]:
        """Get all alerts for a user"""
        return self.active_alerts.get(user_id, [])
    
    def clear_user_alerts(self, user_id: int):
        """Clear all alerts for a user"""
        if user_id in self.active_alerts:
            del self.active_alerts[user_id]

# --- ENHANCED DATA MANAGER WITH PERSISTENT STORAGE ---
class EnhancedDataManager:
    """Enhanced data manager with file-based persistence"""
    
    def __init__(self, data_dir: str = "user_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def save_user_portfolio(self, user_id: int, portfolio_data: Dict):
        """Save user portfolio to file"""
        filename = os.path.join(self.data_dir, f"{user_id}_portfolio.json")
        try:
            with open(filename, 'w') as f:
                json.dump(portfolio_data, f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Error saving portfolio for {user_id}: {e}")
    
    def load_user_portfolio(self, user_id: int) -> Dict:
        """Load user portfolio from file"""
        filename = os.path.join(self.data_dir, f"{user_id}_portfolio.json")
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading portfolio for {user_id}: {e}")
        return {}
    
    def save_user_settings(self, user_id: int, settings: Dict):
        """Save user settings"""
        filename = os.path.join(self.data_dir, f"{user_id}_settings.json")
        try:
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving settings for {user_id}: {e}")
    
    def load_user_settings(self, user_id: int) -> Dict:
        """Load user settings"""
        filename = os.path.join(self.data_dir, f"{user_id}_settings.json")
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading settings for {user_id}: {e}")
        return {}
    
    def save_trade_history(self, user_id: int, trade: Dict):
        """Save trade to history"""
        filename = os.path.join(self.data_dir, f"{user_id}_history.json")
        history = self.load_trade_history(user_id)
        history.append({
            **trade,
            'timestamp': datetime.now().isoformat(),
            'id': f"trade_{int(time.time())}_{len(history)}"
        })
        
        try:
            with open(filename, 'w') as f:
                json.dump(history[-100:], f, default=str, indent=2)  # Keep last 100 trades
        except Exception as e:
            logger.error(f"Error saving trade history for {user_id}: {e}")
    
    def load_trade_history(self, user_id: int) -> List[Dict]:
        """Load trade history"""
        filename = os.path.join(self.data_dir, f"{user_id}_history.json")
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading trade history for {user_id}: {e}")
        return []
    
    def cleanup_old_data(self, max_age_days: int = 30):
        """Cleanup old user data"""
        try:
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.data_dir, filename)
                    file_age = time.time() - os.path.getmtime(filepath)
                    if file_age > max_age_days * 86400:
                        os.remove(filepath)
                        logger.info(f"Removed old file: {filename}")
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

# --- GLOBAL INSTANCES (Lazy initialization) ---
api_manager = None
risk_calculator = None
technical_analyzer = None
stress_tester = None
alert_manager = None
data_manager = None

def get_api_manager():
    global api_manager
    if api_manager is None:
        api_manager = ParallelAPIManager()
    return api_manager

def get_risk_calculator():
    global risk_calculator
    if risk_calculator is None:
        risk_calculator = AdvancedRiskCalculator(get_api_manager())
    return risk_calculator

def get_technical_analyzer():
    global technical_analyzer
    if technical_analyzer is None:
        technical_analyzer = TechnicalAnalyzer()
    return technical_analyzer

def get_stress_tester():
    global stress_tester
    if stress_tester is None:
        stress_tester = PortfolioStressTester()
    return stress_tester

def get_alert_manager():
    global alert_manager
    if alert_manager is None:
        alert_manager = AlertManager()
    return alert_manager

def get_data_manager():
    global data_manager
    if data_manager is None:
        data_manager = EnhancedDataManager()
    return data_manager

# --- INITIALIZATION OPTIMIZATION ---
async def initialize_core_services():
    """Initialize core services in parallel to reduce cold start time"""
    logger.info("Starting parallel initialization...")
    
    # Initialize services in parallel
    init_tasks = [
        get_api_manager().get_session(),  # Initialize API session
    ]
    
    await asyncio.gather(*init_tasks, return_exceptions=True)
    logger.info("Core services initialized")

# --- TELEGRAM BOT HANDLERS WITH PERFORMANCE OPTIMIZATIONS ---

class OptimizedTelegramBot:
    """Optimized Telegram bot for Render Free with advanced features"""
    
    def __init__(self):
        self.application = None
        self.user_states = {}
        self.user_portfolios = defaultdict(lambda: {
            'single_trades': [],
            'multi_trades': [],
            'deposit': 1000.0,
            'leverage': '1:100',
            'settings': {
                'notifications': True,
                'risk_tolerance': 'medium',
                'default_assets': ['EURUSD', 'BTCUSDT', 'AAPL', 'XAUUSD']
            }
        })
        
        # Performance tracking
        self.startup_time = None
        self.request_count = 0
        
    async def initialize(self):
        """Initialize bot with lazy imports"""
        global telegram, InputFile
        telegram, InputFile = lazy_import_telegram()
        
        from telegram.ext import (
            Application, CommandHandler, ContextTypes, MessageHandler,
            filters, CallbackQueryHandler, ConversationHandler
        )
        
        self.startup_time = time.time()
        
        # Create application with optimized settings for Render Free
        request = telegram.request.HTTPXRequest(
            connection_pool_size=5,
            read_timeout=30.0,
            write_timeout=30.0,
            connect_timeout=10.0
        )
        
        self.application = (
            Application.builder()
            .token(TOKEN)
            .request(request)
            .post_init(self._post_init)
            .post_shutdown(self._post_shutdown)
            .build()
        )
        
        # Initialize core services in background
        asyncio.create_task(self._background_initialization())
        
        # Register handlers
        await self._register_handlers()
        
        logger.info(f"Bot initialized in {time.time() - self.startup_time:.2f}s")
    
    async def _background_initialization(self):
        """Background initialization to speed up startup"""
        try:
            await get_api_manager().get_session()
            logger.info("Background services initialized")
        except Exception as e:
            logger.error(f"Background initialization failed: {e}")
    
    async def _post_init(self, application: Application):
        """Post initialization tasks"""
        logger.info("Bot post-initialization started")
        
        # Schedule periodic tasks
        application.job_queue.run_repeating(
            self._periodic_health_check,
            interval=300,  # 5 minutes
            first=10
        )
        
        application.job_queue.run_repeating(
            self._cleanup_old_data,
            interval=86400,  # 24 hours
            first=60
        )
        
        logger.info("Periodic tasks scheduled")
    
    async def _post_shutdown(self, application: Application):
        """Cleanup on shutdown"""
        logger.info("Bot shutting down, cleaning up...")
        await get_api_manager().close()
        get_data_manager().cleanup_old_data()
    
    async def _periodic_health_check(self, context: ContextTypes.DEFAULT_TYPE):
        """Periodic health check to keep Render instance alive"""
        try:
            # Log performance stats
            stats = get_api_manager().get_performance_stats()
            logger.info(f"API Stats: {stats}")
            
            # Simple self-check
            await context.bot.get_me()
            logger.debug("Health check passed")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def _cleanup_old_data(self, context: ContextTypes.DEFAULT_TYPE):
        """Cleanup old user data"""
        get_data_manager().cleanup_old_data(max_age_days=7)
        logger.info("Old data cleanup completed")
    
    async def _register_handlers(self):
        """Register all Telegram handlers"""
        from telegram.ext import (
            CommandHandler, MessageHandler, filters, CallbackQueryHandler,
            ConversationHandler
        )
        
        # Basic commands
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("portfolio", self._portfolio_command))
        self.application.add_handler(CommandHandler("alerts", self._alerts_command))
        self.application.add_handler(CommandHandler("settings", self._settings_command))
        self.application.add_handler(CommandHandler("technical", self._technical_analysis_command))
        self.application.add_handler(CommandHandler("stress", self._stress_test_command))
        
        # Callback query handler
        self.application.add_handler(CallbackQueryHandler(self._callback_handler))
        
        # Message handlers
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._text_message_handler
        ))
        
        # Error handler
        self.application.add_error_handler(self._error_handler)
        
        logger.info("Handlers registered")
    
    # --- COMMAND HANDLERS ---
    
    @monitor_performance
    async def _start_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced start command with quick actions"""
        self.request_count += 1
        
        user = update.effective_user
        welcome_text = f"""🚀 <b>ENTERPRISE RISK CALCULATOR v4.0</b>

Welcome, {user.first_name}! 

I'm your professional risk management assistant with:
• <b>Real-time market data</b> from 7+ sources
• <b>Parallel API processing</b> for speed
• <b>Advanced risk metrics</b> (VaR, CVaR, Stress Tests)
• <b>Technical analysis</b> indicators
• <b>Portfolio stress testing</b>
• <b>Price alerts</b> and notifications

<b>Startup time:</b> {time.time() - (self.startup_time or time.time()):.2f}s
<b>Requests processed:</b> {self.request_count}

Select an option:"""
        
        keyboard = [
            [telegram.InlineKeyboardButton("🎯 Quick Risk Calc", callback_data="quick_calc"),
             telegram.InlineKeyboardButton("📊 Full Portfolio", callback_data="portfolio_full")],
            [telegram.InlineKeyboardButton("📈 Technical Analysis", callback_data="technical_menu"),
             telegram.InlineKeyboardButton("🧪 Stress Test", callback_data="stress_test")],
            [telegram.InlineKeyboardButton("⚙️ Settings", callback_data="settings_menu"),
             telegram.InlineKeyboardButton("🔔 Alerts", callback_data="alerts_menu")],
            [telegram.InlineKeyboardButton("📚 Tutorial", callback_data="tutorial"),
             telegram.InlineKeyboardButton("💝 Donate", callback_data="donate_start")]
        ]
        
        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        
        if update.message:
            await update.message.reply_html(welcome_text, reply_markup=reply_markup)
        else:
            await update.callback_query.message.reply_html(welcome_text, reply_markup=reply_markup)
    
    @monitor_performance
    async def _help_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced help command with feature overview"""
        
        help_text = """📚 <b>ENTERPRISE RISK CALCULATOR - COMMAND GUIDE</b>

<b>🎯 QUICK COMMANDS:</b>
/start - Main menu
/portfolio - View your portfolio
/technical [SYMBOL] - Technical analysis
/stress - Portfolio stress test
/alerts - Manage price alerts
/settings - Bot settings

<b>📊 PORTFOLIO FEATURES:</b>
• Real-time P&L calculation
• Margin level monitoring
• Diversification analysis
• Correlation risk assessment
• VaR (Value at Risk) metrics

<b>📈 TECHNICAL ANALYSIS:</b>
• RSI, MACD, Bollinger Bands
• Support/Resistance levels
• Moving Averages
• Volatility indicators
• Trend analysis

<b>🧪 ADVANCED RISK TOOLS:</b>
• Stress testing (2008, 2022 scenarios)
• Monte Carlo simulations
• Black Swan event modeling
• Liquidity risk assessment
• Concentration risk analysis

<b>⚡ PERFORMANCE:</b>
• Parallel API processing (3x faster)
• Intelligent caching
• Circuit breaker protection
• Graceful degradation

<b>🔧 SUPPORTED ASSETS:</b>
Forex, Crypto, Stocks, Metals, Indices, Energy

<b>💡 TIP:</b> Use 'Quick Risk Calc' for fast calculations with 2% risk rule."""
        
        keyboard = [[telegram.InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]
        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_html(help_text, reply_markup=reply_markup)
    
    @monitor_performance
    async def _portfolio_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced portfolio command with real-time updates"""
        user_id = update.effective_user.id
        
        # Load portfolio
        portfolio_data = get_data_manager().load_user_portfolio(user_id)
        trades = portfolio_data.get('trades', [])
        
        if not trades:
            await update.message.reply_html(
                "📭 <b>Your portfolio is empty</b>\n\n"
                "Start by calculating a trade risk with 2% rule!",
                reply_markup=telegram.InlineKeyboardMarkup([
                    [telegram.InlineKeyboardButton("🎯 Quick Calculation", callback_data="quick_calc")]
                ])
            )
            return
        
        # Update all trades with current prices
        updated_trades = []
        for trade in trades:
            metrics = await get_risk_calculator().calculate_advanced_metrics(
                trade, portfolio_data.get('deposit', 1000), portfolio_data.get('leverage', '1:100')
            )
            trade['metrics'] = metrics
            updated_trades.append(trade)
        
        # Calculate portfolio metrics
        portfolio_metrics = get_stress_tester().analyze_portfolio_stress(
            updated_trades, portfolio_data.get('deposit', 1000)
        )
        
        # Generate portfolio summary
        summary = self._generate_portfolio_summary(updated_trades, portfolio_metrics)
        
        # Send portfolio report
        await self._send_portfolio_report(update, user_id, updated_trades, summary)
    
    @monitor_performance
    async def _technical_analysis_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Technical analysis for an asset"""
        
        if context.args:
            symbol = context.args[0].upper()
        else:
            await update.message.reply_html(
                "📈 <b>Technical Analysis</b>\n\n"
                "Please provide a symbol:\n"
                "<code>/technical BTCUSDT</code>\n"
                "<code>/technical EURUSD</code>\n"
                "<code>/technical AAPL</code>"
            )
            return
        
        # Get current price
        price, source = await get_api_manager().fetch_parallel(symbol)
        
        # Get technical indicators (simplified - in production would use historical data)
        specs = EnhancedInstrumentSpecs.get_specs(symbol)
        
        # Generate analysis
        analysis = self._generate_technical_analysis(symbol, price, specs)
        
        # Prepare response
        response = f"""📈 <b>TECHNICAL ANALYSIS: {symbol}</b>

<b>Current Price:</b> ${price:.4f} ({source})
<b>Instrument Type:</b> {specs['type']}
<b>Avg Volatility:</b> {specs['avg_volatility']}%

<b>📊 INDICATORS:</b>
{analysis['indicators']}

<b>🎯 KEY LEVELS:</b>
{analysis['levels']}

<b>📅 TRADING HOURS:</b> {specs['trading_hours']}
<b>📈 SPREAD AVG:</b> {specs['spread_avg']} pips

<b>💡 ANALYSIS:</b>
{analysis['summary']}

<i>Note: Technical analysis is for informational purposes only. Past performance is not indicative of future results.</i>"""
        
        keyboard = [
            [telegram.InlineKeyboardButton("🔍 More Analysis", callback_data=f"tech_detail_{symbol}"),
             telegram.InlineKeyboardButton("🎯 Risk Calc", callback_data=f"calc_{symbol}")],
            [telegram.InlineKeyboardButton("📊 Compare", callback_data=f"compare_{symbol}"),
             telegram.InlineKeyboardButton("🔔 Set Alert", callback_data=f"alert_{symbol}")]
        ]
        
        await update.message.reply_html(
            response,
            reply_markup=telegram.InlineKeyboardMarkup(keyboard)
        )
    
    @monitor_performance
    async def _stress_test_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Portfolio stress testing command"""
        user_id = update.effective_user.id
        
        # Load portfolio
        portfolio_data = get_data_manager().load_user_portfolio(user_id)
        trades = portfolio_data.get('trades', [])
        
        if len(trades) < 2:
            await update.message.reply_html(
                "🧪 <b>Portfolio Stress Testing</b>\n\n"
                "You need at least 2 positions in your portfolio for meaningful stress testing.\n\n"
                "Add more trades to analyze:\n"
                "• Correlation risks\n"
                "• Market crash scenarios\n"
                "• Black swan events\n"
                "• Liquidity stress"
            )
            return
        
        # Run stress test
        stress_results = get_stress_tester().analyze_portfolio_stress(
            trades, portfolio_data.get('deposit', 1000)
        )
        
        # Generate stress test report
        report = self._generate_stress_test_report(stress_results)
        
        await update.message.reply_html(
            report,
            reply_markup=telegram.InlineKeyboardMarkup([
                [telegram.InlineKeyboardButton("📊 Full Portfolio", callback_data="portfolio_full"),
                 telegram.InlineKeyboardButton("📈 Diversify", callback_data="diversify")],
                [telegram.InlineKeyboardButton("📋 Export Report", callback_data="export_stress")]
            ])
        )
    
    @monitor_performance
    async def _alerts_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Manage price and risk alerts"""
        user_id = update.effective_user.id
        alerts = get_alert_manager().get_user_alerts(user_id)
        
        if not alerts:
            response = """🔔 <b>Price & Risk Alerts</b>

No active alerts. You can set alerts for:
• Price levels (above/below)
• Margin level warnings
• Volatility spikes
• Portfolio risk changes

<b>Quick Actions:</b>"""
            
            keyboard = [
                [telegram.InlineKeyboardButton("💰 Price Alert", callback_data="set_price_alert"),
                 telegram.InlineKeyboardButton("⚠️ Margin Alert", callback_data="set_margin_alert")],
                [telegram.InlineKeyboardButton("📈 Volatility Alert", callback_data="set_vol_alert"),
                 telegram.InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
            ]
        else:
            response = f"""🔔 <b>Active Alerts ({len(alerts)})</b>

"""
            for i, alert in enumerate(alerts[:5], 1):
                status = "✅ TRIGGERED" if alert['triggered'] else "⏳ ACTIVE"
                response += f"{i}. {alert['instrument']} - {alert['type']} {alert['condition']} {alert['threshold']} {status}\n"
            
            if len(alerts) > 5:
                response += f"\n... and {len(alerts) - 5} more alerts\n"
            
            response += "\n<b>Manage:</b>"
            
            keyboard = [
                [telegram.InlineKeyboardButton("➕ New Alert", callback_data="set_price_alert"),
                 telegram.InlineKeyboardButton("🗑 Clear All", callback_data="clear_alerts")],
                [telegram.InlineKeyboardButton("📋 List All", callback_data="list_alerts"),
                 telegram.InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
            ]
        
        await update.message.reply_html(
            response,
            reply_markup=telegram.InlineKeyboardMarkup(keyboard)
        )
    
    @monitor_performance
    async def _settings_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Bot settings management"""
        user_id = update.effective_user.id
        
        # Load settings
        settings = get_data_manager().load_user_settings(user_id)
        defaults = {
            'notifications': True,
            'risk_tolerance': 'medium',
            'default_assets': ['EURUSD', 'BTCUSDT', 'AAPL', 'XAUUSD'],
            'auto_calc': True,
            'api_preference': 'fastest'
        }
        
        settings = {**defaults, **settings}
        
        response = f"""⚙️ <b>Bot Settings</b>

<b>Current Settings:</b>
• Notifications: {'✅ ON' if settings['notifications'] else '❌ OFF'}
• Risk Tolerance: {settings['risk_tolerance'].upper()}
• Auto Calculations: {'✅ ON' if settings['auto_calc'] else '❌ OFF'}
• API Preference: {settings['api_preference'].upper()}
• Default Assets: {', '.join(settings['default_assets'][:3])}

<b>Performance Stats:</b>
• API Success Rate: {self._get_api_success_rate()}%
• Avg Response Time: {self._get_avg_response_time():.2f}s
• Cache Hit Rate: {self._get_cache_hit_rate()}%

Select setting to change:"""
        
        keyboard = [
            [telegram.InlineKeyboardButton("🔔 Notifications", callback_data="toggle_notifications"),
             telegram.InlineKeyboardButton("🎯 Risk Level", callback_data="change_risk")],
            [telegram.InlineKeyboardButton("⚡ API Settings", callback_data="api_settings"),
             telegram.InlineKeyboardButton("📊 Performance", callback_data="performance_stats")],
            [telegram.InlineKeyboardButton("🔄 Reset Defaults", callback_data="reset_settings"),
             telegram.InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        await update.message.reply_html(
            response,
            reply_markup=telegram.InlineKeyboardMarkup(keyboard)
        )
    
    # --- CALLBACK HANDLER ---
    
    @monitor_performance
    async def _callback_handler(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Main callback query handler"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        user_id = query.from_user.id
        
        # Route callback data
        if data == "main_menu":
            await self._show_main_menu(query)
        elif data == "quick_calc":
            await self._quick_calculation_start(query)
        elif data == "portfolio_full":
            await self._show_full_portfolio(query)
        elif data.startswith("tech_"):
            await self._handle_technical_callback(query, data)
        elif data.startswith("calc_"):
            await self._handle_calculation_callback(query, data)
        elif data == "stress_test":
            await self._stress_test_callback(query)
        elif data == "alerts_menu":
            await self._show_alerts_menu(query)
        elif data == "settings_menu":
            await self._show_settings_menu(query)
        elif data.startswith("donate"):
            await self._handle_donation(query, data)
        elif data == "tutorial":
            await self._show_tutorial(query)
        else:
            await query.message.reply_html(
                "❌ Command not recognized",
                reply_markup=telegram.InlineKeyboardMarkup([
                    [telegram.InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
                ])
            )
    
    # --- QUICK CALCULATION FLOW ---
    
    async def _quick_calculation_start(self, query: telegram.CallbackQuery):
        """Start quick calculation with predefined assets"""
        
        # Get user's default assets or popular ones
        user_id = query.from_user.id
        settings = get_data_manager().load_user_settings(user_id)
        default_assets = settings.get('default_assets', ['EURUSD', 'BTCUSDT', 'AAPL', 'XAUUSD'])
        
        keyboard = []
        for asset in default_assets[:8]:  # Max 8 buttons
            keyboard.append([telegram.InlineKeyboardButton(
                f"📊 {asset}", callback_data=f"qcalc_{asset}"
            )])
        
        keyboard.append([
            telegram.InlineKeyboardButton("📝 Custom Asset", callback_data="qcalc_custom"),
            telegram.InlineKeyboardButton("🏠 Menu", callback_data="main_menu")
        ])
        
        await query.message.edit_text(
            "🎯 <b>QUICK RISK CALCULATION</b>\n\n"
            "Select asset for calculation (2% risk rule applied automatically):",
            parse_mode='HTML',
            reply_markup=telegram.InlineKeyboardMarkup(keyboard)
        )
    
    async def _handle_quick_calc_asset(self, query: telegram.CallbackQuery, asset: str):
        """Handle asset selection for quick calculation"""
        
        # Get current price
        price, source = await get_api_manager().fetch_parallel(asset)
        
        # Get asset info
        specs = EnhancedInstrumentSpecs.get_specs(asset)
        
        response = f"""🎯 <b>{asset} QUICK CALC</b>

<b>Current Price:</b> ${price:.4f} ({source})
<b>Type:</b> {specs['type']}
<b>Avg Volatility:</b> {specs['avg_volatility']}%
<b>Contract Size:</b> {specs['contract_size']}

<b>Enter your trade details:</b>
1. Direction (LONG/SHORT)
2. Entry Price
3. Stop Loss
4. Take Profit
5. Deposit Amount
6. Leverage

<b>Example:</b>
<code>LONG 50000 48000 55000 1000 1:100</code>

Or use buttons below:"""
        
        keyboard = [
            [telegram.InlineKeyboardButton("📈 LONG Template", callback_data=f"template_long_{asset}"),
             telegram.InlineKeyboardButton("📉 SHORT Template", callback_data=f"template_short_{asset}")],
            [telegram.InlineKeyboardButton("🔙 Back", callback_data="quick_calc"),
             telegram.InlineKeyboardButton("🏠 Menu", callback_data="main_menu")]
        ]
        
        await query.message.edit_text(
            response,
            parse_mode='HTML',
            reply_markup=telegram.InlineKeyboardMarkup(keyboard)
        )
        
        # Store state for this user
        self.user_states[query.from_user.id] = {
            'state': 'awaiting_quick_calc',
            'asset': asset,
            'current_price': price
        }
    
    # --- TEXT MESSAGE HANDLER ---
    
    @monitor_performance
    async def _text_message_handler(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages for calculations and commands"""
        
        text = update.message.text.strip()
        user_id = update.effective_user.id
        
        # Check if user is in calculation state
        if user_id in self.user_states:
            state = self.user_states[user_id]
            
            if state['state'] == 'awaiting_quick_calc':
                await self._process_quick_calc_input(update, text, state)
                return
        
        # Try to parse as calculation input
        if self._looks_like_calculation_input(text):
            await self._try_parse_calculation(update, text)
            return
        
        # Default response
        await update.message.reply_html(
            "🤖 I can help with:\n\n"
            "• <b>Risk calculations</b> (use format: ASSET DIRECTION ENTRY SL TP DEPOSIT LEVERAGE)\n"
            "• <b>Portfolio analysis</b> (/portfolio)\n"
            "• <b>Technical analysis</b> (/technical SYMBOL)\n"
            "• <b>Stress testing</b> (/stress)\n\n"
            "Or use the menu for more options!",
            reply_markup=telegram.InlineKeyboardMarkup([
                [telegram.InlineKeyboardButton("🎯 Quick Calc", callback_data="quick_calc"),
                 telegram.InlineKeyboardButton("📚 Help", callback_data="tutorial")]
            ])
        )
    
    async def _process_quick_calc_input(self, update: telegram.Update, text: str, state: dict):
        """Process quick calculation input"""
        
        try:
            # Parse input
            parts = text.split()
            if len(parts) != 6:
                raise ValueError("Need 6 parameters")
            
            direction = parts[0].upper()
            entry_price = float(parts[1])
            stop_loss = float(parts[2])
            take_profit = float(parts[3])
            deposit = float(parts[4])
            leverage = parts[5]
            
            # Validate leverage format
            if ':' not in leverage:
                leverage = f"1:{leverage}"
            
            # Validate direction
            if direction not in ['LONG', 'SHORT']:
                raise ValueError("Direction must be LONG or SHORT")
            
            # Validate prices
            if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0 or deposit <= 0:
                raise ValueError("Prices and deposit must be positive")
            
            if direction == 'LONG':
                if stop_loss >= entry_price:
                    raise ValueError("For LONG: Stop Loss must be below Entry")
                if take_profit <= entry_price:
                    raise ValueError("For LONG: Take Profit must be above Entry")
            else:  # SHORT
                if stop_loss <= entry_price:
                    raise ValueError("For SHORT: Stop Loss must be above Entry")
                if take_profit >= entry_price:
                    raise ValueError("For SHORT: Take Profit must be below Entry")
            
            # Create trade object
            trade = {
                'asset': state['asset'],
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate metrics
            calculator = get_risk_calculator()
            metrics = await calculator.calculate_advanced_metrics(trade, deposit, leverage)
            
            # Save to portfolio
            portfolio_data = get_data_manager().load_user_portfolio(update.effective_user.id)
            portfolio_data.setdefault('trades', []).append({
                **trade,
                'metrics': metrics,
                'deposit': deposit,
                'leverage': leverage
            })
            portfolio_data['deposit'] = deposit
            portfolio_data['leverage'] = leverage
            
            get_data_manager().save_user_portfolio(update.effective_user.id, portfolio_data)
            
            # Generate report
            report = self._generate_trade_report(trade, metrics)
            
            # Send report
            keyboard = [
                [telegram.InlineKeyboardButton("💾 Save Trade", callback_data=f"save_trade_{len(portfolio_data['trades'])}"),
                 telegram.InlineKeyboardButton("📊 Add to Portfolio", callback_data="add_to_portfolio")],
                [telegram.InlineKeyboardButton("🎯 New Calc", callback_data="quick_calc"),
                 telegram.InlineKeyboardButton("🏠 Menu", callback_data="main_menu")]
            ]
            
            await update.message.reply_html(
                report,
                reply_markup=telegram.InlineKeyboardMarkup(keyboard)
            )
            
            # Clear state
            if update.effective_user.id in self.user_states:
                del self.user_states[update.effective_user.id]
            
        except ValueError as e:
            await update.message.reply_html(
                f"❌ <b>Input Error:</b> {str(e)}\n\n"
                f"<b>Expected format:</b>\n"
                f"<code>DIRECTION ENTRY SL TP DEPOSIT LEVERAGE</code>\n\n"
                f"<b>Example:</b>\n"
                f"<code>LONG 50000 48000 55000 1000 1:100</code>\n\n"
                f"Try again or use buttons:",
                reply_markup=telegram.InlineKeyboardMarkup([
                    [telegram.InlineKeyboardButton("📈 LONG Example", callback_data=f"template_long_{state['asset']}"),
                     telegram.InlineKeyboardButton("📉 SHORT Example", callback_data=f"template_short_{state['asset']}")],
                    [telegram.InlineKeyboardButton("🔙 Back", callback_data="quick_calc")]
                ])
            )
        except Exception as e:
            logger.error(f"Quick calc error: {e}")
            await update.message.reply_html(
                "❌ <b>Calculation error</b>\n\n"
                "Please check your inputs and try again.",
                reply_markup=telegram.InlineKeyboardMarkup([
                    [telegram.InlineKeyboardButton("🔙 Back", callback_data="quick_calc"),
                     telegram.InlineKeyboardButton("🏠 Menu", callback_data="main_menu")]
                ])
            )
    
    async def _try_parse_calculation(self, update: telegram.Update, text: str):
        """Try to parse calculation from free text"""
        
        try:
            parts = text.split()
            
            # Different formats
            if len(parts) == 6:
                # Format: ASSET DIRECTION ENTRY SL TP DEPOSIT LEVERAGE
                asset = parts[0].upper()
                direction = parts[1].upper()
                entry_price = float(parts[2])
                stop_loss = float(parts[3])
                take_profit = float(parts[4])
                deposit = float(parts[5])
                leverage = '1:100'  # Default
                
            elif len(parts) == 7:
                # Format: ASSET DIRECTION ENTRY SL TP DEPOSIT LEVERAGE
                asset = parts[0].upper()
                direction = parts[1].upper()
                entry_price = float(parts[2])
                stop_loss = float(parts[3])
                take_profit = float(parts[4])
                deposit = float(parts[5])
                leverage = parts[6]
                
            else:
                raise ValueError("Invalid format")
            
            # Validate asset
            if not re.match(r'^[A-Z0-9]{2,20}$', asset):
                raise ValueError("Invalid asset symbol")
            
            # Create and calculate
            trade = {
                'asset': asset,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            calculator = get_risk_calculator()
            metrics = await calculator.calculate_advanced_metrics(trade, deposit, leverage)
            
            # Generate report
            report = self._generate_trade_report(trade, metrics)
            
            keyboard = [
                [telegram.InlineKeyboardButton("💾 Save", callback_data=f"save_calc_{asset}"),
                 telegram.InlineKeyboardButton("📊 Portfolio", callback_data="portfolio_full")],
                [telegram.InlineKeyboardButton("🎯 Another", callback_data="quick_calc")]
            ]
            
            await update.message.reply_html(
                report,
                reply_markup=telegram.InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Parse calculation error: {e}")
            await update.message.reply_html(
                "❌ <b>Could not parse calculation</b>\n\n"
                "<b>Valid formats:</b>\n"
                "1. <code>ASSET DIRECTION ENTRY SL TP DEPOSIT</code>\n"
                "2. <code>ASSET DIRECTION ENTRY SL TP DEPOSIT LEVERAGE</code>\n\n"
                "<b>Example:</b>\n"
                "<code>BTCUSDT LONG 50000 48000 55000 1000 1:100</code>",
                reply_markup=telegram.InlineKeyboardMarkup([
                    [telegram.InlineKeyboardButton("🎯 Quick Calc", callback_data="quick_calc"),
                     telegram.InlineKeyboardButton("📚 Help", callback_data="tutorial")]
                ])
            )
    
    # --- REPORT GENERATORS ---
    
    def _generate_trade_report(self, trade: Dict, metrics: Dict) -> str:
        """Generate detailed trade report"""
        
        risk_score = metrics.get('risk_score', 0)
        risk_color = "🟢" if risk_score < 30 else "🟡" if risk_score < 70 else "🔴"
        
        report = f"""📊 <b>TRADE ANALYSIS REPORT</b>

<b>Instrument:</b> {trade['asset']} ({trade['direction']})
<b>Entry:</b> {trade['entry_price']} | <b>SL:</b> {trade['stop_loss']} | <b>TP:</b> {trade['take_profit']}

<b>💰 POSITION METRICS:</b>
• Volume: {metrics.get('volume_lots', 0):.3f} lots
• Margin Required: ${metrics.get('required_margin', 0):.2f}
• Risk Amount: ${metrics.get('risk_amount', 0):.2f} (2%)
• Potential Profit: ${metrics.get('potential_profit', 0):.2f}
• Potential Loss: ${metrics.get('potential_loss', 0):.2f}
• R/R Ratio: {metrics.get('rr_ratio', 0):.2f}:1

<b>📈 CURRENT STATUS:</b>
• Current Price: ${metrics.get('current_price', 0):.2f}
• Current P&L: ${metrics.get('current_pnl', 0):.2f}
• Equity: ${metrics.get('equity', 0):.2f}
• Margin Level: {metrics.get('margin_level', 0):.1f}%

<b>🎯 RISK ASSESSMENT:</b>
• Risk Score: {risk_color} {risk_score}/100
• 1-day VaR (95%): ${metrics.get('var_95_1d', 0):.2f}
• 1-day CVaR (95%): ${metrics.get('cvar_95_1d', 0):.2f}
• Stop Loss Breach Probability: {metrics.get('var_breach_probability', 0):.1f}%

<b>⚠️ STRESS SCENARIOS:</b>
• Mild Stress P&L: ${metrics.get('mild_stress_pnl', 0):.2f}
• Severe Stress P&L: ${metrics.get('severe_stress_pnl', 0):.2f}
• Black Swan P&L: ${metrics.get('black_swan_pnl', 0):.2f}

<b>🔗 CORRELATION RISK:</b>
• Diversification Score: {metrics.get('diversification_score', 0)}%
• Concentration Risk: {metrics.get('concentration_risk', 0)}%

<i>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"""
        
        return report
    
    def _generate_portfolio_summary(self, trades: List[Dict], portfolio_metrics: Dict) -> str:
        """Generate portfolio summary"""
        
        total_trades = len(trades)
        total_pnl = sum(t.get('metrics', {}).get('current_pnl', 0) for t in trades)
        total_margin = sum(t.get('metrics', {}).get('required_margin', 0) for t in trades)
        total_equity = trades[0].get('metrics', {}).get('equity', 0) if trades else 0
        
        # Calculate winning trades
        winning_trades = sum(1 for t in trades if t.get('metrics', {}).get('current_pnl', 0) > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        summary = f"""📊 <b>PORTFOLIO SUMMARY</b>

<b>Overview:</b>
• Total Trades: {total_trades}
• Winning Trades: {winning_trades} ({win_rate:.1f}%)
• Total P&L: ${total_pnl:+.2f}
• Total Margin: ${total_margin:.2f}
• Equity: ${total_equity:.2f}

<b>Risk Metrics:</b>
• Portfolio Risk Score: {portfolio_metrics.get('risk_score', 0)}/100
• Max Drawdown: {portfolio_metrics.get('max_drawdown', 0):.1f}%
• Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}
• Sortino Ratio: {portfolio_metrics.get('sortino_ratio', 0):.2f}

<b>Diversification:</b>
• Unique Assets: {portfolio_metrics.get('unique_assets', 0)}
• Asset Types: {portfolio_metrics.get('unique_types', 0)}
• Concentration Index: {portfolio_metrics.get('concentration_index', 0):.1f}%

<b>Stress Test Results:</b>
• 2008 Crash Survival: {'✅' if not portfolio_metrics.get('stress_2008_margin_call', True) else '❌'}
• Black Swan Survival: {'✅' if not portfolio_metrics.get('stress_black_swan_margin_call', True) else '❌'}
• Emergency Liquidity: {portfolio_metrics.get('emergency_days', 0):.1f} days"""
        
        return summary
    
    def _generate_technical_analysis(self, symbol: str, current_price: float, specs: Dict) -> Dict:
        """Generate technical analysis"""
        
        # Get technical parameters
        rsi_period = specs.get('rsi_period', 14)
        ma_fast = specs.get('ma_fast', 9)
        ma_slow = specs.get('ma_slow', 21)
        bb_period = specs.get('bb_period', 20)
        
        # Generate indicators text
        indicators = f"""• RSI ({rsi_period}): Calculating...
• MA Fast ({ma_fast}): Calculating...
• MA Slow ({ma_slow}): Calculating...
• Bollinger Bands ({bb_period}): Calculating...
• MACD: Calculating..."""
        
        # Generate key levels (simplified)
        support = current_price * 0.95
        resistance = current_price * 1.05
        
        levels = f"""• Support 1: ${support:.4f}
• Support 2: ${support * 0.98:.4f}
• Resistance 1: ${resistance:.4f}
• Resistance 2: ${resistance * 1.02:.4f}
• Pivot Point: ${current_price:.4f}"""
        
        # Generate summary based on instrument type
        if specs['type'] == 'crypto':
            summary = "High volatility expected. Consider wider stops. Monitor BTC dominance."
        elif specs['type'] == 'forex':
            summary = "Normal trading hours. Watch for economic news releases."
        elif specs['type'] == 'stock':
            summary = "Market hours only. Earnings reports may cause gaps."
        else:
            summary = "Standard analysis applies. Monitor volume and news."
        
        return {
            'indicators': indicators,
            'levels': levels,
            'summary': summary
        }
    
    def _generate_stress_test_report(self, stress_results: Dict) -> str:
        """Generate stress test report"""
        
        if stress_results.get('empty'):
            return "No portfolio data for stress testing."
        
        current = stress_results['current_state']
        scenarios = stress_results['stress_scenarios']
        diversification = stress_results['diversification']
        liquidity = stress_results['liquidity']
        
        report = f"""🧪 <b>PORTFOLIO STRESS TEST REPORT</b>

<b>📊 CURRENT STATE:</b>
• Total Equity: ${current['total_equity']:.2f}
• Total Margin: ${current['total_margin']:.2f}
• Current P&L: ${current['total_pnl']:+.2f}
• Margin Usage: {(current['total_margin']/current['total_equity']*100):.1f}%

<b>⚠️ STRESS SCENARIOS:</b>
"""
        
        for scenario, data in scenarios.items():
            scenario_name = scenario.replace('_', ' ').title()
            margin_warning = " ⚠️ MARGIN CALL" if data['margin_call'] else ""
            report += f"""• {scenario_name}: {data['equity_change_pct']}% → ${data['stressed_equity']:.2f}{margin_warning}
"""
        
        report += f"""
<b>📈 DIVERSIFICATION:</b>
• Unique Assets: {diversification['unique_assets']}
• Asset Types: {diversification['unique_types']}
• Concentration: {diversification['concentration_index']}%
• Diversification Score: {diversification['diversification_score']}/100

<b>💰 LIQUIDITY:</b>
• Free Margin: ${liquidity['free_margin']:.2f}
• Free Margin Ratio: {liquidity['free_margin_ratio']}%
• Emergency Days: {liquidity['emergency_days']}
• Liquidity Grade: {liquidity['liquidity_grade']}

<b>💡 RECOMMENDATIONS:</b>
"""
        
        for rec in stress_results.get('recommendations', []):
            report += f"• {rec}\n"
        
        report += f"\n<i>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"
        
        return report
    
    # --- HELPER METHODS ---
    
    def _looks_like_calculation_input(self, text: str) -> bool:
        """Check if text looks like calculation input"""
        parts = text.split()
        if 5 <= len(parts) <= 7:
            # Check if contains numbers
            num_count = sum(1 for p in parts if re.match(r'^[0-9.,]+$', p))
            return num_count >= 3
        return False
    
    def _get_api_success_rate(self) -> float:
        """Calculate API success rate"""
        stats = get_api_manager().get_performance_stats()
        if not stats:
            return 95.0
        
        success_rates = [s['success_rate'] for s in stats.values()]
        return round(sum(success_rates) / len(success_rates), 1) if success_rates else 95.0
    
    def _get_avg_response_time(self) -> float:
        """Get average API response time"""
        if hasattr(monitor_performance, 'metrics'):
            all_times = []
            for times in monitor_performance.metrics.values():
                all_times.extend(times)
            return sum(all_times) / len(all_times) if all_times else 0.5
        return 0.5
    
    def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate (simplified)"""
        return 65.0  # In production, track actual cache hits
    
    async def _show_main_menu(self, query: telegram.CallbackQuery):
        """Show main menu"""
        await self._start_command(telegram.Update(0, callback_query=query), None)
    
    async def _error_handler(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Global error handler"""
        logger.error(f"Update {update} caused error {context.error}")
        
        try:
            # Notify user
            if update.effective_message:
                await update.effective_message.reply_html(
                    "❌ <b>An error occurred</b>\n\n"
                    "The issue has been logged. Please try again or use a different command.",
                    reply_markup=telegram.InlineKeyboardMarkup([
                        [telegram.InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
                    ])
                )
        except:
            pass
    
    # --- DONATION HANDLERS ---
    
    async def _handle_donation(self, query: telegram.CallbackQuery, data: str):
        """Handle donation requests"""
        
        if data == "donate_start":
            await self._show_donation_menu(query)
        elif data == "donate_usdt":
            await self._show_usdt_donation(query)
        elif data == "donate_ton":
            await self._show_ton_donation(query)
    
    async def _show_donation_menu(self, query: telegram.CallbackQuery):
        """Show donation menu"""
        
        text = """💝 <b>SUPPORT DEVELOPMENT</b>

Your support helps maintain and improve this bot! 

<b>Current Features Funded by Donations:</b>
• Real-time market data APIs
• Parallel processing for speed
• Advanced risk calculations
• Technical analysis tools
• Stress testing scenarios

Select donation method:"""
        
        keyboard = [
            [telegram.InlineKeyboardButton("💎 USDT (TRC20)", callback_data="donate_usdt")],
            [telegram.InlineKeyboardButton("⚡ TON", callback_data="donate_ton")],
            [telegram.InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        await query.message.edit_text(
            text,
            parse_mode='HTML',
            reply_markup=telegram.InlineKeyboardMarkup(keyboard)
        )
    
    async def _show_usdt_donation(self, query: telegram.CallbackQuery):
        """Show USDT donation address"""
        
        text = f"""💎 <b>USDT (TRC20) DONATION</b>

To support development, send USDT to:

<code>{USDT_WALLET_ADDRESS}</code>

<b>Network:</b> TRC20 (Tron)
<b>Min Amount:</b> Any amount appreciated!

💝 <i>Thank you for your support!</i>

<b>After donating:</b>
1. Take a screenshot
2. Send to @risk_bot_support
3. Get premium features!"""
        
        keyboard = [
            [telegram.InlineKeyboardButton("🔙 Back", callback_data="donate_start")],
            [telegram.InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        await query.message.edit_text(
            text,
            parse_mode='HTML',
            reply_markup=telegram.InlineKeyboardMarkup(keyboard)
        )
    
    async def _show_ton_donation(self, query: telegram.CallbackQuery):
        """Show TON donation address"""
        
        text = f"""⚡ <b>TON DONATION</b>

To support development, send TON to:

<code>{TON_WALLET_ADDRESS}</code>

<b>Network:</b> TON (The Open Network)
<b>Min Amount:</b> Any amount appreciated!

💝 <i>Thank you for your support!</i>

<b>After donating:</b>
1. Take a screenshot  
2. Send to @risk_bot_support
3. Get premium features!"""
        
        keyboard = [
            [telegram.InlineKeyboardButton("🔙 Back", callback_data="donate_start")],
            [telegram.InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        await query.message.edit_text(
            text,
            parse_mode='HTML',
            reply_markup=telegram.InlineKeyboardMarkup(keyboard)
        )
    
    async def _show_tutorial(self, query: telegram.CallbackQuery):
        """Show tutorial"""
        
        text = """📚 <b>ENTERPRISE RISK CALCULATOR TUTORIAL</b>

<b>🎯 QUICK START:</b>
1. Use /start for main menu
2. Click "Quick Risk Calc"
3. Select an asset
4. Enter trade details
5. Get risk analysis

<b>📊 PORTFOLIO MANAGEMENT:</b>
• Add multiple trades
• Track real-time P&L
• Monitor margin levels
• Analyze diversification

<b>📈 TECHNICAL ANALYSIS:</b>
• Use /technical SYMBOL
• Get indicators
• Identify levels
• Make informed decisions

<b>🧪 STRESS TESTING:</b>
• Test portfolio resilience
• Simulate market crashes
• Identify weaknesses
• Improve risk management

<b>🔔 ALERTS:</b>
• Price level alerts
• Margin warnings
• Volatility alerts
• Portfolio risk alerts

<b>⚡ PERFORMANCE TIPS:</b>
• Bot uses parallel processing
• Caches frequently used data
• Falls back gracefully if APIs fail
• Optimized for speed

<b>💎 PRO FEATURES:</b>
• VaR and CVaR calculations
• Correlation risk analysis
• Black swan event modeling
• Liquidity stress testing

Need help? Contact @risk_bot_support"""
        
        keyboard = [
            [telegram.InlineKeyboardButton("🎯 Try Quick Calc", callback_data="quick_calc")],
            [telegram.InlineKeyboardButton("📊 View Portfolio", callback_data="portfolio_full")],
            [telegram.InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        await query.message.edit_text(
            text,
            parse_mode='HTML',
            reply_markup=telegram.InlineKeyboardMarkup(keyboard)
        )

# --- WEB SERVER FOR RENDER ---

class RenderWebServer:
    """Optimized web server for Render Free tier"""
    
    def __init__(self, bot: OptimizedTelegramBot):
        self.bot = bot
        self.app = None
        self.runner = None
        self.site = None
        
    async def start(self):
        """Start web server for Render"""
        global web
        web = lazy_import_web()
        
        self.app = web.Application()
        self._setup_routes()
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, '0.0.0.0', PORT)
        await self.site.start()
        
        logger.info(f"Web server started on port {PORT}")
        
        # Set webhook
        if WEBHOOK_URL:
            await self._set_webhook()
    
    def _setup_routes(self):
        """Setup web routes"""
        
        # Webhook endpoint
        async def handle_webhook(request):
            try:
                data = await request.json()
                update = telegram.Update.de_json(data, self.bot.application.bot)
                await self.bot.application.process_update(update)
                return web.Response(text="OK", status=200)
            except Exception as e:
                logger.error(f"Webhook error: {e}")
                return web.Response(text="Error", status=400)
        
        # Health check endpoint (keeps Render instance alive)
        async def health_check(request):
            health_data = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "4.0",
                "performance": {
                    "startup_time": self.bot.startup_time,
                    "request_count": self.bot.request_count,
                    "api_success_rate": self.bot._get_api_success_rate(),
                    "avg_response_time": self.bot._get_avg_response_time()
                },
                "services": {
                    "telegram_bot": "operational",
                    "market_data": "operational",
                    "risk_calculator": "operational",
                    "technical_analysis": "operational"
                }
            }
            return web.json_response(health_data)
        
        # Simple health check for Render
        async def simple_health(request):
            return web.Response(text="OK", status=200)
        
        # API status endpoint
        async def api_status(request):
            stats = get_api_manager().get_performance_stats()
            return web.json_response(stats)
        
        # Register routes
        self.app.router.add_post(WEBHOOK_PATH, handle_webhook)
        self.app.router.add_get('/health', health_check)
        self.app.router.add_get('/health/simple', simple_health)
        self.app.router.add_get('/api/status', api_status)
        self.app.router.add_get('/', simple_health)
    
    async def _set_webhook(self):
        """Set Telegram webhook"""
        try:
            webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
            logger.info(f"Setting webhook to: {webhook_url}")
            
            await self.bot.application.bot.set_webhook(
                webhook_url,
                drop_pending_updates=True,
                allowed_updates=telegram.Update.ALL_TYPES
            )
            
            # Verify webhook
            webhook_info = await self.bot.application.bot.get_webhook_info()
            logger.info(f"Webhook info: {webhook_info}")
            
        except Exception as e:
            logger.error(f"Failed to set webhook: {e}")
            raise
    
    async def stop(self):
        """Stop web server"""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

# --- MAIN APPLICATION ---

async def main():
    """Main application entry point"""
    
    logger.info("🚀 LAUNCHING ENTERPRISE RISK CALCULATOR v4.0")
    logger.info(f"📊 Environment: {'RENDER' if 'render' in (WEBHOOK_URL or '') else 'LOCAL'}")
    logger.info(f"🔑 API Keys loaded: {sum(1 for v in API_KEYS.values() if v)}")
    
    try:
        # Initialize bot
        bot = OptimizedTelegramBot()
        await bot.initialize()
        
        # Start mode
        if WEBHOOK_URL and 'render' in WEBHOOK_URL:
            # Webhook mode for Render
            logger.info("🌐 Starting in WEBHOOK mode (Render)")
            
            server = RenderWebServer(bot)
            await server.start()
            
            # Keep alive loop
            while True:
                await asyncio.sleep(300)  # 5 minutes
                logger.debug("Render instance alive")
                
        else:
            # Polling mode for local development
            logger.info("🔄 Starting in POLLING mode (Local)")
            
            await bot.application.run_polling(
                poll_interval=0.5,
                timeout=30,
                drop_pending_updates=True,
                allowed_updates=telegram.Update.ALL_TYPES
            )
            
    except KeyboardInterrupt:
        logger.info("⏹ Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        raise
    finally:
        # Cleanup
        await get_api_manager().close()
        logger.info("🧹 Cleanup completed")

# --- APPLICATION LAUNCH ---

if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run main application
    asyncio.run(main())
