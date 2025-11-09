# bot_en.py — PRO Risk Calculator v3.0 | ENTERPRISE EDITION - BUGS FIXED
import os
import logging
import asyncio
import time
import functools
import json
import telegram
import io
import re
import aiohttp
import cachetools
import html
from telegram import CallbackQuery
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
from aiohttp import web
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
from decimal import Decimal, ROUND_HALF_UP

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
logger = logging.getLogger("pro_risk_bot")

# ---------------------------
# Timeout and retry settings
# ---------------------------
class RobustApplicationBuilder:
    """Application builder with enhanced error handling"""
    
    @staticmethod
    def create_application(token: str) -> Application:
        """Create application with stability settings"""
        # Configure request parameters
        request = telegram.request.HTTPXRequest(
            connection_pool_size=8,
        )
        
        # Create application with settings
        application = (
            Application.builder()
            .token(token)
            .request(request)
            .build()
        )
        
        return application

# ---------------------------
# Retry Decorator for timeout handling
# ---------------------------
def retry_on_timeout(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying on timeouts"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except telegram.error.TimedOut as e:
                    logger.warning(f"Timeout in {func.__name__}, attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"All retries failed for {func.__name__}")
                        raise
                except telegram.error.NetworkError as e:
                    logger.warning(f"Network error in {func.__name__}, attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))
                    else:
                        logger.error(f"All retries failed for {func.__name__}")
                        raise
            return None
        return wrapper
    return decorator

# ---------------------------
# Safe Message Sender - UPDATED WITH HTML ERROR PROTECTION
# ---------------------------
class SafeMessageSender:
    """Safe message sending with error handling"""
    
    @staticmethod
    def safe_html_text(text: str) -> str:
        """Safe HTML text preparation - IMPROVED VERSION"""
        # First escape all special characters
        text = html.escape(text)
        
        # Then allow only safe HTML tags
        safe_tags = ['b', 'i', 'u', 'em', 'strong', 'code', 'pre']
        
        for tag in safe_tags:
            # Restore allowed tags
            opening_tag = f"&lt;{tag}&gt;"
            closing_tag = f"&lt;/{tag}&gt;"
            text = text.replace(opening_tag, f"<{tag}>").replace(closing_tag, f"</{tag}>")
        
        # Remove multiple line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Truncate too long messages
        if len(text) > 4000:
            text = text[:4000] + "...\n\n[message truncated]"
            
        return text
    
    @staticmethod
    @retry_on_timeout(max_retries=3, delay=1.0)
    async def send_message(
        chat_id: int,
        text: str,
        context: ContextTypes.DEFAULT_TYPE = None,
        reply_markup: InlineKeyboardMarkup = None,
        parse_mode: str = 'HTML'
    ) -> bool:
        """Safe message sending with HTML error protection"""
        try:
            # Clean HTML text
            safe_text = SafeMessageSender.safe_html_text(text)
            
            if context and hasattr(context, 'bot'):
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=safe_text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode
                )
            else:
                # Fallback - create temporary bot
                from telegram import Bot
                bot = Bot(token=TOKEN)
                await bot.send_message(
                    chat_id=chat_id,
                    text=safe_text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode
                )
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")
            # Try to send without HTML
            try:
                plain_text = re.sub(r'<[^>]+>', '', text)
                if context and hasattr(context, 'bot'):
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=plain_text[:4000],
                        reply_markup=reply_markup
                    )
                return True
            except Exception as e2:
                logger.error(f"Failed to send plain text message: {e2}")
                return False
    
    @staticmethod
    @retry_on_timeout(max_retries=2, delay=1.0)
    async def edit_message_text(
        query: 'CallbackQuery',
        text: str,
        reply_markup: InlineKeyboardMarkup = None,
        parse_mode: str = 'HTML'
    ) -> bool:
        """Safe message editing with HTML error protection"""
        try:
            safe_text = SafeMessageSender.safe_html_text(text)
            
            await query.edit_message_text(
                text=safe_text,
                reply_markup=reply_markup,
                parse_mode=parse_mode
            )
            return True
        except telegram.error.BadRequest as e:
            if "Message is not modified" in str(e):
                # Message not changed - not an error
                return True
            logger.warning(f"BadRequest while editing message: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to edit message: {e}")
            return False
    
    @staticmethod
    async def answer_callback_query(
        query: CallbackQuery,
        text: str = None,
        show_alert: bool = False
    ) -> bool:
        """Safe callback query response"""
        try:
            await query.answer(text=text, show_alert=show_alert)
            return True
        except Exception as e:
            logger.error(f"Failed to answer callback query: {e}")
            return False

# ---------------------------
# Donation System - PROFESSIONAL DONATION SYSTEM
# ---------------------------
class DonationSystem:
    """Professional donation system to support development"""
    
    @staticmethod
    async def show_donation_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show currency selection menu for donations"""
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        text = (
            "💝 <b>SUPPORT THE DEVELOPER</b>\n\n"
            "Your support helps develop the bot and add new features!\n\n"
            "Choose donation currency:"
        )
        
        keyboard = [
            [InlineKeyboardButton("💎 USDT (TRC20)", callback_data="donate_usdt")],
            [InlineKeyboardButton("⚡ TON", callback_data="donate_ton")],
            [InlineKeyboardButton("🔙 Back", callback_data="main_menu")]
        ]
        
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup(keyboard)
        )
    
    @staticmethod
    async def show_usdt_donation(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show USDT wallet for donations"""
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        if not USDT_WALLET_ADDRESS:
            await SafeMessageSender.edit_message_text(
                query,
                "❌ USDT wallet temporarily unavailable",
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Back", callback_data="donate_start")]
                ])
            )
            return
        
        text = (
            "💎 <b>USDT (TRC20) DONATION</b>\n\n"
            "To support development, send USDT to the following address:\n\n"
            f"<code>{USDT_WALLET_ADDRESS}</code>\n\n"
            "💝 <i>Any amount will be gratefully accepted!</i>\n\n"
            "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔙 Back to currency selection", callback_data="donate_start")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup(keyboard)
        )
    
    @staticmethod
    async def show_ton_donation(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show TON wallet for donations"""
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        if not TON_WALLET_ADDRESS:
            await SafeMessageSender.edit_message_text(
                query,
                "❌ TON wallet temporarily unavailable",
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Back", callback_data="donate_start")]
                ])
            )
            return
        
        text = (
            "⚡ <b>TON DONATION</b>\n\n"
            "To support development, send TON to the following address:\n\n"
            f"<code>{TON_WALLET_ADDRESS}</code>\n\n"
            "💝 <i>Any amount will be gratefully accepted!</i>\n\n"
            "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔙 Back to currency selection", callback_data="donate_start")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup(keyboard)
        )

# ---------------------------
# Enhanced Market Data Provider - IMPROVED VERSION WITH NEW APIs
# ---------------------------
class EnhancedMarketDataProvider:
    """Universal market data provider with improved metals and new API support"""
    
    def __init__(self):
        self.cache = cachetools.TTLCache(maxsize=500, ttl=300)
        self.session = None
        
    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_real_time_price(self, symbol: str) -> float:
        """Get real-time price with provider prioritization"""
        return await self.get_robust_real_time_price(symbol)
    
    async def get_robust_real_time_price(self, symbol: str) -> float:
        """RELIABLE real price fetching with improved provider queue"""
        try:
            # Check cache
            cached_price = self.cache.get(symbol)
            if cached_price:
                return cached_price
            
            # Determine asset type and select provider
            providers = [
                self._get_fmp_price,               # Financial Modeling Prep - primary
                self._get_metalpriceapi_price,     # Metal Price API - for metals
                self._get_exchangerate_price,      # Forex
                self._get_binance_price,           # Crypto
                self._get_twelvedata_price,        # Stocks, indices
                self._get_alpha_vantage_stock,     # Stocks
                self._get_alpha_vantage_forex,     # Forex backup
                self._get_finnhub_price,           # General backup
                self._get_fallback_price           # Static data
            ]
            
            price = None
            for provider in providers:
                price = await provider(symbol)
                if price and price > 0:
                    break
            
            # Fallback to static data on errors
            if price is None or price <= 0:
                logger.warning(f"Failed to get price for {symbol}, using fallback")
                price = self._get_fallback_price(symbol)
                
            # Save to cache
            if price:
                self.cache[symbol] = price
                
            return price
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return self._get_fallback_price(symbol)
    
    def _is_crypto(self, symbol: str) -> bool:
        """Check if asset is cryptocurrency"""
        crypto_symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'USDT']
        return any(crypto in symbol for crypto in crypto_symbols)
    
    def _is_forex(self, symbol: str) -> bool:
        """Check if asset is Forex pair"""
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        return symbol in forex_pairs
    
    def _is_metal(self, symbol: str) -> bool:
        """Check if asset is metal"""
        metals = ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD']
        return symbol in metals
    
    async def _get_fmp_price(self, symbol: str) -> Optional[float]:
        """Get price via Financial Modeling Prep API"""
        try:
            session = await self.get_session()
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and isinstance(data, list) and len(data) > 0:
                        return data[0]['price']
        except Exception as e:
            logger.error(f"FMP API error for {symbol}: {e}")
        return None
    
    async def _get_metalpriceapi_price(self, symbol: str) -> Optional[float]:
        """Get metal prices via Metal Price API"""
        try:
            if not self._is_metal(symbol):
                return None
                
            session = await self.get_session()
            # Convert symbols for Metal Price API
            metal_map = {
                'XAUUSD': 'XAU',
                'XAGUSD': 'XAG', 
                'XPTUSD': 'XPT',
                'XPDUSD': 'XPD'
            }
            
            metal_code = metal_map.get(symbol)
            if not metal_code:
                return None
                
            url = f"https://api.metalpriceapi.com/v1/latest?api_key={METALPRICE_API_KEY}&base=USD&currencies={metal_code}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('success'):
                        rate = data['rates'].get(f"USD{metal_code}")  # Use direct price if available (e.g., 'USDXAU')
                        if rate:
                            return rate
                        rate = data['rates'].get(metal_code)  # Fallback to inverse
                        if rate:
                            return 1.0 / rate
                else:
                    logger.error(f"Metal Price API response error: {response.status}")
        except Exception as e:
            logger.error(f"Metal Price API error for {symbol}: {e}")
        return None
    
    async def _get_twelvedata_price(self, symbol: str) -> Optional[float]:
        """Get price via Twelve Data API"""
        if not TWELVEDATA_API_KEY:
            return None
            
        try:
            session = await self.get_session()
            url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVEDATA_API_KEY}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'price' in data and data['price'] != '':
                        return float(data['price'])
        except Exception as e:
            logger.error(f"Twelve Data API error for {symbol}: {e}")
        return None
    
    async def _get_exchangerate_price(self, symbol: str) -> Optional[float]:
        """Frankfurter API for accurate Forex prices"""
        try:
            if self._is_forex(symbol):
                # Convert EURUSD -> EUR/USD
                from_curr = symbol[:3]
                to_curr = symbol[3:]
                url = f"https://api.frankfurter.app/latest?from={from_curr}&to={to_curr}"
                
                session = await self.get_session()
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['rates'][to_curr]
        except Exception as e:
            logger.error(f"ExchangeRate API error for {symbol}: {e}")
        return None
    
    async def _get_binance_price(self, symbol: str) -> Optional[float]:
        """Get price from Binance API"""
        try:
            if not self._is_crypto(symbol):
                return None
                
            session = await self.get_session()
            # Format symbol for Binance
            binance_symbol = symbol.replace('USDT', '') + 'USDT'
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data['price'])
        except Exception as e:
            logger.error(f"Binance API error for {symbol}: {e}")
        return None
    
    async def _get_alpha_vantage_stock(self, symbol: str) -> Optional[float]:
        """Get stock prices from Alpha Vantage"""
        if not ALPHA_VANTAGE_API_KEY or self._is_forex(symbol) or self._is_crypto(symbol):
            return None
            
        try:
            session = await self.get_session()
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Global Quote' in data and '05. price' in data['Global Quote']:
                        return float(data['Global Quote']['05. price'])
        except Exception as e:
            logger.error(f"Alpha Vantage stock error for {symbol}: {e}")
        return None
    
    async def _get_alpha_vantage_forex(self, symbol: str) -> Optional[float]:
        """Get Forex prices from Alpha Vantage"""
        if not ALPHA_VANTAGE_API_KEY or not self._is_forex(symbol):
            return None
            
        try:
            session = await self.get_session()
            from_currency = symbol[:3]
            to_currency = symbol[3:]
            url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}&to_currency={to_currency}&apikey={ALPHA_VANTAGE_API_KEY}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Realtime Currency Exchange Rate' in data and '5. Exchange Rate' in data['Realtime Currency Exchange Rate']:
                        return float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])
        except Exception as e:
            logger.error(f"Alpha Vantage forex error for {symbol}: {e}")
        return None
    
    async def _get_finnhub_price(self, symbol: str) -> Optional[float]:
        """Get price from Finnhub (backup)"""
        if not FINNHUB_API_KEY:
            return None
            
        try:
            session = await self.get_session()
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['c']  # current price
        except Exception as e:
            logger.error(f"Finnhub API error for {symbol}: {e}")
        return None
    
    async def _get_fallback_price(self, symbol: str) -> float:
        """UPDATED fallback prices when APIs are unavailable (async version)"""
        current_prices = {
            # Forex (current prices)
            'EURUSD': 1.05, 'GBPUSD': 1.25, 'USDJPY': 150.00, 'USDCHF': 0.90,
            'AUDUSD': 0.65, 'USDCAD': 1.35, 'NZDUSD': 0.60,
            # Crypto (current prices for 2025)
            'BTCUSDT': 100000.0, 'ETHUSDT': 5000.0, 'XRPUSDT': 1.00, 'LTCUSDT': 150.00,
            'BCHUSDT': 600.00, 'ADAUSDT': 1.00, 'DOTUSDT': 10.00,
            # Stocks (current prices)
            'AAPL': 200.00, 'TSLA': 300.00, 'GOOGL': 150.00, 'MSFT': 400.00,
            'AMZN': 200.00, 'META': 500.00, 'NFLX': 600.00,
            # Indices (current prices)
            'NAS100': 20000.0, 'SPX500': 5500.0, 'DJ30': 40000.0, 'FTSE100': 8000.0,
            'DAX40': 19000.0, 'NIKKEI225': 40000.0, 'ASX200': 8000.0,
            # Metals (current prices)
            'XAUUSD': 2500.00, 'XAGUSD': 30.00, 'XPTUSD': 1000.00, 'XPDUSD': 1000.00,
            # Energy (current prices)
            'OIL': 80.00, 'NATURALGAS': 3.00, 'BRENT': 85.00
        }
        return current_prices.get(symbol, 100.0)

    async def get_price_with_fallback(self, symbol: str) -> Tuple[float, str]:
        """Get price with source information"""
        try:
            # First try to get real price
            real_price = await self.get_robust_real_time_price(symbol)
            if real_price and real_price > 0:
                return real_price, "real-time"
            
            # Then use cache
            cached_price = self.cache.get(symbol)
            if cached_price:
                return cached_price, "cached"
            
            # Only then fallback
            fallback_price = self._get_fallback_price(symbol)
            return fallback_price, "fallback"
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return self._get_fallback_price(symbol), "error"

# ---------------------------
# Instrument Specifications - FIXED SPECIFICATIONS BASE
# ---------------------------
class InstrumentSpecs:
    """Fixed base of financial instrument specifications"""
    
    SPECS = {
        # Forex pairs - FIXED VALUES
        "EURUSD": {
            "type": "forex",
            "contract_size": 100000,  # 1 standard lot
            "margin_currency": "USD",
            "pip_value": 10.0,       # 1 pip = $10 for standard lot
            "calculation_formula": "forex",
            "pip_decimal_places": 4,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 1000
        },
        "GBPUSD": {
            "type": "forex",
            "contract_size": 100000,
            "margin_currency": "USD", 
            "pip_value": 10.0,
            "calculation_formula": "forex",
            "pip_decimal_places": 4,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 1000
        },
        "USDJPY": {
            "type": "forex", 
            "contract_size": 100000,
            "margin_currency": "USD",
            "pip_value": 9.09,       # JPY pairs feature
            "calculation_formula": "forex_jpy",
            "pip_decimal_places": 2,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 1000
        },
        
        # Cryptocurrencies - FIXED VALUES
        "BTCUSDT": {
            "type": "crypto",
            "contract_size": 1,      # 1 BTC
            "margin_currency": "USDT",
            "pip_value": 1.0,        # 1 USDT per pip
            "calculation_formula": "crypto",
            "pip_decimal_places": 1,
            "min_volume": 0.001,
            "volume_step": 0.001,
            "max_leverage": 125
        },
        "ETHUSDT": {
            "type": "crypto",
            "contract_size": 1,      # 1 ETH
            "margin_currency": "USDT",
            "pip_value": 1.0, 
            "calculation_formula": "crypto",
            "pip_decimal_places": 2,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 125
        },
        
        # Stocks - FIXED VALUES
        "AAPL": {
            "type": "stock",
            "contract_size": 100,    # 100 shares per lot
            "margin_currency": "USD",
            "pip_value": 1.0,        # $1 per price movement point
            "calculation_formula": "stocks",
            "pip_decimal_places": 2,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 100
        },
        "TSLA": {
            "type": "stock",
            "contract_size": 100,
            "margin_currency": "USD",
            "pip_value": 1.0,
            "calculation_formula": "stocks", 
            "pip_decimal_places": 2,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 100
        },
        
        # Indices - FIXED VALUES
        "NAS100": {
            "type": "index",
            "contract_size": 1,      # 1 index contract
            "margin_currency": "USD",
            "pip_value": 1.0,        # $1 per point
            "calculation_formula": "indices",
            "pip_decimal_places": 1,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 100
        },
        
        # Metals - FIXED VALUES
        "XAUUSD": {
            "type": "metal", 
            "contract_size": 100,    # 100 ounces in standard lot
            "margin_currency": "USD",
            "pip_value": 1.0,        # $1 per point (0.01 price change)
            "calculation_formula": "metals",
            "pip_decimal_places": 2,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 100
        },
        "XAGUSD": {
            "type": "metal", 
            "contract_size": 5000,   # 5000 ounces in standard lot
            "margin_currency": "USD",
            "pip_value": 5.0,        # $5 per point (0.001 price change, corrected for contract size)
            "calculation_formula": "metals",
            "pip_decimal_places": 3, # Corrected for silver precision
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 100
        },
        
        # Energy - FIXED VALUES
        "OIL": {
            "type": "energy",
            "contract_size": 1000,   # 1000 barrels
            "margin_currency": "USD",
            "pip_value": 10.0,       # $10 per point (0.01 price change)
            "calculation_formula": "energy",
            "pip_decimal_places": 2,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 100
        }
    }
    
    @classmethod
    def get_specs(cls, symbol: str) -> Dict[str, Any]:
        """Get specifications for instrument"""
        return cls.SPECS.get(symbol, cls._get_default_specs(symbol))
    
    @classmethod
    def _get_default_specs(cls, symbol: str) -> Dict[str, Any]:
        """Default specifications"""
        if any(currency in symbol for currency in ['USD', 'EUR', 'GBP', 'JPY']):
            return {
                "type": "forex",
                "contract_size": 100000,
                "margin_currency": "USD",
                "pip_value": 10.0,
                "calculation_formula": "forex",
                "pip_decimal_places": 4,
                "min_volume": 0.01,
                "volume_step": 0.01,
                "max_leverage": 1000
            }
        elif 'USDT' in symbol:
            return {
                "type": "crypto",
                "contract_size": 1,
                "margin_currency": "USDT", 
                "pip_value": 1.0,
                "calculation_formula": "crypto",
                "pip_decimal_places": 2,
                "min_volume": 0.01,
                "volume_step": 0.01,
                "max_leverage": 125
            }
        else:
            return {
                "type": "stock",
                "contract_size": 100,
                "margin_currency": "USD",
                "pip_value": 1.0,
                "calculation_formula": "stocks",
                "pip_decimal_places": 2,
                "min_volume": 0.01,
                "volume_step": 0.01,
                "max_leverage": 100
            }

# ---------------------------
# Professional Margin Calculator - FIXED CALCULATION
# ---------------------------
class ProfessionalMarginCalculator:
    """FIXED margin calculation with real quotes"""
    
    def __init__(self):
        self.market_data = EnhancedMarketDataProvider()
    
    async def calculate_professional_margin(self, symbol: str, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Professional margin calculation with real quotes"""
        try:
            specs = InstrumentSpecs.get_specs(symbol)
            formula = specs['calculation_formula']
            
            # Get effective leverage considering limitations
            selected_leverage = int(leverage.split(':')[1])
            max_leverage = specs.get('max_leverage', selected_leverage)
            effective_leverage = min(selected_leverage, max_leverage)
            effective_leverage_str = f"1:{effective_leverage}"
            
            if formula == "forex":
                return await self._calculate_forex_margin(specs, volume, effective_leverage_str, current_price)
            elif formula == "forex_jpy":
                return await self._calculate_forex_jpy_margin(specs, volume, effective_leverage_str, current_price)
            elif formula == "crypto":
                return await self._calculate_crypto_margin(specs, volume, effective_leverage_str, current_price)
            elif formula == "stocks":
                return await self._calculate_stocks_margin(specs, volume, effective_leverage_str, current_price)
            elif formula == "indices":
                return await self._calculate_indices_margin(specs, volume, effective_leverage_str, current_price)
            elif formula == "metals":
                return await self._calculate_metals_margin(specs, volume, effective_leverage_str, current_price)
            elif formula == "energy":
                return await self._calculate_energy_margin(specs, volume, effective_leverage_str, current_price)
            else:
                return await self._calculate_universal_margin(specs, volume, effective_leverage_str, current_price)
                
        except Exception as e:
            logger.error(f"Margin calculation error for {symbol}: {e}")
            return await self._calculate_universal_margin(specs, volume, leverage, current_price)
    
    async def _calculate_forex_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Margin calculation for Forex according to industry standards"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # For Forex: (Volume × Contract Size) / Leverage
        required_margin = (volume * contract_size) / lev_value
        
        return {
            'required_margin': max(required_margin, 0.01),  # Minimum $0.01
            'contract_size': contract_size,
            'calculation_method': 'forex_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size,
            'effective_leverage': leverage
        }
    
    async def _calculate_forex_jpy_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Margin calculation for JPY pairs (calculation specifics)"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # For JPY pairs same formula but consider exchange rate
        required_margin = (volume * contract_size) / lev_value
        
        return {
            'required_margin': max(required_margin, 0.01),
            'contract_size': contract_size,
            'calculation_method': 'forex_jpy_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size,
            'effective_leverage': leverage
        }
    
    async def _calculate_crypto_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Margin calculation for cryptocurrencies"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # For crypto: (Volume × Price) / Leverage
        required_margin = (volume * contract_size * current_price) / lev_value
        
        return {
            'required_margin': max(required_margin, 0.01),
            'contract_size': contract_size,
            'calculation_method': 'crypto_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size * current_price,
            'effective_leverage': leverage
        }
    
    async def _calculate_stocks_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Margin calculation for stocks"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # For stocks: (Volume × Contract Size × Price) / Leverage
        required_margin = (volume * contract_size * current_price) / lev_value
        
        return {
            'required_margin': max(required_margin, 0.01),
            'contract_size': contract_size,
            'calculation_method': 'stocks_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size * current_price,
            'effective_leverage': leverage
        }
    
    async def _calculate_indices_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Margin calculation for indices"""
        return await self._calculate_stocks_margin(specs, volume, leverage, current_price)
    
    async def _calculate_metals_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """FIXED margin calculation for metals"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # FOR METALS: (Volume × Contract Size × Price) / Leverage
        required_margin = (volume * contract_size * current_price) / lev_value
        
        return {
            'required_margin': max(required_margin, 0.01),
            'contract_size': contract_size,
            'calculation_method': 'metals_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size * current_price,
            'effective_leverage': leverage
        }
    
    async def _calculate_energy_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Margin calculation for energy resources"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # For energy: (Volume × Contract Size × Price) / Leverage
        required_margin = (volume * contract_size * current_price) / lev_value
        
        return {
            'required_margin': max(required_margin, 0.01),
            'contract_size': contract_size,
            'calculation_method': 'energy_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size * current_price,
            'effective_leverage': leverage
        }
    
    async def _calculate_universal_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Universal margin calculation"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs.get('contract_size', 1)
        
        required_margin = (volume * contract_size * current_price) / lev_value
        
        return {
            'required_margin': max(required_margin, 0.01),
            'contract_size': contract_size,
            'calculation_method': 'universal',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size * current_price,
            'effective_leverage': leverage
        }

# ---------------------------
# Professional Risk Calculator - FIXED WITH CORRECT VOLUME CALCULATION
# ---------------------------
class ProfessionalRiskCalculator:
    """FIXED calculator with correct volume calculation using 2% rule"""
    
    @staticmethod
    def calculate_pip_distance(entry: float, target: float, direction: str, asset: str) -> float:
        """Professional pip distance calculation"""
        specs = InstrumentSpecs.get_specs(asset)
        pip_decimal_places = specs.get('pip_decimal_places', 4)
        
        if direction.upper() == 'LONG':
            distance = target - entry
        else:  # SHORT
            distance = entry - target
        
        if pip_decimal_places == 2:  # JPY pairs
            return abs(distance) * 100
        elif pip_decimal_places == 1:  # Some indices
            return abs(distance) * 10
        elif pip_decimal_places == 3:  # Silver, etc.
            return abs(distance) * 1000
        else:  # Standard 4 decimal places
            return abs(distance) * 10000

    @staticmethod
    def calculate_pnl_dollar_amount(entry_price: float, exit_price: float, volume: float, pip_value: float, 
                                  direction: str, asset: str, tick_size: float = 0.01) -> float:
        """Professional P&L calculation in dollars"""
        try:
            specs = InstrumentSpecs.get_specs(asset)
            
            if direction.upper() == 'LONG':
                price_diff = exit_price - entry_price
            else:  # SHORT
                price_diff = entry_price - exit_price
            
            # Different calculation for different asset types
            if specs['type'] in ['stock', 'crypto']:
                # For stocks and crypto: price difference × volume × contract size
                pnl = price_diff * volume * specs['contract_size']
            else:
                # For others: via pips
                pip_distance = ProfessionalRiskCalculator.calculate_pip_distance(
                    entry_price, exit_price, direction, asset
                )
                pnl = pip_distance * volume * pip_value
            
            return round(pnl, 2)
        except Exception as e:
            logger.error(f"P&L calculation error: {e}")
            return 0.0

    @staticmethod
    async def calculate_realistic_pnl(trade: Dict, current_price: float, volume: float, pip_value: float, direction: str, asset: str) -> float:
        """REALISTIC P&L calculation considering volume and pip value"""
        entry = trade['entry_price']
        
        return ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            entry, current_price, volume, pip_value, direction, asset
        )

    @staticmethod
    def calculate_margin_level(equity: float, margin: float) -> float:
        """Calculate margin level percentage"""
        if margin == 0:
            return float('inf')  # Infinity at zero margin
        
        margin_level = (equity / margin) * 100
        return round(margin_level, 2)

    @staticmethod
    def calculate_free_margin(equity: float, margin: float) -> float:
        """Calculate free margin"""
        free_margin = equity - margin
        return max(free_margin, 0.0)  # Cannot be negative

    @staticmethod
    async def calculate_professional_metrics(trade: Dict, deposit: float, leverage: str, risk_level: str) -> Dict[str, Any]:
        """
        FIXED calculation with correct volume determination using 2% rule
        """
        try:
            asset = trade['asset']
            entry = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            direction = trade['direction']
            
            current_price = await enhanced_market_data.get_robust_real_time_price(asset)
            specs = InstrumentSpecs.get_specs(asset)
            
            # FIXED 2% RISK according to risk management rules
            risk_percent = 0.02  # Fixed 2% instead of user choice
            risk_amount = deposit * risk_percent
            
            stop_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry, stop_loss, direction, asset)
            profit_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry, take_profit, direction, asset)
            
            pip_value = specs['pip_value']
            
            # CORRECT VOLUME CALCULATION using formula: Volume = Risk Amount / (Stop Distance * Pip Value)
            if stop_distance_pips > 0 and pip_value > 0:
                volume_lots = risk_amount / (stop_distance_pips * pip_value)
                # Round to volume step
                volume_step = specs.get('volume_step', 0.01)
                volume_lots = round(volume_lots / volume_step) * volume_step
                # Limit to minimum volume
                min_volume = specs.get('min_volume', 0.01)
                volume_lots = max(volume_lots, min_volume)
                volume_lots = round(volume_lots, 3)
            else:
                volume_lots = 0
            
            margin_data = await margin_calculator.calculate_professional_margin(
                asset, volume_lots, leverage, current_price
            )
            required_margin = margin_data['required_margin']
            required_margin = round(required_margin, 2)
            
            # Equity calculation (balance + unrealized P&L)
            current_pnl = await ProfessionalRiskCalculator.calculate_realistic_pnl(
                trade, current_price, volume_lots, pip_value, direction, asset
            )
            equity = deposit + current_pnl
            
            # Use professional formulas for margin
            free_margin = ProfessionalRiskCalculator.calculate_free_margin(equity, required_margin)
            margin_level = ProfessionalRiskCalculator.calculate_margin_level(equity, required_margin)
            
            # Potential profit calculation via professional function
            potential_profit = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
                entry, take_profit, volume_lots, pip_value, direction, asset
            )
            potential_profit = round(potential_profit, 2)
            
            rr_ratio = potential_profit / risk_amount if risk_amount > 0 else 0
            rr_ratio = round(rr_ratio, 2)
            
            risk_per_trade_percent = (risk_amount / deposit) * 100 if deposit > 0 else 0
            margin_usage_percent = (required_margin / deposit) * 100 if deposit > 0 else 0
            notional_value = margin_data.get('notional_value', 0)
            
            return {
                'volume_lots': volume_lots,
                'required_margin': required_margin,
                'free_margin': free_margin,
                'margin_level': margin_level,
                'risk_amount': risk_amount,
                'risk_percent': risk_per_trade_percent,
                'potential_profit': potential_profit,
                'rr_ratio': rr_ratio,
                'stop_distance_pips': stop_distance_pips,
                'profit_distance_pips': profit_distance_pips,
                'pip_value': pip_value,
                'contract_size': margin_data['contract_size'],
                'deposit': deposit,
                'leverage': leverage,
                'effective_leverage': margin_data.get('effective_leverage', leverage),
                'risk_per_trade_percent': risk_per_trade_percent,
                'margin_usage_percent': margin_usage_percent,
                'current_price': current_price,
                'calculation_method': margin_data['calculation_method'],
                'notional_value': notional_value,
                'leverage_used': margin_data.get('leverage_used', 1),
                'current_pnl': current_pnl,
                'equity': equity
            }
        except Exception as e:
            logger.error(f"Professional calculation error: {e}")
            return {
                'volume_lots': 0,
                'required_margin': 0,
                'free_margin': deposit,
                'margin_level': 0,
                'risk_amount': 0,
                'risk_percent': 0,
                'potential_profit': 0,
                'rr_ratio': 0,
                'stop_distance_pips': 0,
                'profit_distance_pips': 0,
                'pip_value': 0,
                'contract_size': 0,
                'deposit': deposit,
                'leverage': leverage,
                'effective_leverage': leverage,
                'risk_per_trade_percent': 0,
                'margin_usage_percent': 0,
                'current_price': 0,
                'calculation_method': 'error',
                'notional_value': 0,
                'leverage_used': 1,
                'current_pnl': 0,
                'equity': deposit
            }

# ---------------------------
# Portfolio Analyzer - NEW CLASS FOR PORTFOLIO ANALYSIS
# ---------------------------
class PortfolioAnalyzer:
    """Portfolio analyzer with aggregated metrics"""
    
    @staticmethod
    def calculate_portfolio_metrics(trades: List[Dict], deposit: float) -> Dict[str, Any]:
        """Calculate aggregated portfolio metrics"""
        if not trades:
            return {}
        
        total_risk_usd = sum(trade.get('metrics', {}).get('risk_amount', 0) for trade in trades)
        total_profit = sum(trade.get('metrics', {}).get('potential_profit', 0) for trade in trades)
        total_margin = sum(trade.get('metrics', {}).get('required_margin', 0) for trade in trades)
        total_pnl = sum(trade.get('metrics', {}).get('current_pnl', 0) for trade in trades)
        total_equity = deposit + total_pnl
        avg_rr_ratio = sum(trade.get('metrics', {}).get('rr_ratio', 0) for trade in trades) / len(trades) if trades else 0
        
        total_risk_percent = (total_risk_usd / deposit) * 100 if deposit > 0 else 0
        total_margin_usage = (total_margin / deposit) * 100 if deposit > 0 else 0
        free_margin = max(total_equity - total_margin, 0)
        free_margin_percent = (free_margin / deposit) * 100 if deposit > 0 else 0
        portfolio_margin_level = (total_equity / total_margin * 100) if total_margin > 0 else float('inf')
        
        # Portfolio volatility (weighted average)
        portfolio_volatility = sum(VOLATILITY_DATA.get(trade['asset'], 20) * trade.get('metrics', {}).get('risk_amount', 0) / total_risk_usd for trade in trades) if total_risk_usd > 0 else 20
        
        # Diversification
        unique_assets = len(set(trade['asset'] for trade in trades))
        diversity_score = min(unique_assets / 5, 1.0) # Max 5 unique for 100%
        
        long_positions = sum(1 for trade in trades if trade['direction'] == 'LONG')
        short_positions = len(trades) - long_positions
        
        # Portfolio leverage
        total_notional = sum(trade.get('metrics', {}).get('notional_value', 0) for trade in trades)
        portfolio_leverage = total_notional / deposit if deposit > 0 else 1
        
        return {
            'total_risk_usd': round(total_risk_usd, 2),
            'total_risk_percent': round(total_risk_percent, 1),
            'total_profit': round(total_profit, 2),
            'avg_rr_ratio': round(avg_rr_ratio, 2),
            'total_pnl': round(total_pnl, 2),
            'total_equity': round(total_equity, 2),
            'total_margin': round(total_margin, 2),
            'total_margin_usage': round(total_margin_usage, 1),
            'free_margin': round(free_margin, 2),
            'free_margin_percent': round(free_margin_percent, 1),
            'portfolio_margin_level': round(portfolio_margin_level, 1),
            'portfolio_volatility': round(portfolio_volatility, 1),
            'unique_assets': unique_assets,
            'diversity_score': round(diversity_score * 100, 1),
            'long_positions': long_positions,
            'short_positions': short_positions,
            'portfolio_leverage': round(portfolio_leverage, 1)
        }

    @staticmethod
    def generate_enhanced_recommendations(metrics: Dict, trades: List[Dict]) -> List[str]:
        """Generate enhanced recommendations"""
        recommendations = []
        
        if metrics['total_risk_percent'] > 10:
            recommendations.append("Total risk exceeds 10% - consider reducing positions to decrease volatility.")
        elif metrics['total_risk_percent'] < 2:
            recommendations.append("Risk is low - there may be room for additional positions while maintaining 2% rule.")
        
        if metrics['avg_rr_ratio'] < 2:
            recommendations.append("Average R/R below 2:1 - aim for more favorable ratios for long-term profitability.")
        
        if metrics['diversity_score'] < 0.6:
            recommendations.append("Diversification low - add assets from different categories to reduce correlation risks.")
        
        if metrics['portfolio_margin_level'] < 200:
            recommendations.append("Margin level low - monitor position to avoid margin call.")
        
        if metrics['portfolio_volatility'] > 30:
            recommendations.append("Volatility high - consider hedging or reducing exposure to volatile assets.")
        
        long_short_balance = abs(metrics['long_positions'] - metrics['short_positions']) / len(trades) if trades else 0
        if long_short_balance > 0.7:
            recommendations.append("Portfolio skewed to one side - balance longs and shorts for market neutrality.")
        
        if not recommendations:
            recommendations.append("Portfolio looks balanced - continue monitoring real prices.")
        
        return recommendations

# ---------------------------
# VOLATILITY_DATA - Added for analysis
# ---------------------------
VOLATILITY_DATA = {
    'EURUSD': 8, 'GBPUSD': 10, 'USDJPY': 9, 'BTCUSDT': 50, 'ETHUSDT': 45,
    'AAPL': 25, 'TSLA': 40, 'NAS100': 15, 'XAUUSD': 12, 'OIL': 30
    # Add more as needed
}

# ---------------------------
# Portfolio Manager - PORTFOLIO MANAGEMENT
# ---------------------------
class PortfolioManager:
    """Portfolio manager with data saving"""
    user_data = {}  # Simple in-memory database (use Redis/DB for production)
    
    @staticmethod
    def ensure_user(user_id: int):
        if user_id not in PortfolioManager.user_data:
            PortfolioManager.user_data[user_id] = {
                'single_trades': [],
                'multi_trades': [],
                'deposit': 1000.0,
                'leverage': '1:100'
            }
    
    @staticmethod
    def add_single_trade(user_id: int, trade: Dict):
        PortfolioManager.ensure_user(user_id)
        PortfolioManager.user_data[user_id]['single_trades'].append(trade)
    
    @staticmethod
    def add_multi_trade(user_id: int, trades: List[Dict]):
        PortfolioManager.ensure_user(user_id)
        PortfolioManager.user_data[user_id]['multi_trades'].extend(trades)
    
    @staticmethod
    def set_deposit_leverage(user_id: int, deposit: float, leverage: str):
        PortfolioManager.ensure_user(user_id)
        PortfolioManager.user_data[user_id]['deposit'] = deposit
        PortfolioManager.user_data[user_id]['leverage'] = leverage
    
    @staticmethod
    def clear_portfolio(user_id: int):
        if user_id in PortfolioManager.user_data:
            PortfolioManager.user_data[user_id]['single_trades'] = []
            PortfolioManager.user_data[user_id]['multi_trades'] = []

# ---------------------------
# Data Manager - PROGRESS MANAGEMENT
# ---------------------------
class DataManager:
    """Temporary data manager for progress recovery"""
    @staticmethod
    def load_temporary_data() -> Dict:
        try:
            with open('temporary_progress.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    @staticmethod
    def save_temporary_data(data: Dict):
        with open('temporary_progress.json', 'w') as f:
            json.dump(data, f)
    
    @staticmethod
    def save_progress(user_id: int, state_data: Dict, state_type: str):
        temp_data = DataManager.load_temporary_data()
        temp_data[str(user_id)] = {
            'state_data': state_data,
            'state_type': state_type,
            'timestamp': datetime.now().isoformat()
        }
        DataManager.save_temporary_data(temp_data)
    
    @staticmethod
    def clear_temporary_progress(user_id: int):
        temp_data = DataManager.load_temporary_data()
        temp_data.pop(str(user_id), None)
        DataManager.save_temporary_data(temp_data)

# ---------------------------
# LEVERAGES and ASSET_CATEGORIES - Constants
# ---------------------------
LEVERAGES = {
    "DEFAULT": ["1:100", "1:200", "1:500", "1:1000"]
}

ASSET_CATEGORIES = {
    "Forex": ["EURUSD", "GBPUSD", "USDJPY"],
    "Crypto": ["BTCUSDT", "ETHUSDT"],
    "Stocks": ["AAPL", "TSLA"],
    "Indices": ["NAS100"],
    "Metals": ["XAUUSD", "XAGUSD"],
    "Energy": ["OIL"]
}

# ---------------------------
# ENUM STATES - For single and multi trade
# ---------------------------
class SingleTradeState(Enum):
    DEPOSIT = 1
    LEVERAGE = 2
    ASSET_CATEGORY = 3
    ASSET = 4
    DIRECTION = 5
    ENTRY = 6
    STOP_LOSS = 7
    TAKE_PROFIT = 8

class MultiTradeState(Enum):
    DEPOSIT = 1
    LEVERAGE = 2
    ASSET_CATEGORY = 3
    ASSET = 4
    DIRECTION = 5
    ENTRY = 6
    STOP_LOSS = 7
    TAKE_PROFIT = 8
    ADD_MORE = 9

# ---------------------------
# GLOBAL INSTANCES
# ---------------------------
enhanced_market_data = EnhancedMarketDataProvider()
margin_calculator = ProfessionalMarginCalculator()

# ---------------------------
# COMMANDS
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command /start"""
    text = (
        "🚀 <b>Welcome to PRO RISK CALCULATOR v3.0 ENTERPRISE</b>\n\n"
        "Professional risk calculation tool with fixed 2% rule.\n"
        "Use real quotes and accurate margin calculations.\n\n"
        "Start from main menu:"
    )
    
    keyboard = [
        [InlineKeyboardButton("🎯 Professional Calculation", callback_data="pro_calculation")],
        [InlineKeyboardButton("📊 Portfolio", callback_data="portfolio")],
        [InlineKeyboardButton("📚 Instructions", callback_data="pro_info")],
        [InlineKeyboardButton("💖 Support Developer", callback_data="donate_start")]
    ]
    
    await SafeMessageSender.send_message(
        update.message.chat_id,
        text,
        context,
        InlineKeyboardMarkup(keyboard)
    )

@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command /pro_info - Part 1"""
    text = (
        "📚 <b>PRO RISK CALCULATOR v3.0 INSTRUCTIONS</b>\n\n"
        "1. <b>Fixed Risk</b>: All calculations use 2% rule per trade.\n"
        "2. <b>Real Prices</b>: Bot gets quotes from multiple APIs.\n"
        "3. <b>Margin</b>: Calculated according to standards (volume * contract * price / leverage).\n"
        "4. <b>Volume</b>: Automatically adjusted for 2% risk.\n"
        "5. <b>Portfolio</b>: Aggregates metrics for multiple trades.\n\n"
        "Click 'Next' for details."
    )
    
    keyboard = [
        [InlineKeyboardButton("▶️ Next", callback_data="pro_info_part2")],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
    ]
    
    if update.callback_query:
        query = update.callback_query
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup(keyboard)
        )
    else:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            text,
            context,
            InlineKeyboardMarkup(keyboard)
        )

@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_info_part2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Part 2 instructions"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = (
        "📚 <b>INSTRUCTIONS - PART 2</b>\n\n"
        "• <b>Single Trade</b>: Calculate risk for one position.\n"
        "• <b>Multi-position</b>: Add multiple trades to portfolio.\n"
        "• <b>Recommendations</b>: Bot gives diversification and risk advice.\n"
        "• <b>Export</b>: Download portfolio report in TXT.\n"
        "• <b>Recovery</b>: Continue interrupted calculation.\n\n"
        "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
    )
    
    keyboard = [
        [InlineKeyboardButton("🔙 Back", callback_data="pro_info")],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

@retry_on_timeout(max_retries=2, delay=1.0)
async def future_features_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Future features"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = (
        "🚀 <b>FUTURE FEATURES v4.0</b>\n\n"
        "• Exchange integration (Binance, MT5)\n"
        "• Automatic position tracking\n"
        "• AI entry recommendations\n"
        "• Risk charts\n"
        "• Mobile application\n\n"
        "Support development with donations!"
    )
    
    keyboard = [
        [InlineKeyboardButton("💖 Donate", callback_data="donate_start")],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

# ---------------------------
# SINGLE TRADE HANDLERS - FIXED
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start single trade"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    context.user_data.clear()
    
    text = (
        "🎯 <b>SINGLE TRADE v3.0</b>\n\n"
        "Step 1/7: Enter deposit in USD (minimum $100):"
    )
    
    keyboard = [[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )
    return SingleTradeState.DEPOSIT.value

async def single_trade_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Deposit for single trade"""
    text = update.message.text.strip()
    
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "single")
    
    try:
        deposit = float(text.replace(',', '.'))
        if deposit < 100:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Minimum deposit: $100\nTry again:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.DEPOSIT.value
        
        context.user_data['deposit'] = deposit
        
        keyboard = []
        for leverage in LEVERAGES["DEFAULT"]:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"lev_{leverage}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")])
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Deposit: ${deposit:,.2f}\n\n"
            "Step 2/7: <b>Choose leverage:</b>",
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return SingleTradeState.LEVERAGE.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Enter a number (example: 1000)\nTry again:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.DEPOSIT.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Leverage handling"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "single")
    
    leverage = query.data.replace('lev_', '')
    context.user_data['leverage'] = leverage
    
    keyboard = []
    for category in ASSET_CATEGORIES.keys():
        keyboard.append([InlineKeyboardButton(category, callback_data=f"cat_{category}")])
    
    keyboard.append([InlineKeyboardButton("📝 Enter asset manually", callback_data="asset_manual")])
    keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")])
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Leverage: {leverage}\n\n"
        "Step 3/7: <b>Choose asset category:</b>",
        InlineKeyboardMarkup(keyboard)
    )
    return SingleTradeState.ASSET_CATEGORY.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_asset_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Category handling"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "single")
    
    if query.data == "asset_manual":
        await SafeMessageSender.edit_message_text(
            query,
            "Step 4/7: ✍️ Enter asset name (example: BTCUSDT):",
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data="back_to_categories")],
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.ASSET.value
    
    category = query.data.replace('cat_', '')
    context.user_data['asset_category'] = category
    
    assets = ASSET_CATEGORIES.get(category, [])
    
    keyboard = []
    for asset in assets:
        keyboard.append([InlineKeyboardButton(asset, callback_data=f"asset_{asset}")])
    
    keyboard.append([InlineKeyboardButton("🔙 Back to categories", callback_data="back_to_categories")])
    keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")])
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Category: {category}\n\n"
        "Step 4/7: <b>Choose asset:</b>",
        InlineKeyboardMarkup(keyboard)
    )
    return SingleTradeState.ASSET.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def enhanced_single_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asset handler"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "single")
    
    if query.data == "back_to_categories":
        keyboard = []
        for category in ASSET_CATEGORIES.keys():
            keyboard.append([InlineKeyboardButton(category, callback_data=f"cat_{category}")])
        
        keyboard.append([InlineKeyboardButton("📝 Enter asset manually", callback_data="asset_manual")])
        keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")])
        
        await SafeMessageSender.edit_message_text(
            query,
            "Step 3/7: <b>Choose asset category:</b>",
            InlineKeyboardMarkup(keyboard)
        )
        return SingleTradeState.ASSET_CATEGORY.value
    
    asset = query.data.replace('asset_', '')
    context.user_data['asset'] = asset
    
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Asset: {asset}\n{price_info}\n\n"
        "Step 5/7: <b>Choose trade direction:</b>",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🔙 Back to categories", callback_data="back_to_categories")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.DIRECTION.value

async def single_trade_asset_manual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Manual asset input"""
    asset = update.message.text.strip().upper()
    
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "single")
    
    if not re.match(r'^[A-Z0-9]{2,20}$', asset):
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Invalid asset format. Try again:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.ASSET.value
    
    context.user_data['asset'] = asset
    
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.send_message(
        update.message.chat_id,
        f"✅ Asset: {asset}\n{price_info}\n\n"
        "Step 5/7: <b>Choose trade direction:</b>",
        context,
        InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🔙 Back", callback_data="back_to_categories")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.DIRECTION.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def enhanced_single_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Direction handler"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "single")
    
    direction = query.data.replace('dir_', '')
    context.user_data['direction'] = direction
    
    asset = context.user_data['asset']
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Direction: {direction}\n{price_info}\n\n"
        "Step 6/7: <b>Enter entry price:</b>",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Back", callback_data="back_to_asset")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.ENTRY.value

async def single_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Entry price"""
    text = update.message.text.strip()
    
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "single")
    
    try:
        entry_price = float(text.replace(',', '.'))
        if entry_price <= 0:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Price must be greater than 0\nTry again:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.ENTRY.value
        
        context.user_data['entry_price'] = entry_price
        
        asset = context.user_data['asset']
        price_info = await show_asset_price_in_realtime(asset)
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Entry price: {entry_price}\n{price_info}\n\n"
            "Step 7/7: <b>Enter stop-loss level:</b>",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.STOP_LOSS.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Enter a number (example: 50000)\nTry again:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.ENTRY.value

async def single_trade_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stop-loss"""
    if 'entry_price' not in context.user_data or 'direction' not in context.user_data or 'asset' not in context.user_data:
        logger.error("Missing data in single_trade_stop_loss")
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Data error. Restarting calculation.",
            context,
            InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Start Over", callback_data="single_trade")]])
        )
        return ConversationHandler.END
    
    text = update.message.text.strip()
    
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "single")
    
    try:
        stop_loss = float(text.replace(',', '.'))
        entry_price = context.user_data['entry_price']
        direction = context.user_data['direction']
        asset = context.user_data['asset']
        
        sl_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            entry_price, stop_loss, 1.0, 1.0, direction, asset
        )
        
        if direction == 'LONG' and stop_loss >= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ For LONG stop-loss must be BELOW entry price\nTry again:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.STOP_LOSS.value
        elif direction == 'SHORT' and stop_loss <= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ For SHORT stop-loss must be ABOVE entry price\nTry again:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.STOP_LOSS.value
        
        context.user_data['stop_loss'] = stop_loss
        
        stop_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry_price, stop_loss, direction, asset)
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Stop-loss: {stop_loss} ({stop_distance_pips:.0f} pips)\n"
            f"💵 SL Amount: ${abs(sl_amount):.2f}\n\n"
            "📊 <b>Risk level fixed: 2%</b>\n\n"
            "<b>Enter take-profit level:</b>",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.TAKE_PROFIT.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Enter a number (example: 48000)\nTry again:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.STOP_LOSS.value

async def single_trade_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Take-profit and calculation"""
    if 'entry_price' not in context.user_data or 'stop_loss' not in context.user_data:
        logger.error("Missing data in single_trade_take_profit")
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Data error. Restarting calculation.",
            context,
            InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Start Over", callback_data="single_trade")]])
        )
        return ConversationHandler.END
    
    text = update.message.text.strip()
    
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "single")
    
    try:
        take_profit = float(text.replace(',', '.'))
        entry_price = context.user_data['entry_price']
        direction = context.user_data['direction']
        asset = context.user_data['asset']
        
        tp_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            entry_price, take_profit, 1.0, 1.0, direction, asset
        )
        
        if direction == 'LONG' and take_profit <= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ For LONG take-profit must be ABOVE entry price\nTry again:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.TAKE_PROFIT.value
        elif direction == 'SHORT' and take_profit >= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ For SHORT take-profit must be BELOW entry price\nTry again:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.TAKE_PROFIT.value
        
        context.user_data['take_profit'] = take_profit
        
        trade = context.user_data.copy()
        
        metrics = await ProfessionalRiskCalculator.calculate_professional_metrics(
            trade, trade['deposit'], trade['leverage'], "2%"
        )
        
        trade['metrics'] = metrics
        
        sl_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            trade['entry_price'], trade['stop_loss'], metrics['volume_lots'], 
            metrics['pip_value'], trade['direction'], trade['asset']
        )
        
        tp_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            trade['entry_price'], trade['take_profit'], metrics['volume_lots'],
            metrics['pip_value'], trade['direction'], trade['asset']
        )
        
        user_id = update.message.from_user.id
        PortfolioManager.ensure_user(user_id)
        PortfolioManager.add_single_trade(user_id, trade)
        PortfolioManager.set_deposit_leverage(user_id, trade['deposit'], trade['leverage'])
        
        text = (
            "📊 <b>SINGLE TRADE CALCULATION v3.0</b>\n\n"
            f"Asset: {trade['asset']} | {trade['direction']}\n"
            f"Entry: {trade['entry_price']} | SL: {trade['stop_loss']} (${abs(sl_amount):.2f})\n"
            f"TP: {trade['take_profit']} (${tp_amount:.2f})\n\n"
            f"💰 <b>METRICS:</b>\n"
            f"Volume: {metrics['volume_lots']:.2f} lots\n"
            f"Margin: ${metrics['required_margin']:.2f}\n"
            f"Risk: ${metrics['risk_amount']:.2f} (2%)\n"
            f"Profit: ${metrics['potential_profit']:.2f}\n"
            f"R/R: {metrics['rr_ratio']:.2f}\n"
            f"Current P&L: ${metrics['current_pnl']:.2f}\n\n"
            "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
        )
        
        keyboard = [
            [InlineKeyboardButton("📊 Portfolio", callback_data="portfolio")],
            [InlineKeyboardButton("🎯 New Trade", callback_data="single_trade")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            text,
            context,
            InlineKeyboardMarkup(keyboard)
        )
        
        DataManager.clear_temporary_progress(user_id)
        context.user_data.clear()
        return ConversationHandler.END
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Enter a number (example: 55000)\nTry again:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.TAKE_PROFIT.value

async def single_trade_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Calculation cancellation"""
    await SafeMessageSender.send_message(
        update.message.chat_id if update.message else update.callback_query.message.chat_id,
        "❌ Calculation cancelled",
        context,
        InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]])
    )
    context.user_data.clear()
    DataManager.clear_temporary_progress(update.effective_user.id)
    return ConversationHandler.END

# ---------------------------
# MULTI TRADE HANDLERS - FIXED
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start multi-position"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    context.user_data.clear()
    context.user_data['current_multi_trades'] = []
    
    text = (
        "📊 <b>MULTI-POSITION v3.0</b>\n\n"
        "Step 1/7: Enter deposit in USD (minimum $100):"
    )
    
    keyboard = [[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )
    return MultiTradeState.DEPOSIT.value

async def multi_trade_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Deposit for multi-position"""
    text = update.message.text.strip()
    
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "multi")
    
    try:
        deposit = float(text.replace(',', '.'))
        if deposit < 100:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Minimum deposit: $100\nTry again:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
                ])
            )
            return MultiTradeState.DEPOSIT.value
        
        context.user_data['deposit'] = deposit
        
        keyboard = []
        for leverage in LEVERAGES["DEFAULT"]:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"mlev_{leverage}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")])
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Deposit: ${deposit:,.2f}\n\n"
            "Step 2/7: <b>Choose leverage:</b>",
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return MultiTradeState.LEVERAGE.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Enter a number (example: 1000)\nTry again:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.DEPOSIT.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Leverage handling for multi-position"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "multi")
    
    leverage = query.data.replace('mlev_', '')
    context.user_data['leverage'] = leverage
    
    keyboard = []
    for category in ASSET_CATEGORIES.keys():
        keyboard.append([InlineKeyboardButton(category, callback_data=f"mcat_{category}")])
    
    keyboard.append([InlineKeyboardButton("📝 Enter asset manually", callback_data="massset_manual")])
    keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")])
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Leverage: {leverage}\n\n"
        "Step 3/7: <b>Choose asset category:</b>",
        InlineKeyboardMarkup(keyboard)
    )
    return MultiTradeState.ASSET_CATEGORY.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_asset_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Category handling for multi-position"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "multi")
    
    if query.data == "massset_manual":
        await SafeMessageSender.edit_message_text(
            query,
            "Step 4/7: ✍️ Enter asset name (example: BTCUSDT):",
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data="mback_to_categories")],
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.ASSET.value
    
    category = query.data.replace('mcat_', '')
    context.user_data['asset_category'] = category
    
    assets = ASSET_CATEGORIES.get(category, [])
    
    keyboard = []
    for asset in assets:
        keyboard.append([InlineKeyboardButton(asset, callback_data=f"massset_{asset}")])
    
    keyboard.append([InlineKeyboardButton("🔙 Back to categories", callback_data="mback_to_categories")])
    keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")])
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Category: {category}\n\n"
        "Step 4/7: <b>Choose asset:</b>",
        InlineKeyboardMarkup(keyboard)
    )
    return MultiTradeState.ASSET.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def enhanced_multi_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asset handler for multi-position"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "multi")
    
    if query.data == "mback_to_categories":
        keyboard = []
        for category in ASSET_CATEGORIES.keys():
            keyboard.append([InlineKeyboardButton(category, callback_data=f"mcat_{category}")])
        
        keyboard.append([InlineKeyboardButton("📝 Enter asset manually", callback_data="massset_manual")])
        keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")])
        
        await SafeMessageSender.edit_message_text(
            query,
            "Step 3/7: <b>Choose asset category:</b>",
            InlineKeyboardMarkup(keyboard)
        )
        return MultiTradeState.ASSET_CATEGORY.value
    
    asset = query.data.replace('massset_', '')
    context.user_data['asset'] = asset
    
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Asset: {asset}\n{price_info}\n\n"
        "Step 5/7: <b>Choose trade direction:</b>",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="mdir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="mdir_SHORT")],
            [InlineKeyboardButton("🔙 Back to categories", callback_data="mback_to_categories")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
        ])
    )
    return MultiTradeState.DIRECTION.value

async def multi_trade_asset_manual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Manual asset input for multi-position"""
    asset = update.message.text.strip().upper()
    
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "multi")
    
    if not re.match(r'^[A-Z0-9]{2,20}$', asset):
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Invalid asset format. Try again:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.ASSET.value
    
    context.user_data['asset'] = asset
    
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.send_message(
        update.message.chat_id,
        f"✅ Asset: {asset}\n{price_info}\n\n"
        "Step 5/7: <b>Choose trade direction:</b>",
        context,
        InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="mdir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="mdir_SHORT")],
            [InlineKeyboardButton("🔙 Back", callback_data="mback_to_categories")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
        ])
    )
    return MultiTradeState.DIRECTION.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def enhanced_multi_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Direction handler for multi-position"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "multi")
    
    direction = query.data.replace('mdir_', '')
    context.user_data['direction'] = direction
    
    asset = context.user_data['asset']
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Direction: {direction}\n{price_info}\n\n"
        "Step 6/7: <b>Enter entry price:</b>",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Back", callback_data="mback_to_asset")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
        ])
    )
    return MultiTradeState.ENTRY.value

async def multi_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Entry price for multi-position"""
    text = update.message.text.strip()
    
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "multi")
    
    try:
        entry_price = float(text.replace(',', '.'))
        if entry_price <= 0:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Price must be greater than 0\nTry again:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
                ])
            )
            return MultiTradeState.ENTRY.value
        
        context.user_data['entry_price'] = entry_price
        
        asset = context.user_data['asset']
        price_info = await show_asset_price_in_realtime(asset)
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Entry price: {entry_price}\n{price_info}\n\n"
            "Step 7/7: <b>Enter stop-loss level:</b>",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.STOP_LOSS.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Enter a number (example: 50000)\nTry again:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.ENTRY.value

async def multi_trade_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stop-loss for multi-position"""
    if 'entry_price' not in context.user_data or 'direction' not in context.user_data or 'asset' not in context.user_data:
        logger.error("Missing data in multi_trade_stop_loss")
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Data error. Restarting multi-position.",
            context,
            InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Start Over", callback_data="multi_trade_start")]])
        )
        return ConversationHandler.END
    
    text = update.message.text.strip()
    
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "multi")
    
    try:
        stop_loss = float(text.replace(',', '.'))
        entry_price = context.user_data['entry_price']
        direction = context.user_data['direction']
        asset = context.user_data['asset']
        
        sl_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            entry_price, stop_loss, 1.0, 1.0, direction, asset
        )
        
        if direction == 'LONG' and stop_loss >= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ For LONG stop-loss must be BELOW entry price\nTry again:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
                ])
            )
            return MultiTradeState.STOP_LOSS.value
        elif direction == 'SHORT' and stop_loss <= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ For SHORT stop-loss must be ABOVE entry price\nTry again:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
                ])
            )
            return MultiTradeState.STOP_LOSS.value
        
        context.user_data['stop_loss'] = stop_loss
        
        stop_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry_price, stop_loss, direction, asset)
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Stop-loss: {stop_loss} ({stop_distance_pips:.0f} pips)\n"
            f"💵 SL Amount: ${abs(sl_amount):.2f}\n\n"
            "📊 <b>Risk level fixed: 2%</b>\n\n"
            "<b>Enter take-profit level:</b>",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.TAKE_PROFIT.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Enter a number (example: 48000)\nTry again:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.STOP_LOSS.value

async def multi_trade_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Take-profit for multi-position"""
    if 'entry_price' not in context.user_data or 'stop_loss' not in context.user_data:
        logger.error("Missing data in multi_trade_take_profit")
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Data error. Restarting multi-position.",
            context,
            InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Start Over", callback_data="multi_trade_start")]])
        )
        return ConversationHandler.END
    
    text = update.message.text.strip()
    
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "multi")
    
    try:
        take_profit = float(text.replace(',', '.'))
        entry_price = context.user_data['entry_price']
        direction = context.user_data['direction']
        asset = context.user_data['asset']
        
        tp_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            entry_price, take_profit, 1.0, 1.0, direction, asset
        )
        
        if direction == 'LONG' and take_profit <= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ For LONG take-profit must be ABOVE entry price\nTry again:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
                ])
            )
            return MultiTradeState.TAKE_PROFIT.value
        elif direction == 'SHORT' and take_profit >= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ For SHORT take-profit must be BELOW entry price\nTry again:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
                ])
            )
            return MultiTradeState.TAKE_PROFIT.value
        
        context.user_data['take_profit'] = take_profit
        
        # Create trade and add to current_multi_trades
        trade = context.user_data.copy()
        trade.pop('current_multi_trades', None)
        context.user_data['current_multi_trades'].append(trade)
        
        # Calculate metrics
        metrics = await ProfessionalRiskCalculator.calculate_professional_metrics(
            trade, trade['deposit'], trade['leverage'], "2%"
        )
        
        trade['metrics'] = metrics
        
        sl_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            trade['entry_price'], trade['stop_loss'], metrics['volume_lots'], 
            metrics['pip_value'], trade['direction'], trade['asset']
        )
        
        tp_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            trade['entry_price'], trade['take_profit'], metrics['volume_lots'],
            metrics['pip_value'], trade['direction'], trade['asset']
        )
        
        text = (
            f"✅ <b>Trade #{len(context.user_data['current_multi_trades'])} added!</b>\n\n"
            f"Asset: {trade['asset']} | {trade['direction']}\n"
            f"Entry: {trade['entry_price']} | SL: {trade['stop_loss']} (${abs(sl_amount):.2f})\n"
            f"TP: {trade['take_profit']} (${tp_amount:.2f}) | Volume: {metrics['volume_lots']:.2f}\n"
            f"Risk: ${metrics['risk_amount']:.2f} (2% of deposit)\n\n"
            "<b>Add another trade or finish?</b>"
        )
        
        keyboard = [
            [InlineKeyboardButton("➕ Add Trade", callback_data="madd_more")],
            [InlineKeyboardButton("✅ Finish Multi-position", callback_data="mfinish_multi")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
        ]
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            text,
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return MultiTradeState.ADD_MORE.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Enter a number (example: 55000)\nTry again:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.TAKE_PROFIT.value

async def multi_trade_add_more(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Adding another trade to multi-position - FIXED VERSION"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Save current trades
    current_multi = context.user_data.get('current_multi_trades', [])
    
    # Clear user_data for new trade, but keep deposit, leverage and current_multi_trades
    deposit = context.user_data.get('deposit')
    leverage = context.user_data.get('leverage')
    
    context.user_data.clear()
    
    # Restore necessary data
    context.user_data['deposit'] = deposit
    context.user_data['leverage'] = leverage
    context.user_data['current_multi_trades'] = current_multi
    
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "multi")
    
    keyboard = []
    for category in ASSET_CATEGORIES.keys():
        keyboard.append([InlineKeyboardButton(category, callback_data=f"mcat_{category}")])
    
    keyboard.append([InlineKeyboardButton("📝 Enter asset manually", callback_data="massset_manual")])
    keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu_save")])
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Trades added: {len(current_multi)}\n\n"
        "Step 3/7: <b>Choose asset category for next trade:</b>",
        InlineKeyboardMarkup(keyboard)
    )
    return MultiTradeState.ASSET_CATEGORY.value

async def multi_trade_finish(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Finish multi-position and portfolio calculation"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    current_multi = context.user_data.get('current_multi_trades', [])
    
    if not current_multi:
        await SafeMessageSender.edit_message_text(
            query,
            "❌ No trades to finish",
            InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]])
        )
        return ConversationHandler.END
    
    # Add to portfolio
    PortfolioManager.ensure_user(user_id)
    PortfolioManager.add_multi_trade(user_id, current_multi)
    deposit = current_multi[0].get('deposit', 1000)
    leverage = current_multi[0].get('leverage', '1:100')
    PortfolioManager.set_deposit_leverage(user_id, deposit, leverage)
    
    # Calculate portfolio metrics
    metrics = PortfolioAnalyzer.calculate_portfolio_metrics(current_multi, deposit)
    recommendations = PortfolioAnalyzer.generate_enhanced_recommendations(metrics, current_multi)
    
    text = (
        "📊 <b>MULTI-POSITION COMPLETED v3.0</b>\n\n"
        f"Trades added: {len(current_multi)}\n"
        f"Deposit: ${deposit:,.2f} | Leverage: {leverage}\n\n"
        f"💰 <b>PORTFOLIO METRICS:</b>\n"
        f"Total risk: ${metrics['total_risk_usd']:.2f} ({metrics['total_risk_percent']:.1f}%)\n"
        f"Potential profit: ${metrics['total_profit']:.2f}\n"
        f"Current P&L: ${metrics['total_pnl']:.2f}\n"
        f"Equity: ${metrics['total_equity']:.2f}\n\n"
        f"🛡 <b>MARGIN:</b>\n"
        f"Required: ${metrics['total_margin']:.2f} ({metrics['total_margin_usage']:.1f}%)\n"
        f"Free: ${metrics['free_margin']:.2f}\n"
        f"Margin level: {metrics['portfolio_margin_level']:.1f}%\n\n"
        f"<b>💡 RECOMMENDATIONS:</b>\n" + "\n".join(f"• {rec}" for rec in recommendations) + "\n\n"
        "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
    )
    
    keyboard = [
        [InlineKeyboardButton("📊 Full Portfolio", callback_data="portfolio")],
        [InlineKeyboardButton("📤 Export Report", callback_data="export_portfolio")],
        [InlineKeyboardButton("🎯 New Multi-position", callback_data="multi_trade_start")],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )
    
    DataManager.clear_temporary_progress(user_id)
    context.user_data.clear()
    return ConversationHandler.END

# ---------------------------
# CALLBACK ROUTER - COMPLETELY FIXED
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def callback_router_fixed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """GUARANTEED WORKING HANDLERS"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    try:
        if data == "main_menu" or data == "main_menu_save":
            await main_menu_save_handler(update, context)
        elif data == "portfolio":
            await show_portfolio(update, context)
        elif data == "pro_calculation":
            await pro_calculation_handler(update, context)
        elif data == "pro_info":
            await pro_info_command(update, context)
        elif data == "pro_info_part2":
            await pro_info_part2(update, context)
        elif data == "future_features":
            await future_features_handler(update, context)
        elif data == "clear_portfolio":
            await clear_portfolio_handler(update, context)
        elif data == "export_portfolio":
            await export_portfolio_handler(update, context)
        elif data == "restore_progress":
            await restore_progress_handler(update, context)
        elif data == "donate_start":
            await DonationSystem.show_donation_menu(update, context)
        elif data == "donate_usdt":
            await DonationSystem.show_usdt_donation(update, context)
        elif data == "donate_ton":
            await DonationSystem.show_ton_donation(update, context)
        elif data == "single_trade":
            await single_trade_start(update, context)
        elif data == "multi_trade_start":
            await multi_trade_start(update, context)
        # Single Trade Callbacks
        elif data.startswith("asset_"):
            await enhanced_single_trade_asset(update, context)
        elif data.startswith("dir_"):
            await enhanced_single_trade_direction(update, context)
        elif data == "back_to_asset":
            await enhanced_single_trade_asset(update, context)
        elif data.startswith("lev_"):
            await single_trade_leverage(update, context)
        elif data.startswith("cat_"):
            await single_trade_asset_category(update, context)
        elif data == "asset_manual":
            await single_trade_asset_category(update, context)
        elif data == "back_to_categories":
            await single_trade_leverage(update, context)
        # Multi Trade Callbacks
        elif data.startswith("massset_"):
            await enhanced_multi_trade_asset(update, context)
        elif data.startswith("mdir_"):
            await enhanced_multi_trade_direction(update, context)
        elif data == "mback_to_asset":
            await enhanced_multi_trade_asset(update, context)
        elif data.startswith("mlev_"):
            await multi_trade_leverage(update, context)
        elif data.startswith("mcat_"):
            await multi_trade_asset_category(update, context)
        elif data == "massset_manual":
            await multi_trade_asset_category(update, context)
        elif data == "mback_to_categories":
            await multi_trade_leverage(update, context)
        elif data == "madd_more":
            await multi_trade_add_more(update, context)
        elif data == "mfinish_multi":
            await multi_trade_finish(update, context)
        else:
            await query.answer("Command not recognized")
            
    except Exception as e:
        logger.error(f"Error in callback router: {e}")
        await query.answer("❌ An error occurred")

# ---------------------------
# ADDITIONAL HANDLERS
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_calculation_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Professional trades handler"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = (
        "🎯 <b>PROFESSIONAL TRADES v3.0</b>\n\n"
        "Choose calculation type:\n\n"
        "• <b>Single Trade</b> - calculation for one position\n"
        "• <b>Multi-position</b> - portfolio calculation from multiple trades\n\n"
        "<i>All cases use fixed 2% risk per trade</i>"
    )
    
    keyboard = [
        [InlineKeyboardButton("🎯 Single Trade", callback_data="single_trade")],
        [InlineKeyboardButton("📊 Multi-position", callback_data="multi_trade_start")],
        [InlineKeyboardButton("🔙 Back", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

@retry_on_timeout(max_retries=2, delay=1.0)
async def main_menu_save_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main menu"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Clear progress on menu access
    DataManager.clear_temporary_progress(query.from_user.id)
    context.user_data.clear()
    
    text = (
        "🏠 <b>MAIN MENU</b>\n\n"
        "Professional risk management calculator with fixed 2% risk\n\n"
        "Choose action:"
    )
    
    keyboard = [
        [InlineKeyboardButton("🎯 Professional Calculation", callback_data="pro_calculation")],
        [InlineKeyboardButton("📊 Portfolio", callback_data="portfolio")],
        [InlineKeyboardButton("📚 Instructions", callback_data="pro_info")],
        [InlineKeyboardButton("💖 Support Developer", callback_data="donate_start")],
        [InlineKeyboardButton("🔄 Restore Progress", callback_data="restore_progress")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

@retry_on_timeout(max_retries=2, delay=1.0)
async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int = None):
    """Show portfolio with real data"""
    query = update.callback_query if update.callback_query else None
    if query:
        await SafeMessageSender.answer_callback_query(query)
    
    if not user_id:
        user_id = query.from_user.id if query else update.message.from_user.id
    
    PortfolioManager.ensure_user(user_id)
    user_portfolio = PortfolioManager.user_data[user_id]
    
    trades = user_portfolio.get('multi_trades', []) + user_portfolio.get('single_trades', [])
    
    if not trades:
        text = (
            "📊 <b>Your portfolio is empty</b>\n\n"
            "Start with trade calculation with fixed 2% risk!"
        )
        keyboard = [
            [InlineKeyboardButton("🎯 New Trade", callback_data="single_trade")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        if query:
            await SafeMessageSender.edit_message_text(
                query,
                text,
                InlineKeyboardMarkup(keyboard)
            )
        else:
            await SafeMessageSender.send_message(
                user_id,
                text,
                context,
                InlineKeyboardMarkup(keyboard)
            )
        return
    
    deposit = user_portfolio['deposit']
    
    # Update metrics with real prices
    for trade in trades:
        metrics = await ProfessionalRiskCalculator.calculate_professional_metrics(
            trade, deposit, user_portfolio['leverage'], "2%"
        )
        trade['metrics'] = metrics
    
    metrics = PortfolioAnalyzer.calculate_portfolio_metrics(trades, deposit)
    recommendations = PortfolioAnalyzer.generate_enhanced_recommendations(metrics, trades)
    
    text = (
        "📊 <b>PORTFOLIO v3.0</b>\n\n"
        f"💰 <b>KEY METRICS:</b>\n"
        f"Deposit: ${deposit:,.2f}\n"
        f"Leverage: {user_portfolio['leverage']}\n"
        f"Trades: {len(trades)}\n"
        f"Equity: ${metrics['total_equity']:.2f}\n\n"
        f"🎯 <b>RISKS AND PROFIT:</b>\n"
        f"Total risk: ${metrics['total_risk_usd']:.2f} ({metrics['total_risk_percent']:.1f}%)\n"
        f"Potential profit: ${metrics['total_profit']:.2f}\n"
        f"Average R/R: {metrics['avg_rr_ratio']:.2f}\n"
        f"Current P&L: ${metrics['total_pnl']:.2f}\n\n"
        f"🛡 <b>MARGIN METRICS:</b>\n"
        f"Required margin: ${metrics['total_margin']:.2f} ({metrics['total_margin_usage']:.1f}%)\n"
        f"Free margin: ${metrics['free_margin']:.2f} ({metrics['free_margin_percent']:.1f}%)\n"
        f"Margin level: {metrics['portfolio_margin_level']:.1f}%\n"
        f"Portfolio leverage: {metrics['portfolio_leverage']:.1f}x\n\n"
        f"📈 <b>ANALYTICS:</b>\n"
        f"Volatility: {metrics['portfolio_volatility']:.1f}%\n"
        f"Longs: {metrics['long_positions']} | Shorts: {metrics['short_positions']}\n"
        f"Unique assets: {metrics['unique_assets']}\n"
        f"Diversification: {metrics['diversity_score']}%\n\n"
        "<b>💡 RECOMMENDATIONS:</b>\n" + "\n".join(f"• {rec}" for rec in recommendations) + "\n\n"
        "<b>📋 TRADES:</b>\n"
    )
    
    for i, trade in enumerate(trades, 1):
        metrics = trade.get('metrics', {})
        pnl = metrics.get('current_pnl', 0)
        pnl_sign = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
        
        sl_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            trade['entry_price'], trade['stop_loss'], metrics.get('volume_lots', 0),
            metrics.get('pip_value', 1), trade['direction'], trade['asset']
        )
        
        tp_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            trade['entry_price'], trade['take_profit'], metrics.get('volume_lots', 0),
            metrics.get('pip_value', 1), trade['direction'], trade['asset']
        )
        
        text += (
            f"{pnl_sign} <b>#{i}</b> {trade['asset']} {trade['direction']}\n"
            f"   Entry: {trade['entry_price']} | SL: {trade['stop_loss']} (${abs(sl_amount):.2f}) | TP: {trade['take_profit']} (${tp_amount:.2f})\n"
            f"   Volume: {metrics.get('volume_lots', 0):.2f} | Risk: ${metrics.get('risk_amount', 0):.2f}\n"
            f"   P&L: ${pnl:.2f} | Margin: ${metrics.get('required_margin', 0):.2f}\n\n"
        )
    
    text += "\n💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
    
    keyboard = [
        [InlineKeyboardButton("🗑 Clear Portfolio", callback_data="clear_portfolio")],
        [InlineKeyboardButton("📤 Export Report", callback_data="export_portfolio")],
        [InlineKeyboardButton("🎯 New Trade", callback_data="single_trade")],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
    ]
    
    if query:
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup(keyboard)
        )
    else:
        await SafeMessageSender.send_message(
            user_id,
            text,
            context,
            InlineKeyboardMarkup(keyboard)
        )

@retry_on_timeout(max_retries=2, delay=1.0)
async def clear_portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear portfolio"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    PortfolioManager.clear_portfolio(user_id)
    
    text = "✅ Portfolio cleared"
    keyboard = [
        [InlineKeyboardButton("🎯 New Trade", callback_data="single_trade")],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

@retry_on_timeout(max_retries=2, delay=1.0)
async def export_portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Export portfolio"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    PortfolioManager.ensure_user(user_id)
    
    user_portfolio = PortfolioManager.user_data[user_id]
    trades = user_portfolio.get('multi_trades', []) + user_portfolio.get('single_trades', [])
    
    if not trades:
        await SafeMessageSender.answer_callback_query(query, "❌ Portfolio is empty")
        return
    
    report = f"📊 PORTFOLIO REPORT v3.0\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    report += f"Deposit: ${user_portfolio['deposit']:,.2f}\n"
    report += f"Leverage: {user_portfolio['leverage']}\n"
    report += f"Total trades: {len(trades)}\n\n"
    
    for i, trade in enumerate(trades, 1):
        report += f"TRADE #{i}:\n"
        report += f"Asset: {trade['asset']}\n"
        report += f"Direction: {trade['direction']}\n"
        report += f"Entry: {trade['entry_price']}\n"
        report += f"SL: {trade['stop_loss']}\n"
        report += f"TP: {trade['take_profit']}\n"
        
        if 'metrics' in trade:
            metrics = trade['metrics']
            sl_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
                trade['entry_price'], trade['stop_loss'], metrics['volume_lots'],
                metrics['pip_value'], trade['direction'], trade['asset']
            )
            
            tp_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
                trade['entry_price'], trade['take_profit'], metrics['volume_lots'],
                metrics['pip_value'], trade['direction'], trade['asset']
            )
            
            report += f"Volume: {metrics['volume_lots']:.2f} lots\n"
            report += f"Risk: ${metrics['risk_amount']:.2f} (2% of deposit)\n"
            report += f"Margin: ${metrics['required_margin']:.2f}\n"
            report += f"Profit: ${metrics['potential_profit']:.2f}\n"
            report += f"R/R: {metrics['rr_ratio']:.2f}\n"
            report += f"P&L: ${metrics['current_pnl']:.2f}\n"
            report += f"SL amount: ${abs(sl_amount):.2f}\n"
            report += f"TP amount: ${tp_amount:.2f}\n"
        
        report += "\n"
    
    report += "💎 PRO v3.0 | Smart • Fast • Reliable 🚀\n"
    
    bio = io.BytesIO()
    bio.write(report.encode('utf-8'))
    bio.seek(0)
    
    try:
        await context.bot.send_document(
            chat_id=query.message.chat_id,
            document=InputFile(bio, filename=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"),
            caption="📊 Your portfolio report"
        )
    except Exception as e:
        logger.error(f"Error sending portfolio report: {e}")
        await SafeMessageSender.answer_callback_query(query, "❌ Export error")

@retry_on_timeout(max_retries=2, delay=1.0)
async def restore_progress_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Progress recovery"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    temp_data = DataManager.load_temporary_data()
    saved_progress = temp_data.get(str(user_id))
    
    if not saved_progress:
        await SafeMessageSender.edit_message_text(
            query,
            "❌ No saved progress",
            InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]])
        )
        return
    
    context.user_data.update(saved_progress['state_data'])
    state_type = saved_progress['state_type']
    
    text = "✅ Progress restored! Continue calculation."
    keyboard = []
    
    if state_type == "single":
        keyboard = [[InlineKeyboardButton("🔄 Continue", callback_data="single_trade")]]
    else:
        keyboard = [[InlineKeyboardButton("🔄 Continue", callback_data="multi_trade_start")]]
    
    keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")])
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

# ---------------------------
# CONVERSATION HANDLERS SETUP
# ---------------------------
def setup_conversation_handlers(application: Application):
    """Setup conversation handlers with real data"""
    
    # Single trade
    single_trade_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(single_trade_start, pattern="^single_trade$")],
        states={
            SingleTradeState.DEPOSIT.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_deposit),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            SingleTradeState.LEVERAGE.value: [
                CallbackQueryHandler(single_trade_leverage, pattern="^lev_"),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            SingleTradeState.ASSET_CATEGORY.value: [
                CallbackQueryHandler(single_trade_asset_category, pattern="^(cat_|asset_manual)"),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            SingleTradeState.ASSET.value: [
                CallbackQueryHandler(enhanced_single_trade_asset, pattern="^(asset_|back_to_categories)"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_asset_manual),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            SingleTradeState.DIRECTION.value: [
                CallbackQueryHandler(enhanced_single_trade_direction, pattern="^dir_"),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            SingleTradeState.ENTRY.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_entry),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            SingleTradeState.STOP_LOSS.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_stop_loss),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            SingleTradeState.TAKE_PROFIT.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_take_profit),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ]
        },
        fallbacks=[
            CommandHandler("cancel", single_trade_cancel),
            MessageHandler(filters.TEXT, single_trade_cancel),
            CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
        ],
        name="single_trade_conversation"
    )
    
    # Multi-position
    multi_trade_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(multi_trade_start, pattern="^multi_trade_start$")],
        states={
            MultiTradeState.DEPOSIT.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_deposit),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            MultiTradeState.LEVERAGE.value: [
                CallbackQueryHandler(multi_trade_leverage, pattern="^mlev_"),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            MultiTradeState.ASSET_CATEGORY.value: [
                CallbackQueryHandler(multi_trade_asset_category, pattern="^(mcat_|massset_manual)"),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            MultiTradeState.ASSET.value: [
                CallbackQueryHandler(enhanced_multi_trade_asset, pattern="^(massset_|mback_to_categories)"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_asset_manual),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            MultiTradeState.DIRECTION.value: [
                CallbackQueryHandler(enhanced_multi_trade_direction, pattern="^mdir_"),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            MultiTradeState.ENTRY.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_entry),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            MultiTradeState.STOP_LOSS.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_stop_loss),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            MultiTradeState.TAKE_PROFIT.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_take_profit),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            MultiTradeState.ADD_MORE.value: [
                CallbackQueryHandler(multi_trade_add_more, pattern="^madd_more$"),
                CallbackQueryHandler(multi_trade_finish, pattern="^mfinish_multi$"),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ]
        },
        fallbacks=[
            CommandHandler("cancel", single_trade_cancel),  # Shared cancel
            CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
        ],
        name="multi_trade_conversation"
    )
    
    application.add_handler(single_trade_conv)
    application.add_handler(multi_trade_conv)

# ---------------------------
# WEBHOOK AND HTTP SERVER
# ---------------------------
async def set_webhook(application: Application) -> bool:
    """Set webhook with verification"""
    try:
        webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
        logger.info(f"Setting webhook: {webhook_url}")
        await application.bot.set_webhook(
            webhook_url,
            allowed_updates=Update.ALL_TYPES
        )
        webhook_info = await application.bot.get_webhook_info()
        logger.info(f"Webhook info: {webhook_info}")
        return True
    except Exception as e:
        logger.error(f"Failed to set webhook: {e}")
        return False

async def start_http_server(application: Application) -> web.AppRunner:
    """Start HTTP server with improved handling"""
    app = web.Application()
    
    async def handle_webhook(request):
        """Webhook handling with timeout"""
        try:
            async with asyncio.timeout(10.0):
                data = await request.text()
                update = Update.de_json(json.loads(data), application.bot)
                await application.process_update(update)
                return web.Response(status=200)
        except asyncio.TimeoutError:
            logger.error("Webhook request timeout")
            return web.Response(status=408)
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return web.Response(status=400)
    
    async def health_check(request):
        """Comprehensive health check"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "3.0",
            "services": {
                "telegram_bot": "operational",
                "market_data": "operational", 
                "database": "operational"
            }
        }
        
        try:
            await application.bot.get_me()
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["services"]["telegram_bot"] = f"error: {str(e)}"
            
        return web.json_response(health_status)
    
    async def render_health_check(request):
        """Simplified health check for Render"""
        return web.Response(text="OK", status=200)
    
    app.router.add_post(WEBHOOK_PATH, handle_webhook)
    app.router.add_get('/health', health_check)
    app.router.add_get('/health/simple', render_health_check)
    app.router.add_get('/', render_health_check)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    
    logger.info(f"HTTP server started on port {PORT}")
    return runner

# ---------------------------
# MAIN LAUNCH FUNCTION
# ---------------------------
async def main_enhanced():
    """Enhanced main function with complete error fixes"""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to start enhanced bot...")
            
            application = RobustApplicationBuilder.create_application(TOKEN)
            
            # Register command handlers
            application.add_handler(CommandHandler("start", start_command))
            application.add_handler(CommandHandler("pro_info", pro_info_command))
            
            # Setup conversations
            setup_conversation_handlers(application)
            
            # Callback router
            application.add_handler(CallbackQueryHandler(callback_router_fixed))
            
            # Handler for any messages (fallback)
            application.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND, 
                lambda update, context: SafeMessageSender.send_message(
                    update.message.chat_id,
                    "🤖 Use menu for navigation or /start to begin",
                    context,
                    InlineKeyboardMarkup([
                        [InlineKeyboardButton("🏠 Main Menu", callback_data="main
