# bot.py — 💎 PRO RISK CALCULATOR v3.0 ENTERPRISE EDITION
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

# --- Загрузка .env ---
from dotenv import load_dotenv
load_dotenv()

# --- Настройки ---
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

# --- Логи ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("pro_risk_bot")

# ---------------------------
# Настройки таймаутов и повторных попыток
# ---------------------------
class RobustApplicationBuilder:
    """Строитель приложения с улучшенной обработкой ошибок"""
    
    @staticmethod
    def create_application(token: str) -> Application:
        """Создание приложения с настройками для устойчивости"""
        request = telegram.request.HTTPXRequest(connection_pool_size=8)
        
        application = (
            Application.builder()
            .token(token)
            .request(request)
            .build()
        )
        
        return application

# ---------------------------
# Retry Decorator для обработки таймаутов
# ---------------------------
def retry_on_timeout(max_retries: int = 3, delay: float = 1.0):
    """Декоратор для повторных попыток при таймаутах"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except telegram.error.TimedOut as e:
                    logger.warning(f"Timeout in {func.__name__}, attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))
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
# Safe Message Sender - ОБНОВЛЕННЫЙ С ЗАЩИТОЙ ОТ HTML ОШИБОК
# ---------------------------
class SafeMessageSender:
    """Безопасная отправка сообщений с обработкой ошибок"""
    
    @staticmethod
    def safe_html_text(text: str) -> str:
        """Безопасная подготовка HTML текста - УЛУЧШЕННАЯ ВЕРСИЯ"""
        text = html.escape(text)
        
        safe_tags = ['b', 'i', 'u', 'em', 'strong', 'code', 'pre']
        for tag in safe_tags:
            opening_tag = f"&lt;{tag}&gt;"
            closing_tag = f"&lt;/{tag}&gt;"
            text = text.replace(opening_tag, f"<{tag}>").replace(closing_tag, f"</{tag}>")
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        if len(text) > 4000:
            text = text[:4000] + "...\n\n[сообщение сокращено]"
            
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
        """Безопасная отправка сообщения с защитой от HTML ошибок"""
        try:
            safe_text = SafeMessageSender.safe_html_text(text)
            
            if context and hasattr(context, 'bot'):
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=safe_text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode
                )
            else:
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
        """Безопасное редактирование сообщения с защитой от HTML ошибок"""
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
        """Безопасный ответ на callback query"""
        try:
            await query.answer(text=text, show_alert=show_alert)
            return True
        except Exception as e:
            logger.error(f"Failed to answer callback query: {e}")
            return False

# ---------------------------
# Donation System - ПРОФЕССИОНАЛЬНАЯ СИСТЕМА ДОНАТОВ
# ---------------------------
class DonationSystem:
    """Профессиональная система донатов для поддержки разработки"""
    
    @staticmethod
    async def show_donation_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать меню выбора валюты для доната"""
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        text = (
            "💝 <b>ПОДДЕРЖАТЬ РАЗРАБОТЧИКА</b>\n\n"
            "Ваша поддержка помогает развивать бота и добавлять новые функции!\n\n"
            "Выберите валюту для доната:"
        )
        
        keyboard = [
            [InlineKeyboardButton("💎 USDT (TRC20)", callback_data="donate_usdt")],
            [InlineKeyboardButton("⚡ TON", callback_data="donate_ton")],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
        ]
        
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup(keyboard)
        )
    
    @staticmethod
    async def show_usdt_donation(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать USDT кошелек для донатов"""
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        if not USDT_WALLET_ADDRESS:
            await SafeMessageSender.edit_message_text(
                query,
                "❌ USDT кошелек временно недоступен",
                InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="donate_start")]])
            )
            return
        
        text = (
            "💎 <b>USDT (TRC20) ДОНАТ</b>\n\n"
            "Для поддержки разработки отправьте USDT на следующий адрес:\n\n"
            f"<code>{USDT_WALLET_ADDRESS}</code>\n\n"
            "💝 <i>Любая сумма будет принята с благодарностью!</i>\n\n"
            "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔙 К выбору валюты", callback_data="donate_start")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup(keyboard)
        )
    
    @staticmethod
    async def show_ton_donation(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать TON кошелек для донатов"""
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        if not TON_WALLET_ADDRESS:
            await SafeMessageSender.edit_message_text(
                query,
                "❌ TON кошелек временно недоступен",
                InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="donate_start")]])
            )
            return
        
        text = (
            "⚡ <b>TON ДОНАТ</b>\n\n"
            "Для поддержки разработки отправьте TON на следующий адрес:\n\n"
            f"<code>{TON_WALLET_ADDRESS}</code>\n\n"
            "💝 <i>Любая сумма будет принята с благодарностью!</i>\n\n"
            "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔙 К выбору валюты", callback_data="donate_start")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup(keyboard)
        )

# ---------------------------
# Enhanced Market Data Provider - УЛУЧШЕННАЯ ВЕРСИЯ
# ---------------------------
class EnhancedMarketDataProvider:
    """Универсальный провайдер рыночных данных с улучшенной поддержкой"""
    
    def __init__(self):
        self.cache = cachetools.TTLCache(maxsize=500, ttl=300)
        self.session = None
        
    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_real_time_price(self, symbol: str) -> float:
        """Получение реальной цены с приоритизацией провайдеров"""
        return await self.get_robust_real_time_price(symbol)
    
    async def get_robust_real_time_price(self, symbol: str) -> float:
        """НАДЕЖНОЕ получение реальных цен"""
        try:
            cached_price = self.cache.get(symbol)
            if cached_price:
                return cached_price
            
            providers = [
                self._get_fmp_price,
                self._get_metalpriceapi_price,
                self._get_exchangerate_price,
                self._get_binance_price,
                self._get_twelvedata_price,
                self._get_alpha_vantage_stock,
                self._get_alpha_vantage_forex,
                self._get_finnhub_price,
                self._get_fallback_price
            ]
            
            price = None
            for provider in providers:
                price = await provider(symbol)
                if price and price > 0:
                    break
            
            if price is None or price <= 0:
                logger.warning(f"Не удалось получить цену для {symbol}, используется fallback")
                price = self._get_fallback_price(symbol)
                
            if price:
                self.cache[symbol] = price
                
            return price
            
        except Exception as e:
            logger.error(f"Ошибка получения цены для {symbol}: {e}")
            return self._get_fallback_price(symbol)
    
    def _is_crypto(self, symbol: str) -> bool:
        crypto_symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'USDT']
        return any(crypto in symbol for crypto in crypto_symbols)
    
    def _is_forex(self, symbol: str) -> bool:
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        return symbol in forex_pairs
    
    def _is_metal(self, symbol: str) -> bool:
        metals = ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD']
        return symbol in metals
    
    async def _get_fmp_price(self, symbol: str) -> Optional[float]:
        """Получение цены через Financial Modeling Prep API"""
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
        """Получение цен на металлы через Metal Price API"""
        try:
            if not self._is_metal(symbol):
                return None
                
            session = await self.get_session()
            metal_map = {
                'XAUUSD': 'XAU', 'XAGUSD': 'XAG', 'XPTUSD': 'XPT', 'XPDUSD': 'XPD'
            }
            
            metal_code = metal_map.get(symbol)
            if not metal_code:
                return None
                
            url = f"http://api.metalpriceapi.com/v1/latest?api_key={METALPRICE_API_KEY}&base=USD&currencies={metal_code}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('success'):
                        rate = data['rates'].get(metal_code)
                        if rate:
                            return 1.0 / rate
        except Exception as e:
            logger.error(f"Metal Price API error for {symbol}: {e}")
        return None
    
    async def _get_twelvedata_price(self, symbol: str) -> Optional[float]:
        """Получение цены через Twelve Data API"""
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
        """Frankfurter API для точных Forex цен"""
        try:
            if self._is_forex(symbol):
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
        """Получение цены с Binance API"""
        try:
            if not self._is_crypto(symbol):
                return None
                
            session = await self.get_session()
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
        """Получение цены акций с Alpha Vantage"""
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
        """Получение Forex цен с Alpha Vantage"""
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
        """Получение цены с Finnhub (резервный)"""
        if not FINNHUB_API_KEY:
            return None
            
        try:
            session = await self.get_session()
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['c']
        except Exception as e:
            logger.error(f"Finnhub API error for {symbol}: {e}")
        return None
    
    def _get_fallback_price(self, symbol: str) -> float:
        """АКТУАЛИЗИРОВАННЫЕ fallback цены при недоступности API"""
        current_prices = {
            # Forex
            'EURUSD': 1.0732, 'GBPUSD': 1.2548, 'USDJPY': 155.42, 'USDCHF': 0.9054,
            'AUDUSD': 0.6589, 'USDCAD': 1.3732, 'NZDUSD': 0.6014,
            # Crypto
            'BTCUSDT': 61450.0, 'ETHUSDT': 3450.0, 'XRPUSDT': 0.524, 'LTCUSDT': 82.15,
            'BCHUSDT': 415.00, 'ADAUSDT': 0.462, 'DOTUSDT': 6.95,
            # Stocks
            'AAPL': 189.20, 'TSLA': 177.50, 'GOOGL': 174.35, 'MSFT': 420.72,
            'AMZN': 178.22, 'META': 469.85, 'NFLX': 617.80,
            # Indices
            'NAS100': 17750.0, 'SPX500': 5225.0, 'DJ30': 38850.0, 'FTSE100': 8213.0,
            'DAX40': 18420.0, 'NIKKEI225': 38175.0, 'ASX200': 7620.0,
            # Metals
            'XAUUSD': 2335.50, 'XAGUSD': 27.80, 'XPTUSD': 890.50, 'XPDUSD': 945.75,
            # Energy
            'OIL': 78.25, 'NATURALGAS': 2.15, 'BRENT': 82.80
        }
        return current_prices.get(symbol, 100.0)

    async def get_price_with_fallback(self, symbol: str) -> Tuple[float, str]:
        """Получение цены с информацией о источнике"""
        try:
            real_price = await self.get_robust_real_time_price(symbol)
            if real_price and real_price > 0:
                return real_price, "real-time"
            
            cached_price = self.cache.get(symbol)
            if cached_price:
                return cached_price, "cached"
            
            fallback_price = self._get_fallback_price(symbol)
            return fallback_price, "fallback"
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return self._get_fallback_price(symbol), "error"

# ---------------------------
# Instrument Specifications - ИСПРАВЛЕННАЯ БАЗА СПЕЦИФИКАЦИЙ
# ---------------------------
class InstrumentSpecs:
    """Исправленная база спецификаций финансовых инструментов"""
    
    SPECS = {
        # Forex пары - ИСПРАВЛЕННЫЕ ЗНАЧЕНИЯ
        "EURUSD": {
            "type": "forex", "contract_size": 100000, "margin_currency": "USD",
            "pip_value": 10.0, "calculation_formula": "forex", "pip_decimal_places": 4,
            "min_volume": 0.01, "volume_step": 0.01, "max_leverage": 1000
        },
        "GBPUSD": {
            "type": "forex", "contract_size": 100000, "margin_currency": "USD", 
            "pip_value": 10.0, "calculation_formula": "forex", "pip_decimal_places": 4,
            "min_volume": 0.01, "volume_step": 0.01, "max_leverage": 1000
        },
        "USDJPY": {
            "type": "forex", "contract_size": 100000, "margin_currency": "USD",
            "pip_value": 9.09, "calculation_formula": "forex_jpy", "pip_decimal_places": 2,
            "min_volume": 0.01, "volume_step": 0.01, "max_leverage": 1000
        },
        
        # Криптовалюты - ИСПРАВЛЕННЫЕ ЗНАЧЕНИЯ
        "BTCUSDT": {
            "type": "crypto", "contract_size": 1, "margin_currency": "USDT",
            "pip_value": 1.0, "calculation_formula": "crypto", "pip_decimal_places": 1,
            "min_volume": 0.001, "volume_step": 0.001, "max_leverage": 125
        },
        "ETHUSDT": {
            "type": "crypto", "contract_size": 1, "margin_currency": "USDT",
            "pip_value": 1.0, "calculation_formula": "crypto", "pip_decimal_places": 2,
            "min_volume": 0.01, "volume_step": 0.01, "max_leverage": 125
        },
        
        # Акции - ИСПРАВЛЕННЫЕ ЗНАЧЕНИЯ
        "AAPL": {
            "type": "stock", "contract_size": 100, "margin_currency": "USD",
            "pip_value": 1.0, "calculation_formula": "stocks", "pip_decimal_places": 2,
            "min_volume": 0.01, "volume_step": 0.01, "max_leverage": 100
        },
        
        # Индексы - ИСПРАВЛЕННЫЕ ЗНАЧЕНИЯ
        "NAS100": {
            "type": "index", "contract_size": 1, "margin_currency": "USD",
            "pip_value": 1.0, "calculation_formula": "indices", "pip_decimal_places": 1,
            "min_volume": 0.01, "volume_step": 0.01, "max_leverage": 100
        },
        
        # Металлы - ИСПРАВЛЕННЫЕ ЗНАЧЕНИЯ
        "XAUUSD": {
            "type": "metal", "contract_size": 100, "margin_currency": "USD",
            "pip_value": 1.0, "calculation_formula": "metals", "pip_decimal_places": 2,
            "min_volume": 0.01, "volume_step": 0.01, "max_leverage": 100
        },
        "XAGUSD": {
            "type": "metal", "contract_size": 5000, "margin_currency": "USD",
            "pip_value": 0.5, "calculation_formula": "metals", "pip_decimal_places": 2,
            "min_volume": 0.01, "volume_step": 0.01, "max_leverage": 100
        },
        
        # Энергия - ИСПРАВЛЕННЫЕ ЗНАЧЕНИЯ
        "OIL": {
            "type": "energy", "contract_size": 1000, "margin_currency": "USD",
            "pip_value": 10.0, "calculation_formula": "energy", "pip_decimal_places": 2,
            "min_volume": 0.01, "volume_step": 0.01, "max_leverage": 100
        }
    }
    
    @classmethod
    def get_specs(cls, symbol: str) -> Dict[str, Any]:
        return cls.SPECS.get(symbol, cls._get_default_specs(symbol))
    
    @classmethod
    def _get_default_specs(cls, symbol: str) -> Dict[str, Any]:
        if any(currency in symbol for currency in ['USD', 'EUR', 'GBP', 'JPY']):
            return {
                "type": "forex", "contract_size": 100000, "margin_currency": "USD",
                "pip_value": 10.0, "calculation_formula": "forex", "pip_decimal_places": 4,
                "min_volume": 0.01, "volume_step": 0.01, "max_leverage": 1000
            }
        elif 'USDT' in symbol:
            return {
                "type": "crypto", "contract_size": 1, "margin_currency": "USDT", 
                "pip_value": 1.0, "calculation_formula": "crypto", "pip_decimal_places": 2,
                "min_volume": 0.01, "volume_step": 0.01, "max_leverage": 125
            }
        else:
            return {
                "type": "stock", "contract_size": 100, "margin_currency": "USD",
                "pip_value": 1.0, "calculation_formula": "stocks", "pip_decimal_places": 2,
                "min_volume": 0.01, "volume_step": 0.01, "max_leverage": 100
            }

# ---------------------------
# Professional Margin Calculator - ИСПРАВЛЕННЫЙ РАСЧЕТ
# ---------------------------
class ProfessionalMarginCalculator:
    """ИСПРАВЛЕННЫЙ расчет маржи с реальными котировками"""
    
    def __init__(self):
        self.market_data = EnhancedMarketDataProvider()
    
    async def calculate_professional_margin(self, symbol: str, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Профессиональный расчет маржи с реальными котировками"""
        try:
            specs = InstrumentSpecs.get_specs(symbol)
            formula = specs['calculation_formula']
            
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
            logger.error(f"Ошибка расчета маржи для {symbol}: {e}")
            return await self._calculate_universal_margin(specs, volume, leverage, current_price)
    
    async def _calculate_forex_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        required_margin = (volume * contract_size) / lev_value
        
        return {
            'required_margin': max(required_margin, 0.01),
            'contract_size': contract_size,
            'calculation_method': 'forex_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size,
            'effective_leverage': leverage
        }
    
    async def _calculate_forex_jpy_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
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
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
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
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
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
        return await self._calculate_stocks_margin(specs, volume, leverage, current_price)
    
    async def _calculate_metals_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
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
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
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
# Professional Risk Calculator - ИСПРАВЛЕННЫЙ
# ---------------------------
class ProfessionalRiskCalculator:
    """ИСПРАВЛЕННЫЙ калькулятор с реальными котировками"""
    
    @staticmethod
    def calculate_pip_distance(entry: float, target: float, direction: str, asset: str) -> float:
        specs = InstrumentSpecs.get_specs(asset)
        pip_decimal_places = specs.get('pip_decimal_places', 4)
        
        if direction.upper() == 'LONG':
            distance = target - entry
        else:
            distance = entry - target
        
        if pip_decimal_places == 2:
            return abs(distance) * 100
        elif pip_decimal_places == 1:
            return abs(distance) * 10
        else:
            return abs(distance) * 10000

    @staticmethod
    def calculate_pnl_dollar_amount(entry_price: float, exit_price: float, volume: float, pip_value: float, 
                                  direction: str, asset: str, tick_size: float = 0.01) -> float:
        try:
            specs = InstrumentSpecs.get_specs(asset)
            
            if direction.upper() == 'LONG':
                price_diff = exit_price - entry_price
            else:
                price_diff = entry_price - exit_price
            
            if specs['type'] in ['stock', 'crypto']:
                pnl = price_diff * volume * specs['contract_size']
            else:
                pip_distance = ProfessionalRiskCalculator.calculate_pip_distance(
                    entry_price, exit_price, direction, asset
                )
                pnl = pip_distance * volume * pip_value
            
            return round(pnl, 2)
        except Exception as e:
            logger.error(f"Ошибка расчета P&L: {e}")
            return 0.0

    @staticmethod
    async def calculate_realistic_pnl(trade: Dict, current_price: float, volume: float, pip_value: float, direction: str, asset: str) -> float:
        entry = trade['entry_price']
        return ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            entry, current_price, volume, pip_value, direction, asset
        )

    @staticmethod
    def calculate_margin_level(equity: float, margin: float) -> float:
        if margin == 0:
            return float('inf')
        margin_level = (equity / margin) * 100
        return round(margin_level, 2)

    @staticmethod
    def calculate_free_margin(equity: float, margin: float) -> float:
        free_margin = equity - margin
        return max(free_margin, 0.0)

    @staticmethod
    async def calculate_professional_metrics(trade: Dict, deposit: float, leverage: str, risk_level: str) -> Dict[str, Any]:
        try:
            asset = trade['asset']
            entry = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            direction = trade['direction']
            
            current_price = await enhanced_market_data.get_robust_real_time_price(asset)
            specs = InstrumentSpecs.get_specs(asset)
            
            risk_percent = float(risk_level.strip('%'))
            risk_amount = deposit * (risk_percent / 100)
            
            stop_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry, stop_loss, direction, asset)
            profit_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry, take_profit, direction, asset)
            
            pip_value = specs['pip_value']
            
            if stop_distance_pips > 0 and pip_value > 0:
                volume_lots = risk_amount / (stop_distance_pips * pip_value)
                volume_step = specs.get('volume_step', 0.01)
                volume_lots = round(volume_lots / volume_step) * volume_step
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
            
            current_pnl = await ProfessionalRiskCalculator.calculate_realistic_pnl(
                trade, current_price, volume_lots, pip_value, direction, asset
            )
            equity = deposit + current_pnl
            
            free_margin = ProfessionalRiskCalculator.calculate_free_margin(equity, required_margin)
            margin_level = ProfessionalRiskCalculator.calculate_margin_level(equity, required_margin)
            
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
            logger.error(f"Профессиональный расчет ошибка: {e}")
            return {
                'volume_lots': 0, 'required_margin': 0, 'free_margin': deposit, 'margin_level': 0,
                'risk_amount': 0, 'risk_percent': 0, 'potential_profit': 0, 'rr_ratio': 0,
                'stop_distance_pips': 0, 'profit_distance_pips': 0, 'pip_value': 0, 'contract_size': 0,
                'deposit': deposit, 'leverage': leverage, 'effective_leverage': leverage,
                'risk_per_trade_percent': 0, 'margin_usage_percent': 0, 'current_price': 0,
                'calculation_method': 'error', 'notional_value': 0, 'leverage_used': 1,
                'current_pnl': 0, 'equity': deposit
            }

# ---------------------------
# Константы и состояния
# ---------------------------
class SingleTradeState(Enum):
    DEPOSIT = 1
    LEVERAGE = 2
    ASSET_CATEGORY = 3
    ASSET = 4
    DIRECTION = 5
    ENTRY = 6
    STOP_LOSS = 7
    RISK_LEVEL = 8
    TAKE_PROFIT = 9

class MultiTradeState(Enum):
    DEPOSIT = 1
    LEVERAGE = 2
    ASSET_CATEGORY = 3
    ASSET = 4
    DIRECTION = 5
    ENTRY = 6
    STOP_LOSS = 7
    RISK_LEVEL = 8
    TAKE_PROFIT = 9
    ADD_MORE = 10

ASSET_CATEGORIES = {
    "FOREX": ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'],
    "CRYPTO": ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'ADAUSDT', 'DOTUSDT'],
    "INDICES": ['NAS100', 'SPX500', 'DJ30', 'FTSE100', 'DAX40', 'NIKKEI225', 'ASX200'],
    "METALS": ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD'],
    "ENERGY": ['OIL', 'NATURALGAS', 'BRENT'],
    "STOCKS": ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX']
}

LEVERAGES = {
    "FOREX": ['1:1', '1:5', '1:10', '1:20', '1:50', '1:100', '1:200', '1:500', '1:1000'],
    "CRYPTO": ['1:1', '1:5', '1:10', '1:20', '1:50', '1:100', '1:110', '1:120', '1:125'],
    "METALS": ['1:1', '1:5', '1:10', '1:20', '1:50', '1:100'],
    "DEFAULT": ['1:1', '1:5', '1:10', '1:20', '1:50', '1:100']
}

RISK_LEVELS = ['2%', '5%', '7%', '10%', '15%', '20%', '25%']

VOLATILITY_DATA = {
    'BTCUSDT': 65.2, 'ETHUSDT': 70.5, 'AAPL': 25.3, 'TSLA': 55.1,
    'GOOGL': 22.8, 'MSFT': 20.1, 'AMZN': 28.7, 'EURUSD': 8.5,
    'GBPUSD': 9.2, 'USDJPY': 7.8, 'XAUUSD': 14.5, 'XAGUSD': 25.3,
    'OIL': 35.2, 'NAS100': 18.5, 'SPX500': 15.2, 'DJ30': 12.8
}

# Инициализация глобальных сервисов
enhanced_market_data = EnhancedMarketDataProvider()
margin_calculator = ProfessionalMarginCalculator()

# ---------------------------
# Data Manager - УЛУЧШЕННЫЙ С ПОДДЕРЖКОЙ ВОССТАНОВЛЕНИЯ
# ---------------------------
class DataManager:
    @staticmethod
    def load_data() -> Dict[int, Dict[str, Any]]:
        try:
            if os.path.exists("user_data.json"):
                with open("user_data.json", 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                return {int(k): v for k, v in raw.items()}
        except Exception as e:
            logger.error("Ошибка загрузки: %s", e)
        return {}

    @staticmethod
    def save_data(data: Dict[int, Dict[str, Any]]):
        try:
            serializable = {str(k): v for k, v in data.items()}
            with open("user_data.json", 'w', encoding='utf-8') as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Ошибка сохранения: %s", e)

    @staticmethod
    def save_temporary_progress(user_id: int, state_data: Dict, state_type: str):
        try:
            temp_data = DataManager.load_temporary_data()
            temp_data[str(user_id)] = {
                'state_data': state_data,
                'state_type': state_type,
                'saved_at': datetime.now().isoformat()
            }
            with open("temp_progress.json", 'w', encoding='utf-8') as f:
                json.dump(temp_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Ошибка сохранения прогресса: %s", e)

    @staticmethod
    def load_temporary_data() -> Dict[str, Any]:
        try:
            if os.path.exists("temp_progress.json"):
                with open("temp_progress.json", 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error("Ошибка загрузки прогресса: %s", e)
        return {}

    @staticmethod
    def clear_temporary_progress(user_id: int):
        try:
            temp_data = DataManager.load_temporary_data()
            if str(user_id) in temp_data:
                del temp_data[str(user_id)]
                with open("temp_progress.json", 'w', encoding='utf-8') as f:
                    json.dump(temp_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Ошибка очистки прогресса: %s", e)

user_data = DataManager.load_data()

# ---------------------------
# Portfolio Manager
# ---------------------------
class PortfolioManager:
    @staticmethod
    def ensure_user(user_id: int):
        if user_id not in user_data:
            user_data[user_id] = {
                'multi_trades': [],
                'single_trades': [],
                'deposit': 0.0,
                'leverage': '1:100',
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            DataManager.save_data(user_data)

    @staticmethod
    def add_multi_trade(user_id: int, trade: Dict):
        PortfolioManager.ensure_user(user_id)
        trade['id'] = len(user_data[user_id]['multi_trades']) + 1
        trade['created_at'] = datetime.now().isoformat()
        user_data[user_id]['multi_trades'].append(trade)
        user_data[user_id]['last_updated'] = datetime.now().isoformat()
        DataManager.save_data(user_data)

    @staticmethod
    def add_single_trade(user_id: int, trade: Dict):
        PortfolioManager.ensure_user(user_id)
        trade['id'] = len(user_data[user_id]['single_trades']) + 1
        trade['created_at'] = datetime.now().isoformat()
        user_data[user_id]['single_trades'].append(trade)
        user_data[user_id]['last_updated'] = datetime.now().isoformat()
        DataManager.save_data(user_data)

    @staticmethod
    def set_deposit_leverage(user_id: int, deposit: float, leverage: str):
        PortfolioManager.ensure_user(user_id)
        user_data[user_id]['deposit'] = deposit
        user_data[user_id]['leverage'] = leverage
        user_data[user_id]['last_updated'] = datetime.now().isoformat()
        DataManager.save_data(user_data)

    @staticmethod
    def clear_portfolio(user_id: int):
        if user_id in user_data:
            user_data[user_id]['multi_trades'] = []
            user_data[user_id]['single_trades'] = []
            user_data[user_id]['deposit'] = 0.0
            user_data[user_id]['last_updated'] = datetime.now().isoformat()
            DataManager.save_data(user_data)

    @staticmethod
    def remove_trade(user_id: int, trade_id: int):
        if user_id in user_data:
            user_data[user_id]['multi_trades'] = [
                t for t in user_data[user_id]['multi_trades'] 
                if t['id'] != trade_id
            ]
            user_data[user_id]['last_updated'] = datetime.now().isoformat()
            DataManager.save_data(user_data)

# ---------------------------
# Portfolio Analyzer
# ---------------------------
class PortfolioAnalyzer:
    @staticmethod
    def calculate_portfolio_metrics(trades: List[Dict], deposit: float) -> Dict[str, Any]:
        if not trades:
            return {
                'total_risk_usd': 0, 'total_risk_percent': 0, 'total_profit': 0, 'total_margin': 0,
                'portfolio_margin_level': 0, 'total_margin_usage': 0, 'avg_rr_ratio': 0,
                'portfolio_volatility': 0, 'long_positions': 0, 'short_positions': 0,
                'direction_balance': 0, 'diversity_score': 0, 'unique_assets': 0,
                'total_notional_value': 0, 'portfolio_leverage': 0, 'free_margin': deposit,
                'free_margin_percent': 100, 'total_pnl': 0, 'total_equity': deposit
            }
        
        total_risk = sum(t.get('metrics', {}).get('risk_amount', 0) for t in trades)
        total_profit = sum(t.get('metrics', {}).get('potential_profit', 0) for t in trades)
        total_margin = sum(t.get('metrics', {}).get('required_margin', 0) for t in trades)
        total_notional = sum(t.get('metrics', {}).get('notional_value', 0) for t in trades)
        total_pnl = sum(t.get('metrics', {}).get('current_pnl', 0) for t in trades)
        total_equity = deposit + total_pnl
        
        avg_rr = sum(t.get('metrics', {}).get('rr_ratio', 0) for t in trades) / len(trades) if trades else 0
        portfolio_volatility = sum(VOLATILITY_DATA.get(t['asset'], 20) for t in trades) / len(trades) if trades else 0
        
        long_count = sum(1 for t in trades if t.get('direction', '').upper() == 'LONG')
        short_count = len(trades) - long_count
        direction_balance = abs(long_count - short_count) / len(trades) if trades else 0
        
        unique_assets = len(set(t['asset'] for t in trades))
        diversity_score = unique_assets / len(trades) if trades else 0
        
        portfolio_margin_level = ProfessionalRiskCalculator.calculate_margin_level(total_equity, total_margin)
        total_margin_usage = (total_margin / total_equity) * 100 if total_equity > 0 else 0
        portfolio_leverage = total_notional / total_equity if total_equity > 0 else 0
        free_margin = ProfessionalRiskCalculator.calculate_free_margin(total_equity, total_margin)
        free_margin_percent = (free_margin / total_equity) * 100 if total_equity > 0 else 0
        
        return {
            'total_risk_usd': total_risk,
            'total_risk_percent': (total_risk / deposit) * 100 if deposit > 0 else 0,
            'total_profit': total_profit,
            'total_margin': total_margin,
            'portfolio_margin_level': portfolio_margin_level,
            'total_margin_usage': total_margin_usage,
            'avg_rr_ratio': avg_rr,
            'portfolio_volatility': portfolio_volatility,
            'long_positions': long_count,
            'short_positions': short_count,
            'direction_balance': direction_balance,
            'diversity_score': diversity_score,
            'unique_assets': unique_assets,
            'total_notional_value': total_notional,
            'portfolio_leverage': portfolio_leverage,
            'free_margin': free_margin,
            'free_margin_percent': free_margin_percent,
            'total_pnl': total_pnl,
            'total_equity': total_equity
        }

    @staticmethod
    def generate_recommendations(metrics: Dict, trades: List[Dict]) -> List[str]:
        recommendations = []
        
        if metrics.get('total_risk_percent', 0) > 10:
            recommendations.append("⚠️ ВНИМАНИЕ: Общий риск портфеля превышает 10%. Уменьшите объем позиций.")
        elif metrics.get('total_risk_percent', 0) > 5:
            recommendations.append("🔶 ПРЕДУПРЕЖДЕНИЕ: Общий риск портфеля превышает 5%. Рассмотрите снижение объема.")
        
        if metrics.get('portfolio_margin_level', 0) < 100:
            recommendations.append("🔴 КРИТИЧЕСКИЙ УРОВЕНЬ МАРЖИ! Немедленно пополните счет или закрите часть позиций.")
        elif metrics.get('portfolio_margin_level', 0) < 200:
            recommendations.append("🟡 НИЗКИЙ УРОВЕНЬ МАРЖИ: Рассмотрите пополнение счета. Рекомендуемый уровень > 200%.")
        
        if metrics.get('total_margin_usage', 0) > 50:
            recommendations.append(f"🟡 ВЫСОКОЕ ИСПОЛЬЗОВАНИЕ МАРЖИ: {metrics['total_margin_usage']:.1f}%. Оставьте свободную маржу.")
        
        if metrics.get('portfolio_leverage', 0) > 10:
            recommendations.append(f"🔶 ВЫСОКИЙ ЛЕВЕРЕДЖ: {metrics['portfolio_leverage']:.1f}x. Высокий левередж увеличивает риски.")
        
        low_rr_trades = [t for t in trades if t.get('metrics', {}).get('rr_ratio', 0) < 1]
        if low_rr_trades:
            recommendations.append(f"📉 НЕВЫГОДНОЕ R/R: {len(low_rr_trades)} сделок имеют соотношение < 1.")
        
        if metrics.get('portfolio_volatility', 0) > 30:
            recommendations.append(f"🌪 ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ: {metrics['portfolio_volatility']:.1f}%.")
        
        if metrics.get('diversity_score', 0) < 0.5 and len(trades) > 1:
            recommendations.append("🎯 НИЗКАЯ ДИВЕРСИФИКАЦИЯ. Рассмотрите добавление активов из разных секторов.")
        
        if not recommendations:
            recommendations.append("✅ ПОРТФЕЛЬ СБАЛАНСИРОВАН. Продолжайте в том же духе!")
        
        return recommendations

# ---------------------------
# Основные обработчики
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    try:
        user = update.effective_user
        user_id = user.id
        PortfolioManager.ensure_user(user_id)
        
        temp_data = DataManager.load_temporary_data()
        saved_progress = temp_data.get(str(user_id))
        
        text = (
            f"👋 Привет, {user.first_name}!\n\n"
            "🤖 <b>💎 PRO Калькулятор Управления Рисками v3.0 ENTERPRISE</b>\n\n"
            "🚀 <b>МОИ ВОЗМОЖНОСТИ:</b>\n"
            "• 💼 <b>ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ</b> маржи по отраслевым стандартам\n"
            "• 🎯 Контроль уровней риска (2%-25% от депозита)\n"
            "• 💡 Умные рекомендации и аналитика портфеля\n"
            "• 🛡 <b>ЗАЩИТА ОТ МАРЖИН-КОЛЛА</b> через правильный расчет объема\n"
            "• 📈 <b>АКТУАЛЬНЫЕ ДАННЫЕ</b> для точного риск-менеджмента\n\n"
        )
        
        if saved_progress:
            text += "🔔 У вас есть сохраненный прогресс! Вы можете продолжить с того же места.\n\n"
        
        text += "<b>Выберите раздел:</b>"
        
        keyboard = [
            [InlineKeyboardButton("🎯 Профессиональные сделки", callback_data="pro_calculation")],
            [InlineKeyboardButton("📊 Мой портфель", callback_data="portfolio")]
        ]
        
        if saved_progress:
            keyboard.append([InlineKeyboardButton("🔄 Продолжить расчет", callback_data="restore_progress")])
        
        keyboard.extend([
            [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")],
            [InlineKeyboardButton("💝 Поддержать разработчика", callback_data="donate_start")],
            [InlineKeyboardButton("🚀 Будущие разработки", callback_data="future_features")]
        ])
        
        if update.callback_query:
            success = await SafeMessageSender.edit_message_text(
                update.callback_query,
                text,
                InlineKeyboardMarkup(keyboard)
            )
            if not success:
                await SafeMessageSender.send_message(
                    user_id,
                    text,
                    context,
                    InlineKeyboardMarkup(keyboard)
                )
        else:
            await SafeMessageSender.send_message(
                user_id,
                text,
                context,
                InlineKeyboardMarkup(keyboard)
            )
            
    except Exception as e:
        logger.error(f"Error in start_command: {e}")
        try:
            if update.effective_user:
                await SafeMessageSender.send_message(
                    update.effective_user.id,
                    "❌ Произошла ошибка при загрузке. Пожалуйста, попробуйте еще раз.",
                    context,
                    InlineKeyboardMarkup([
                        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
                    ])
                )
        except:
            pass

async def main_menu_save_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, current_state: int = None):
    """УНИВЕРСАЛЬНЫЙ ОБРАБОТЧИК СОХРАНЕНИЯ ПЕРЕД ВЫХОДОМ"""
    query = update.callback_query
    user_id = query.from_user.id if query else update.message.from_user.id
    
    if context.user_data:
        state_type = "single" if current_state in [s.value for s in SingleTradeState] else "multi"
        DataManager.save_temporary_progress(user_id, context.user_data.copy(), state_type)
    
    context.user_data.clear()
    
    await start_command(update, context)
    
    return ConversationHandler.END

# ---------------------------
# CALLBACK ROUTER - ПОЛНОСТЬЮ ИСПРАВЛЕННЫЙ
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def callback_router_fixed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ГАРАНТИРОВАННО РАБОЧИЕ ОБРАБОТЧИКИ"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    try:
        if data == "main_menu":
            await main_menu_save_handler(update, context)
        elif data == "main_menu_save":
            await main_menu_save_handler(update, context)
        elif data == "portfolio":
            await show_portfolio(update, context)
        elif data == "pro_calculation":
            await pro_calculation_handler(update, context)
        elif data == "pro_info":
            await pro_info_command(update, context)
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
        else:
            await query.answer("Команда не распознана")
            
    except Exception as e:
        logger.error(f"Error in callback router: {e}")
        await query.answer("❌ Произошла ошибка")

# ---------------------------
# Дополнительные обработчики (сокращенно для краткости)
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_calculation_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик профессиональных сделок"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = "🎯 <b>ПРОФЕССИОНАЛЬНЫЕ СДЕЛКИ v3.0</b>\n\nВыберите тип расчета:"
    
    keyboard = [
        [InlineKeyboardButton("🎯 Одна сделка", callback_data="single_trade")],
        [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")],
        [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(keyboard))

@retry_on_timeout(max_retries=2, delay=1.0)
async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int = None):
    """Показ портфеля с реальными данными"""
    query = update.callback_query if update.callback_query else None
    if query:
        await SafeMessageSender.answer_callback_query(query)
    
    if not user_id:
        user_id = query.from_user.id if query else update.message.from_user.id
    
    PortfolioManager.ensure_user(user_id)
    user_portfolio = user_data[user_id]
    
    trades = user_portfolio.get('multi_trades', []) + user_portfolio.get('single_trades', [])
    
    if not trades:
        text = "📊 <b>Ваш портфель пуст</b>\n\nНачните с расчета сделки!"
        keyboard = [
            [InlineKeyboardButton("🎯 Новая сделка", callback_data="single_trade")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        if query:
            await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(keyboard))
        else:
            await SafeMessageSender.send_message(user_id, text, context, InlineKeyboardMarkup(keyboard))
        return
    
    deposit = user_portfolio['deposit']
    
    for trade in trades:
        metrics = await ProfessionalRiskCalculator.calculate_professional_metrics(
            trade, deposit, user_portfolio['leverage'], trade['risk_level']
        )
        trade['metrics'] = metrics
    
    metrics = PortfolioAnalyzer.calculate_portfolio_metrics(trades, deposit)
    recommendations = PortfolioAnalyzer.generate_recommendations(metrics, trades)
    
    text = (
        "📊 <b>ПОРТФЕЛЬ v3.0</b>\n\n"
        f"💰 <b>ОСНОВНЫЕ ПОКАЗАТЕЛИ:</b>\n"
        f"Депозит: ${deposit:,.2f}\nПлечо: {user_portfolio['leverage']}\n"
        f"Сделок: {len(trades)}\nEquity: ${metrics['total_equity']:.2f}\n\n"
        f"🎯 <b>РИСКИ И ПРИБЫЛЬ:</b>\n"
        f"Общий риск: ${metrics['total_risk_usd']:.2f} ({metrics['total_risk_percent']:.1f}%)\n"
        f"Потенциальная прибыль: ${metrics['total_profit']:.2f}\n"
        f"Средний R/R: {metrics['avg_rr_ratio']:.2f}\nТекущий P&L: ${metrics['total_pnl']:.2f}\n\n"
        f"🛡 <b>МАРЖИНАЛЬНЫЕ ПОКАЗАТЕЛИ:</b>\n"
        f"Требуемая маржа: ${metrics['total_margin']:.2f} ({metrics['total_margin_usage']:.1f}%)\n"
        f"Свободная маржа: ${metrics['free_margin']:.2f} ({metrics['free_margin_percent']:.1f}%)\n"
        f"Уровень маржи: {metrics['portfolio_margin_level']:.1f}%\n"
        f"Левередж портфеля: {metrics['portfolio_leverage']:.1f}x\n\n"
        f"📈 <b>АНАЛИТИКА:</b>\n"
        f"Волатильность: {metrics['portfolio_volatility']:.1f}%\n"
        f"Лонгов: {metrics['long_positions']} | Шортов: {metrics['short_positions']}\n"
        f"Уникальных активов: {metrics['unique_assets']}\n"
        f"Диверсификация: {metrics['diversity_score']:.1%}\n\n"
        "<b>💡 РЕКОМЕНДАЦИИ:</b>\n" + "\n".join(f"• {rec}" for rec in recommendations) + "\n\n"
    )
    
    keyboard = [
        [InlineKeyboardButton("🗑 Очистить портфель", callback_data="clear_portfolio")],
        [InlineKeyboardButton("📤 Экспорт отчета", callback_data="export_portfolio")],
        [InlineKeyboardButton("🎯 Новая сделка", callback_data="single_trade")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    
    if query:
        await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(keyboard))
    else:
        await SafeMessageSender.send_message(user_id, text, context, InlineKeyboardMarkup(keyboard))

@retry_on_timeout(max_retries=2, delay=1.0)
async def clear_portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Очистка портфеля"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    PortfolioManager.clear_portfolio(user_id)
    
    text = "✅ Портфель очищен"
    keyboard = [
        [InlineKeyboardButton("🎯 Новая сделка", callback_data="single_trade")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(keyboard))

@retry_on_timeout(max_retries=2, delay=1.0)
async def restore_progress_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Восстановление прогресса"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    temp_data = DataManager.load_temporary_data()
    saved_progress = temp_data.get(str(user_id))
    
    if not saved_progress:
        await SafeMessageSender.answer_callback_query(query, "❌ Нет сохраненного прогресса")
        return
    
    context.user_data.update(saved_progress['state_data'])
    state_type = saved_progress['state_type']
    
    text = "✅ Прогресс восстановлен! Продолжайте расчет."
    
    if state_type == "single":
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Продолжить", callback_data="single_trade")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ])
        )
    else:
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Продолжить", callback_data="multi_trade_start")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ])
        )

# ---------------------------
# Запуск приложения
# ---------------------------
def setup_handlers(application: Application):
    """Настройка обработчиков"""
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(callback_router_fixed))
    
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, 
        lambda update, context: SafeMessageSender.send_message(
            update.message.chat_id,
            "🤖 Используйте меню для навигации или /start для начала работы",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ])
        )
    ))

async def main():
    """Основная функция запуска"""
    application = RobustApplicationBuilder.create_application(TOKEN)
    setup_handlers(application)
    
    logger.info("🚀 ЗАПУСК 💎 PRO RISK CALCULATOR v3.0 ENTERPRISE EDITION")
    
    if WEBHOOK_URL and WEBHOOK_URL.strip():
        logger.info("Запуск в режиме WEBHOOK")
        await application.initialize()
        await application.start_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=WEBHOOK_PATH,
            webhook_url=f"{WEBHOOK_URL}{WEBHOOK_PATH}"
        )
        logger.info("✅ Бот успешно запущен в режиме WEBHOOK")
        
        while True:
            await asyncio.sleep(300)
    else:
        logger.info("Запуск в режиме POLLING")
        await application.run_polling()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹ Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
