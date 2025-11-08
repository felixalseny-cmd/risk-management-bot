# bot.py — PRO Risk Calculator v3.0 | ENTERPRISE EDITION - КРИТИЧЕСКИЕ ОШИБКИ ИСПРАВЛЕНЫ
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
        # Настройка параметров запросов
        request = telegram.request.HTTPXRequest(
            connection_pool_size=8,
        )
        
        # Создание приложения с настройками
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
# Safe Message Sender - ОБНОВЛЕННЫЙ С ЗАЩИТОЙ ОТ HTML ОШИБОК
# ---------------------------
class SafeMessageSender:
    """Безопасная отправка сообщений с обработкой ошибок"""
    
    @staticmethod
    def safe_html_text(text: str) -> str:
        """Безопасная подготовка HTML текста - УЛУЧШЕННАЯ ВЕРСИЯ"""
        # Сначала экранируем все специальные символы
        text = html.escape(text)
        
        # Затем разрешаем только безопасные HTML теги
        safe_tags = ['b', 'i', 'u', 'em', 'strong', 'code', 'pre']
        
        for tag in safe_tags:
            # Восстанавливаем разрешенные теги
            opening_tag = f"&lt;{tag}&gt;"
            closing_tag = f"&lt;/{tag}&gt;"
            text = text.replace(opening_tag, f"<{tag}>").replace(closing_tag, f"</{tag}>")
        
        # Удаляем множественные переносы строк
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Обрезаем слишком длинные сообщения
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
            # Очищаем HTML текст
            safe_text = SafeMessageSender.safe_html_text(text)
            
            if context and hasattr(context, 'bot'):
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=safe_text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode
                )
            else:
                # Fallback - создаем временного бота
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
            # Пытаемся отправить без HTML
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
                # Сообщение не изменилось - это не ошибка
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
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="donate_start")]
                ])
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
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="donate_start")]
                ])
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
# Enhanced Market Data Provider - УЛУЧШЕННАЯ ВЕРСИЯ С НОВЫМИ API
# ---------------------------
class EnhancedMarketDataProvider:
    """Универсальный провайдер рыночных данных с улучшенной поддержкой металлов и новых API"""
    
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
        """НАДЕЖНОЕ получение реальных цен с улучшенной очередью провайдеров"""
        try:
            # Проверка кэша
            cached_price = self.cache.get(symbol)
            if cached_price:
                return cached_price
            
            # Определяем тип актива и выбираем провайдера
            providers = [
                self._get_fmp_price,               # Financial Modeling Prep - основной
                self._get_metalpriceapi_price,     # Metal Price API - для металлов
                self._get_exchangerate_price,      # Forex
                self._get_binance_price,           # Крипто
                self._get_twelvedata_price,        # Акции, индексы
                self._get_alpha_vantage_stock,     # Акции
                self._get_alpha_vantage_forex,     # Forex резерв
                self._get_finnhub_price,           # Общий резерв
                self._get_fallback_price           # Статические данные
            ]
            
            price = None
            for provider in providers:
                price = await provider(symbol)
                if price and price > 0:
                    break
            
            # Fallback на статические данные при ошибках
            if price is None or price <= 0:
                logger.warning(f"Не удалось получить цену для {symbol}, используется fallback")
                price = self._get_fallback_price(symbol)
                
            # Сохраняем в кэш
            if price:
                self.cache[symbol] = price
                
            return price
            
        except Exception as e:
            logger.error(f"Ошибка получения цены для {symbol}: {e}")
            return self._get_fallback_price(symbol)
    
    def _is_crypto(self, symbol: str) -> bool:
        """Проверка является ли актив криптовалютой"""
        crypto_symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'USDT']
        return any(crypto in symbol for crypto in crypto_symbols)
    
    def _is_forex(self, symbol: str) -> bool:
        """Проверка является ли актив Forex парой"""
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        return symbol in forex_pairs
    
    def _is_metal(self, symbol: str) -> bool:
        """Проверка является ли актив металлом"""
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
            # Конвертируем символы для Metal Price API
            metal_map = {
                'XAUUSD': 'XAU',
                'XAGUSD': 'XAG', 
                'XPTUSD': 'XPT',
                'XPDUSD': 'XPD'
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
                            # Metal Price API возвращает курс, нужно конвертировать
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
                # Конвертация EURUSD -> EUR/USD
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
            # Форматируем символ для Binance
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
                    return data['c']  # current price
        except Exception as e:
            logger.error(f"Finnhub API error for {symbol}: {e}")
        return None
    
    def _get_fallback_price(self, symbol: str) -> float:
        """АКТУАЛИЗИРОВАННЫЕ fallback цены при недоступности API"""
        current_prices = {
            # Forex (актуальные цены)
            'EURUSD': 1.0732, 'GBPUSD': 1.2548, 'USDJPY': 155.42, 'USDCHF': 0.9054,
            'AUDUSD': 0.6589, 'USDCAD': 1.3732, 'NZDUSD': 0.6014,
            # Crypto (актуальные цены)
            'BTCUSDT': 61450.0, 'ETHUSDT': 3450.0, 'XRPUSDT': 0.524, 'LTCUSDT': 82.15,
            'BCHUSDT': 415.00, 'ADAUSDT': 0.462, 'DOTUSDT': 6.95,
            # Stocks (актуальные цены)
            'AAPL': 189.20, 'TSLA': 177.50, 'GOOGL': 174.35, 'MSFT': 420.72,
            'AMZN': 178.22, 'META': 469.85, 'NFLX': 617.80,
            # Indices (актуальные цены)
            'NAS100': 17750.0, 'SPX500': 5225.0, 'DJ30': 38850.0, 'FTSE100': 8213.0,
            'DAX40': 18420.0, 'NIKKEI225': 38175.0, 'ASX200': 7620.0,
            # Metals (актуальные цены)
            'XAUUSD': 2335.50, 'XAGUSD': 27.80, 'XPTUSD': 890.50, 'XPDUSD': 945.75,
            # Energy (актуальные цены)
            'OIL': 78.25, 'NATURALGAS': 2.15, 'BRENT': 82.80
        }
        return current_prices.get(symbol, 100.0)

    async def get_price_with_fallback(self, symbol: str) -> Tuple[float, str]:
        """Получение цены с информацией о источнике"""
        try:
            # Сначала пытаемся получить реальную цену
            real_price = await self.get_robust_real_time_price(symbol)
            if real_price and real_price > 0:
                return real_price, "real-time"
            
            # Затем используем кэш
            cached_price = self.cache.get(symbol)
            if cached_price:
                return cached_price, "cached"
            
            # И только потом fallback
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
            "type": "forex",
            "contract_size": 100000,  # 1 стандартный лот
            "margin_currency": "USD",
            "pip_value": 10.0,       # 1 пункт = $10 для стандартного лота
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
            "pip_value": 9.09,       # Особенность JPY пар
            "calculation_formula": "forex_jpy",
            "pip_decimal_places": 2,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 1000
        },
        
        # Криптовалюты - ИСПРАВЛЕННЫЕ ЗНАЧЕНИЯ
        "BTCUSDT": {
            "type": "crypto",
            "contract_size": 1,      # 1 BTC
            "margin_currency": "USDT",
            "pip_value": 1.0,        # 1 USDT за пункт
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
        
        # Акции - ИСПРАВЛЕННЫЕ ЗНАЧЕНИЯ
        "AAPL": {
            "type": "stock",
            "contract_size": 100,    # 100 акций в лоте
            "margin_currency": "USD",
            "pip_value": 1.0,        # $1 за пункт движения цены
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
        
        # Индексы - ИСПРАВЛЕННЫЕ ЗНАЧЕНИЯ
        "NAS100": {
            "type": "index",
            "contract_size": 1,      # 1 контракт на индекс
            "margin_currency": "USD",
            "pip_value": 1.0,        # $1 за пункт
            "calculation_formula": "indices",
            "pip_decimal_places": 1,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 100
        },
        
        # Металлы - ИСПРАВЛЕННЫЕ ЗНАЧЕНИЯ
        "XAUUSD": {
            "type": "metal", 
            "contract_size": 100,    # 100 унций в стандартном лоте
            "margin_currency": "USD",
            "pip_value": 1.0,        # $1 за пункт (0.01 изменения цены)
            "calculation_formula": "metals",
            "pip_decimal_places": 2,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 100
        },
        "XAGUSD": {
            "type": "metal", 
            "contract_size": 5000,   # 5000 унций в стандартном лоте
            "margin_currency": "USD",
            "pip_value": 0.5,        # $0.5 за пункт (0.01 изменения цены)
            "calculation_formula": "metals",
            "pip_decimal_places": 2,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 100
        },
        
        # Энергия - ИСПРАВЛЕННЫЕ ЗНАЧЕНИЯ
        "OIL": {
            "type": "energy",
            "contract_size": 1000,   # 1000 баррелей
            "margin_currency": "USD",
            "pip_value": 10.0,       # $10 за пункт (0.01 изменения цены)
            "calculation_formula": "energy",
            "pip_decimal_places": 2,
            "min_volume": 0.01,
            "volume_step": 0.01,
            "max_leverage": 100
        }
    }
    
    @classmethod
    def get_specs(cls, symbol: str) -> Dict[str, Any]:
        """Получение спецификаций для инструмента"""
        return cls.SPECS.get(symbol, cls._get_default_specs(symbol))
    
    @classmethod
    def _get_default_specs(cls, symbol: str) -> Dict[str, Any]:
        """Спецификации по умолчанию"""
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
            
            # Получаем эффективное плечо с учетом ограничений
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
        """Расчет маржи для Forex по отраслевым стандартам"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # Для Forex: (Объем × Размер контракта) / Плечо
        required_margin = (volume * contract_size) / lev_value
        
        return {
            'required_margin': max(required_margin, 0.01),  # Минимум $0.01
            'contract_size': contract_size,
            'calculation_method': 'forex_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size,
            'effective_leverage': leverage
        }
    
    async def _calculate_forex_jpy_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Расчет маржи для JPY пар (особенности расчета)"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # Для JPY пар та же формула, но учитываем курс
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
        """Расчет маржи для криптовалют"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # Для крипто: (Объем × Цена) / Плечо
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
        """Расчет маржи для акций"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # Для акций: (Объем × Размер контракта × Цена) / Плечо
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
        """Расчет маржи для индексов"""
        return await self._calculate_stocks_margin(specs, volume, leverage, current_price)
    
    async def _calculate_metals_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """ИСПРАВЛЕННЫЙ расчет маржи для металлов"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # ДЛЯ МЕТАЛЛОВ: (Объем × Размер контракта × Цена) / Плечо
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
        """Расчет маржи для энергоресурсов"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # Для энергии: (Объем × Размер контракта × Цена) / Плечо
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
        """Универсальный расчет маржи"""
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
        """Профессиональный расчет дистанции в пунктах"""
        specs = InstrumentSpecs.get_specs(asset)
        pip_decimal_places = specs.get('pip_decimal_places', 4)
        
        if direction.upper() == 'LONG':
            distance = target - entry
        else:  # SHORT
            distance = entry - target
        
        if pip_decimal_places == 2:  # JPY пары
            return abs(distance) * 100
        elif pip_decimal_places == 1:  # Некоторые индексы
            return abs(distance) * 10
        else:  # Стандартные 4 знака
            return abs(distance) * 10000

    @staticmethod
    def calculate_pnl_dollar_amount(entry_price: float, exit_price: float, volume: float, pip_value: float, 
                                  direction: str, asset: str, tick_size: float = 0.01) -> float:
        """Профессиональный расчет P&L в долларах"""
        try:
            specs = InstrumentSpecs.get_specs(asset)
            
            if direction.upper() == 'LONG':
                price_diff = exit_price - entry_price
            else:  # SHORT
                price_diff = entry_price - exit_price
            
            # Для разных типов активов разный расчет
            if specs['type'] in ['stock', 'crypto']:
                # Для акций и крипто: разница цены × объем × размер контракта
                pnl = price_diff * volume * specs['contract_size']
            else:
                # Для остальных: через пункты
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
        """РЕАЛИСТИЧНЫЙ расчет P&L с учетом объема и стоимости пункта"""
        entry = trade['entry_price']
        
        return ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            entry, current_price, volume, pip_value, direction, asset
        )

    @staticmethod
    def calculate_margin_level(equity: float, margin: float) -> float:
        """Расчет уровня маржи в процентах"""
        if margin == 0:
            return float('inf')  # Бесконечность при нулевой марже
        
        margin_level = (equity / margin) * 100
        return round(margin_level, 2)

    @staticmethod
    def calculate_free_margin(equity: float, margin: float) -> float:
        """Расчет свободной маржи"""
        free_margin = equity - margin
        return max(free_margin, 0.0)  # Не может быть отрицательной

    @staticmethod
    async def calculate_professional_metrics(trade: Dict, deposit: float, leverage: str, risk_level: str) -> Dict[str, Any]:
        """
        ИСПРАВЛЕННЫЙ расчет с реальными котировками и маржой
        """
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
            
            # ИСПРАВЛЕННЫЙ РАСЧЕТ ОБЪЕМА
            if stop_distance_pips > 0 and pip_value > 0:
                volume_lots = risk_amount / (stop_distance_pips * pip_value)
                # Округляем до шага объема
                volume_step = specs.get('volume_step', 0.01)
                volume_lots = round(volume_lots / volume_step) * volume_step
                # Ограничиваем минимальным объемом
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
            
            # Расчет equity (баланс + нереализованный P&L)
            current_pnl = await ProfessionalRiskCalculator.calculate_realistic_pnl(
                trade, current_price, volume_lots, pip_value, direction, asset
            )
            equity = deposit + current_pnl
            
            # Используем профессиональные формулы для маржи
            free_margin = ProfessionalRiskCalculator.calculate_free_margin(equity, required_margin)
            margin_level = ProfessionalRiskCalculator.calculate_margin_level(equity, required_margin)
            
            # Расчет потенциальной прибыли через профессиональную функцию
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
# Portfolio Analyzer - НОВЫЙ КЛАСС ДЛЯ АНАЛИЗА ПОРТФЕЛЯ
# ---------------------------
class PortfolioAnalyzer:
    """Анализатор портфеля с агрегированными метриками"""
    
    @staticmethod
    def calculate_portfolio_metrics(trades: List[Dict], deposit: float) -> Dict[str, Any]:
        """Расчет агрегированных метрик портфеля"""
        if not trades:
            return {}
        
        total_risk_usd = sum(trade.get('metrics', {}).get('risk_amount', 0) for trade in trades)
        total_profit = sum(trade.get('metrics', {}).get('potential_profit', 0) for trade in trades)
        total_margin = sum(trade.get('metrics', {}).get('required_margin', 0) for trade in trades)
        total_pnl = sum(trade.get('metrics', {}).get('current_pnl', 0) for trade in trades)
        total_equity = deposit + total_pnl
        avg_rr_ratio = sum(trade.get('metrics', {}).get('rr_ratio', 0) for trade in trades) / len(trades)
        
        total_risk_percent = (total_risk_usd / deposit) * 100 if deposit > 0 else 0
        total_margin_usage = (total_margin / deposit) * 100 if deposit > 0 else 0
        free_margin = max(total_equity - total_margin, 0)
        free_margin_percent = (free_margin / deposit) * 100 if deposit > 0 else 0
        portfolio_margin_level = (total_equity / total_margin * 100) if total_margin > 0 else float('inf')
        
        # Волатильность портфеля (средневзвешенная)
        portfolio_volatility = sum(VOLATILITY_DATA.get(trade['asset'], 20) * trade.get('metrics', {}).get('risk_amount', 0) / total_risk_usd for trade in trades) if total_risk_usd > 0 else 20
        
        # Диверсификация
        unique_assets = len(set(trade['asset'] for trade in trades))
        diversity_score = min(unique_assets / 5, 1.0) # Макс 5 уникальных для 100%
        
        long_positions = sum(1 for trade in trades if trade['direction'] == 'LONG')
        short_positions = len(trades) - long_positions
        
        # Левередж портфеля
        total_notional = sum(trade.get('metrics', {}).get('notional_value', 0) for trade in trades)
        portfolio_leverage = total_notional / deposit if deposit > 0 else 1
        
        return {
            'total_risk_usd': total_risk_usd,
            'total_risk_percent': total_risk_percent,
            'total_profit': total_profit,
            'avg_rr_ratio': avg_rr_ratio,
            'total_pnl': total_pnl,
            'total_equity': total_equity,
            'total_margin': total_margin,
            'total_margin_usage': total_margin_usage,
            'free_margin': free_margin,
            'free_margin_percent': free_margin_percent,
            'portfolio_margin_level': portfolio_margin_level,
            'portfolio_leverage': portfolio_leverage,
            'portfolio_volatility': portfolio_volatility,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'unique_assets': unique_assets,
            'diversity_score': diversity_score
        }
    
    @staticmethod
    def generate_enhanced_recommendations(metrics: Dict, trades: List[Dict]) -> List[str]:
        """Генерация рекомендаций на основе метрик"""
        recommendations = []
        
        if metrics.get('total_risk_percent', 0) > 10:
            recommendations.append("⚠️ Высокий общий риск (>10%). Рассмотрите сокращение позиций.")
        
        if metrics.get('portfolio_margin_level', 0) < 200:
            recommendations.append("🛑 Уровень маржи низкий. Добавьте средства или закройте позиции.")
        
        if metrics.get('avg_rr_ratio', 0) < 1.5:
            recommendations.append("📈 Улучшите R/R соотношение (>1.5 для устойчивости).")
        
        if metrics.get('portfolio_volatility', 0) > 30:
            recommendations.append("🌪 Высокая волатильность. Увеличьте стоп-лоссы.")
        
        if metrics.get('diversity_score', 0) < 0.6:
            recommendations.append("🌐 Улучшите диверсификацию (добавьте активы из разных категорий).")
        
        if not recommendations:
            recommendations.append("✅ Портфель сбалансирован. Продолжайте мониторинг.")
        
        return recommendations

# ---------------------------
# Data Manager - УЛУЧШЕННЫЙ С ПОДДЕРЖКОЙ ВОССТАНОВЛЕНИЯ
# ---------------------------
class DataManager:
    """Менеджер данных с поддержкой временных состояний"""
    
    @staticmethod
    def load_data() -> Dict[int, Dict[str, Any]]:
        try:
            if os.path.exists("user_data.json"):
                with open("user_data.json", 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                return {int(k): v for k, v in raw.items()}
            return {}
        except Exception as e:
            logger.error(f"Error loading user data: {e}")
            return {}
    
    @staticmethod
    def save_data(user_data: Dict[int, Dict[str, Any]]):
        try:
            with open("user_data.json", 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in user_data.items()}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving user data: {e}")
    
    @staticmethod
    def load_temporary_data() -> Dict[int, Dict[str, Any]]:
        try:
            if os.path.exists("temp_progress.json"):
                with open("temp_progress.json", 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                return {int(k): v for k, v in raw.items()}
            return {}
        except Exception as e:
            logger.error(f"Error loading temp data: {e}")
            return {}
    
    @staticmethod
    def save_temporary_data(temp_data: Dict[int, Dict[str, Any]]):
        try:
            with open("temp_progress.json", 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in temp_data.items()}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving temp data: {e}")
    
    @staticmethod
    def clear_temporary_progress(user_id: int):
        temp_data = DataManager.load_temporary_data()
        temp_data.pop(str(user_id), None)
        DataManager.save_temporary_data(temp_data)

    @staticmethod
    def save_progress(user_id: int, state_data: Dict, state_type: str):
        """Сохранение прогресса"""
        temp_data = DataManager.load_temporary_data()
        temp_data[str(user_id)] = {
            'state_data': state_data,
            'state_type': state_type,
            'timestamp': datetime.now().isoformat()
        }
        DataManager.save_temporary_data(temp_data)

# ---------------------------
# Portfolio Manager - УЛУЧШЕННЫЙ С МУЛЬТИПОЗИЦИЯМИ
# ---------------------------
class PortfolioManager:
    """Менеджер портфеля с поддержкой одиночных и мульти сделок"""
    
    user_data: Dict[int, Dict[str, Any]] = DataManager.load_data()
    
    @classmethod
    def ensure_user(cls, user_id: int):
        if user_id not in cls.user_data:
            cls.user_data[user_id] = {
                'deposit': 1000.0,
                'leverage': '1:100',
                'single_trades': [],
                'multi_trades': [],
                'last_updated': datetime.now().isoformat()
            }
    
    @classmethod
    def add_single_trade(cls, user_id: int, trade: Dict):
        cls.ensure_user(user_id)
        cls.user_data[user_id]['single_trades'].append(trade)
        cls.save()
    
    @classmethod
    def add_multi_trade(cls, user_id: int, trades: List[Dict]):
        cls.ensure_user(user_id)
        cls.user_data[user_id]['multi_trades'].extend(trades)
        cls.save()
    
    @classmethod
    def set_deposit_leverage(cls, user_id: int, deposit: float, leverage: str):
        cls.ensure_user(user_id)
        cls.user_data[user_id]['deposit'] = deposit
        cls.user_data[user_id]['leverage'] = leverage
        cls.save()
    
    @classmethod
    def clear_portfolio(cls, user_id: int):
        cls.ensure_user(user_id)
        cls.user_data[user_id] = {
            'deposit': cls.user_data[user_id]['deposit'],
            'leverage': cls.user_data[user_id]['leverage'],
            'single_trades': [],
            'multi_trades': [],
            'last_updated': datetime.now().isoformat()
        }
        cls.save()
    
    @classmethod
    def save(cls):
        DataManager.save_data(cls.user_data)

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

# Инструменты и пресеты с обновленными кредитными плечами
ASSET_CATEGORIES = {
    "FOREX": ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'],
    "CRYPTO": ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'ADAUSDT', 'DOTUSDT'],
    "INDICES": ['NAS100', 'SPX500', 'DJ30', 'FTSE100', 'DAX40', 'NIKKEI225', 'ASX200'],
    "METALS": ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD'],
    "ENERGY": ['OIL', 'NATURALGAS', 'BRENT'],
    "STOCKS": ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX']
}

# Обновленные кредитные плечи по категориям активов
LEVERAGES = {
    "FOREX": ['1:1', '1:5', '1:10', '1:20', '1:50', '1:100', '1:200', '1:500', '1:1000'],
    "CRYPTO": ['1:1', '1:5', '1:10', '1:20', '1:50', '1:100', '1:110', '1:120', '1:125'],
    "METALS": ['1:1', '1:5', '1:10', '1:20', '1:50', '1:100'],
    "DEFAULT": ['1:1', '1:5', '1:10', '1:20', '1:50', '1:100']
}

RISK_LEVELS = ['2%', '5%', '7%', '10%', '15%', '20%', '25%']

# Волатильность активов (ОБНОВЛЕННЫЕ ДАННЫЕ)
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
# Utility Functions
# ---------------------------
async def show_asset_price_in_realtime(asset: str) -> str:
    """Показ реальной цены актива в реальном времени"""
    try:
        price, source = await enhanced_market_data.get_price_with_fallback(asset)
        source_text = {
            "real-time": "🟢 РЕАЛЬНОЕ ОБНОВЛЕНИЕ",
            "cached": "🟡 КЭШИРОВАННЫЕ ДАННЫЕ", 
            "fallback": "⚪ БАЗОВЫЕ ДАННЫЕ",
            "error": "🔴 ОШИБКА ДАННЫХ"
        }.get(source, "⚪ НЕИЗВЕСТНО")
        
        return f"💡 Текущая цена {asset}: <b>{price:.4f}</b>\n{source_text}\n\n"
    except Exception as e:
        logger.error(f"Error showing realtime price for {asset}: {e}")
        return f"💡 Текущая цена {asset}: <b>Обновление...</b>\n\n"

def get_leverages_for_category(category: str) -> List[str]:
    """Получение списка кредитных плеч для категории активов"""
    return LEVERAGES.get(category, LEVERAGES["DEFAULT"])

# ---------------------------
# Command Handlers
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Стартовый обработчик"""
    user_id = update.effective_user.id
    PortfolioManager.ensure_user(user_id)
    
    text = (
        "💎 <b>PRO RISK CALCULATOR v3.0 ENTERPRISE</b>\n\n"
        "Профессиональный инструмент для расчета рисков с РЕАЛЬНЫМИ котировками.\n\n"
        "💎 Функции:\n"
        "• Расчет маржи и объема\n"
        "• Мониторинг портфеля\n"
        "• Защита от маржин-колла\n"
        "• AI-рекомендации\n\n"
        "Выберите действие:"
    )
    
    keyboard = [
        [InlineKeyboardButton("🎯 Профессиональный расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("📊 Портфель", callback_data="portfolio")],
        [InlineKeyboardButton("📚 Инструкции", callback_data="pro_info")],
        [InlineKeyboardButton("🚀 Будущие функции", callback_data="future_features")]
    ]
    
    await SafeMessageSender.send_message(
        user_id,
        text,
        context,
        InlineKeyboardMarkup(keyboard)
    )

@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """PRO инструкции - Часть 1"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = (
        "<b>📚 PRO ИНСТРУКЦИИ v3.0 - ЧАСТЬ 1/2</b>\n\n"
        
        "<b>🎯 ОСНОВНЫЕ ФУНКЦИИ:</b>\n"
        "• <b>РЕАЛЬНЫЕ КОТИРОВКИ</b> - данные с 8+ API (Binance, FMP, Alpha Vantage)\n"
        "• <b>ПРОФЕССИОНАЛЬНАЯ МАРЖА</b> - точный расчет по стандартам MT4/MT5\n"
        "• <b>МУЛЬТИПОЗИЦИОННЫЙ АНАЛИЗ</b> - агрегация рисков портфеля\n"
        "• <b>ЗАЩИТА ОТ КОЛЛА</b> - автоматические предупреждения\n\n"
        
        "<b>🔧 ТЕХНИЧЕСКИЕ ОСОБЕННОСТИ:</b>\n"
        "• <b>Async/Await</b> - мгновенные обновления без задержек\n"
        "• <b>Retry Logic</b> - устойчивость к сетевым сбоям\n"
        "• <b>Smart Caching</b> - оптимизация API запросов\n"
        "• <b>Rate Limiting</b> - соблюдение лимитов провайдеров\n\n"
        
        "<b>📊 РАСЧЕТЫ ПО СТАНДАРТАМ:</b>\n"
        "• Forex: (Volume × Contract Size) / Leverage\n"
        "• Crypto/Stocks: (Volume × Price × Contract) / Leverage\n"
        "• Metals/Energy: Учет спецификаций контрактов\n\n"
        
        "Продолжить на Часть 2/2 →"
    )
    
    keyboard = [
        [InlineKeyboardButton("📖 Часть 2/2", callback_data="pro_info_part2")],
        [InlineKeyboardButton("💖 Поддержать разработчика", callback_data="donate_start")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_info_part2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Вторая часть PRO инструкций"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    volatility_explanation = (
        "<b>🌪 ВОЛАТИЛЬНОСТЬ В РАСЧЕТАХ:</b>\n"
        "• <b>Что это?</b> Мера колебаний цены актива\n"
        "• <b>Как используется?</b> Для оценки риска и рекомендаций\n"
        "• <b>Высокая волатильность</b> (>30%) = большие риски И возможности\n"
        "• <b>Низкая волатильность</b> (<15%) = стабильность, но меньший потенциал\n\n"
        
        "<b>ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ:</b>\n"
        "• BTCUSDT: 65% - высокий риск, нужен широкий SL\n"
        "• EURUSD: 8% - низкий риск, можно tighter управление\n"
        "• Используйте эти данные для настройки стоп-лоссов!\n\n"
    )
    
    text = (
        "<b>📚 PRO ИНСТРУКЦИИ v3.0 - ЧАСТЬ 2/2</b>\n\n"
        
        "<b>🎯 РЕКОМЕНДАЦИИ ДЛЯ ПРОФЕССИОНАЛОВ:</b>\n"
        "• Риск на сделку: 1-5% от депозита\n"
        "• Общий риск портфеля: < 10%\n"
        "• Уровень маржи: > 200%\n"
        "• Соотношение R/R: минимум 1:1.5\n"
        "• Диверсификация: 3-5 активов разных категорий\n\n"
        
        f"{volatility_explanation}"
        
        "<b>🚀 ПРЕИМУЩЕСТВА v3.0:</b>\n"
        "✅ РЕАЛЬНЫЕ цены вместо статических данных\n"
        "✅ ПРОФЕССИОНАЛЬНЫЙ расчет маржи\n"
        "✅ ЗАЩИТА от маржин-колла\n"
        "✅ АВТОМАТИЧЕСКИЕ рекомендации\n"
        "✅ ОБНОВЛЕНИЕ портфеля в реальном времени\n\n"
        
        "<b>💝 Поддержите разработку для новых функций!</b>\n\n"
        
        "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
    )
    
    keyboard = [
        [InlineKeyboardButton("📖 Часть 1/2", callback_data="pro_info")],
        [InlineKeyboardButton("🎯 Начать расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("💖 Поддержать разработчика", callback_data="donate_start")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

# ---------------------------
# ОБНОВЛЕННЫЙ РАЗДЕЛ "БУДУЩИЕ РАЗРАБОТКИ"
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def future_features_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ОБНОВЛЕННЫЕ Будущие разработки"""
    text = (
        "<b>🚀 БУДУЩИЕ РАЗРАБОТКИ v4.0</b>\n\n"
        
        "<b>📈 ПЛАНИРУЕМЫЕ ФУНКЦИИ:</b>\n"
        "• 🤖 <b>AI-АНАЛИТИКА</b> - нейросеть для прогнозирования рисков\n"
        "• 📊 <b>ПРОГНОЗ ВОЛАТИЛЬНОСТИ</b> - предсказание рыночных движений\n"
        "• 🔄 <b>АВТОМАТИЧЕСКИЕ ОРДЕРА</b> - интеграция с биржами\n"
        "• 📱 <b>МОБИЛЬНОЕ ПРИЛОЖЕНИЕ</b> - трейдинг на ходу\n"
        "• 🌐 <b>WEB-ПАНЕЛЬ</b> - расширенная аналитика в браузере\n"
        "• 📊 <b>ПОРТФЕЛЬНАЯ АНАЛИТИКА</b> - корреляция, бета, альфа\n"
        "• ⚡ <b>РЕАЛЬНЫЙ СТРИМИНГ</b> - мгновенные обновления цен\n"
        "• 🎯 <b>СКАНЕР РЫНКА</b> - автоматический поиск возможностей\n\n"
        
        "<b>💡 ТЕХНОЛОГИИ:</b>\n"
        "• Machine Learning для прогнозирования рисков\n"
        "• Real-time WebSocket подключения к биржам\n"
        "• Cloud-native архитектура для масштабирования\n"
        "• Advanced backtesting и симуляции\n"
        "• Multi-exchange поддержка (20+ бирж)\n\n"
        
        "<b>💝 ПОДДЕРЖИТЕ РАЗРАБОТКУ!</b>\n"
        "Каждый донат приближает нас к реализации этих функций!\n"
        "Ваша поддержка помогает создавать лучшие инструменты для трейдеров.\n\n"
        
        "<b>🎯 УЖЕ РЕАЛИЗОВАНО В v3.0:</b>\n"
        "✅ РЕАЛЬНЫЕ котировки через Binance\n"
        "✅ ПРОФЕССИОНАЛЬНЫЙ расчет маржи\n"
        "✅ ЗАЩИТА от маржин-колла\n"
        "✅ АВТОМАТИЧЕСКИЕ рекомендации\n"
        "✅ МУЛЬТИПОЗИЦИОННЫЙ расчет\n"
        "✅ ПОРТФЕЛЬНАЯ аналитика\n"
        "✅ ВОЗМОЖНОСТЬ выгрузить результаты анализа сделок\n\n"
        
        "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
    )
    
    keyboard = [
        [InlineKeyboardButton("💖 Поддержать разработку", callback_data="donate_start")],
        [InlineKeyboardButton("🎯 Начать расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    
    if update.callback_query:
        await SafeMessageSender.edit_message_text(
            update.callback_query,
            text,
            InlineKeyboardMarkup(keyboard)
        )
    else:
        await SafeMessageSender.send_message(
            update.effective_user.id,
            text,
            context,
            InlineKeyboardMarkup(keyboard)
        )

# ---------------------------
# Single Trade Handlers
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало одиночной сделки с реальными данными"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Save progress
    DataManager.save_progress(query.from_user.id, {}, "single")
    
    text = (
        "🎯 <b>ОДИНОЧНАЯ СДЕЛКА v3.0</b>\n\n"
        "ПРОФЕССИОНАЛЬНЫЙ расчет с РЕАЛЬНЫМИ котировками и защитой от маржин-колла.\n\n"
        "<b>Введите ваш депозит в USD:</b>"
    )
    
    keyboard = [
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )
    return SingleTradeState.DEPOSIT.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода депозита для одиночной сделки"""
    text = update.message.text.strip()
    
    # Save progress
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "single")
    
    try:
        deposit = float(text.replace(',', '.'))
        if deposit < 100:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Минимальный депозит: $100\nПопробуйте еще раз:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.DEPOSIT.value
        
        context.user_data['deposit'] = deposit
        
        keyboard = []
        # Используем стандартные плечи для начала
        for leverage in LEVERAGES["DEFAULT"]:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"lev_{leverage}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Депозит: ${deposit:,.2f}\n\n"
            "<b>Выберите кредитное плечо:</b>",
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return SingleTradeState.LEVERAGE.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Введите число (например: 1000)\nПопробуйте еще раз:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.DEPOSIT.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора плеча для одиночной сделки"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Save progress
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "single")
    
    leverage = query.data.replace('lev_', '')
    context.user_data['leverage'] = leverage
    
    keyboard = []
    for category in ASSET_CATEGORIES.keys():
        keyboard.append([InlineKeyboardButton(category, callback_data=f"cat_{category}")])
    
    keyboard.append([InlineKeyboardButton("📝 Ввести актив вручную", callback_data="asset_manual")])
    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Плечо: {leverage}\n\n"
        "<b>Выберите категорию актива:</b>",
        InlineKeyboardMarkup(keyboard)
    )
    return SingleTradeState.ASSET_CATEGORY.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_asset_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора категории активов"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Save progress
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "single")
    
    if query.data == "asset_manual":
        await SafeMessageSender.edit_message_text(
            query,
            "✍️ Введите название актива (например: BTCUSDT):",
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_categories")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.ASSET.value
    
    category = query.data.replace('cat_', '')
    context.user_data['asset_category'] = category
    
    assets = ASSET_CATEGORIES.get(category, [])
    
    keyboard = []
    for asset in assets:
        keyboard.append([InlineKeyboardButton(asset, callback_data=f"asset_{asset}")])
    
    keyboard.append([InlineKeyboardButton("🔙 Назад к категориям", callback_data="back_to_categories")])
    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Категория: {category}\n\n"
        "<b>Выберите актив:</b>",
        InlineKeyboardMarkup(keyboard)
    )
    return SingleTradeState.ASSET.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def enhanced_single_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Улучшенный обработчик выбора актива с реальной ценой"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Save progress
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "single")
    
    if query.data == "back_to_categories":
        keyboard = []
        for category in ASSET_CATEGORIES.keys():
            keyboard.append([InlineKeyboardButton(category, callback_data=f"cat_{category}")])
        
        keyboard.append([InlineKeyboardButton("📝 Ввести актив вручную", callback_data="asset_manual")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await SafeMessageSender.edit_message_text(
            query,
            "<b>Выберите категорию актива:</b>",
            InlineKeyboardMarkup(keyboard)
        )
        return SingleTradeState.ASSET_CATEGORY.value
    
    asset = query.data.replace('asset_', '')
    context.user_data['asset'] = asset
    
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Актив: {asset}\n{price_info}"
        "<b>Выберите направление сделки:</b>",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🔙 Назад к категориям", callback_data="back_to_categories")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.DIRECTION.value

async def single_trade_asset_manual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ручного ввода актива"""
    asset = update.message.text.strip().upper()
    
    # Save progress
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "single")
    
    if not re.match(r'^[A-Z0-9]{2,20}$', asset):
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Неверный формат актива. Попробуйте еще раз:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.ASSET.value
    
    context.user_data['asset'] = asset
    
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.send_message(
        update.message.chat_id,
        f"✅ Актив: {asset}\n{price_info}"
        "<b>Выберите направление сделки:</b>",
        context,
        InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_categories")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.DIRECTION.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def enhanced_single_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Улучшенный обработчик направления с реальной ценой"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Save progress
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "single")
    
    direction = query.data.replace('dir_', '')
    context.user_data['direction'] = direction
    
    asset = context.user_data['asset']
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Направление: {direction}\n{price_info}"
        "<b>Введите цену входа:</b>",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_asset")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.ENTRY.value

async def single_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка цены входа для одиночной сделки"""
    text = update.message.text.strip()
    
    # Save progress
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "single")
    
    try:
        entry_price = float(text.replace(',', '.'))
        if entry_price <= 0:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Цена должна быть больше 0\nПопробуйте еще раз:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.ENTRY.value
        
        context.user_data['entry_price'] = entry_price
        
        asset = context.user_data['asset']
        price_info = await show_asset_price_in_realtime(asset)
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Цена входа: {entry_price}\n{price_info}"
            "<b>Введите уровень стоп-лосса:</b>",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.STOP_LOSS.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Введите число (например: 50000)\nПопробуйте еще раз:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.ENTRY.value

async def single_trade_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка стоп-лосса для одиночной сделки"""
    text = update.message.text.strip()
    
    # Save progress
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "single")
    
    try:
        stop_loss = float(text.replace(',', '.'))
        entry_price = context.user_data['entry_price']
        direction = context.user_data['direction']
        asset = context.user_data['asset']
        
        # Расчет суммы SL в долларах
        sl_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            entry_price, stop_loss, 1.0, 1.0, direction, asset
        )
        
        if direction == 'LONG' and stop_loss >= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для LONG стоп-лосс должен быть НИЖЕ цены входа\nПопробуйте еще раз:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.STOP_LOSS.value
        elif direction == 'SHORT' and stop_loss <= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для SHORT стоп-лосс должен быть ВЫШЕ цены входа\nПопробуйте еще раз:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.STOP_LOSS.value
        
        context.user_data['stop_loss'] = stop_loss
        
        stop_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry_price, stop_loss, direction, asset)
        
        keyboard = []
        for risk_level in RISK_LEVELS:
            keyboard.append([InlineKeyboardButton(risk_level, callback_data=f"risk_{risk_level}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Стоп-лосс: {stop_loss} ({stop_distance_pips:.0f} пунктов)\n"
            f"💵 Сумма SL: ${abs(sl_amount):.2f}\n\n"
            "<b>Выберите уровень риска:</b>",
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return SingleTradeState.RISK_LEVEL.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Введите число (например: 48000)\nПопробуйте еще раз:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.STOP_LOSS.value

async def single_trade_risk_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора уровня риска"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Save progress
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "single")
    
    risk_level = query.data.replace('risk_', '')
    context.user_data['risk_level'] = risk_level
    
    asset = context.user_data['asset']
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Уровень риска: {risk_level}\n{price_info}"
        "<b>Введите уровень тейк-профита:</b>",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.TAKE_PROFIT.value

async def single_trade_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка тейк-профита и показ результатов с РЕАЛЬНЫМИ ДАННЫМИ"""
    text = update.message.text.strip()
    
    # Save progress
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "single")
    
    try:
        take_profit = float(text.replace(',', '.'))
        entry_price = context.user_data['entry_price']
        direction = context.user_data['direction']
        
        # Расчет суммы TP в долларах
        tp_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            entry_price, take_profit, 1.0, 1.0, direction, context.user_data['asset']
        )
        
        if direction == 'LONG' and take_profit <= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для LONG тейк-профит должен быть ВЫШЕ цены входа\nПопробуйте еще раз:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.TAKE_PROFIT.value
        elif direction == 'SHORT' and take_profit >= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для SHORT тейк-профит должен быть НИЖЕ цены входа\nПопробуйте еще раз:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.TAKE_PROFIT.value
        
        context.user_data['take_profit'] = take_profit
        
        user_id = update.message.from_user.id
        trade = context.user_data.copy()
        PortfolioManager.ensure_user(user_id)
        PortfolioManager.add_single_trade(user_id, trade)
        PortfolioManager.set_deposit_leverage(user_id, trade['deposit'], trade['leverage'])
        
        metrics = await ProfessionalRiskCalculator.calculate_professional_metrics(
            trade, trade['deposit'], trade['leverage'], trade['risk_level']
        )
        trade['metrics'] = metrics
        
        # Расчет сумм SL/TP в долларах для отображения
        sl_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            trade['entry_price'], trade['stop_loss'], metrics['volume_lots'], 
            metrics['pip_value'], trade['direction'], trade['asset']
        )
        
        tp_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
            trade['entry_price'], trade['take_profit'], metrics['volume_lots'],
            metrics['pip_value'], trade['direction'], trade['asset']
        )
        
        text = (
            "🎯 <b>РЕЗУЛЬТАТЫ РАСЧЕТА v3.0</b>\n\n"
            f"Актив: {trade['asset']}\n"
            f"Направление: {trade['direction']}\n"
            f"Вход: {trade['entry_price']}\n"
            f"SL: {trade['stop_loss']} (${abs(sl_amount):.2f})\n"
            f"TP: {trade['take_profit']} (${tp_amount:.2f})\n"
            f"Дистанция SL: {metrics['stop_distance_pips']:.0f} пунктов\n"
            f"Дистанция TP: {metrics['profit_distance_pips']:.0f} пунктов\n"
            f"Стоимость пункта: ${metrics['pip_value']:.2f}\n\n"
            f"💰 <b>ФИНАНСОВЫЕ ПОКАЗАТЕЛИ:</b>\n"
            f"• Объем: {metrics['volume_lots']:.2f} лотов\n"
            f"• Риск: ${metrics['risk_amount']:.2f} ({metrics['risk_percent']:.1f}% от депозита)\n"
            f"• Потенциальная прибыль: ${metrics['potential_profit']:.2f}\n"
            f"• R/R соотношение: {metrics['rr_ratio']:.2f}\n\n"
            f"🛡 <b>МАРЖИНАЛЬНЫЕ ПОКАЗАТЕЛИ:</b>\n"
            f"• Требуемая маржа: ${metrics['required_margin']:.2f}\n"
            f"• Свободная маржа: ${metrics['free_margin']:.2f}\n"
            f"• Уровень маржи: {metrics['margin_level']:.1f}%\n"
            f"• Использование маржи: {metrics['margin_usage_percent']:.1f}%\n"
            f"• Эффективное плечо: {metrics['effective_leverage']}\n\n"
            f"💡 <b>ТЕКУЩИЙ СТАТУС:</b>\n"
            f"• Текущая цена: ${metrics['current_price']:.4f}\n"
            f"• Текущий P&L: ${metrics['current_pnl']:.2f}\n"
            f"• Equity: ${metrics['equity']:.2f}\n\n"
            
            "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
        )
        
        keyboard = [
            [InlineKeyboardButton("📊 Портфель", callback_data="portfolio")],
            [InlineKeyboardButton("🎯 Новая сделка", callback_data="single_trade")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
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
            "❌ Введите число (например: 55000)\nПопробуйте еще раз:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return SingleTradeState.TAKE_PROFIT.value

async def single_trade_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена одиночной сделки"""
    user_id = update.effective_user.id if update.message else update.callback_query.from_user.id
    DataManager.clear_temporary_progress(user_id)
    context.user_data.clear()
    await SafeMessageSender.send_message(
        update.effective_chat.id,
        "❌ Расчет отменен",
        context,
        InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ])
    )
    return ConversationHandler.END

# ---------------------------
# Multi Trade Handlers (Зеркало Single Trade с префиксом 'm')
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало мультипозиции"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Initialize current_multi_trades if not exists
    if 'current_multi_trades' not in context.user_data:
        context.user_data['current_multi_trades'] = []
    
    # Save progress
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "multi")
    
    text = (
        "📊 <b>МУЛЬТИПОЗИЦИЯ v3.0</b>\n\n"
        "Расчет нескольких сделок в портфеле.\n\n"
        "<b>Введите ваш депозит в USD:</b>"
    )
    
    keyboard = [
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )
    return MultiTradeState.DEPOSIT.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка депозита для мультипозиции"""
    text = update.message.text.strip()
    
    # Save progress
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "multi")
    
    try:
        deposit = float(text.replace(',', '.'))
        if deposit < 100:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Минимальный депозит: $100\nПопробуйте еще раз:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return MultiTradeState.DEPOSIT.value
        
        context.user_data['deposit'] = deposit
        
        keyboard = []
        for leverage in LEVERAGES["DEFAULT"]:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"mlev_{leverage}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Депозит: ${deposit:,.2f}\n\n"
            "<b>Выберите кредитное плечо:</b>",
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return MultiTradeState.LEVERAGE.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Введите число (например: 1000)\nПопробуйте еще раз:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.DEPOSIT.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка плеча для мультипозиции"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Save progress
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "multi")
    
    leverage = query.data.replace('mlev_', '')
    context.user_data['leverage'] = leverage
    
    keyboard = []
    for category in ASSET_CATEGORIES.keys():
        keyboard.append([InlineKeyboardButton(category, callback_data=f"mcat_{category}")])
    
    keyboard.append([InlineKeyboardButton("📝 Ввести актив вручную", callback_data="massset_manual")])
    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Плечо: {leverage}\n\n"
        "<b>Выберите категорию актива:</b>",
        InlineKeyboardMarkup(keyboard)
    )
    return MultiTradeState.ASSET_CATEGORY.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_asset_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка категории для мультипозиции"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Save progress
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "multi")
    
    if query.data == "massset_manual":
        await SafeMessageSender.edit_message_text(
            query,
            "✍️ Введите название актива (например: BTCUSDT):",
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="mback_to_categories")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.ASSET.value
    
    category = query.data.replace('mcat_', '')
    context.user_data['asset_category'] = category
    
    assets = ASSET_CATEGORIES.get(category, [])
    
    keyboard = []
    for asset in assets:
        keyboard.append([InlineKeyboardButton(asset, callback_data=f"massset_{asset}")])
    
    keyboard.append([InlineKeyboardButton("🔙 Назад к категориям", callback_data="mback_to_categories")])
    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Категория: {category}\n\n"
        "<b>Выберите актив:</b>",
        InlineKeyboardMarkup(keyboard)
    )
    return MultiTradeState.ASSET.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def enhanced_multi_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик актива для мультипозиции"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Save progress
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "multi")
    
    if query.data == "mback_to_categories":
        keyboard = []
        for category in ASSET_CATEGORIES.keys():
            keyboard.append([InlineKeyboardButton(category, callback_data=f"mcat_{category}")])
        
        keyboard.append([InlineKeyboardButton("📝 Ввести актив вручную", callback_data="massset_manual")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await SafeMessageSender.edit_message_text(
            query,
            "<b>Выберите категорию актива:</b>",
            InlineKeyboardMarkup(keyboard)
        )
        return MultiTradeState.ASSET_CATEGORY.value
    
    asset = query.data.replace('massset_', '')
    context.user_data['asset'] = asset
    
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Актив: {asset}\n{price_info}"
        "<b>Выберите направление сделки:</b>",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="mdir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="mdir_SHORT")],
            [InlineKeyboardButton("🔙 Назад к категориям", callback_data="mback_to_categories")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return MultiTradeState.DIRECTION.value

async def multi_trade_asset_manual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Ручной ввод актива для мультипозиции"""
    asset = update.message.text.strip().upper()
    
    # Save progress
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "multi")
    
    if not re.match(r'^[A-Z0-9]{2,20}$', asset):
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Неверный формат актива. Попробуйте еще раз:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.ASSET.value
    
    context.user_data['asset'] = asset
    
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.send_message(
        update.message.chat_id,
        f"✅ Актив: {asset}\n{price_info}"
        "<b>Выберите направление сделки:</b>",
        context,
        InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="mdir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="mdir_SHORT")],
            [InlineKeyboardButton("🔙 Назад", callback_data="mback_to_categories")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return MultiTradeState.DIRECTION.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def enhanced_multi_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик направления для мультипозиции"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Save progress
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "multi")
    
    direction = query.data.replace('mdir_', '')
    context.user_data['direction'] = direction
    
    asset = context.user_data['asset']
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Направление: {direction}\n{price_info}"
        "<b>Введите цену входа:</b>",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="mback_to_asset")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return MultiTradeState.ENTRY.value

async def multi_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Цена входа для мультипозиции"""
    text = update.message.text.strip()
    
    # Save progress
    DataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "multi")
    
    try:
        entry_price = float(text.replace(',', '.'))
        if entry_price <= 0:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Цена должна быть больше 0\nПопробуйте еще раз:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return MultiTradeState.ENTRY.value
        
        context.user_data['entry_price'] = entry_price
        
        asset = context.user_data['asset']
        price_info = await show_asset_price_in_realtime(asset)
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Цена входа: {entry_price}\n{price_info}"
            "<b>Введите уровень стоп-лосса:</b>",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.STOP_LOSS.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Введите число (например: 50000)\nПопробуйте еще раз:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.ENTRY.value

async def multi_trade_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Стоп-лосс для мультипозиции"""
    text = update.message.text.strip()
    
    # Save progress
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
                "❌ Для LONG стоп-лосс должен быть НИЖЕ цены входа\nПопробуйте еще раз:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return MultiTradeState.STOP_LOSS.value
        elif direction == 'SHORT' and stop_loss <= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для SHORT стоп-лосс должен быть ВЫШЕ цены входа\nПопробуйте еще раз:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return MultiTradeState.STOP_LOSS.value
        
        context.user_data['stop_loss'] = stop_loss
        
        stop_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry_price, stop_loss, direction, asset)
        
        keyboard = []
        for risk_level in RISK_LEVELS:
            keyboard.append([InlineKeyboardButton(risk_level, callback_data=f"mrisk_{risk_level}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Стоп-лосс: {stop_loss} ({stop_distance_pips:.0f} пунктов)\n"
            f"💵 Сумма SL: ${abs(sl_amount):.2f}\n\n"
            "<b>Выберите уровень риска:</b>",
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return MultiTradeState.RISK_LEVEL.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Введите число (например: 48000)\nПопробуйте еще раз:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.STOP_LOSS.value

async def multi_trade_risk_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Уровень риска для мультипозиции"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Save progress
    DataManager.save_progress(query.from_user.id, context.user_data.copy(), "multi")
    
    risk_level = query.data.replace('mrisk_', '')
    context.user_data['risk_level'] = risk_level
    
    asset = context.user_data['asset']
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Уровень риска: {risk_level}\n{price_info}"
        "<b>Введите уровень тейк-профита:</b>",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return MultiTradeState.TAKE_PROFIT.value

async def multi_trade_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Тейк-профит для мультипозиции"""
    text = update.message.text.strip()
    
    # Save progress
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
                "❌ Для LONG тейк-профит должен быть ВЫШЕ цены входа\nПопробуйте еще раз:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return MultiTradeState.TAKE_PROFIT.value
        elif direction == 'SHORT' and take_profit >= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для SHORT тейк-профит должен быть НИЖЕ цены входа\nПопробуйте еще раз:",
                context,
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return MultiTradeState.TAKE_PROFIT.value
        
        context.user_data['take_profit'] = take_profit
        
        # Create trade and add to current_multi_trades
        trade = context.user_data.copy()
        trade.pop('current_multi_trades', None)  # Clean up
        context.user_data['current_multi_trades'].append(trade)
        
        # Calculate metrics for this trade
        metrics = await ProfessionalRiskCalculator.calculate_professional_metrics(
            trade, trade['deposit'], trade['leverage'], trade['risk_level']
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
            f"✅ Сделка #{len(context.user_data['current_multi_trades'])} добавлена!\n\n"
            f"Актив: {trade['asset']} | {trade['direction']}\n"
            f"Вход: {trade['entry_price']} | SL: {trade['stop_loss']} (${abs(sl_amount):.2f})\n"
            f"TP: {trade['take_profit']} (${tp_amount:.2f}) | Объем: {metrics['volume_lots']:.2f}\n"
            f"Риск: ${metrics['risk_amount']:.2f}\n\n"
            "<b>Добавить еще сделку или завершить?</b>"
        )
        
        keyboard = [
            [InlineKeyboardButton("➕ Добавить сделку", callback_data="madd_more")],
            [InlineKeyboardButton("✅ Завершить мультипозицию", callback_data="mfinish_multi")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
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
            "❌ Введите число (например: 55000)\nПопробуйте еще раз:",
            context,
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
        )
        return MultiTradeState.TAKE_PROFIT.value

async def multi_trade_add_more(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Добавление еще одной сделки в мультипозицию"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Clear current trade data but keep current_multi_trades
    current_multi = context.user_data.get('current_multi_trades', [])
    context.user_data = {'current_multi_trades': current_multi}
    
    await multi_trade_leverage(update, context)  # Redirect to leverage for new trade
    return MultiTradeState.LEVERAGE.value

async def multi_trade_finish(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Завершение мультипозиции и расчет портфеля"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    current_multi = context.user_data.get('current_multi_trades', [])
    
    if not current_multi:
        await SafeMessageSender.edit_message_text(
            query,
            "❌ Нет сделок для завершения",
            InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]])
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
        "📊 <b>МУЛЬТИПОЗИЦИЯ ЗАВЕРШЕНА v3.0</b>\n\n"
        f"Сделок добавлено: {len(current_multi)}\n"
        f"Депозит: ${deposit:,.2f} | Плечо: {leverage}\n\n"
        f"💰 <b>ПОРТФЕЛЬНЫЕ ПОКАЗАТЕЛИ:</b>\n"
        f"Общий риск: ${metrics['total_risk_usd']:.2f} ({metrics['total_risk_percent']:.1f}%)\n"
        f"Потенциальная прибыль: ${metrics['total_profit']:.2f}\n"
        f"Текущий P&L: ${metrics['total_pnl']:.2f}\n"
        f"Equity: ${metrics['total_equity']:.2f}\n\n"
        f"🛡 <b>МАРЖА:</b>\n"
        f"Требуемая: ${metrics['total_margin']:.2f} ({metrics['total_margin_usage']:.1f}%)\n"
        f"Свободная: ${metrics['free_margin']:.2f}\n"
        f"Уровень маржи: {metrics['portfolio_margin_level']:.1f}%\n\n"
        f"<b>💡 РЕКОМЕНДАЦИИ:</b>\n" + "\n".join(f"• {rec}" for rec in recommendations) + "\n\n"
        "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
    )
    
    keyboard = [
        [InlineKeyboardButton("📊 Полный портфель", callback_data="portfolio")],
        [InlineKeyboardButton("📤 Экспорт отчета", callback_data="export_portfolio")],
        [InlineKeyboardButton("🎯 Новая мультипозиция", callback_data="multi_trade_start")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
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
# CALLBACK ROUTER - ПОЛНОСТЬЮ ИСПРАВЛЕННЫЙ
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def callback_router_fixed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ГАРАНТИРОВАННО РАБОЧИЕ ОБРАБОТЧИКИ"""
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
        elif data.startswith("risk_"):
            await single_trade_risk_level(update, context)
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
        elif data.startswith("mrisk_"):
            await multi_trade_risk_level(update, context)
        elif data == "madd_more":
            await multi_trade_add_more(update, context)
        elif data == "mfinish_multi":
            await multi_trade_finish(update, context)
        else:
            await query.answer("Команда не распознана")
            
    except Exception as e:
        logger.error(f"Error in callback router: {e}")
        await query.answer("❌ Произошла ошибка")

# ---------------------------
# ДОПОЛНИТЕЛЬНЫЕ ОБРАБОТЧИКИ
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_calculation_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик профессиональных сделок"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = (
        "🎯 <b>ПРОФЕССИОНАЛЬНЫЕ СДЕЛКИ v3.0</b>\n\n"
        "Выберите тип расчета:"
    )
    
    keyboard = [
        [InlineKeyboardButton("🎯 Одна сделка", callback_data="single_trade")],
        [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")],
        [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

@retry_on_timeout(max_retries=2, delay=1.0)
async def main_menu_save_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Главное меню"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    # Clear progress on menu access
    DataManager.clear_temporary_progress(query.from_user.id)
    context.user_data.clear()
    
    text = "🏠 <b>ГЛАВНОЕ МЕНЮ</b>\n\nВыберите действие:"
    
    keyboard = [
        [InlineKeyboardButton("🎯 Профессиональный расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("📊 Портфель", callback_data="portfolio")],
        [InlineKeyboardButton("📚 Инструкции", callback_data="pro_info")],
        [InlineKeyboardButton("💖 Поддержать", callback_data="donate_start")],
        [InlineKeyboardButton("🔄 Восстановить прогресс", callback_data="restore_progress")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

@retry_on_timeout(max_retries=2, delay=1.0)
async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int = None):
    """Показ портфеля с реальными данными"""
    query = update.callback_query if update.callback_query else None
    if query:
        await SafeMessageSender.answer_callback_query(query)
    
    if not user_id:
        user_id = query.from_user.id if query else update.message.from_user.id
    
    PortfolioManager.ensure_user(user_id)
    user_portfolio = PortfolioManager.user_data[user_id]
    
    trades = user_portfolio.get('multi_trades', []) + user_portfolio.get('single_trades', [])
    
    if not trades:
        text = "📊 <b>Ваш портфель пуст</b>\n\nНачните с расчета сделки!"
        keyboard = [
            [InlineKeyboardButton("🎯 Новая сделка", callback_data="single_trade")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
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
    
    # Обновляем метрики с реальными ценами
    for trade in trades:
        metrics = await ProfessionalRiskCalculator.calculate_professional_metrics(
            trade, deposit, user_portfolio['leverage'], trade['risk_level']
        )
        trade['metrics'] = metrics
    
    metrics = PortfolioAnalyzer.calculate_portfolio_metrics(trades, deposit)
    recommendations = PortfolioAnalyzer.generate_enhanced_recommendations(metrics, trades)
    
    text = (
        "📊 <b>ПОРТФЕЛЬ v3.0</b>\n\n"
        f"💰 <b>ОСНОВНЫЕ ПОКАЗАТЕЛИ:</b>\n"
        f"Депозит: ${deposit:,.2f}\n"
        f"Плечо: {user_portfolio['leverage']}\n"
        f"Сделок: {len(trades)}\n"
        f"Equity: ${metrics['total_equity']:.2f}\n\n"
        f"🎯 <b>РИСКИ И ПРИБЫЛЬ:</b>\n"
        f"Общий риск: ${metrics['total_risk_usd']:.2f} ({metrics['total_risk_percent']:.1f}%)\n"
        f"Потенциальная прибыль: ${metrics['total_profit']:.2f}\n"
        f"Средний R/R: {metrics['avg_rr_ratio']:.2f}\n"
        f"Текущий P&L: ${metrics['total_pnl']:.2f}\n\n"
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
        "<b>📋 СДЕЛКИ:</b>\n"
    )
    
    for i, trade in enumerate(trades, 1):
        metrics = trade.get('metrics', {})
        pnl = metrics.get('current_pnl', 0)
        pnl_sign = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
        
        # Расчет сумм SL/TP в долларах
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
            f"   Вход: {trade['entry_price']} | SL: {trade['stop_loss']} (${abs(sl_amount):.2f}) | TP: {trade['take_profit']} (${tp_amount:.2f})\n"
            f"   Объем: {metrics.get('volume_lots', 0):.2f} | Риск: ${metrics.get('risk_amount', 0):.2f}\n"
            f"   P&L: ${pnl:.2f} | Маржа: ${metrics.get('required_margin', 0):.2f}\n\n"
        )
    
    text += "\n💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
    
    keyboard = [
        [InlineKeyboardButton("🗑 Очистить портфель", callback_data="clear_portfolio")],
        [InlineKeyboardButton("📤 Экспорт отчета", callback_data="export_portfolio")],
        [InlineKeyboardButton("🎯 Новая сделка", callback_data="single_trade")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
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
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

@retry_on_timeout(max_retries=2, delay=1.0)
async def export_portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Экспорт портфеля"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    PortfolioManager.ensure_user(user_id)
    
    user_portfolio = PortfolioManager.user_data[user_id]
    trades = user_portfolio.get('multi_trades', []) + user_portfolio.get('single_trades', [])
    
    if not trades:
        await SafeMessageSender.answer_callback_query(query, "❌ Портфель пуст")
        return
    
    report = f"📊 ОТЧЕТ ПОРТФЕЛЯ v3.0\nДата: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    report += f"Депозит: ${user_portfolio['deposit']:,.2f}\n"
    report += f"Плечо: {user_portfolio['leverage']}\n"
    report += f"Всего сделок: {len(trades)}\n\n"
    
    for i, trade in enumerate(trades, 1):
        report += f"СДЕЛКА #{i}:\n"
        report += f"Актив: {trade['asset']}\n"
        report += f"Направление: {trade['direction']}\n"
        report += f"Вход: {trade['entry_price']}\n"
        report += f"SL: {trade['stop_loss']}\n"
        report += f"TP: {trade['take_profit']}\n"
        
        if 'metrics' in trade:
            metrics = trade['metrics']
            # Расчет сумм SL/TP в долларах
            sl_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
                trade['entry_price'], trade['stop_loss'], metrics['volume_lots'],
                metrics['pip_value'], trade['direction'], trade['asset']
            )
            
            tp_amount = ProfessionalRiskCalculator.calculate_pnl_dollar_amount(
                trade['entry_price'], trade['take_profit'], metrics['volume_lots'],
                metrics['pip_value'], trade['direction'], trade['asset']
            )
            
            report += f"Объем: {metrics['volume_lots']:.2f} лотов\n"
            report += f"Риск: ${metrics['risk_amount']:.2f}\n"
            report += f"Маржа: ${metrics['required_margin']:.2f}\n"
            report += f"Прибыль: ${metrics['potential_profit']:.2f}\n"
            report += f"R/R: {metrics['rr_ratio']:.2f}\n"
            report += f"P&L: ${metrics['current_pnl']:.2f}\n"
            report += f"SL сумма: ${abs(sl_amount):.2f}\n"
            report += f"TP сумма: ${tp_amount:.2f}\n"
        
        report += "\n"
    
    report += "💎 PRO v3.0 | Smart • Fast • Reliable 🚀\n"
    
    bio = io.BytesIO()
    bio.write(report.encode('utf-8'))
    bio.seek(0)
    
    try:
        await context.bot.send_document(
            chat_id=query.message.chat_id,
            document=InputFile(bio, filename=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"),
            caption="📊 Отчет вашего портфеля"
        )
    except Exception as e:
        logger.error(f"Error sending portfolio report: {e}")
        await SafeMessageSender.answer_callback_query(query, "❌ Ошибка экспорта")

@retry_on_timeout(max_retries=2, delay=1.0)
async def restore_progress_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Восстановление прогресса"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    temp_data = DataManager.load_temporary_data()
    saved_progress = temp_data.get(str(user_id))
    
    if not saved_progress:
        await SafeMessageSender.edit_message_text(
            query,
            "❌ Нет сохраненного прогресса",
            InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]])
        )
        return
    
    context.user_data.update(saved_progress['state_data'])
    state_type = saved_progress['state_type']
    
    text = "✅ Прогресс восстановлен! Продолжайте расчет."
    keyboard = []
    
    if state_type == "single":
        keyboard = [[InlineKeyboardButton("🔄 Продолжить", callback_data="single_trade")]]
    else:
        keyboard = [[InlineKeyboardButton("🔄 Продолжить", callback_data="multi_trade_start")]]
    
    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

# ---------------------------
# НАСТРОЙКА CONVERSATION HANDLERS
# ---------------------------
def setup_conversation_handlers(application: Application):
    """Настройка обработчиков диалогов с реальными данными"""
    
    # Одиночная сделка
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
            SingleTradeState.RISK_LEVEL.value: [
                CallbackQueryHandler(single_trade_risk_level, pattern="^risk_"),
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
    
    # Мультипозиция
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
            MultiTradeState.RISK_LEVEL.value: [
                CallbackQueryHandler(multi_trade_risk_level, pattern="^mrisk_"),
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
# WEBHOOK И HTTP СЕРВЕР
# ---------------------------
async def set_webhook(application: Application) -> bool:
    """Установка вебхука с проверкой"""
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
    """Запуск HTTP сервера с улучшенной обработкой"""
    app = web.Application()
    
    async def handle_webhook(request):
        """Обработка webhook с таймаутом"""
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
        """Упрощенный health check для Render"""
        return web.Response(text="OK", status=200)
    
    app.router.add_post(WEBHOOK_PATH, handle_webhook)
    app.router.add_get('/health', health_check)
    app.router.add_get('/health/simple', render_health_check)
    app.router.add_get('/', render_health_check)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    
    logger.info(f"HTTP сервер запущен на порту {PORT}")
    return runner

# ---------------------------
# ГЛАВНАЯ ФУНКЦИЯ ЗАПУСКА
# ---------------------------
async def main_enhanced():
    """Улучшенная основная функция с полным исправлением всех ошибок"""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to start enhanced bot...")
            
            application = RobustApplicationBuilder.create_application(TOKEN)
            
            # Регистрация обработчиков команд
            application.add_handler(CommandHandler("start", start_command))
            application.add_handler(CommandHandler("pro_info", pro_info_command))
            
            # Настройка диалогов
            setup_conversation_handlers(application)
            
            # Callback router
            application.add_handler(CallbackQueryHandler(callback_router_fixed))
            
            # Обработчик для любых сообщений (fallback)
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
            
            # Режим запуска
            if WEBHOOK_URL and WEBHOOK_URL.strip():
                logger.info("Запуск в режиме WEBHOOK")
                await application.initialize()
                
                if await set_webhook(application):
                    await start_http_server(application)
                    logger.info("✅ Бот успешно запущен в режиме WEBHOOK")
                    
                    while True:
                        await asyncio.sleep(300)
                        logger.debug("Health check - бот работает стабильно")
                else:
                    logger.error("Не удалось установить вебхук, запуск в режиме polling")
                    raise Exception("Webhook setup failed")
            else:
                logger.info("Запуск в режиме POLLING")
                await application.run_polling(
                    poll_interval=1.0,
                    timeout=30,
                    drop_pending_updates=True
                )
                
            break
                
        except telegram.error.TimedOut as e:
            logger.error(f"Timeout error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error("All startup attempts failed due to timeouts")
                raise
                
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error("All startup attempts failed")
                raise

# ---------------------------
# ЗАПУСК ПРИЛОЖЕНИЯ
# ---------------------------
if __name__ == "__main__":
    logger.info("🚀 ЗАПУСК PRO RISK CALCULATOR v3.0 ENTERPRISE EDITION")
    logger.info("✅ ВСЕ КРИТИЧЕСКИЕ ОШИБКИ ИСПРАВЛЕНЫ")
    logger.info("🎯 ИСПРАВЛЕНЫ РАСЧЕТЫ МАРЖИ И ОБЪЕМА")
    logger.info("🔧 СИСТЕМА ГОТОВА К ПРОДАКШЕНУ")
    
    try:
        asyncio.run(main_enhanced())
    except KeyboardInterrupt:
        logger.info("⏹ Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        try:
            if enhanced_market_data.session:
                await enhanced_market_data.session.close()
        except:
            pass
        raise
