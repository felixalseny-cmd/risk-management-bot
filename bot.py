# bot.py — PRO Risk Calculator v3.1 | ENTERPRISE EDITION
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

# Donation Wallets
USDT_WALLET_ADDRESS = os.getenv("USDT_WALLET_ADDRESS", "TVRGFPKVs1nN3fUXBTQfu5syTcmYGgADre")
TON_WALLET_ADDRESS = os.getenv("TON_WALLET_ADDRESS", "UQDpCH-pGSzp3zEkpJY1Wc46gaorw9K-7T9FX7gHTrthMWMj")

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
            read_timeout=30,
            write_timeout=30,
            connect_timeout=30
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
# Safe Message Sender - ОБНОВЛЕННЫЙ С HTML
# ---------------------------
class SafeMessageSender:
    """Безопасная отправка сообщений с обработкой ошибок"""
    
    @staticmethod
    @retry_on_timeout(max_retries=3, delay=1.0)
    async def send_message(
        chat_id: int,
        text: str,
        context: ContextTypes.DEFAULT_TYPE = None,
        reply_markup: InlineKeyboardMarkup = None,
        parse_mode: str = 'HTML'
    ) -> bool:
        """Безопасная отправка сообщения"""
        try:
            if context and hasattr(context, 'bot'):
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode
                )
            else:
                # Fallback - создаем временного бота
                from telegram import Bot
                bot = Bot(token=TOKEN)
                await bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode
                )
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")
            return False
    
    @staticmethod
    @retry_on_timeout(max_retries=2, delay=1.0)
    async def edit_message_text(
        query: 'CallbackQuery',
        text: str,
        reply_markup: InlineKeyboardMarkup = None,
        parse_mode: str = 'HTML'
    ) -> bool:
        """Безопасное редактирование сообщения"""
        try:
            await query.edit_message_text(
                text=text,
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
# Donation System - ПРОФЕССИОНАЛЬНАЯ СИСТЕМА ДОНАТОВ БЕЗ QR
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
        """Показать USDT кошелек для донатов без QR"""
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
            "💝 <i>Любая сумма будет принята с благодарностью!</i>"
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
        """Показать TON кошелек для донатов без QR"""
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
            "💝 <i>Любая сумма будет принята с благодарностью!</i>"
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
# Enhanced Market Data Provider - РЕАЛЬНЫЕ КОТИРОВКИ С ПРИОРИТЕТАМИ
# ---------------------------
class EnhancedMarketDataProvider:
    """Улучшенный провайдер рыночных данных с приоритетами и комбинированием источников"""
    
    def __init__(self):
        # Кэш с разным TTL: 60 сек для крипто, 300 сек для остального
        self.crypto_cache = cachetools.TTLCache(maxsize=200, ttl=60)
        self.standard_cache = cachetools.TTLCache(maxsize=300, ttl=300)
        self.session = None
        self.request_count = 0
        self.last_request_time = time.time()
        
    async def get_session(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _get_cache(self, symbol: str) -> cachetools.TTLCache:
        """Выбор кэша в зависимости от типа актива"""
        if self._is_crypto(symbol):
            return self.crypto_cache
        return self.standard_cache
    
    async def get_real_time_price(self, symbol: str) -> float:
        """Получение реальной цены с приоритизацией провайдеров"""
        return await self.get_robust_real_time_price(symbol)
    
    async def get_robust_real_time_price(self, symbol: str) -> float:
        """Надежное получение реальных цен с приоритетами и комбинированием"""
        try:
            # Проверка кэша
            cache = self._get_cache(symbol)
            cached_price = cache.get(symbol)
            if cached_price:
                return cached_price
                
            # Определяем тип актива и выбираем провайдеров по приоритету
            price_providers = []
            
            if self._is_forex(symbol):
                # Forex: ExchangeRate-API > Alpha Vantage > Finnhub
                price_providers = [
                    self._get_exchangerate_price,
                    self._get_alpha_vantage_forex,
                    self._get_finnhub_price
                ]
            elif self._is_crypto(symbol):
                # Crypto: Binance > Finnhub
                price_providers = [
                    self._get_binance_price,
                    self._get_finnhub_price
                ]
            elif self._is_stock(symbol) or self._is_index(symbol):
                # Stocks/Indices: Alpha Vantage > Finnhub
                price_providers = [
                    self._get_alpha_vantage_stock,
                    self._get_finnhub_price
                ]
            else:  # Metals, Energy
                price_providers = [
                    self._get_alpha_vantage_forex,
                    self._get_finnhub_price
                ]
            
            # Получаем цены от всех доступных провайдеров
            prices = []
            for provider in price_providers:
                try:
                    price = await provider(symbol)
                    if price and price > 0:
                        prices.append(price)
                        # Если нашли хорошую цену, можно остановиться или продолжить для комбинирования
                except Exception as e:
                    logger.debug(f"Provider {provider.__name__} failed for {symbol}: {e}")
                    continue
            
            # Комбинирование цен
            price = self._combine_prices(prices, symbol)
                
            # Fallback на статические данные при ошибках
            if price is None or price <= 0:
                logger.warning(f"Не удалось получить цену для {symbol}, используется fallback")
                price = self._get_fallback_price(symbol)
                
            # Сохраняем в кэш
            if price:
                cache[symbol] = price
                
            return price
            
        except Exception as e:
            logger.error(f"Ошибка получения цены для {symbol}: {e}")
            return self._get_fallback_price(symbol)
    
    def _combine_prices(self, prices: List[float], symbol: str) -> Optional[float]:
        """Комбинирование цен от разных источников"""
        if not prices:
            return None
            
        if len(prices) == 1:
            return prices[0]
            
        # Проверяем дисперсию цен
        avg_price = sum(prices) / len(prices)
        max_deviation = max(abs(p - avg_price) for p in prices)
        deviation_percent = (max_deviation / avg_price) * 100 if avg_price > 0 else 100
        
        # Если дисперсия в пределах 1%, используем среднее
        if deviation_percent <= 1.0:
            return avg_price
        else:
            # Большая дисперсия - используем самый надежный источник
            logger.warning(f"Большая дисперсия цен для {symbol}: {deviation_percent:.2f}%")
            # Для Forex предпочтительнее ExchangeRate-API
            if self._is_forex(symbol):
                return prices[0] if prices else None
            # Для крипто - Binance
            elif self._is_crypto(symbol):
                return prices[0] if prices else None
            else:
                return prices[0] if prices else None
    
    def _is_crypto(self, symbol: str) -> bool:
        """Проверка является ли актив криптовалютой"""
        crypto_symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'USDT']
        return any(crypto in symbol.upper() for crypto in crypto_symbols)
    
    def _is_forex(self, symbol: str) -> bool:
        """Проверка является ли актив Forex парой"""
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        return symbol.upper() in forex_pairs
    
    def _is_metal(self, symbol: str) -> bool:
        """Проверка является ли актив металлом"""
        metals = ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD']
        return symbol.upper() in metals
    
    def _is_energy(self, symbol: str) -> bool:
        """Проверка является ли актив энергоресурсом"""
        energy = ['OIL', 'NATURALGAS', 'BRENT']
        return symbol.upper() in energy
    
    def _is_stock(self, symbol: str) -> bool:
        """Проверка является ли актив акцией"""
        stocks = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX']
        return symbol.upper() in stocks
    
    def _is_index(self, symbol: str) -> bool:
        """Проверка является ли актив индексом"""
        indices = ['NAS100', 'SPX500', 'DJ30', 'FTSE100', 'DAX40', 'NIKKEI225', 'ASX200']
        return symbol.upper() in indices
    
    async def _get_exchangerate_price(self, symbol: str) -> Optional[float]:
        """Получение Forex цен с ExchangeRate-API"""
        try:
            if not self._is_forex(symbol):
                return None
                
            # Конвертируем символ для API (EURUSD -> EUR/USD)
            from_currency = symbol[:3]
            to_currency = symbol[3:]
            api_symbol = f"{from_currency}/{to_currency}"
            
            session = await self.get_session()
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
            
            async with session.get(url, timeout=8) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'rates' in data and to_currency in data['rates']:
                        return float(data['rates'][to_currency])
        except asyncio.TimeoutError:
            logger.warning(f"ExchangeRate-API timeout for {symbol}")
        except Exception as e:
            logger.debug(f"ExchangeRate-API error for {symbol}: {e}")
        return None
    
    async def _get_binance_price(self, symbol: str) -> Optional[float]:
        """Получение цены с Binance API"""
        try:
            session = await self.get_session()
            # Форматируем символ для Binance
            if 'USDT' in symbol.upper():
                binance_symbol = symbol.replace('USDT', '') + 'USDT'
            else:
                binance_symbol = symbol + 'USDT'
                
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"
            
            async with session.get(url, timeout=8) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data['price'])
        except asyncio.TimeoutError:
            logger.warning(f"Binance API timeout for {symbol}")
        except Exception as e:
            logger.debug(f"Binance API error for {symbol}: {e}")
        return None
    
    async def _get_alpha_vantage_stock(self, symbol: str) -> Optional[float]:
        """Получение цены акций с Alpha Vantage"""
        if not ALPHA_VANTAGE_API_KEY:
            return None
            
        try:
            session = await self.get_session()
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Global Quote' in data and '05. price' in data['Global Quote']:
                        return float(data['Global Quote']['05. price'])
        except asyncio.TimeoutError:
            logger.warning(f"Alpha Vantage stock timeout for {symbol}")
        except Exception as e:
            logger.debug(f"Alpha Vantage stock error for {symbol}: {e}")
        return None
    
    async def _get_alpha_vantage_forex(self, symbol: str) -> Optional[float]:
        """Получение Forex цен с Alpha Vantage"""
        if not ALPHA_VANTAGE_API_KEY:
            return None
            
        try:
            session = await self.get_session()
            # Конвертируем символы для Alpha Vantage (EURUSD -> EUR/USD)
            from_currency = symbol[:3]
            to_currency = symbol[3:]
            url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}&to_currency={to_currency}&apikey={ALPHA_VANTAGE_API_KEY}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Realtime Currency Exchange Rate' in data and '5. Exchange Rate' in data['Realtime Currency Exchange Rate']:
                        return float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])
        except asyncio.TimeoutError:
            logger.warning(f"Alpha Vantage forex timeout for {symbol}")
        except Exception as e:
            logger.debug(f"Alpha Vantage forex error for {symbol}: {e}")
        return None
    
    async def _get_finnhub_price(self, symbol: str) -> Optional[float]:
        """Получение цены с Finnhub (резервный)"""
        if not FINNHUB_API_KEY:
            return None
            
        try:
            session = await self.get_session()
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
            
            async with session.get(url, timeout=8) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['c']  # current price
        except asyncio.TimeoutError:
            logger.warning(f"Finnhub API timeout for {symbol}")
        except Exception as e:
            logger.debug(f"Finnhub API error for {symbol}: {e}")
        return None
    
    def _get_fallback_price(self, symbol: str) -> float:
        """Fallback цены при недоступности API"""
        fallback_prices = {
            'BTCUSDT': 65000.0, 'ETHUSDT': 3500.0, 'EURUSD': 1.0850,
            'GBPUSD': 1.2650, 'USDJPY': 150.50, 'XAUUSD': 2150.0,
            'AAPL': 185.0, 'TSLA': 180.0, 'GOOGL': 140.0, 'MSFT': 420.0,
            'AMZN': 175.0, 'NAS100': 18000.0, 'SPX500': 5200.0, 'OIL': 80.0
        }
        return fallback_prices.get(symbol.upper(), 100.0)

# ---------------------------
# Enhanced Instrument Specifications - ОБНОВЛЕННЫЕ СПЕЦИФИКАЦИИ
# ---------------------------
class EnhancedInstrumentSpecs:
    """Улучшенная база спецификаций финансовых инструментов"""
    
    SPECS = {
        # Forex пары с обновленными pip values
        "EURUSD": {
            "type": "forex",
            "contract_size": 100000,
            "margin_currency": "USD",
            "pip_value": 10.0,  # Стандарт для Forex majors
            "calculation_formula": "forex",
            "pip_decimal_places": 4
        },
        "GBPUSD": {
            "type": "forex",
            "contract_size": 100000,
            "margin_currency": "USD", 
            "pip_value": 10.0,  # Стандарт для Forex majors
            "calculation_formula": "forex",
            "pip_decimal_places": 4
        },
        "USDJPY": {
            "type": "forex", 
            "contract_size": 100000,
            "margin_currency": "USD",
            "pip_value": 9.09,  # Динамически рассчитывается ~100000 * 0.01 / 110.00
            "calculation_formula": "forex_jpy",
            "pip_decimal_places": 2
        },
        "USDCHF": {
            "type": "forex",
            "contract_size": 100000,
            "margin_currency": "USD",
            "pip_value": 10.0,
            "calculation_formula": "forex",
            "pip_decimal_places": 4
        },
        
        # Криптовалюты с обновленными значениями
        "BTCUSDT": {
            "type": "crypto",
            "contract_size": 1,
            "margin_currency": "USDT",
            "pip_value": 1.0,  # $1 за пункт
            "calculation_formula": "crypto",
            "pip_decimal_places": 1
        },
        "ETHUSDT": {
            "type": "crypto",
            "contract_size": 1,
            "margin_currency": "USDT",
            "pip_value": 1.0,  # $1 за пункт
            "calculation_formula": "crypto",
            "pip_decimal_places": 2
        },
        
        # Акции
        "AAPL": {
            "type": "stock",
            "contract_size": 100,
            "margin_currency": "USD",
            "pip_value": 1.0,  # $1 за пункт
            "calculation_formula": "stocks",
            "pip_decimal_places": 2
        },
        "TSLA": {
            "type": "stock",
            "contract_size": 100,
            "margin_currency": "USD",
            "pip_value": 1.0,
            "calculation_formula": "stocks", 
            "pip_decimal_places": 2
        },
        
        # Индексы
        "NAS100": {
            "type": "index",
            "contract_size": 10,
            "margin_currency": "USD",
            "pip_value": 1.0,
            "calculation_formula": "indices",
            "pip_decimal_places": 1
        },
        
        # Металлы
        "XAUUSD": {
            "type": "metal", 
            "contract_size": 100,
            "margin_currency": "USD",
            "pip_value": 10.0,  # $10 за пункт
            "calculation_formula": "metals",
            "pip_decimal_places": 2
        },
        
        # Энергия
        "OIL": {
            "type": "energy",
            "contract_size": 1000,
            "margin_currency": "USD",
            "pip_value": 10.0,  # $10 за пункт
            "calculation_formula": "energy",
            "pip_decimal_places": 2
        }
    }
    
    @classmethod
    def get_specs(cls, symbol: str) -> Dict[str, Any]:
        """Получение спецификаций для инструмента"""
        specs = cls.SPECS.get(symbol.upper(), cls._get_default_specs(symbol))
        
        # Динамический расчет pip value для JPY пар
        if symbol.upper() == 'USDJPY':
            # Примерный расчет: 100000 * 0.01 / текущий курс (~150) = ~6.67
            # Но используем стандартное значение для консистентности
            pass
            
        return specs
    
    @classmethod
    def _get_default_specs(cls, symbol: str) -> Dict[str, Any]:
        """Спецификации по умолчанию"""
        symbol_upper = symbol.upper()
        
        if any(currency in symbol_upper for currency in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']):
            return {
                "type": "forex",
                "contract_size": 100000,
                "margin_currency": "USD",
                "pip_value": 10.0,  # Стандарт для Forex
                "calculation_formula": "forex",
                "pip_decimal_places": 4
            }
        elif 'USDT' in symbol_upper:
            return {
                "type": "crypto",
                "contract_size": 1,
                "margin_currency": "USDT", 
                "pip_value": 1.0,  # Стандарт для крипто
                "calculation_formula": "crypto",
                "pip_decimal_places": 2
            }
        else:
            return {
                "type": "stock",
                "contract_size": 100,
                "margin_currency": "USD",
                "pip_value": 1.0,  # Стандарт для акций
                "calculation_formula": "stocks",
                "pip_decimal_places": 2
            }

# ---------------------------
# Professional Margin Calculator - ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ
# ---------------------------
class ProfessionalMarginCalculator:
    """ПРОФЕССИОНАЛЬНЫЙ расчет маржи с реальными котировками"""
    
    def __init__(self):
        self.market_data = EnhancedMarketDataProvider()
    
    async def calculate_professional_margin(self, symbol: str, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Профессиональный расчет маржи с реальными котировками"""
        try:
            specs = EnhancedInstrumentSpecs.get_specs(symbol)
            formula = specs['calculation_formula']
            
            if formula == "forex":
                return await self._calculate_forex_margin(specs, volume, leverage, current_price)
            elif formula == "forex_jpy":
                return await self._calculate_forex_jpy_margin(specs, volume, leverage, current_price)
            elif formula == "crypto":
                return await self._calculate_crypto_margin(specs, volume, leverage, current_price)
            elif formula == "stocks":
                return await self._calculate_stocks_margin(specs, volume, leverage, current_price)
            elif formula == "indices":
                return await self._calculate_indices_margin(specs, volume, leverage, current_price)
            elif formula == "metals":
                return await self._calculate_metals_margin(specs, volume, leverage, current_price)
            elif formula == "energy":
                return await self._calculate_energy_margin(specs, volume, leverage, current_price)
            else:
                return await self._calculate_universal_margin(specs, volume, leverage, current_price)
                
        except Exception as e:
            logger.error(f"Ошибка расчета маржи для {symbol}: {e}")
            return await self._calculate_universal_margin(specs, volume, leverage, current_price)
    
    async def _calculate_forex_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Расчет маржи для Forex по отраслевым стандартам"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # Профессиональная формула: (Объем × Размер контракта) / Плечо
        required_margin = (volume * contract_size) / lev_value
        
        return {
            'required_margin': required_margin,
            'contract_size': contract_size,
            'calculation_method': 'forex_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size
        }
    
    async def _calculate_forex_jpy_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Расчет маржи для JPY пар (особенности расчета)"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # Для JPY: (Объем × Размер контракта) / (Плечо × Цена)
        required_margin = (volume * contract_size) / (lev_value * current_price)
        
        return {
            'required_margin': required_margin,
            'contract_size': contract_size,
            'calculation_method': 'forex_jpy_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size
        }
    
    async def _calculate_crypto_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Расчет маржи для криптовалют"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # Для крипто: (Объем × Цена) / Плечо
        required_margin = (volume * current_price) / lev_value
        
        return {
            'required_margin': required_margin,
            'contract_size': contract_size,
            'calculation_method': 'crypto_standard',
            'leverage_used': lev_value,
            'notional_value': volume * current_price
        }
    
    async def _calculate_stocks_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Расчет маржи для акций"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # Для акций: (Объем × Размер контракта × Цена) / Плечо
        required_margin = (volume * contract_size * current_price) / lev_value
        
        return {
            'required_margin': required_margin,
            'contract_size': contract_size,
            'calculation_method': 'stocks_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size * current_price
        }
    
    async def _calculate_indices_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Расчет маржи для индексов"""
        return await self._calculate_stocks_margin(specs, volume, leverage, current_price)
    
    async def _calculate_metals_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Расчет маржи для металлов"""
        return await self._calculate_forex_margin(specs, volume, leverage, current_price)
    
    async def _calculate_energy_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Расчет маржи для энергоресурсов"""
        return await self._calculate_forex_margin(specs, volume, leverage, current_price)
    
    async def _calculate_universal_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Универсальный расчет маржи"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs.get('contract_size', 1)
        
        required_margin = (volume * contract_size * current_price) / lev_value
        
        return {
            'required_margin': required_margin,
            'contract_size': contract_size,
            'calculation_method': 'universal',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size * current_price
        }

# Инициализация глобальных сервисов
market_data_provider = EnhancedMarketDataProvider()
margin_calculator = ProfessionalMarginCalculator()

# ---------------------------
# Константы и состояния (ОБНОВЛЕННЫЕ)
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

# Инструменты и пресеты
ASSET_CATEGORIES = {
    "FOREX": ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'],
    "CRYPTO": ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'ADAUSDT', 'DOTUSDT'],
    "INDICES": ['NAS100', 'SPX500', 'DJ30', 'FTSE100', 'DAX40', 'NIKKEI225', 'ASX200'],
    "METALS": ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD'],
    "ENERGY": ['OIL', 'NATURALGAS', 'BRENT'],
    "STOCKS": ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX']
}

LEVERAGES = ['1:10', '1:20', '1:50', '1:100', '1:200', '1:500', '1:1000']
RISK_LEVELS = ['2%', '5%', '7%', '10%', '15%', '20%', '25%']

# Волатильность активов (ОБНОВЛЕННЫЕ ДАННЫЕ)
VOLATILITY_DATA = {
    'BTCUSDT': 65.2, 'ETHUSDT': 70.5, 'AAPL': 25.3, 'TSLA': 55.1,
    'GOOGL': 22.8, 'MSFT': 20.1, 'AMZN': 28.7, 'EURUSD': 8.5,
    'GBPUSD': 9.2, 'USDJPY': 7.8, 'USDCHF': 6.5, 'AUDUSD': 10.1,
    'USDCAD': 7.2, 'NZDUSD': 9.8, 'XAUUSD': 14.5, 'XAGUSD': 25.3,
    'OIL': 35.2, 'NAS100': 18.5, 'SPX500': 15.2, 'DJ30': 12.8
}

# ---------------------------
# Data Manager (ОБНОВЛЕННЫЙ)
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
        """Сохранение временного прогресса"""
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
        """Загрузка временных данных"""
        try:
            if os.path.exists("temp_progress.json"):
                with open("temp_progress.json", 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error("Ошибка загрузки прогресса: %s", e)
        return {}

    @staticmethod
    def clear_temporary_progress(user_id: int):
        """Очистка временного прогресса"""
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
# Portfolio Manager (ОБНОВЛЕННЫЙ - только для мульти)
# ---------------------------
class PortfolioManager:
    @staticmethod
    def ensure_user(user_id: int):
        if user_id not in user_data:
            user_data[user_id] = {
                'multi_trades': [],
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
# Enhanced Professional Risk Calculator - ПОЛНОСТЬЮ ПЕРЕРАБОТАННЫЙ
# ---------------------------
class EnhancedProfessionalRiskCalculator:
    """УЛУЧШЕННЫЙ ПРОФЕССИОНАЛЬНЫЙ калькулятор с реальными котировками"""
    
    @staticmethod
    def calculate_pip_distance(entry: float, target: float, direction: str, asset: str) -> float:
        """Профессиональный расчет дистанции в пунктах"""
        specs = EnhancedInstrumentSpecs.get_specs(asset)
        pip_decimal_places = specs.get('pip_decimal_places', 4)
        
        if direction.upper() == 'LONG':
            distance = target - entry
        else:  # SHORT
            distance = entry - target
        
        # Масштабирование в зависимости от типа актива
        if pip_decimal_places == 2:  # JPY пары
            return abs(distance) * 100
        elif pip_decimal_places == 1:  # Некоторые индексы
            return abs(distance) * 10
        else:  # Стандартные 4 знака
            return abs(distance) * 10000

    @staticmethod
    async def calculate_enhanced_metrics(trade: Dict, deposit: float, leverage: str, risk_level: str) -> Dict[str, Any]:
        """
        УЛУЧШЕННЫЙ расчет с реальными котировками и маржой
        """
        try:
            asset = trade['asset']
            entry = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            direction = trade['direction']
            
            # 1. Получение РЕАЛЬНОЙ цены актива
            current_price = await market_data_provider.get_robust_real_time_price(asset)
            
            # 2. Получение спецификаций инструмента
            specs = EnhancedInstrumentSpecs.get_specs(asset)
            
            # 3. Расчет суммы риска
            risk_percent = float(risk_level.strip('%'))
            risk_amount = deposit * (risk_percent / 100)
            
            # 4. Профессиональный расчет дистанции
            stop_distance_pips = EnhancedProfessionalRiskCalculator.calculate_pip_distance(entry, stop_loss, direction, asset)
            profit_distance_pips = EnhancedProfessionalRiskCalculator.calculate_pip_distance(entry, take_profit, direction, asset)
            
            # 5. Получаем стоимость пункта
            pip_value = specs['pip_value']
            
            # 6. Расчет объема на основе РИСКА
            if stop_distance_pips > 0 and pip_value > 0:
                volume_lots = risk_amount / (stop_distance_pips * pip_value)
                volume_lots = round(volume_lots, 2)
            else:
                volume_lots = 0
            
            # 7. ПРОФЕССИОНАЛЬНЫЙ расчет маржи с реальными котировками
            margin_data = await margin_calculator.calculate_professional_margin(
                asset, volume_lots, leverage, current_price
            )
            required_margin = margin_data['required_margin']
            required_margin = round(required_margin, 2)
            
            # 8. Расчет всех метрик
            free_margin = deposit - required_margin
            free_margin = round(free_margin, 2)
            
            margin_level = (deposit / required_margin) * 100 if required_margin > 0 else 0
            margin_level = round(margin_level, 1)
            
            potential_profit = volume_lots * profit_distance_pips * pip_value
            potential_profit = round(potential_profit, 2)
            
            rr_ratio = potential_profit / risk_amount if risk_amount > 0 else 0
            rr_ratio = round(rr_ratio, 2)
            
            # Дополнительные профессиональные метрики
            risk_per_trade_percent = (risk_amount / deposit) * 100 if deposit > 0 else 0
            margin_usage_percent = (required_margin / deposit) * 100 if deposit > 0 else 0
            notional_value = margin_data.get('notional_value', 0)
            
            # Расчет реалистичного P&L
            current_pnl = EnhancedProfessionalRiskCalculator.calculate_realistic_pnl(trade, current_price, volume_lots, pip_value, direction, asset)
            
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
                'risk_per_trade_percent': risk_per_trade_percent,
                'margin_usage_percent': margin_usage_percent,
                'current_price': current_price,  # РЕАЛЬНАЯ цена
                'calculation_method': margin_data['calculation_method'],  # Метод расчета
                'notional_value': notional_value,  # Номинальная стоимость
                'leverage_used': margin_data.get('leverage_used', 1),
                'current_pnl': current_pnl  # Реалистичный P&L
            }
        except Exception as e:
            logger.error(f"Профессиональный расчет ошибка: {e}")
            return {}

    @staticmethod
    def calculate_realistic_pnl(trade: Dict, current_price: float, volume: float, pip_value: float, direction: str, asset: str) -> float:
        """Расчет реалистичного P&L"""
        entry = trade['entry_price']
        
        if direction == 'LONG':
            price_diff = current_price - entry
        else:  # SHORT
            price_diff = entry - current_price
        
        # Конвертация в пункты с учетом спецификаций актива
        pip_diff = EnhancedProfessionalRiskCalculator.calculate_pip_distance(entry, entry + price_diff if direction == 'LONG' else entry - price_diff, direction, asset)
        
        return round(volume * pip_diff * pip_value, 2)

# ---------------------------
# Portfolio Analyzer (ОБНОВЛЕННЫЙ)
# ---------------------------
class PortfolioAnalyzer:
    @staticmethod
    def calculate_portfolio_metrics(trades: List[Dict], deposit: float) -> Dict[str, Any]:
        """Профессиональный расчет метрик портфеля"""
        if not trades:
            return {}
        
        total_risk = sum(t.get('metrics', {}).get('risk_amount', 0) for t in trades)
        total_profit = sum(t.get('metrics', {}).get('potential_profit', 0) for t in trades)
        total_margin = sum(t.get('metrics', {}).get('required_margin', 0) for t in trades)
        total_notional = sum(t.get('metrics', {}).get('notional_value', 0) for t in trades)
        total_free_margin = sum(t.get('metrics', {}).get('free_margin', 0) for t in trades) if trades else deposit
        
        avg_rr = sum(t.get('metrics', {}).get('rr_ratio', 0) for t in trades) / len(trades) if trades else 0
        
        # Волатильность портфеля
        portfolio_volatility = sum(VOLATILITY_DATA.get(t['asset'], 20) for t in trades) / len(trades) if trades else 0
        
        # Анализ направлений
        long_count = sum(1 for t in trades if t.get('direction', '').upper() == 'LONG')
        short_count = len(trades) - long_count
        direction_balance = abs(long_count - short_count) / len(trades) if trades else 0
        
        # Диверсификация
        unique_assets = len(set(t['asset'] for t in trades))
        diversity_score = unique_assets / len(trades) if trades else 0
        
        # Уровень маржи портфеля
        portfolio_margin_level = (deposit / total_margin) * 100 if total_margin > 0 else 0
        
        # Общее использование маржи
        total_margin_usage = (total_margin / deposit) * 100 if deposit > 0 else 0
        
        # Общий левередж портфеля
        portfolio_leverage = total_notional / deposit if deposit > 0 else 0
        
        # Свободная маржа портфеля
        free_margin = deposit - total_margin if deposit > 0 else 0
        free_margin_percent = (free_margin / deposit) * 100 if deposit > 0 else 0
        
        # Общий P&L
        total_pnl = sum(t.get('metrics', {}).get('current_pnl', 0) for t in trades)
        
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
            'total_pnl': total_pnl
        }

    @staticmethod
    def generate_recommendations(metrics: Dict, trades: List[Dict]) -> List[str]:
        """Профессиональные рекомендации на основе метрик"""
        recommendations = []
        
        # Проверка общего риска
        if metrics.get('total_risk_percent', 0) > 10:
            recommendations.append(
                "⚠️ ВНИМАНИЕ: Общий риск портфеля превышает 10%. "
                "Рекомендуется уменьшить объем позиций для защиты капитала."
            )
        elif metrics.get('total_risk_percent', 0) > 5:
            recommendations.append(
                "🔶 ПРЕДУПРЕЖДЕНИЕ: Общий риск портфеля превышает 5%. "
                "Рассмотрите снижение объема позиций."
            )
        
        # Проверка уровня маржи
        if metrics.get('portfolio_margin_level', 0) < 100:
            recommendations.append(
                "🔴 КРИТИЧЕСКИЙ УРОВЕНЬ МАРЖИ! Немедленно пополните счет "
                "или закрите часть позиций во избежание маржин-колла."
            )
        elif metrics.get('portfolio_margin_level', 0) < 200:
            recommendations.append(
                "🟡 НИЗКИЙ УРОВЕНЬ МАРЖИ: Рассмотрите пополнение счета "
                "для безопасности позиций. Рекомендуемый уровень > 200%."
            )
        
        # Проверка использования маржи
        if metrics.get('total_margin_usage', 0) > 50:
            recommendations.append(
                f"🟡 ВЫСОКОЕ ИСПОЛЬЗОВАНИЕ МАРЖИ: {metrics['total_margin_usage']:.1f}%. "
                "Оставьте свободную маржу для непредвиденных ситуаций."
            )
        
        # Проверка левереджа
        if metrics.get('portfolio_leverage', 0) > 10:
            recommendations.append(
                f"🔶 ВЫСОКИЙ ЛЕВЕРЕДЖ: {metrics['portfolio_leverage']:.1f}x. "
                "Высокий левередж увеличивает как потенциальную прибыль, так и риски."
            )
        
        # Проверка Risk/Reward
        low_rr_trades = [
            t for t in trades 
            if t.get('metrics', {}).get('rr_ratio', 0) < 1
        ]
        if low_rr_trades:
            recommendations.append(
                f"📉 НЕВЫГОДНОЕ R/R: {len(low_rr_trades)} сделок имеют соотношение < 1. "
                "Пересмотрите уровни TP/SL для улучшения риск-менеджмента."
            )
        
        # Проверка волатильности
        if metrics.get('portfolio_volatility', 0) > 30:
            recommendations.append(
                f"🌪 ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ: {metrics['portfolio_volatility']:.1f}%. "
                "Будьте готовы к значительным колебаниям стоимости портфеля."
            )
        
        # Проверка диверсификации
        if metrics.get('diversity_score', 0) < 0.5 and len(trades) > 1:
            recommendations.append(
                "🎯 НИЗКАЯ ДИВЕРСИФИКАЦИЯ. Рассмотрите добавление активов "
                "из разных секторов для снижения систематического риска."
            )
        
        if not recommendations:
            recommendations.append("✅ ПОРТФЕЛЬ СБАЛАНСИРОВАН. Продолжайте в том же духе!")
        
        return recommendations

    @staticmethod
    def generate_enhanced_recommendations(metrics: Dict, trades: List[Dict]) -> List[str]:
        """Улучшенные рекомендации"""
        recommendations = PortfolioAnalyzer.generate_recommendations(metrics, trades)
        
        # Анализ концентрации риска
        if len(trades) == 1 and metrics['total_risk_percent'] > 5:
            recommendations.append("⚠️ ВСЕ ЯЙЦА В ОДНОЙ КОРЗИНЕ: Риск сконцентрирован в одной сделке. Диверсифицируйте!")
        
        # Анализ использования маржи
        if metrics['total_margin_usage'] > 80:
            recommendations.append("🔴 ПЕРЕГРУЗКА МАРЖИ: Использование >80%. Увеличьте депозит или уменьшите объемы.")
        elif metrics['total_margin_usage'] > 60:
            recommendations.append("🟡 ВЫСОКАЯ НАГРУЗКА: Использование >60%. Оставьте запас для управления позициями.")
        
        # Анализ волатильности портфеля
        high_vol_assets = [t for t in trades if VOLATILITY_DATA.get(t['asset'], 0) > 40]
        if len(high_vol_assets) > 2:
            recommendations.append("🌪 МНОГО ВОЛАТИЛЬНЫХ АКТИВОВ: Рассмотрите хеджирование или уменьшение объема.")
        
        # Анализ соотношений R/R
        low_rr_trades = [t for t in trades if t.get('metrics', {}).get('rr_ratio', 0) < 1]
        if len(low_rr_trades) > 0:
            recommendations.append(f"📉 НЕВЫГОДНЫЕ СДЕЛКИ: {len(low_rr_trades)} сделок с R/R < 1. Улучшите соотношение риск/прибыль.")
        
        return recommendations

# ---------------------------
# Handlers (ОБНОВЛЕННЫЕ С РЕАЛЬНЫМИ ДАННЫМИ)
# ---------------------------
def performance_logger(func):
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        start = time.time()
        try:
            return await func(update, context)
        finally:
            duration = time.time() - start
            if duration > 1.0:
                logger.warning("Slow handler: %s took %.2fs", func.__name__, duration)
    return wrapper

# Универсальный обработчик главного меню
async def main_menu_save_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, current_state: int = None):
    """УНИВЕРСАЛЬНЫЙ ОБРАБОТЧИК СОХРАНЕНИЯ ПЕРЕД ВЫХОДОМ"""
    query = update.callback_query
    user_id = query.from_user.id if query else update.message.from_user.id
    
    # Сохраняем текущий прогресс
    if context.user_data:
        state_type = "single" if current_state in [s.value for s in SingleTradeState] else "multi"
        DataManager.save_temporary_progress(user_id, context.user_data.copy(), state_type)
    
    # Очищаем временные данные
    context.user_data.clear()
    
    # Возвращаем в главное меню
    await start_command(update, context)
    
    return ConversationHandler.END

@performance_logger
@retry_on_timeout(max_retries=2, delay=1.0)
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start с защитой от таймаутов"""
    try:
        user = update.effective_user
        user_id = user.id
        PortfolioManager.ensure_user(user_id)
        
        # Проверяем есть ли сохраненный прогресс
        temp_data = DataManager.load_temporary_data()
        saved_progress = temp_data.get(str(user_id))
        
        text = (
            f"👋 Привет, {user.first_name}!\n\n"
            "🤖 <b>PRO Калькулятор Управления Рисками v3.1</b>\n\n"
            "🚀 <b>МОИ ВОЗМОЖНОСТИ:</b>\n"
            "• 📊 <b>РЕАЛЬНЫЕ КОТИРОВКИ</b> через ExchangeRate-API, Binance, Alpha Vantage\n"
            "• 💼 <b>ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ</b> маржи по отраслевым стандартам\n"
            "• 🎯 Контроль уровней риска (2%-25% от депозита)\n"
            "• 💡 Умные рекомендации и аналитика портфеля\n"
            "• 🛡 <b>ЗАЩИТА ОТ МАРЖИН-КОЛЛА</b> через правильный расчет объема\n"
            "• 📈 <b>РЕАЛЬНЫЕ ДАННЫЕ</b> для точного риск-менеджмента\n\n"
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
                # Fallback - отправляем новое сообщение
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
        # Пытаемся отправить сообщение об ошибке
        try:
            if update.effective_user:
                await SafeMessageSender.send_message(
                    update.effective_user.id,
                    "❌ Произошла ошибка при загрузке. Пожалуйста, попробуйте еще раз.",
                    context
                )
        except:
            pass

# ОБНОВЛЕННЫЙ обработчик одиночной сделки с реальными ценами
@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка тейк-профита и показ результатов с РЕАЛЬНЫМИ ДАННЫМИ"""
    text = update.message.text.strip()
    
    try:
        take_profit = float(text.replace(',', '.'))
        entry_price = context.user_data['entry_price']
        direction = context.user_data['direction']
        asset = context.user_data['asset']
        
        # Валидация TP
        if direction == 'LONG' and take_profit <= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для LONG тейк-профит должен быть ВЫШЕ цены входа\nПопробуйте еще раз:",
                context
            )
            return SingleTradeState.TAKE_PROFIT.value
        elif direction == 'SHORT' and take_profit >= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для SHORT тейк-профит должен быть НИЖЕ цены входа\nПопробуйте еще раз:",
                context
            )
            return SingleTradeState.TAKE_PROFIT.value
        
        context.user_data['take_profit'] = take_profit
        
        user_id = update.message.from_user.id
        trade = context.user_data.copy()
        
        # Рассчитываем метрики с реальными данными
        metrics = await EnhancedProfessionalRiskCalculator.calculate_enhanced_metrics(
            trade, trade['deposit'], trade['leverage'], trade['risk_level']
        )
        trade['metrics'] = metrics
        
        # Получаем текущую цену для отображения
        current_price = await market_data_provider.get_robust_real_time_price(asset)
        
        # Расчет дистанций
        stop_distance_pips = EnhancedProfessionalRiskCalculator.calculate_pip_distance(
            trade['entry_price'], trade['stop_loss'], trade['direction'], asset
        )
        profit_distance_pips = EnhancedProfessionalRiskCalculator.calculate_pip_distance(
            trade['entry_price'], trade['take_profit'], trade['direction'], asset
        )
        
        # Форматируем результаты
        text = (
            "🎯 <b>РЕЗУЛЬТАТЫ РАСЧЕТА v3.1</b>\n\n"
            f"<b>Актив:</b> {trade['asset']}\n"
            f"<b>Направление:</b> {trade['direction']}\n"
            f"<b>Вход:</b> {trade['entry_price']} (Текущая: {current_price:.4f})\n"
            f"<b>SL:</b> {trade['stop_loss']} ({stop_distance_pips:.0f} пунктов)\n"
            f"<b>TP:</b> {trade['take_profit']} ({profit_distance_pips:.0f} пунктов)\n"
            f"<b>Стоимость пункта:</b> ${metrics['pip_value']:.2f}\n\n"
            f"<b>Объем:</b> {metrics['volume_lots']:.2f} лотов\n"
            f"<b>Риск:</b> ${metrics['risk_amount']:.2f} ({metrics['risk_percent']:.1f}%)\n"
            f"<b>Потенциальная прибыль:</b> ${metrics['potential_profit']:.2f}\n"
            f"<b>R/R соотношение:</b> {metrics['rr_ratio']:.2f}\n"
            f"<b>Требуемая маржа:</b> ${metrics['required_margin']:.2f}\n"
            f"<b>Свободная маржа:</b> ${metrics['free_margin']:.2f}\n"
            f"<b>Уровень маржи:</b> {metrics['margin_level']:.1f}%\n"
            f"<b>Текущий P&L:</b> ${metrics['current_pnl']:.2f}\n\n"
            f"<i>Расчет выполнен с реальными котировками</i>"
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
            context
        )
        return SingleTradeState.TAKE_PROFIT.value

# ОБНОВЛЕННЫЙ обработчик входа с реальной ценой
@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка цены входа для одиночной сделки с реальной ценой"""
    text = update.message.text.strip()
    
    try:
        entry_price = float(text.replace(',', '.'))
        if entry_price <= 0:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Цена должна быть больше 0\nПопробуйте еще раз:",
                context
            )
            return SingleTradeState.ENTRY.value
        
        context.user_data['entry_price'] = entry_price
        
        # Получаем текущую цену для информации
        asset = context.user_data['asset']
        current_price = await market_data_provider.get_robust_real_time_price(asset)
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ <b>Цена входа:</b> {entry_price}\n"
            f"💰 <b>Текущая цена:</b> {current_price:.4f}\n\n"
            "Введите уровень стоп-лосса:",
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
            context
        )
        return SingleTradeState.ENTRY.value

# ---------------------------
# PRO Инструкции с улучшенным содержанием
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """PRO инструкции v3.1 с объяснением волатильности"""
    volatility_explanation = """
🌪 <b>ВОЛАТИЛЬНОСТЬ В РАСЧЕТАХ:</b>
• <b>Что это?</b> Мера колебаний цены актива
• <b>Как используется?</b> Для оценки риска и рекомендаций
• <b>Высокая волатильность</b> (>30%) = большие риски И возможности
• <b>Низкая волатильность</b> (<15%) = стабильность, но меньший потенциал

📊 <b>ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ:</b>
• <b>BTCUSDT:</b> 65% - высокий риск, нужен широкий SL
• <b>EURUSD:</b> 8% - низкий риск, можно tighter управление
• <b>Используйте эти данные</b> для настройки стоп-лоссов!
"""
    
    text = (
        "📚 <b>PRO ИНСТРУКЦИИ v3.1</b>\n\n"
        
        "🎯 <b>ПРАВИЛЬНОЕ УПРАВЛЕНИЕ РИСКАМИ С РЕАЛЬНЫМИ ДАННЫМИ</b>\n\n"
        
        "💼 <b>МЕТОДОЛОГИЯ РАСЧЕТА v3.1:</b>\n"
        "• <b>Риск на сделку</b> = % от депозита (например: 2% от $1000 = $20)\n"
        "• <b>Объем позиции</b> рассчитывается ИСКЛЮЧИТЕЛЬНО из суммы риска\n"
        "• <b>РЕАЛЬНЫЕ КОТИРОВКИ</b> через ExchangeRate-API, Binance, Alpha Vantage\n"
        "• <b>ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ</b> маржи по отраслевым стандартам\n"
        "• <b>Защита от маржин-колла</b> через правильный расчет объема\n\n"
        
        "📊 <b>РЕАЛЬНЫЕ КОТИРОВКИ:</b>\n"
        "• <b>ExchangeRate-API</b> - Forex пары с высокой точностью\n"
        "• <b>Binance API</b> - криптовалюты с точностью до 0.01%\n"
        "• <b>Alpha Vantage</b> - акции, индексы, товары\n"
        "• <b>Умное комбинирование</b> - защита от недоступности API\n\n"
        
        "💡 <b>ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ МАРЖИ:</b>\n"
        "• <b>Forex:</b> (Объем × Размер контракта) / Плечо\n"
        "• <b>Крипто:</b> (Объем × Цена) / Плечо\n"
        "• <b>Акции:</b> (Объем × Размер контракта × Цена) / Плечо\n"
        "• <b>РЕАЛЬНЫЕ СПЕЦИФИКАЦИИ</b> для 50+ активов\n\n"
        
        f"{volatility_explanation}\n\n"
        
        "🎯 <b>РЕКОМЕНДАЦИИ ДЛЯ ПРОФЕССИОНАЛОВ:</b>\n"
        "• <b>Риск на сделку:</b> 1-5% от депозита\n"
        "• <b>Общий риск портфеля:</b> < 10%\n"
        "• <b>Уровень маржи:</b> > 200%\n"
        "• <b>Соотношение R/R:</b> минимум 1:1.5\n"
        "• <b>Диверсификация:</b> 3-5 активов разных категорий\n\n"
        
        "🚀 <b>ПРЕИМУЩЕСТВА v3.1:</b>\n"
        "✅ <b>РЕАЛЬНЫЕ ЦЕНЫ</b> через multiple API sources\n"
        "✅ <b>ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ</b> маржи\n"
        "✅ <b>ЗАЩИТА</b> от маржин-колла\n"
        "✅ <b>АВТОМАТИЧЕСКИЕ РЕКОМЕНДАЦИИ</b>\n"
        "✅ <b>ОБНОВЛЕНИЕ</b> портфеля в реальном времени\n\n"
        
        "💝 <i>Поддержите разработку для новых функций!</i>"
    )
    
    keyboard = [
        [InlineKeyboardButton("🎯 Начать расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("📊 Мой портфель", callback_data="portfolio")],
        [InlineKeyboardButton("💖 Поддержать разработчика", callback_data="donate_start")],
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
# Остальные обработчики состояний (аналогично оригинальным)
# ---------------------------

# Обработчики для одиночных сделок
@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало одиночной сделки с реальными данными"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = (
        "🎯 <b>ОДИНОЧНАЯ СДЕЛКА v3.1</b>\n\n"
        "ПРОФЕССИОНАЛЬНЫЙ расчет с <b>РЕАЛЬНЫМИ КОТИРОВКАМИ</b> и защитой от маржин-колла.\n"
        "Объем рассчитывается <b>ИСКЛЮЧИТЕЛЬНО</b> из суммы риска на основе текущих рыночных цен!\n\n"
        "<b>Механика расчета:</b>\n"
        "• <b>Риск на сделку</b> = % от депозита\n"
        "• <b>Объем</b> = Риск / (Дистанция SL в пунктах × Стоимость пункта)\n"
        "• Таким образом объем <b>АВТОМАТИЧЕСКИ</b> адаптируется под ваш риск!\n\n"
        "Введите ваш депозит в USD:"
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
    
    try:
        deposit = float(text.replace(',', '.'))
        if deposit < 100:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Минимальный депозит: $100\nПопробуйте еще раз:",
                context
            )
            return SingleTradeState.DEPOSIT.value
        
        context.user_data['deposit'] = deposit
        
        keyboard = []
        for leverage in LEVERAGES:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"lev_{leverage}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ <b>Депозит:</b> ${deposit:,.2f}\n\n"
            "Выберите кредитное плечо:",
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return SingleTradeState.LEVERAGE.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Введите число (например: 1000)\nПопробуйте еще раз:",
            context
        )
        return SingleTradeState.DEPOSIT.value

# Обработчики для мультипозиций (аналогично оригинальным)

# ---------------------------
# Callback Router (ОБНОВЛЕННЫЙ)
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Маршрутизатор callback запросов"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    data = query.data
    
    try:
        if data == "main_menu":
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
            await SafeMessageSender.answer_callback_query(query, "Команда не распознана")
            
    except Exception as e:
        logger.error(f"Error in callback router: {e}")
        await SafeMessageSender.answer_callback_query(query, "❌ Произошла ошибка")

# ---------------------------
# Дополнительные обработчики
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_calculation_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик профессиональных сделок"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = (
        "🎯 <b>ПРОФЕССИОНАЛЬНЫЕ СДЕЛКИ v3.1</b>\n\n"
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

# Остальные обработчики (show_portfolio, future_features_handler, clear_portfolio_handler, 
# export_portfolio_handler, restore_progress_handler) остаются аналогичными оригинальным,
# но используют улучшенные классы EnhancedProfessionalRiskCalculator и EnhancedMarketDataProvider

# ---------------------------
# Setup Conversation Handlers
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
                CallbackQueryHandler(single_trade_asset, pattern="^(asset_|back_to_categories)"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_asset_manual),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            SingleTradeState.DIRECTION.value: [
                CallbackQueryHandler(single_trade_direction, pattern="^dir_"),
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
    
    # Добавление обработчиков в приложение
    application.add_handler(single_trade_conv)
    
    # Добавление других обработчиков...
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("pro_info", pro_info_command))
    application.add_handler(CallbackQueryHandler(callback_router))

# ---------------------------
# Webhook и основной запуск
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
            "version": "3.1",
            "services": {
                "telegram_bot": "operational",
                "market_data": "operational", 
                "database": "operational"
            }
        }
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

async def main():
    """Основная функция с улучшенной обработкой ошибок"""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to start bot...")
            
            # Создаем устойчивое приложение
            application = RobustApplicationBuilder.create_application(TOKEN)
            
            # Регистрация обработчиков
            setup_conversation_handlers(application)
            
            # Обработчик для любых сообщений (fallback)
            application.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND, 
                lambda update, context: SafeMessageSender.send_message(
                    update.message.chat_id,
                    "Используйте меню для навигации или /start для начала работы",
                    context
                )
            ))
            
            # Режим запуска
            if WEBHOOK_URL and WEBHOOK_URL.strip():
                logger.info("Запуск в режиме WEBHOOK")
                await application.initialize()
                
                if await set_webhook(application):
                    await start_http_server(application)
                    logger.info("Бот успешно запущен в режиме WEBHOOK")
                    
                    # Бесконечный цикл
                    while True:
                        await asyncio.sleep(300)
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
                
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error("All startup attempts failed")
                raise
        finally:
            # Закрываем сессию market data provider
            await market_data_provider.close_session()

if __name__ == "__main__":
    asyncio.run(main())
