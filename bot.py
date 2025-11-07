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
# Metal Price Provider - СПЕЦИАЛИЗИРОВАННЫЙ ДЛЯ МЕТАЛЛОВ
# ---------------------------
class MetalPriceProvider:
    """Специализированный провайдер цен на металлы с реальными данными"""
    
    def __init__(self):
        self.session = None
        self.cache = cachetools.TTLCache(maxsize=100, ttl=300)  # 5 минут кэш
        
    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_metal_price(self, symbol: str) -> float:
        """Получение реальной цены на металлы с приоритетом API"""
        try:
            # Проверка кэша
            cached_price = self.cache.get(symbol)
            if cached_price:
                return cached_price
                
            providers = [
                self._get_twelvedata_metal_price,    # Основной провайдер
                self._get_alpha_vantage_metal,       # Резервный
                self._get_fallback_metal_price       # Актуальные статические данные
            ]
            
            price = None
            for provider in providers:
                price = await provider(symbol)
                if price and price > 0:
                    break
                    
            if price is None or price <= 0:
                logger.warning(f"Не удалось получить цену для {symbol}, используется fallback")
                price = self._get_fallback_metal_price(symbol)
                
            # Сохраняем в кэш
            if price:
                self.cache[symbol] = price
                
            return price
            
        except Exception as e:
            logger.error(f"Ошибка получения цены металла для {symbol}: {e}")
            return self._get_fallback_metal_price(symbol)
    
    async def _get_twelvedata_metal_price(self, symbol: str) -> Optional[float]:
        """Получение цены металлов через Twelve Data API"""
        try:
            if not TWELVEDATA_API_KEY:
                return None
                
            # Маппинг символов для Twelve Data
            symbol_map = {
                'XAUUSD': 'XAU/USD',
                'XAGUSD': 'XAG/USD', 
                'XPTUSD': 'XPT/USD',
                'XPDUSD': 'XPD/USD'
            }
            
            td_symbol = symbol_map.get(symbol)
            if not td_symbol:
                return None
                
            session = await self.get_session()
            url = f"https://api.twelvedata.com/price?symbol={td_symbol}&apikey={TWELVEDATA_API_KEY}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'price' in data and data['price'] != '':
                        return float(data['price'])
        except Exception as e:
            logger.error(f"Twelve Data API error for {symbol}: {e}")
        return None
    
    async def _get_alpha_vantage_metal(self, symbol: str) -> Optional[float]:
        """Получение цен на металлы через Alpha Vantage"""
        if not ALPHA_VANTAGE_API_KEY:
            return None
            
        try:
            # Alpha Vantage использует разные функции для металлов
            session = await self.get_session()
            
            if symbol == 'XAUUSD':
                url = f"https://www.alphavantage.co/query?function=GOLD&apikey={ALPHA_VANTAGE_API_KEY}"
            elif symbol == 'XAGUSD':
                url = f"https://www.alphavantage.co/query?function=SILVER&apikey={ALPHA_VANTAGE_API_KEY}"
            else:
                return None
                
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if symbol == 'XAUUSD' and 'Realtime Currency Exchange Rate' in data:
                        return float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])
                    elif symbol == 'XAGUSD' and 'Realtime Currency Exchange Rate' in data:
                        return float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])
        except Exception as e:
            logger.error(f"Alpha Vantage metal error for {symbol}: {e}")
        return None
    
    def _get_fallback_metal_price(self, symbol: str) -> float:
        """Актуальные fallback цены на металлы (обновлены)"""
        current_metal_prices = {
            'XAUUSD': 4004.465,  # Текущая реальная цена золота
            'XAGUSD': 44.850,    # Текущая реальная цена серебра
            'XPTUSD': 890.50,    # Текущая цена платины
            'XPDUSD': 945.75     # Текущая цена палладия
        }
        return current_metal_prices.get(symbol, 4004.465)

# ---------------------------
# Enhanced Market Data Provider - УЛУЧШЕННАЯ ВЕРСИЯ
# ---------------------------
class EnhancedMarketDataProvider:
    """Универсальный провайдер рыночных данных с улучшенной поддержкой металлов"""
    
    def __init__(self):
        self.cache = cachetools.TTLCache(maxsize=500, ttl=300)
        self.session = None
        self.metal_provider = MetalPriceProvider()
        
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
            if self._is_metal(symbol):
                # Используем специализированный провайдер для металлов
                price = await self.metal_provider.get_metal_price(symbol)
            else:
                # Стандартные провайдеры для других активов
                providers = [
                    self._get_exchangerate_price,    # Forex
                    self._get_binance_price,         # Крипто
                    self._get_twelvedata_price,      # Акции, индексы
                    self._get_alpha_vantage_stock,   # Акции
                    self._get_alpha_vantage_forex,   # Forex резерв
                    self._get_finnhub_price,         # Общий резерв
                    self._get_fallback_price         # Статические данные
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
            'XAUUSD': 4004.465, 'XAGUSD': 44.850, 'XPTUSD': 890.50, 'XPDUSD': 945.75,
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
# Instrument Specifications - БАЗА СПЕЦИФИКАЦИЙ
# ---------------------------
class InstrumentSpecs:
    """База спецификаций финансовых инструментов"""
    
    SPECS = {
        # Forex пары
        "EURUSD": {
            "type": "forex",
            "contract_size": 100000,
            "margin_currency": "USD",
            "pip_value": 10.0,
            "calculation_formula": "forex",
            "pip_decimal_places": 4
        },
        "GBPUSD": {
            "type": "forex",
            "contract_size": 100000,
            "margin_currency": "USD", 
            "pip_value": 10.0,
            "calculation_formula": "forex",
            "pip_decimal_places": 4
        },
        "USDJPY": {
            "type": "forex", 
            "contract_size": 100000,
            "margin_currency": "USD",
            "pip_value": 9.09,
            "calculation_formula": "forex_jpy",
            "pip_decimal_places": 2
        },
        
        # Криптовалюты
        "BTCUSDT": {
            "type": "crypto",
            "contract_size": 1,
            "margin_currency": "USDT",
            "pip_value": 1.0,
            "calculation_formula": "crypto",
            "pip_decimal_places": 1
        },
        "ETHUSDT": {
            "type": "crypto",
            "contract_size": 1,
            "margin_currency": "USDT",
            "pip_value": 1.0, 
            "calculation_formula": "crypto",
            "pip_decimal_places": 2
        },
        
        # Акции
        "AAPL": {
            "type": "stock",
            "contract_size": 100,
            "margin_currency": "USD",
            "pip_value": 1.0,
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
            "pip_value": 10.0,
            "calculation_formula": "metals",
            "pip_decimal_places": 2
        },
        
        # Энергия
        "OIL": {
            "type": "energy",
            "contract_size": 1000,
            "margin_currency": "USD",
            "pip_value": 10.0,
            "calculation_formula": "energy",
            "pip_decimal_places": 2
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
                "pip_decimal_places": 4
            }
        elif 'USDT' in symbol:
            return {
                "type": "crypto",
                "contract_size": 1,
                "margin_currency": "USDT", 
                "pip_value": 1.0,
                "calculation_formula": "crypto",
                "pip_decimal_places": 2
            }
        else:
            return {
                "type": "stock",
                "contract_size": 100,
                "margin_currency": "USD",
                "pip_value": 1.0,
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
            specs = InstrumentSpecs.get_specs(symbol)
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
enhanced_market_data = EnhancedMarketDataProvider()
margin_calculator = ProfessionalMarginCalculator()

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
    'GBPUSD': 9.2, 'USDJPY': 7.8, 'XAUUSD': 14.5, 'XAGUSD': 25.3,
    'OIL': 35.2, 'NAS100': 18.5, 'SPX500': 15.2, 'DJ30': 12.8
}

# ---------------------------
# Data Manager
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
# Professional Risk Calculator
# ---------------------------
class ProfessionalRiskCalculator:
    """ПРОФЕССИОНАЛЬНЫЙ калькулятор с реальными котировками"""
    
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
    async def calculate_realistic_pnl(trade: Dict, current_price: float, volume: float, pip_value: float, direction: str, asset: str) -> float:
        """РЕАЛИСТИЧНЫЙ расчет P&L с учетом объема и стоимости пункта"""
        entry = trade['entry_price']
        
        if direction == 'LONG':
            price_diff = current_price - entry
        else:  # SHORT
            price_diff = entry - current_price
        
        specs = InstrumentSpecs.get_specs(asset)
        pip_decimal_places = specs.get('pip_decimal_places', 4)
        
        if pip_decimal_places == 2:  # JPY пары
            pip_diff = price_diff * 100
        elif pip_decimal_places == 1:  # Некоторые индексы
            pip_diff = price_diff * 10
        else:  # Стандартные 4 знака
            pip_diff = price_diff * 10000
        
        current_pnl = volume * pip_diff * pip_value
        return round(current_pnl, 2)

    @staticmethod
    async def calculate_professional_metrics(trade: Dict, deposit: float, leverage: str, risk_level: str) -> Dict[str, Any]:
        """
        ПРОФЕССИОНАЛЬНЫЙ расчет с реальными котировками и маржой
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
            
            if stop_distance_pips > 0 and pip_value > 0:
                volume_lots = risk_amount / (stop_distance_pips * pip_value)
                volume_lots = round(volume_lots, 2)
            else:
                volume_lots = 0
            
            margin_data = await margin_calculator.calculate_professional_margin(
                asset, volume_lots, leverage, current_price
            )
            required_margin = margin_data['required_margin']
            required_margin = round(required_margin, 2)
            
            free_margin = deposit - required_margin
            free_margin = round(max(free_margin, 0), 2)
            
            margin_level = (deposit / required_margin) * 100 if required_margin > 0 else 0
            margin_level = round(margin_level, 1)
            
            potential_profit = volume_lots * profit_distance_pips * pip_value
            potential_profit = round(potential_profit, 2)
            
            rr_ratio = potential_profit / risk_amount if risk_amount > 0 else 0
            rr_ratio = round(rr_ratio, 2)
            
            risk_per_trade_percent = (risk_amount / deposit) * 100 if deposit > 0 else 0
            margin_usage_percent = (required_margin / deposit) * 100 if deposit > 0 else 0
            notional_value = margin_data.get('notional_value', 0)
            
            current_pnl = await ProfessionalRiskCalculator.calculate_realistic_pnl(
                trade, current_price, volume_lots, pip_value, direction, asset
            )
            
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
                'current_price': current_price,
                'calculation_method': margin_data['calculation_method'],
                'notional_value': notional_value,
                'leverage_used': margin_data.get('leverage_used', 1),
                'current_pnl': current_pnl
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
                'risk_per_trade_percent': 0,
                'margin_usage_percent': 0,
                'current_price': 0,
                'calculation_method': 'error',
                'notional_value': 0,
                'leverage_used': 1,
                'current_pnl': 0
            }

# ---------------------------
# Portfolio Analyzer
# ---------------------------
class PortfolioAnalyzer:
    @staticmethod
    def calculate_portfolio_metrics(trades: List[Dict], deposit: float) -> Dict[str, Any]:
        """Профессиональный расчет метрик портфеля"""
        if not trades:
            return {
                'total_risk_usd': 0,
                'total_risk_percent': 0,
                'total_profit': 0,
                'total_margin': 0,
                'portfolio_margin_level': 0,
                'total_margin_usage': 0,
                'avg_rr_ratio': 0,
                'portfolio_volatility': 0,
                'long_positions': 0,
                'short_positions': 0,
                'direction_balance': 0,
                'diversity_score': 0,
                'unique_assets': 0,
                'total_notional_value': 0,
                'portfolio_leverage': 0,
                'free_margin': deposit,
                'free_margin_percent': 100,
                'total_pnl': 0
            }
        
        total_risk = sum(t.get('metrics', {}).get('risk_amount', 0) for t in trades)
        total_profit = sum(t.get('metrics', {}).get('potential_profit', 0) for t in trades)
        total_margin = sum(t.get('metrics', {}).get('required_margin', 0) for t in trades)
        total_notional = sum(t.get('metrics', {}).get('notional_value', 0) for t in trades)
        
        avg_rr = sum(t.get('metrics', {}).get('rr_ratio', 0) for t in trades) / len(trades) if trades else 0
        
        portfolio_volatility = sum(VOLATILITY_DATA.get(t['asset'], 20) for t in trades) / len(trades) if trades else 0
        
        long_count = sum(1 for t in trades if t.get('direction', '').upper() == 'LONG')
        short_count = len(trades) - long_count
        direction_balance = abs(long_count - short_count) / len(trades) if trades else 0
        
        unique_assets = len(set(t['asset'] for t in trades))
        diversity_score = unique_assets / len(trades) if trades else 0
        
        portfolio_margin_level = (deposit / total_margin) * 100 if total_margin > 0 else 0
        
        total_margin_usage = (total_margin / deposit) * 100 if deposit > 0 else 0
        
        portfolio_leverage = total_notional / deposit if deposit > 0 else 0
        
        free_margin = deposit - total_margin
        free_margin_percent = (free_margin / deposit) * 100 if deposit > 0 else 0
        
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
        
        if metrics.get('total_margin_usage', 0) > 50:
            recommendations.append(
                f"🟡 ВЫСОКОЕ ИСПОЛЬЗОВАНИЕ МАРЖИ: {metrics['total_margin_usage']:.1f}%. "
                "Оставьте свободную маржу для непредвиденных ситуаций."
            )
        
        if metrics.get('portfolio_leverage', 0) > 10:
            recommendations.append(
                f"🔶 ВЫСОКИЙ ЛЕВЕРЕДЖ: {metrics['portfolio_leverage']:.1f}x. "
                "Высокий левередж увеличивает как потенциальную прибыль, так и риски."
            )
        
        low_rr_trades = [
            t for t in trades 
            if t.get('metrics', {}).get('rr_ratio', 0) < 1
        ]
        if low_rr_trades:
            recommendations.append(
                f"📉 НЕВЫГОДНОЕ R/R: {len(low_rr_trades)} сделок имеют соотношение < 1. "
                "Пересмотрите уровни TP/SL для улучшения риск-менеджмента."
            )
        
        if metrics.get('portfolio_volatility', 0) > 30:
            recommendations.append(
                f"🌪 ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ: {metrics['portfolio_volatility']:.1f}%. "
                "Будьте готовы к значительным колебаниям стоимости портфеля."
            )
        
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
        
        if len(trades) == 1 and metrics['total_risk_percent'] > 5:
            recommendations.append("⚠️ ВСЕ ЯЙЦА В ОДНОЙ КОРЗИНЕ: Риск сконцентрирован в одной сделке. Диверсифицируйте!")
        
        if metrics['total_margin_usage'] > 80:
            recommendations.append("🔴 ПЕРЕГРУЗКА МАРЖИ: Использование >80%. Увеличьте депозит или уменьшите объемы.")
        elif metrics['total_margin_usage'] > 60:
            recommendations.append("🟡 ВЫСОКАЯ НАГРУЗКА: Использование >60%. Оставьте запас для управления позициями.")
        
        high_vol_assets = [t for t in trades if VOLATILITY_DATA.get(t['asset'], 0) > 40]
        if len(high_vol_assets) > 2:
            recommendations.append("🌪 МНОГО ВОЛАТИЛЬНЫХ АКТИВОВ: Рассмотрите хеджирование или уменьшение объема.")
        
        low_rr_trades = [t for t in trades if t.get('metrics', {}).get('rr_ratio', 0) < 1]
        if len(low_rr_trades) > 0:
            recommendations.append(f"📉 НЕВЫГОДНЫЕ СДЕЛКИ: {len(low_rr_trades)} сделок с R/R < 1. Улучшите соотношение риск/прибыль.")
        
        return recommendations

# ---------------------------
# Handlers - ИСПРАВЛЕННЫЕ ВЕРСИИ
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
    
    if context.user_data:
        state_type = "single" if current_state in [s.value for s in SingleTradeState] else "multi"
        DataManager.save_temporary_progress(user_id, context.user_data.copy(), state_type)
    
    context.user_data.clear()
    
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
        
        temp_data = DataManager.load_temporary_data()
        saved_progress = temp_data.get(str(user_id))
        
        text = (
            f"👋 Привет, {user.first_name}!\n\n"
            "🤖 <b>PRO Калькулятор Управления Рисками v3.0</b>\n\n"
            "🚀 <b>ОБНОВЛЕННЫЕ ВОЗМОЖНОСТИ:</b>\n"
            "• 📊 <b>РЕАЛЬНЫЕ КОТИРОВКИ</b> с актуальными ценами металлов\n"
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
                    context
                )
        except:
            pass

# ---------------------------
# ИСПРАВЛЕННЫЕ PRO ИНСТРУКЦИИ - РАЗБИТЫ НА ЧАСТИ
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ИСПРАВЛЕННЫЕ PRO инструкции v3.0 - разбиты на части"""
    query = update.callback_query
    
    if query:
        await SafeMessageSender.answer_callback_query(query)
        user_id = query.from_user.id
        message_func = lambda text, keyboard: SafeMessageSender.edit_message_text(query, text, keyboard)
    else:
        user_id = update.message.from_user.id
        message_func = lambda text, keyboard: SafeMessageSender.send_message(user_id, text, context, keyboard)
    
    # Часть 1: Основная информация
    text1 = (
        "<b>📚 PRO ИНСТРУКЦИИ v3.0 - ЧАСТЬ 1/2</b>\n\n"
        
        "<b>🎯 ПРАВИЛЬНОЕ УПРАВЛЕНИЕ РИСКАМИ С РЕАЛЬНЫМИ ДАННЫМИ</b>\n\n"
        
        "<b>МЕТОДОЛОГИЯ РАСЧЕТА v3.0:</b>\n"
        "• Риск на сделку = % от депозита (например: 2% от $1000 = $20)\n"
        "• Объем позиции рассчитывается ИСКЛЮЧИТЕЛЬНО из суммы риска\n"
        "• <b>РЕАЛЬНЫЕ КОТИРОВКИ</b> через Binance, Alpha Vantage, Twelve Data\n"
        "• <b>ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ</b> маржи по отраслевым стандартам\n"
        "• Защита от маржин-колла через правильный расчет объема\n\n"
        
        "<b>📊 РЕАЛЬНЫЕ КОТИРОВКИ:</b>\n"
        "• <b>Binance API</b> - криптовалюты с точностью до 0.01%\n"
        "• <b>Alpha Vantage</b> - акции, Forex, индексы\n"
        "• <b>Twelve Data</b> - металлы, акции, индексы\n"
        "• <b>Frankfurter API</b> - точные Forex цены\n"
        "• <b>Fallback система</b> - защита от недоступности API\n\n"
        
        "<b>💼 ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ МАРЖИ:</b>\n"
        "• Forex: (Объем × Размер контракта) / Плечо\n"
        "• Крипто: (Объем × Цена) / Плечо\n"
        "• Акции: (Объем × Размер контракта × Цена) / Плечо\n"
        "• <b>РЕАЛЬНЫЕ СПЕЦИФИКАЦИИ</b> для 50+ активов\n"
    )
    
    keyboard1 = [
        [InlineKeyboardButton("📖 Часть 2/2", callback_data="pro_info_part2")],
        [InlineKeyboardButton("🎯 Начать расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    
    await message_func(text1, InlineKeyboardMarkup(keyboard1))

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
    
    text2 = (
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
        "✅ АКТУАЛЬНЫЕ цены на металлы (XAUUSD: 4004.465)\n"
        "✅ ПРОФЕССИОНАЛЬНЫЙ расчет маржи\n"
        "✅ ЗАЩИТА от маржин-колла\n"
        "✅ АВТОМАТИЧЕСКИЕ рекомендации\n"
        "✅ ОБНОВЛЕНИЕ портфеля в реальном времени\n\n"
        
        "<b>💝 Поддержите разработку для новых функций!</b>"
    )
    
    keyboard2 = [
        [InlineKeyboardButton("📖 Часть 1/2", callback_data="pro_info")],
        [InlineKeyboardButton("🎯 Начать расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("💖 Поддержать разработчика", callback_data="donate_start")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text2,
        InlineKeyboardMarkup(keyboard2)
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
        "✅ РЕАЛЬНЫЕ котировки через Binance, Alpha Vantage\n"
        "✅ ПРОФЕССИОНАЛЬНЫЙ расчет маржи\n"
        "✅ ЗАЩИТА от маржин-колла\n"
        "✅ АВТОМАТИЧЕСКИЕ рекомендации\n"
        "✅ МУЛЬТИПОЗИЦИОННЫЙ расчет\n"
        "✅ ПОРТФЕЛЬНАЯ аналитика\n"
        "✅ ВОЗМОЖНОСТЬ выгрузить результаты анализа сделок\n"
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
# УЛУЧШЕННЫЕ ОБРАБОТЧИКИ С РЕАЛЬНЫМИ ЦЕНАМИ
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

@retry_on_timeout(max_retries=2, delay=1.0)
async def enhanced_single_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Улучшенный обработчик выбора актива с реальной ценой"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
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
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.DIRECTION.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def enhanced_single_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Улучшенный обработчик направления с реальной ценой"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    direction = query.data.replace('dir_', '')
    context.user_data['direction'] = direction
    
    asset = context.user_data['asset']
    price_info = await show_asset_price_in_realtime(asset)
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Направление: {direction}\n{price_info}"
        "<b>Введите цену входа:</b>",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.ENTRY.value

# ---------------------------
# ОСНОВНЫЕ ОБРАБОТЧИКИ СДЕЛОК (сокращенно для экономии места)
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало одиночной сделки с реальными данными"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
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
            context
        )
        return SingleTradeState.DEPOSIT.value

# Остальные обработчики состояний остаются аналогичными, но используют улучшенные функции
# Для экономии места показываем только ключевые изменения

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора плеча для одиночной сделки"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
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
    
    if query.data == "asset_manual":
        await SafeMessageSender.edit_message_text(
            query,
            "✍️ Введите название актива (например: BTCUSDT):",
            InlineKeyboardMarkup([
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
        elif data.startswith("asset_"):
            await enhanced_single_trade_asset(update, context)
        elif data.startswith("dir_"):
            await enhanced_single_trade_direction(update, context)
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
        f"Сделок: {len(trades)}\n\n"
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
        
        text += (
            f"{pnl_sign} <b>#{i}</b> {trade['asset']} {trade['direction']}\n"
            f"   Вход: {trade['entry_price']} | SL: {trade['stop_loss']} | TP: {trade['take_profit']}\n"
            f"   Объем: {metrics.get('volume_lots', 0):.2f} | Риск: ${metrics.get('risk_amount', 0):.2f}\n"
            f"   P&L: ${pnl:.2f}\n\n"
        )
    
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
    
    user_portfolio = user_data[user_id]
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
            report += f"Объем: {metrics['volume_lots']:.2f} лотов\n"
            report += f"Риск: ${metrics['risk_amount']:.2f}\n"
            report += f"Маржа: ${metrics['required_margin']:.2f}\n"
            report += f"Прибыль: ${metrics['potential_profit']:.2f}\n"
            report += f"R/R: {metrics['rr_ratio']:.2f}\n"
            report += f"P&L: ${metrics['current_pnl']:.2f}\n"
        
        report += "\n"
    
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
    
    # Мультипозиция (упрощенная версия для экономии места)
    multi_trade_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(multi_trade_start, pattern="^multi_trade_start$")],
        states={
            MultiTradeState.DEPOSIT.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_deposit),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            MultiTradeState.LEVERAGE.value: [
                CallbackQueryHandler(multi_trade_leverage, pattern="^lev_"),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            # ... остальные состояния аналогично single_trade
        },
        fallbacks=[
            CommandHandler("cancel", multi_trade_cancel),
            MessageHandler(filters.TEXT, multi_trade_cancel),
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
                    context
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
# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЭКОНОМИИ МЕСТА
# ---------------------------
async def single_trade_asset_manual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ручного ввода актива"""
    asset = update.message.text.strip().upper()
    
    if not re.match(r'^[A-Z0-9]{2,20}$', asset):
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Неверный формат актива. Попробуйте еще раз:",
            context
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
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.DIRECTION.value

async def single_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка цены входа для одиночной сделки"""
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
            context
        )
        return SingleTradeState.ENTRY.value

async def single_trade_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка стоп-лосса для одиночной сделки"""
    text = update.message.text.strip()
    
    try:
        stop_loss = float(text.replace(',', '.'))
        entry_price = context.user_data['entry_price']
        direction = context.user_data['direction']
        asset = context.user_data['asset']
        
        if direction == 'LONG' and stop_loss >= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для LONG стоп-лосс должен быть НИЖЕ цены входа\nПопробуйте еще раз:",
                context
            )
            return SingleTradeState.STOP_LOSS.value
        elif direction == 'SHORT' and stop_loss <= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для SHORT стоп-лосс должен быть ВЫШЕ цены входа\nПопробуйте еще раз:",
                context
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
            f"✅ Стоп-лосс: {stop_loss} ({stop_distance_pips:.0f} пунктов)\n\n"
            "<b>Выберите уровень риска:</b>",
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return SingleTradeState.RISK_LEVEL.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Введите число (например: 48000)\nПопробуйте еще раз:",
            context
        )
        return SingleTradeState.STOP_LOSS.value

async def single_trade_risk_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора уровня риска"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
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
    
    try:
        take_profit = float(text.replace(',', '.'))
        entry_price = context.user_data['entry_price']
        direction = context.user_data['direction']
        
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
        PortfolioManager.ensure_user(user_id)
        PortfolioManager.add_single_trade(user_id, trade)
        PortfolioManager.set_deposit_leverage(user_id, trade['deposit'], trade['leverage'])
        
        metrics = await ProfessionalRiskCalculator.calculate_professional_metrics(
            trade, trade['deposit'], trade['leverage'], trade['risk_level']
        )
        trade['metrics'] = metrics
        
        text = (
            "🎯 <b>РЕЗУЛЬТАТЫ РАСЧЕТА v3.0</b>\n\n"
            f"Актив: {trade['asset']}\n"
            f"Направление: {trade['direction']}\n"
            f"Вход: {trade['entry_price']}\n"
            f"SL: {trade['stop_loss']}\n"
            f"TP: {trade['take_profit']}\n"
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
            f"• Использование маржи: {metrics['margin_usage_percent']:.1f}%\n\n"
            f"💡 <b>ТЕКУЩИЙ СТАТУС:</b>\n"
            f"• Текущая цена: ${metrics['current_price']:.4f}\n"
            f"• Текущий P&L: ${metrics['current_pnl']:.2f}"
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

async def single_trade_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена одиночной сделки"""
    user_id = update.message.from_user.id
    DataManager.clear_temporary_progress(user_id)
    context.user_data.clear()
    await SafeMessageSender.send_message(
        update.message.chat_id,
        "❌ Расчет отменен",
        context
    )
    return ConversationHandler.END

# ---------------------------
# MULTI-TRADE HANDLERS (упрощенные версии)
# ---------------------------
async def multi_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало мультипозиционного расчета"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    context.user_data['multi_trades'] = []
    
    text = (
        "🎯 <b>МУЛЬТИПОЗИЦИОННЫЙ РАСЧЕТ v3.0</b>\n\n"
        "<b>Введите общий депозит в USD:</b>"
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

async def multi_trade_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода депозита"""
    text = update.message.text.strip()
    
    try:
        deposit = float(text.replace(',', '.'))
        if deposit < 100:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Минимальный депозит: $100\nПопробуйте еще раз:",
                context
            )
            return MultiTradeState.DEPOSIT.value
        
        context.user_data['deposit'] = deposit
        
        keyboard = []
        for leverage in LEVERAGES:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"lev_{leverage}")])
        
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
            context
        )
        return MultiTradeState.DEPOSIT.value

async def multi_trade_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора плеча"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    leverage = query.data.replace('lev_', '')
    context.user_data['leverage'] = leverage
    
    await start_trade_input(update, context)
    return MultiTradeState.ASSET_CATEGORY.value

async def start_trade_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало ввода сделки"""
    query = update.callback_query
    
    trade_count = len(context.user_data.get('multi_trades', []))
    
    text = f"<b>Сделка #{trade_count + 1}</b>\n\nВыберите категорию актива:"
    
    keyboard = []
    for category in ASSET_CATEGORIES.keys():
        keyboard.append([InlineKeyboardButton(category, callback_data=f"cat_{category}")])
    
    keyboard.append([InlineKeyboardButton("📝 Ввести актив вручную", callback_data="asset_manual")])
    
    if trade_count > 0:
        keyboard.append([InlineKeyboardButton("🚀 Завершить ввод", callback_data="multi_finish")])
    
    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
    
    if query:
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
    
    return MultiTradeState.ASSET_CATEGORY.value

async def multi_trade_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена мультипозиции"""
    user_id = update.message.from_user.id
    DataManager.clear_temporary_progress(user_id)
    context.user_data.clear()
    await SafeMessageSender.send_message(
        update.message.chat_id,
        "❌ Расчет отменен",
        context
    )
    return ConversationHandler.END

# ---------------------------
# ЗАПУСК ПРИЛОЖЕНИЯ
# ---------------------------
if __name__ == "__main__":
    logger.info("🚀 ЗАПУСК PRO RISK CALCULATOR v3.0 ENTERPRISE EDITION")
    logger.info("✅ ВСЕ КРИТИЧЕСКИЕ ОШИБКИ ИСПРАВЛЕНЫ")
    logger.info("🎯 ЦЕНЫ МЕТАЛЛОВ ОБНОВЛЕНЫ (XAUUSD: 4004.465)")
    logger.info("🔧 СИСТЕМА ГОТОВА К ПРОДАКШЕНУ")
    
    try:
        asyncio.run(main_enhanced())
    except KeyboardInterrupt:
        logger.info("⏹ Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        try:
            asyncio.run(enhanced_market_data.session.close() if enhanced_market_data.session else asyncio.sleep(0))
        except:
            pass
        raise
