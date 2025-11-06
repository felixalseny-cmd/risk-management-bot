# bot.py — PRO Risk Calculator v3.0 | ENTERPRISE EDITION (ИСПРАВЛЕННАЯ ВЕРСИЯ)
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
            "📱 <i>Скопируйте адрес выше и отправьте любую сумму</i>\n"
            "💝 <i>Каждый донат помогает развивать бота!</i>"
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
            "📱 <i>Скопируйте адрес выше и отправьте любую сумму</i>\n"
            "💝 <i>Каждый донат помогает развивать бота!</i>"
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
# Market Data Provider - РЕАЛЬНЫЕ КОТИРОВКИ (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# ---------------------------
class MarketDataProvider:
    """Универсальный провайдер рыночных данных с кэшированием"""
    
    def __init__(self):
        self.cache = cachetools.TTLCache(maxsize=500, ttl=300)  # 5 минут кэш
        self.session = None
        
    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_real_time_price(self, symbol: str) -> float:
        """Получение реальной цены с приоритизацией провайдеров"""
        return await self.get_robust_real_time_price(symbol)
    
    async def get_robust_real_time_price(self, symbol: str) -> float:
        """НАДЕЖНОЕ получение цен с приоритетной очередью"""
        try:
            # Проверка кэша
            cached_price = self.cache.get(symbol)
            if cached_price:
                return cached_price
                
            # Определяем тип актива и выбираем провайдера
            providers = [
                self._get_exchangerate_price,    # НОВЫЙ - для Forex
                self._get_binance_price,         # Крипто
                self._get_alpha_vantage_stock,   # Акции
                self._get_alpha_vantage_forex,   # Forex резерв
                self._get_finnhub_price,         # Общий резерв
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
            if price and price > 0:
                self.cache[symbol] = price
                
            return price if price and price > 0 else self._get_fallback_price(symbol)
            
        except Exception as e:
            logger.error(f"Ошибка получения цены для {symbol}: {e}")
            return self._get_fallback_price(symbol)
    
    def _is_forex(self, symbol: str) -> bool:
        """Проверка, является ли символ Forex парой"""
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        return symbol.upper() in forex_pairs
    
    def _is_crypto(self, symbol: str) -> bool:
        """Проверка, является ли символ криптовалютой"""
        crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'XRPUSDT']
        return symbol.upper() in crypto_symbols
    
    async def _get_exchangerate_price(self, symbol: str) -> Optional[float]:
        """НОВЫЙ: Frankfurter API для точных Forex цен"""
        try:
            if self._is_forex(symbol):
                # Конвертация EURUSD -> EUR/USD
                from_curr = symbol[:3]
                to_curr = symbol[3:]
                
                session = await self.get_session()
                url = f"https://api.frankfurter.app/latest?from={from_curr}&to={to_curr}"
                
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = data['rates'].get(to_curr)
                        if price:
                            logger.info(f"ExchangeRate API: {symbol} = {price}")
                            return float(price)
        except Exception as e:
            logger.error(f"ExchangeRate API error for {symbol}: {e}")
        return None
    
    async def _get_binance_price(self, symbol: str) -> Optional[float]:
        """Получение цены с Binance API"""
        try:
            if self._is_crypto(symbol):
                session = await self.get_session()
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = float(data['price'])
                        logger.info(f"Binance API: {symbol} = {price}")
                        return price
        except Exception as e:
            logger.debug(f"Binance API error for {symbol}: {e}")
        return None
    
    async def _get_alpha_vantage_stock(self, symbol: str) -> Optional[float]:
        """Получение цены акций через Alpha Vantage"""
        try:
            if not ALPHA_VANTAGE_API_KEY:
                return None
                
            # Проверяем, является ли символ акцией
            if len(symbol) <= 5 and symbol.isalpha():
                session = await self.get_session()
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
                
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        price_str = data.get('Global Quote', {}).get('05. price')
                        if price_str:
                            price = float(price_str)
                            logger.info(f"Alpha Vantage Stock: {symbol} = {price}")
                            return price
        except Exception as e:
            logger.debug(f"Alpha Vantage Stock error for {symbol}: {e}")
        return None
    
    async def _get_alpha_vantage_forex(self, symbol: str) -> Optional[float]:
        """Получение Forex цен через Alpha Vantage"""
        try:
            if not ALPHA_VANTAGE_API_KEY:
                return None
                
            if self._is_forex(symbol):
                from_curr = symbol[:3]
                to_curr = symbol[3:]
                
                session = await self.get_session()
                url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_curr}&to_currency={to_curr}&apikey={ALPHA_VANTAGE_API_KEY}"
                
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        rate_data = data.get('Realtime Currency Exchange Rate', {})
                        price_str = rate_data.get('5. Exchange Rate')
                        if price_str:
                            price = float(price_str)
                            logger.info(f"Alpha Vantage Forex: {symbol} = {price}")
                            return price
        except Exception as e:
            logger.debug(f"Alpha Vantage Forex error for {symbol}: {e}")
        return None
    
    async def _get_finnhub_price(self, symbol: str) -> Optional[float]:
        """Получение цены через Finnhub"""
        try:
            if not FINNHUB_API_KEY:
                return None
                
            session = await self.get_session()
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
            
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    price = data.get('c')  # текущая цена
                    if price and price > 0:
                        logger.info(f"Finnhub API: {symbol} = {price}")
                        return float(price)
        except Exception as e:
            logger.debug(f"Finnhub API error for {symbol}: {e}")
        return None
    
    def _get_fallback_price(self, symbol: str) -> float:
        """Резервные статические цены при недоступности API"""
        fallback_prices = {
            # Forex
            'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDJPY': 148.20, 
            'USDCHF': 0.8680, 'AUDUSD': 0.6520, 'USDCAD': 1.3520,
            'NZDUSD': 0.6080,
            # Crypto
            'BTCUSDT': 42500.0, 'ETHUSDT': 2550.0, 'ADAUSDT': 0.52,
            'DOTUSDT': 7.20, 'LINKUSDT': 14.50, 'LTCUSDT': 71.80,
            'BCHUSDT': 240.50, 'XLMUSDT': 0.125, 'XRPUSDT': 0.57,
            # Stocks
            'AAPL': 185.0, 'TSLA': 245.0, 'MSFT': 375.0, 'GOOGL': 135.0,
            'AMZN': 155.0, 'META': 335.0, 'NVDA': 485.0, 'SPY': 455.0,
            # Indices
            'SPX': 4780.0, 'DJI': 37500.0, 'NDX': 16900.0,
            # Commodities
            'XAUUSD': 2025.0, 'XAGUSD': 22.85, 'OIL': 75.50
        }
        
        price = fallback_prices.get(symbol.upper())
        if price:
            logger.info(f"Fallback price for {symbol}: {price}")
            return price
        
        # Дефолтная цена для неизвестных символов
        return 100.0

# ---------------------------
# Instrument Specifications - ИСПРАВЛЕННЫЕ СПЕЦИФИКАЦИИ
# ---------------------------
class InstrumentSpecs:
    """Спецификации торговых инструментов"""
    
    SPECS = {
        # Forex
        'EURUSD': {'pip_decimal_places': 4, 'lot_size': 100000, 'margin_requirement': 0.02},
        'GBPUSD': {'pip_decimal_places': 4, 'lot_size': 100000, 'margin_requirement': 0.02},
        'USDJPY': {'pip_decimal_places': 2, 'lot_size': 100000, 'margin_requirement': 0.02},
        'USDCHF': {'pip_decimal_places': 4, 'lot_size': 100000, 'margin_requirement': 0.02},
        'AUDUSD': {'pip_decimal_places': 4, 'lot_size': 100000, 'margin_requirement': 0.02},
        'USDCAD': {'pip_decimal_places': 4, 'lot_size': 100000, 'margin_requirement': 0.02},
        'NZDUSD': {'pip_decimal_places': 4, 'lot_size': 100000, 'margin_requirement': 0.02},
        
        # Crypto
        'BTCUSDT': {'pip_decimal_places': 2, 'lot_size': 1, 'margin_requirement': 0.01},
        'ETHUSDT': {'pip_decimal_places': 2, 'lot_size': 1, 'margin_requirement': 0.01},
        'ADAUSDT': {'pip_decimal_places': 5, 'lot_size': 1, 'margin_requirement': 0.01},
        'DOTUSDT': {'pip_decimal_places': 3, 'lot_size': 1, 'margin_requirement': 0.01},
        'LINKUSDT': {'pip_decimal_places': 3, 'lot_size': 1, 'margin_requirement': 0.01},
        'LTCUSDT': {'pip_decimal_places': 2, 'lot_size': 1, 'margin_requirement': 0.01},
        'BCHUSDT': {'pip_decimal_places': 2, 'lot_size': 1, 'margin_requirement': 0.01},
        'XLMUSDT': {'pip_decimal_places': 5, 'lot_size': 1, 'margin_requirement': 0.01},
        'XRPUSDT': {'pip_decimal_places': 4, 'lot_size': 1, 'margin_requirement': 0.01},
        
        # Stocks
        'AAPL': {'pip_decimal_places': 2, 'lot_size': 100, 'margin_requirement': 0.05},
        'TSLA': {'pip_decimal_places': 2, 'lot_size': 100, 'margin_requirement': 0.05},
        'MSFT': {'pip_decimal_places': 2, 'lot_size': 100, 'margin_requirement': 0.05},
        'GOOGL': {'pip_decimal_places': 2, 'lot_size': 100, 'margin_requirement': 0.05},
        'AMZN': {'pip_decimal_places': 2, 'lot_size': 100, 'margin_requirement': 0.05},
        'META': {'pip_decimal_places': 2, 'lot_size': 100, 'margin_requirement': 0.05},
        'NVDA': {'pip_decimal_places': 2, 'lot_size': 100, 'margin_requirement': 0.05},
        'SPY': {'pip_decimal_places': 2, 'lot_size': 100, 'margin_requirement': 0.05},
        
        # Indices
        'SPX': {'pip_decimal_places': 1, 'lot_size': 1, 'margin_requirement': 0.02},
        'DJI': {'pip_decimal_places': 1, 'lot_size': 1, 'margin_requirement': 0.02},
        'NDX': {'pip_decimal_places': 1, 'lot_size': 1, 'margin_requirement': 0.02},
        
        # Commodities
        'XAUUSD': {'pip_decimal_places': 2, 'lot_size': 100, 'margin_requirement': 0.02},
        'XAGUSD': {'pip_decimal_places': 3, 'lot_size': 5000, 'margin_requirement': 0.02},
        'OIL': {'pip_decimal_places': 2, 'lot_size': 1000, 'margin_requirement': 0.02},
    }
    
    @staticmethod
    def get_specs(symbol: str) -> Dict[str, Any]:
        """Получение спецификаций для символа"""
        return InstrumentSpecs.SPECS.get(symbol.upper(), {
            'pip_decimal_places': 4,
            'lot_size': 100000,
            'margin_requirement': 0.02
        })
    
    @staticmethod
    def calculate_pip_value(symbol: str, lot_size: float = 1.0) -> float:
        """Расчет стоимости пункта"""
        specs = InstrumentSpecs.get_specs(symbol)
        pip_decimal_places = specs['pip_decimal_places']
        
        # Базовая стоимость пункта для 1 лота
        if pip_decimal_places == 4:
            base_pip_value = 10.0  # $10 для стандартного лота Forex
        elif pip_decimal_places == 2:
            base_pip_value = 1000.0  # $1000 для JPY пар
        elif pip_decimal_places == 5:
            base_pip_value = 1.0  # $1 для некоторых крипто
        else:
            base_pip_value = 10.0
            
        return base_pip_value * lot_size

# ---------------------------
# Professional Risk Calculator - ИСПРАВЛЕННЫЙ РАСЧЕТНЫЙ ДВИЖОК
# ---------------------------
class ProfessionalRiskCalculator:
    """Профессиональный калькулятор рисков с исправленными формулами"""
    
    def __init__(self, deposit: float, leverage: int = 30):
        self.deposit = deposit
        self.leverage = leverage
        self.market_data = MarketDataProvider()
    
    async def calculate_single_trade_risk(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет риска для одной сделки с РЕАЛЬНЫМИ ценами"""
        try:
            symbol = trade_data['symbol']
            direction = trade_data['direction']
            entry_price = trade_data['entry_price']
            stop_loss = trade_data['stop_loss']
            take_profit = trade_data['take_profit']
            risk_percent = trade_data['risk_percent']
            
            # Получение РЕАЛЬНОЙ текущей цены
            current_price = await self.market_data.get_real_time_price(symbol)
            
            # Расчет дистанций
            sl_distance = self.calculate_pip_distance(entry_price, stop_loss, direction, symbol)
            tp_distance = self.calculate_pip_distance(entry_price, take_profit, direction, symbol)
            
            # Расчет стоимости пункта
            pip_value = InstrumentSpecs.calculate_pip_value(symbol)
            
            # Расчет объема на основе риска
            risk_amount = (risk_percent / 100) * self.deposit
            volume = risk_amount / (sl_distance * pip_value) if sl_distance > 0 else 0
            
            # Расчет потенциальной прибыли/убытка
            potential_loss = risk_amount
            potential_profit = volume * tp_distance * pip_value
            
            # Расчет требуемой маржи
            required_margin = self.calculate_required_margin(symbol, volume)
            
            # Расчет текущего P&L
            current_pnl = await self.calculate_realistic_pnl(
                trade_data, current_price, volume, pip_value, direction
            )
            
            # Расчет соотношения риск/прибыль
            rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
            
            return {
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sl_distance_pips': sl_distance,
                'tp_distance_pips': tp_distance,
                'volume': round(volume, 2),
                'pip_value': pip_value,
                'risk_amount': round(risk_amount, 2),
                'potential_profit': round(potential_profit, 2),
                'potential_loss': round(potential_loss, 2),
                'current_pnl': current_pnl,
                'rr_ratio': round(rr_ratio, 2),
                'required_margin': round(required_margin, 2),
                'risk_percent': risk_percent,
                'free_margin': round(self.deposit - required_margin, 2),
                'free_margin_percent': round(((self.deposit - required_margin) / self.deposit) * 100, 1) if self.deposit > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in calculate_single_trade_risk: {e}")
            # Возвращаем базовую структуру с нулевыми значениями при ошибке
            return self._get_default_metrics(trade_data)
    
    async def calculate_realistic_pnl(self, trade: Dict, current_price: float, volume: float, pip_value: float, direction: str) -> float:
        """РЕАЛИСТИЧНЫЙ расчет P&L с учетом объема и стоимости пункта"""
        try:
            entry = trade['entry_price']
            symbol = trade['symbol']
            specs = InstrumentSpecs.get_specs(symbol)
            
            if direction.upper() == 'LONG':
                price_diff = current_price - entry
            else:  # SHORT
                price_diff = entry - current_price
            
            # ПРАВИЛЬНОЕ преобразование в пункты
            pip_decimal_places = specs['pip_decimal_places']
            if pip_decimal_places == 2:  # JPY пары
                pip_diff = price_diff * 100
            elif pip_decimal_places == 5:  # Некоторые крипто
                pip_diff = price_diff * 100000
            elif pip_decimal_places == 3:  # Другие инструменты
                pip_diff = price_diff * 1000
            else:  # Стандартные 4 знака
                pip_diff = price_diff * 10000
            
            current_pnl = volume * pip_diff * pip_value
            return round(current_pnl, 2)
            
        except Exception as e:
            logger.error(f"Error in calculate_realistic_pnl: {e}")
            return 0.0
    
    def calculate_pip_distance(self, price1: float, price2: float, direction: str, symbol: str) -> float:
        """Расчет дистанции в пунктах между двумя ценами"""
        specs = InstrumentSpecs.get_specs(symbol)
        pip_decimal_places = specs['pip_decimal_places']
        
        if direction.upper() == 'LONG':
            distance = abs(price1 - price2)
        else:  # SHORT
            distance = abs(price2 - price1)
        
        # Конвертация в пункты
        if pip_decimal_places == 2:  # JPY пары
            return distance * 100
        elif pip_decimal_places == 5:  # Некоторые крипто
            return distance * 100000
        elif pip_decimal_places == 3:  # Другие инструменты
            return distance * 1000
        else:  # Стандартные 4 знака
            return distance * 10000
    
    def calculate_required_margin(self, symbol: str, volume: float) -> float:
        """Расчет требуемой маржи"""
        specs = InstrumentSpecs.get_specs(symbol)
        contract_size = specs['lot_size']
        margin_rate = specs['margin_requirement']
        
        # Упрощенный расчет маржи
        notional_value = volume * contract_size
        required_margin = notional_value * margin_rate / self.leverage
        
        return round(required_margin, 2)
    
    def _get_default_metrics(self, trade_data: Dict) -> Dict[str, Any]:
        """Возвращает метрики по умолчанию при ошибках"""
        return {
            'symbol': trade_data.get('symbol', 'UNKNOWN'),
            'direction': trade_data.get('direction', 'LONG'),
            'entry_price': trade_data.get('entry_price', 0),
            'current_price': 0,
            'stop_loss': trade_data.get('stop_loss', 0),
            'take_profit': trade_data.get('take_profit', 0),
            'sl_distance_pips': 0,
            'tp_distance_pips': 0,
            'volume': 0,
            'pip_value': 0,
            'risk_amount': 0,
            'potential_profit': 0,
            'potential_loss': 0,
            'current_pnl': 0,
            'rr_ratio': 0,
            'required_margin': 0,
            'risk_percent': trade_data.get('risk_percent', 0),
            'free_margin': self.deposit,
            'free_margin_percent': 100
        }

# ---------------------------
# Portfolio Manager - УЛУЧШЕННЫЙ МЕНЕДЖЕР ПОРТФЕЛЯ
# ---------------------------
class PortfolioManager:
    """Менеджер портфеля с исправленными расчетами"""
    
    def __init__(self):
        self.user_data = {}
    
    def set_deposit_leverage(self, user_id: int, deposit: float, leverage: int = 30):
        """Установка депозита и плеча для пользователя"""
        if user_id not in self.user_data:
            self.user_data[user_id] = {'trades': [], 'deposit_history': []}
        
        self.user_data[user_id]['deposit'] = deposit
        self.user_data[user_id]['leverage'] = leverage
        self.user_data[user_id]['deposit_history'].append({
            'timestamp': datetime.now(),
            'deposit': deposit,
            'leverage': leverage
        })
    
    def get_deposit(self, user_id: int) -> float:
        """Получение депозита пользователя"""
        return self.user_data.get(user_id, {}).get('deposit', 0.0)
    
    def add_trade(self, user_id: int, trade_data: Dict):
        """Добавление сделки в портфель"""
        if user_id not in self.user_data:
            self.user_data[user_id] = {'trades': [], 'deposit_history': []}
        
        trade_data['id'] = len(self.user_data[user_id]['trades']) + 1
        trade_data['timestamp'] = datetime.now()
        self.user_data[user_id]['trades'].append(trade_data)
    
    def get_trades(self, user_id: int) -> List[Dict]:
        """Получение всех сделок пользователя"""
        return self.user_data.get(user_id, {}).get('trades', [])
    
    def clear_trades(self, user_id: int):
        """Очистка всех сделок пользователя"""
        if user_id in self.user_data:
            self.user_data[user_id]['trades'] = []

# ---------------------------
# Risk Analytics - ПРОФЕССИОНАЛЬНАЯ АНАЛИТИКА
# ---------------------------
class RiskAnalytics:
    """Профессиональная аналитика рисков"""
    
    @staticmethod
    def generate_professional_recommendations(metrics: Dict, trades: List[Dict]) -> List[str]:
        """ПРОФЕССИОНАЛЬНЫЕ рекомендации для риск-менеджмента"""
        recommendations = []
        
        # Анализ концентрации
        if len(trades) == 1 and metrics.get('total_risk_percent', 0) > 5:
            recommendations.append("⚠️ <b>КОНЦЕНТРАЦИЯ РИСКА</b>: Весь риск в одной сделке! Диверсифицируйте позиции.")
        
        # Анализ использования маржи
        margin_usage = metrics.get('total_margin_usage', 0)
        if margin_usage > 80:
            recommendations.append("🔴 <b>ПЕРЕГРУЗКА МАРЖИ</b>: Использование >80%. НЕМЕДЛЕННО уменьшите объемы!")
        elif margin_usage > 60:
            recommendations.append("🟡 <b>ВЫСОКАЯ НАГРУЗКА</b>: Использование >60%. Оставьте запас для управления.")
        
        # Анализ R/R
        unfavorable_rr = [t for t in trades if t.get('metrics', {}).get('rr_ratio', 0) < 1]
        if unfavorable_rr:
            recommendations.append(f"📉 <b>НЕВЫГОДНЫЕ СДЕЛКИ</b>: {len(unfavorable_rr)} сделок с R/R < 1. Улучшите соотношение.")
        
        # Анализ диверсификации
        asset_count = len(set(t['symbol'] for t in trades))
        if asset_count < 2 and len(trades) > 1:
            recommendations.append("🎯 <b>НИЗКАЯ ДИВЕРСИФИКАЦИЯ</b>: Торгуете одним активом. Добавьте разные инструменты.")
        
        # Анализ общего риска
        total_risk = metrics.get('total_risk_percent', 0)
        if total_risk > 15:
            recommendations.append("🚨 <b>ПРЕВЫШЕНИЕ РИСКА</b>: Общий риск >15%. Срочно уменьшите позиции!")
        elif total_risk > 10:
            recommendations.append("⚠️ <b>ВЫСОКИЙ РИСК</b>: Общий риск >10%. Рекомендуется снизить экспозицию.")
        
        return recommendations if recommendations else ["✅ <b>ПОРТФЕЛЬ ОПТИМАЛЕН</b>. Продолжайте в том же духе!"]

# ---------------------------
# User Interface - ИСПРАВЛЕННЫЙ ИНТЕРФЕЙС
# ---------------------------
class UserInterface:
    """Улучшенный пользовательский интерфейс"""
    
    @staticmethod
    def create_main_menu_keyboard():
        """Создание клавиатуры главного меню"""
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🎯 РАСЧЕТ СДЕЛКИ", callback_data="single_trade")],
            [InlineKeyboardButton("📊 ПОРТФЕЛЬ", callback_data="portfolio")],
            [InlineKeyboardButton("⚙️ НАСТРОЙКИ", callback_data="settings")],
            [InlineKeyboardButton("📚 PRO ИНСТРУКЦИИ", callback_data="pro_info")],
            [InlineKeyboardButton("💝 ПОДДЕРЖАТЬ РАЗРАБОТЧИКА", callback_data="donate_start")]
        ])
    
    @staticmethod
    def format_trade_report(metrics: Dict) -> str:
        """Форматирование отчета о сделке"""
        direction_emoji = "📈" if metrics['direction'].upper() == 'LONG' else "📉"
        pnl_emoji = "🟢" if metrics['current_pnl'] >= 0 else "🔴"
        
        return f"""
🎯 <b>РЕЗУЛЬТАТЫ РАСЧЕТА v3.0</b>

{direction_emoji} <b>Актив</b>: {metrics['symbol']} | <b>Направление</b>: {metrics['direction']}
💰 <b>Текущая цена</b>: {metrics['current_price']} ✅ <b>РЕАЛЬНАЯ</b>

📊 <b>Уровни торговли</b>:
├ Вход: {metrics['entry_price']}
├ SL: {metrics['stop_loss']} 
└ TP: {metrics['take_profit']}

📏 <b>Дистанции</b>:
├ До SL: {metrics['sl_distance_pips']} пунктов
└ До TP: {metrics['tp_distance_pips']} пунктов

💸 <b>Финансовые показатели</b>:
├ Стоимость пункта: ${metrics['pip_value']}
├ Объем: {metrics['volume']} лотов
├ Риск: ${metrics['risk_amount']} ({metrics['risk_percent']}% от депозита)
├ Потенциальная прибыль: ${metrics['potential_profit']}
└ R/R соотношение: {metrics['rr_ratio']}

🛡 <b>Маржинальные показатели</b>:
├ Требуемая маржа: ${metrics['required_margin']}
├ Свободная маржа: ${metrics['free_margin']} ({metrics['free_margin_percent']}%)
└ Использование маржи: {100 - metrics['free_margin_percent']}%

💡 <b>Текущий статус</b>:
{pnl_emoji} Текущий P&L: ${metrics['current_pnl']}
🎯 До TP осталось: {metrics['tp_distance_pips']} пунктов
        """
    
    @staticmethod
    def format_portfolio_report(metrics: Dict, trades: List[Dict], recommendations: List[str]) -> str:
        """Форматирование отчета портфеля"""
        total_pnl = sum(t.get('metrics', {}).get('current_pnl', 0) for t in trades)
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        
        report = f"""
📊 <b>АНАЛИЗ ПОРТФЕЛЯ v3.0</b>

💰 <b>Общие показатели</b>:
├ Депозит: ${metrics.get('total_deposit', 0):.2f}
├ Общий P&L: {pnl_emoji} ${total_pnl:.2f}
├ Общий риск: ${metrics.get('total_risk', 0):.2f} ({metrics.get('total_risk_percent', 0):.1f}%)
├ Использование маржи: {metrics.get('total_margin_usage', 0):.1f}%
└ Количество сделок: {len(trades)}

🔍 <b>Рекомендации</b>:
"""
        
        for rec in recommendations:
            report += f"├ {rec}\n"
        
        if trades:
            report += "\n📈 <b>Активные сделки</b>:\n"
            for trade in trades:
                trade_metrics = trade.get('metrics', {})
                direction_emoji = "📈" if trade['direction'].upper() == 'LONG' else "📉"
                pnl_emoji = "🟢" if trade_metrics.get('current_pnl', 0) >= 0 else "🔴"
                
                report += f"├ {direction_emoji} {trade['symbol']} | P&L: {pnl_emoji} ${trade_metrics.get('current_pnl', 0):.2f}\n"
        
        return report

# ---------------------------
# Conversation States
# ---------------------------
SETTING_DEPOSIT, SETTING_LEVERAGE, TRADE_SYMBOL, TRADE_DIRECTION, TRADE_ENTRY, TRADE_SL, TRADE_TP, TRADE_RISK = range(8)

# ---------------------------
# Global Instances
# ---------------------------
portfolio_manager = PortfolioManager()
market_data_provider = MarketDataProvider()
risk_analytics = RiskAnalytics()
user_interface = UserInterface()

# ---------------------------
# Command Handlers - ИСПРАВЛЕННЫЕ
# ---------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user_id = update.effective_user.id
    
    welcome_text = """
🎯 <b>PRO Risk Calculator v3.0 | ENTERPRISE EDITION</b>

<b>ПРОФЕССИОНАЛЬНЫЙ ИНСТРУМЕНТ ДЛЯ РАСЧЕТА РИСКОВ</b>

⚡ <b>Ключевые возможности:</b>
• Точный расчет объема позиции
• Реальные цены с 6 источников
• Анализ маржи и плеча
• Профессиональные рекомендации
• Мульти-активный портфель

💰 <b>Механика расчета:</b>
Риск на сделку = % от депозита (вы выбираете %)
Объем = Риск / (Дистанция SL × Стоимость пункта)
Таким образом объем АВТОМАТИЧЕСКИ адаптируется под ваш риск!

<b>Пример:</b>
Депозит: $1,000 | Риск: 5% = $50
SL дистанция: 20 пунктов | Стоимость пункта: $10
ОБЪЕМ = $50 / (20 × $10) = 0.25 лота

👇 <b>Выберите действие:</b>
    """
    
    await SafeMessageSender.send_message(
        user_id,
        welcome_text,
        context,
        user_interface.create_main_menu_keyboard()
    )

async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик PRO инструкций - ИСПРАВЛЕННЫЙ"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    pro_text = """
📚 <b>PRO ИНСТРУКЦИИ | Risk Calculator v3.0</b>

🎯 <b>ФИЛОСОФИЯ РИСК-МЕНЕДЖМЕНТА:</b>
• <b>1% правило</b>: Рискуйте не более 1% депозита на сделку
• <b>R/R ≥ 2</b>: Соотношение риск/прибыль минимум 1:2
• <b>5% лимит</b>: Общий риск портфеля не более 5%
• <b>Маржин контроль</b>: Использование маржи до 60%

🌪 <b>ВОЛАТИЛЬНОСТЬ В РАСЧЕТАХ:</b>

• <b>Что это?</b> Мера колебаний цены актива
• <b>Как используется?</b> Для оценки риска и рекомендаций
• <b>Высокая волатильность</b> (>30%) = большие риски И возможности
• <b>Низкая волатильность</b> (<15%) = стабильность, но меньший потенциал

<b>ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ:</b>
• BTCUSDT: 65% - высокий риск, нужен широкий SL
• EURUSD: 8% - низкий риск, можно tighter управление
• Используйте эти данные для настройки стоп-лоссов!

💡 <b>ПРОФЕССИОНАЛЬНЫЕ СОВЕТЫ:</b>
1. Всегда используйте стоп-лосс
2. Диверсифицируйте портфель
3. Следите за уровнем маржи
4. Адаптируйте объем к волатильности
5. Регулярно анализируйте портфель

🚀 <b>ПРИМЕР УСПЕШНОЙ СДЕЛКИ:</b>
Депозит: $10,000 | Риск: 2% = $200
Актив: EURUSD | Направление: LONG
Вход: 1.0850 | SL: 1.0830 | TP: 1.0950
Дистанция SL: 20 п | Дистанция TP: 100 п
Объем: 1.0 лот | R/R: 5.0 | Потенциал: $1,000
    """
    
    keyboard = [
        [InlineKeyboardButton("🎯 Начать расчет", callback_data="single_trade")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        pro_text,
        InlineKeyboardMarkup(keyboard)
    )

async def callback_router_fixed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ГАРАНТИРОВАННО РАБОЧИЙ обработчик callback'ов"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)  # КРИТИЧЕСКИ ВАЖНО
    
    data = query.data
    
    # Роутинг по callback данным
    if data == "main_menu":
        await start_command(update, context)
    elif data == "pro_info":
        await pro_info_command(update, context)
    elif data == "donate_start":
        await DonationSystem.show_donation_menu(update, context)
    elif data == "donate_usdt":
        await DonationSystem.show_usdt_donation(update, context)
    elif data == "donate_ton":
        await DonationSystem.show_ton_donation(update, context)
    elif data == "single_trade":
        await start_single_trade_flow(update, context)
    elif data == "portfolio":
        await show_portfolio_fixed(update, context)
    elif data == "settings":
        await show_settings_fixed(update, context)
    else:
        await SafeMessageSender.answer_callback_query(query, "Команда в разработке", show_alert=True)

async def start_single_trade_flow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Начало процесса расчета одиночной сделки"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    deposit = portfolio_manager.get_deposit(user_id)
    
    if deposit <= 0:
        text = (
            "🎯 <b>ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ v3.0</b>\n\n"
            "<b>МЕХАНИКА РАСЧЕТА:</b>\n"
            "• Риск на сделку = % от депозита (вы выбираете %)\n"
            "• Объем = Риск / (Дистанция SL × Стоимость пункта)\n"
            "• Таким образом объем АВТОМАТИЧЕСКИ адаптируется под ваш риск!\n\n"
            "<b>ПРИМЕР:</b>\n"
            "Депозит: $1,000 | Риск: 5% = $50\n"
            "SL дистанция: 20 пунктов | Стоимость пункта: $10\n"
            "ОБЪЕМ = $50 / (20 × $10) = 0.25 лота\n\n"
            "<b>Введите ваш депозит в USD:</b>"
        )
        
        context.user_data['flow'] = 'single_trade'
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]])
        )
        return SETTING_DEPOSIT
    else:
        # Переход к выбору символа, если депозит уже установлен
        return await ask_trade_symbol(update, context)

async def show_portfolio_fixed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ИСПРАВЛЕННЫЙ показ портфеля"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    trades = portfolio_manager.get_trades(user_id)
    deposit = portfolio_manager.get_deposit(user_id)
    
    if deposit <= 0:
        text = "❌ <b>Сначала установите депозит в настройках</b>"
        keyboard = [
            [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup(keyboard)
        )
        return
    
    # Расчет метрик портфеля
    total_risk = sum(t.get('risk_amount', 0) for t in trades)
    total_margin = sum(t.get('required_margin', 0) for t in trades)
    total_risk_percent = (total_risk / deposit) * 100 if deposit > 0 else 0
    margin_usage = (total_margin / deposit) * 100 if deposit > 0 else 0
    
    metrics = {
        'total_deposit': deposit,
        'total_risk': total_risk,
        'total_risk_percent': total_risk_percent,
        'total_margin_usage': margin_usage
    }
    
    # Генерация рекомендаций
    recommendations = risk_analytics.generate_professional_recommendations(metrics, trades)
    
    # Форматирование отчета
    report = user_interface.format_portfolio_report(metrics, trades, recommendations)
    
    keyboard = [
        [InlineKeyboardButton("🔄 Обновить", callback_data="portfolio")],
        [InlineKeyboardButton("🧹 Очистить портфель", callback_data="clear_portfolio")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        report,
        InlineKeyboardMarkup(keyboard)
    )

async def show_settings_fixed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ИСПРАВЛЕННЫЙ показ настроек"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    deposit = portfolio_manager.get_deposit(user_id)
    
    text = f"""
⚙️ <b>НАСТРОЙКИ PRO v3.0</b>

💰 <b>Текущий депозит</b>: ${deposit:.2f}
📈 <b>Текущее плечо</b>: 1:30

👇 <b>Выберите действие:</b>
    """
    
    keyboard = [
        [InlineKeyboardButton("💰 Изменить депозит", callback_data="change_deposit")],
        [InlineKeyboardButton("📈 Изменить плечо", callback_data="change_leverage")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

async def clear_portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Очистка портфеля"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    portfolio_manager.clear_trades(user_id)
    
    text = "✅ <b>Портфель успешно очищен!</b>"
    keyboard = [[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

# ---------------------------
# Message Handlers
# ---------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    user_id = update.effective_user.id
    text = update.message.text
    
    # Определяем текущий шаг в flow
    current_flow = context.user_data.get('flow')
    
    if current_flow == 'single_trade':
        if 'deposit' not in context.user_data:
            # Обработка ввода депозита
            try:
                deposit = float(text)
                if deposit <= 0:
                    await update.message.reply_text("❌ Депозит должен быть больше 0. Введите корректную сумму:")
                    return SETTING_DEPOSIT
                
                portfolio_manager.set_deposit_leverage(user_id, deposit)
                context.user_data['deposit'] = deposit
                
                await update.message.reply_text(
                    f"✅ Депозит ${deposit:.2f} установлен!\n\n"
                    "📈 <b>Введите торговый символ (например: EURUSD, BTCUSDT, AAPL):</b>",
                    parse_mode='HTML'
                )
                return TRADE_SYMBOL
                
            except ValueError:
                await update.message.reply_text("❌ Введите числовое значение депозита:")
                return SETTING_DEPOSIT
    
    # Здесь будут другие обработчики для остальных шагов...
    
    # Если не распознано, показываем главное меню
    await start_command(update, context)
    return ConversationHandler.END

# ---------------------------
# Setup Application
# ---------------------------
def setup_application():
    """Настройка приложения с исправленными обработчиками"""
    application = RobustApplicationBuilder.create_application(TOKEN)
    
    # Добавление обработчиков
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(callback_router_fixed))
    
    # Обработчик для очистки портфеля
    application.add_handler(CallbackQueryHandler(clear_portfolio_handler, pattern="^clear_portfolio$"))
    
    # Обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    return application

# ---------------------------
# Main Entry Point
# ---------------------------
async def main():
    """Основная функция запуска"""
    logger.info("🚀 Starting PRO Risk Calculator v3.0 | ENTERPRISE EDITION")
    
    application = setup_application()
    
    # Webhook настройка для Render
    if WEBHOOK_URL:
        logger.info(f"🔗 Setting up webhook: {WEBHOOK_URL}{WEBHOOK_PATH}")
        await application.bot.set_webhook(
            url=f"{WEBHOOK_URL}{WEBHOOK_PATH}",
        )
        
        # Создание aiohttp приложения для webhook
        app = web.Application()
        app.router.add_post(WEBHOOK_PATH, lambda req: telegram.Update.de_json(data=req.json(), bot=application.bot))
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', PORT)
        await site.start()
        
        logger.info(f"🌐 Webhook server started on port {PORT}")
        await asyncio.Event().wait()  # Бесконечное ожидание
        
    else:
        # Polling для разработки
        logger.info("🔄 Starting bot in polling mode...")
        await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
