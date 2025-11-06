# bot.py — PRO Risk Calculator v3.0 | ENTERPRISE EDITION
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

# --- Категории активов ---
ASSET_CATEGORIES = {
    "Криптовалюта": ["BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT"],
    "Forex": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
    "Акции": ["AAPL", "TSLA", "GOOGL", "MSFT"],
    "Индексы": ["NAS100", "SPX500", "DJ30"],
    "Металлы": ["XAUUSD", "XAGUSD"],
    "Энергия": ["OIL", "BRENT"]
}

# --- Уровни риска ---
RISK_LEVELS = ["1%", "2%", "3%", "4%", "5%", "10%"]

# --- Состояния для ConversationHandler ---
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

# --- Настройки таймаутов и повторных попыток ---
class RobustApplicationBuilder:
    """Строитель приложения с улучшенной обработкой ошибок"""
    
    @staticmethod
    def create_application(token: str) -> Application:
        """Создание приложения с настройками для устойчивости"""
        request = telegram.request.HTTPXRequest(
            connection_pool_size=8,
            read_timeout=30,
            write_timeout=30,
            connect_timeout=30
        )
        
        application = (
            Application.builder()
            .token(token)
            .request(request)
            .build()
        )
        
        return application

# --- Retry Decorator ---
def retry_on_timeout(max_retries: int = 3, delay: float = 1.0):
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
                        raise
                except telegram.error.NetworkError as e:
                    logger.warning(f"Network error in {func.__name__}, attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))
                    else:
                        raise
            return None
        return wrapper
    return decorator

# --- Safe Message Sender ---
class SafeMessageSender:
    @staticmethod
    @retry_on_timeout(max_retries=3, delay=1.0)
    async def send_message(
        chat_id: int,
        text: str,
        context: ContextTypes.DEFAULT_TYPE = None,
        reply_markup: InlineKeyboardMarkup = None,
        parse_mode: str = 'HTML'
    ) -> bool:
        try:
            if context and hasattr(context, 'bot'):
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode
                )
            else:
                bot = telegram.Bot(token=TOKEN)
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
        query: CallbackQuery,
        text: str,
        reply_markup: InlineKeyboardMarkup = None,
        parse_mode: str = 'HTML'
    ) -> bool:
        try:
            await query.edit_message_text(
                text=text,
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
        try:
            await query.answer(text=text, show_alert=show_alert)
            return True
        except Exception as e:
            logger.error(f"Failed to answer callback query: {e}")
            return False

# --- Donation System ---
class DonationSystem:
    @staticmethod
    async def show_donation_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

# --- Market Data Provider ---
class MarketDataProvider:
    def __init__(self):
        self.cache = cachetools.TTLCache(maxsize=500, ttl=300)
        self.session = None
        
    async def get_session(self):
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def get_real_time_price(self, symbol: str) -> float:
        return await self.get_robust_real_time_price(symbol)
    
    async def get_robust_real_time_price(self, symbol: str) -> float:
        cached_price = self.cache.get(symbol)
        if cached_price:
            return cached_price
                
        providers = []
        if self._is_forex(symbol):
            providers = [
                self._get_exchangerate_price,
                self._get_alpha_vantage_forex,
                self._get_finnhub_price,
                self._get_fallback_price
            ]
        elif self._is_crypto(symbol):
            providers = [
                self._get_binance_price,
                self._get_finnhub_price,
                self._get_fallback_price
            ]
        else:
            providers = [
                self._get_alpha_vantage_stock,
                self._get_finnhub_price,
                self._get_fallback_price
            ]
            
        price = None
        for provider in providers:
            price = await provider(symbol)
            if price and price > 0:
                break
                    
        if price is None or price <= 0:
            price = self._get_fallback_price(symbol)
                
        if price:
            self.cache[symbol] = price
                
        return price
            
    def _is_crypto(self, symbol: str) -> bool:
        crypto_symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'USDT']
        return any(crypto in symbol for crypto in crypto_symbols)
    
    def _is_forex(self, symbol: str) -> bool:
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        return symbol in forex_pairs
    
    async def _get_exchangerate_price(self, symbol: str) -> Optional[float]:
        try:
            if self._is_forex(symbol):
                from_curr = symbol[:3]
                to_curr = symbol[3:]
                url = f"https://api.frankfurter.app/latest?from={from_curr}&to={to_curr}"
                
                session = await self.get_session()
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'rates' in data and to_curr in data['rates']:
                            return data['rates'][to_curr]
        except Exception as e:
            logger.error(f"ExchangeRate API error for {symbol}: {e}")
        return None
    
    async def _get_binance_price(self, symbol: str) -> Optional[float]:
        try:
            session = await self.get_session()
            if 'USDT' in symbol:
                binance_symbol = symbol.replace('/', '')
            else:
                binance_symbol = symbol.replace('USDT', '') + 'USDT'
            
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data['price'])
        except Exception as e:
            logger.error(f"Binance API error for {symbol}: {e}")
        return None
    
    async def _get_alpha_vantage_stock(self, symbol: str) -> Optional[float]:
        if not ALPHA_VANTAGE_API_KEY:
            return None
            
        try:
            session = await self.get_session()
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Global Quote' in data and '05. price' in data['Global Quote']:
                        return float(data['Global Quote']['05. price'])
        except Exception as e:
            logger.error(f"Alpha Vantage stock error for {symbol}: {e}")
        return None
    
    async def _get_alpha_vantage_forex(self, symbol: str) -> Optional[float]:
        if not ALPHA_VANTAGE_API_KEY:
            return None
            
        try:
            session = await self.get_session()
            from_currency = symbol[:3]
            to_currency = symbol[3:]
            url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}&to_currency={to_currency}&apikey={ALPHA_VANTAGE_API_KEY}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Realtime Currency Exchange Rate' in data and '5. Exchange Rate' in data['Realtime Currency Exchange Rate']:
                        return float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])
        except Exception as e:
            logger.error(f"Alpha Vantage forex error for {symbol}: {e}")
        return None
    
    async def _get_finnhub_price(self, symbol: str) -> Optional[float]:
        if not FINNHUB_API_KEY:
            return None
            
        try:
            session = await self.get_session()
            finnhub_symbol = symbol
            if self._is_forex(symbol):
                finnhub_symbol = f"OANDA:{symbol[:3]}_{symbol[3:]}"
            elif self._is_crypto(symbol) and 'USDT' in symbol:
                finnhub_symbol = f"BINANCE:{symbol.replace('USDT', '')}-USDT"
            
            url = f"https://finnhub.io/api/v1/quote?symbol={finnhub_symbol}&token={FINNHUB_API_KEY}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'c' in data and data['c'] > 0:
                        return data['c']
        except Exception as e:
            logger.error(f"Finnhub API error for {symbol}: {e}")
        return None
    
    def _get_fallback_price(self, symbol: str) -> float:
        fallback_prices = {
            'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDJPY': 147.50, 'USDCHF': 0.8800,
            'AUDUSD': 0.6520, 'USDCAD': 1.3500, 'NZDUSD': 0.6100,
            'BTCUSDT': 45000.0, 'ETHUSDT': 3000.0, 'XRPUSDT': 0.62, 'LTCUSDT': 72.0,
            'AAPL': 185.0, 'TSLA': 240.0, 'GOOGL': 138.0, 'MSFT': 330.0,
            'NAS100': 16200.0, 'SPX500': 4800.0, 'DJ30': 37500.0,
            'XAUUSD': 1980.0, 'XAGUSD': 23.50,
            'OIL': 75.0, 'BRENT': 80.0
        }
        return fallback_prices.get(symbol, 100.0)

# --- Instrument Specs ---
class InstrumentSpecs:
    SPECS = {
        "EURUSD": {"type": "forex", "contract_size": 100000, "margin_currency": "USD", "pip_value": 10.0, "calculation_formula": "forex", "pip_decimal_places": 4},
        "GBPUSD": {"type": "forex", "contract_size": 100000, "margin_currency": "USD", "pip_value": 10.0, "calculation_formula": "forex", "pip_decimal_places": 4},
        "USDJPY": {"type": "forex", "contract_size": 100000, "margin_currency": "USD", "pip_value": 9.09, "calculation_formula": "forex_jpy", "pip_decimal_places": 2},
        "BTCUSDT": {"type": "crypto", "contract_size": 1, "margin_currency": "USDT", "pip_value": 1.0, "calculation_formula": "crypto", "pip_decimal_places": 1},
        "ETHUSDT": {"type": "crypto", "contract_size": 1, "margin_currency": "USDT", "pip_value": 1.0, "calculation_formula": "crypto", "pip_decimal_places": 2},
        "AAPL": {"type": "stock", "contract_size": 100, "margin_currency": "USD", "pip_value": 1.0, "calculation_formula": "stocks", "pip_decimal_places": 2},
        "TSLA": {"type": "stock", "contract_size": 100, "margin_currency": "USD", "pip_value": 1.0, "calculation_formula": "stocks", "pip_decimal_places": 2},
        "NAS100": {"type": "index", "contract_size": 10, "margin_currency": "USD", "pip_value": 1.0, "calculation_formula": "indices", "pip_decimal_places": 1},
        "XAUUSD": {"type": "metal", "contract_size": 100, "margin_currency": "USD", "pip_value": 10.0, "calculation_formula": "metals", "pip_decimal_places": 2},
        "OIL": {"type": "energy", "contract_size": 1000, "margin_currency": "USD", "pip_value": 10.0, "calculation_formula": "energy", "pip_decimal_places": 2}
    }
    
    @classmethod
    def get_specs(cls, symbol: str) -> Dict[str, Any]:
        return cls.SPECS.get(symbol, cls._get_default_specs(symbol))
    
    @classmethod
    def _get_default_specs(cls, symbol: str) -> Dict[str, Any]:
        if any(currency in symbol for currency in ['USD', 'EUR', 'GBP', 'JPY']):
            return {"type": "forex", "contract_size": 100000, "margin_currency": "USD", "pip_value": 10.0, "calculation_formula": "forex", "pip_decimal_places": 4}
        elif 'USDT' in symbol:
            return {"type": "crypto", "contract_size": 1, "margin_currency": "USDT", "pip_value": 1.0, "calculation_formula": "crypto", "pip_decimal_places": 2}
        else:
            return {"type": "stock", "contract_size": 100, "margin_currency": "USD", "pip_value": 1.0, "calculation_formula": "stocks", "pip_decimal_places": 2}

# --- Professional Margin Calculator ---
class ProfessionalMarginCalculator:
    def __init__(self):
        self.market_data = MarketDataProvider()
    
    async def calculate_professional_margin(self, symbol: str, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        specs = InstrumentSpecs.get_specs(symbol)
        formula = specs['calculation_formula']
        
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        if formula == "forex":
            required_margin = (volume * contract_size) / lev_value
        elif formula == "forex_jpy":
            required_margin = (volume * contract_size) / (lev_value * current_price)
        elif formula == "crypto":
            required_margin = (volume * current_price) / lev_value
        elif formula == "stocks":
            required_margin = (volume * contract_size * current_price) / lev_value
        elif formula == "indices":
            required_margin = (volume * contract_size * current_price) / lev_value
        elif formula == "metals":
            required_margin = (volume * contract_size * current_price) / lev_value
        elif formula == "energy":
            required_margin = (volume * contract_size * current_price) / lev_value
        else:
            required_margin = (volume * contract_size * current_price) / lev_value
        
        return {
            'required_margin': required_margin,
            'contract_size': contract_size,
            'calculation_method': formula,
            'leverage_used': lev_value,
            'notional_value': volume * contract_size * current_price if current_price else volume * contract_size
        }

# --- Portfolio Manager (Добавлен для управления портфелем) ---
class PortfolioManager:
    @staticmethod
    def ensure_user(user_id: int):
        if 'user_data' not in globals():
            global user_data
            user_data = {}
        if user_id not in user_data:
            user_data[user_id] = {'deposit': 0, 'leverage': '1:100', 'trades': []}

    @staticmethod
    def add_trade(user_id: int, trade: Dict):
        PortfolioManager.ensure_user(user_id)
        user_data[user_id]['trades'].append(trade)

    @staticmethod
    def clear_portfolio(user_id: int):
        PortfolioManager.ensure_user(user_id)
        user_data[user_id]['trades'] = []

    @staticmethod
    def set_deposit_leverage(user_id: int, deposit: float, leverage: str):
        PortfolioManager.ensure_user(user_id)
        user_data[user_id]['deposit'] = deposit
        user_data[user_id]['leverage'] = leverage

# --- Data Manager (Добавлен для временных данных) ---
class DataManager:
    @staticmethod
    def load_temporary_data():
        return {}  # Заглушка, можно реализовать с файлом или DB

    @staticmethod
    def clear_temporary_progress(user_id: int):
        pass  # Заглушка

# --- State Handlers Class (Новый класс для оптимизации) ---
class StateHandlers:
    @staticmethod
    async def handle_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE, is_single: bool = True) -> int:
        try:
            deposit = float(update.message.text.replace(',', '.'))
            if deposit <= 0:
                await SafeMessageSender.send_message(update.message.chat_id, "❌ Депозит должен быть больше 0\nПопробуйте еще раз:", context)
                return SingleTradeState.DEPOSIT.value if is_single else MultiTradeState.DEPOSIT.value
            
            context.user_data['deposit'] = deposit
            
            keyboard = [[InlineKeyboardButton(f"1:{lev}", callback_data=f"lev_1:{lev}") for lev in [100, 500, 1000]]]
            keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
            
            await SafeMessageSender.send_message(update.message.chat_id, f"✅ Депозит: ${deposit:,.2f}\n\n<b>Выберите кредитное плечо:</b>", context, InlineKeyboardMarkup(keyboard))
            return SingleTradeState.LEVERAGE.value if is_single else MultiTradeState.LEVERAGE.value
        except ValueError:
            await SafeMessageSender.send_message(update.message.chat_id, "❌ Введите число (например: 1000)\nПопробуйте еще раз:", context)
            return SingleTradeState.DEPOSIT.value if is_single else MultiTradeState.DEPOSIT.value

    @staticmethod
    async def handle_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE, is_single: bool = True) -> int:
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        leverage = query.data.replace('lev_', '')
        context.user_data['leverage'] = leverage
        
        return await StateHandlers.start_asset_input(update, context, is_single)

    @staticmethod
    async def start_asset_input(update: Update, context: ContextTypes.DEFAULT_TYPE, is_single: bool = True) -> int:
        query = update.callback_query
        
        trade_count = len(context.user_data.get('trades', [])) if not is_single else 0
        
        text = f"<b>Сделка #{trade_count + 1}</b>\n\nВыберите категорию актива:" if not is_single else "<b>Выберите категорию актива:</b>"
        
        keyboard = [[InlineKeyboardButton(cat, callback_data=f"cat_{cat}")] for cat in ASSET_CATEGORIES.keys()]
        keyboard.append([InlineKeyboardButton("📝 Ввести актив вручную", callback_data="asset_manual")])
        
        if trade_count > 0 and not is_single:
            keyboard.append([InlineKeyboardButton("🚀 Завершить ввод", callback_data="multi_finish")])
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        if query:
            await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(keyboard))
        else:
            await SafeMessageSender.send_message(update.message.chat_id, text, context, InlineKeyboardMarkup(keyboard))
        
        return SingleTradeState.ASSET_CATEGORY.value if is_single else MultiTradeState.ASSET_CATEGORY.value

    @staticmethod
    async def handle_asset_category(update: Update, context: ContextTypes.DEFAULT_TYPE, is_single: bool = True) -> int:
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        if query.data == "asset_manual":
            await SafeMessageSender.edit_message_text(query, "✍️ Введите название актива (например: BTCUSDT):", InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]]))
            return SingleTradeState.ASSET.value if is_single else MultiTradeState.ASSET.value
        
        if query.data == "multi_finish" and not is_single:
            return await StateHandlers.finish_multi_trade(update, context)
        
        category = query.data.replace('cat_', '')
        context.user_data['current_trade'] = {'asset_category': category}
        
        assets = ASSET_CATEGORIES.get(category, [])
        
        keyboard = [[InlineKeyboardButton(asset, callback_data=f"asset_{asset}")] for asset in assets]
        keyboard.append([InlineKeyboardButton("🔙 Назад к категориям", callback_data="back_to_categories")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await SafeMessageSender.edit_message_text(query, f"✅ Категория: {category}\n\n<b>Выберите актив:</b>", InlineKeyboardMarkup(keyboard))
        return SingleTradeState.ASSET.value if is_single else MultiTradeState.ASSET.value

    @staticmethod
    async def handle_asset(update: Update, context: ContextTypes.DEFAULT_TYPE, is_single: bool = True) -> int:
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        if query.data == "back_to_categories":
            return await StateHandlers.start_asset_input(update, context, is_single)
        
        asset = query.data.replace('asset_', '')
        context.user_data['current_trade']['asset'] = asset
        
        await SafeMessageSender.edit_message_text(query, f"✅ Актив: {asset}\n\n<b>Выберите направление сделки:</b>", InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ]))
        return SingleTradeState.DIRECTION.value if is_single else MultiTradeState.DIRECTION.value

    @staticmethod
    async def handle_asset_manual(update: Update, context: ContextTypes.DEFAULT_TYPE, is_single: bool = True) -> int:
        asset = update.message.text.strip().upper()
        
        if not re.match(r'^[A-Z0-9]{2,20}$', asset):
            await SafeMessageSender.send_message(update.message.chat_id, "❌ Неверный формат актива. Попробуйте еще раз:", context)
            return SingleTradeState.ASSET.value if is_single else MultiTradeState.ASSET.value
        
        context.user_data['current_trade']['asset'] = asset
        
        await SafeMessageSender.send_message(update.message.chat_id, f"✅ Актив: {asset}\n\n<b>Выберите направление сделки:</b>", context, InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ]))
        return SingleTradeState.DIRECTION.value if is_single else MultiTradeState.DIRECTION.value

    @staticmethod
    async def handle_direction(update: Update, context: ContextTypes.DEFAULT_TYPE, is_single: bool = True) -> int:
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        direction = query.data.replace('dir_', '')
        context.user_data['current_trade']['direction'] = direction
        
        await SafeMessageSender.edit_message_text(query, f"✅ Направление: {direction}\n\n<b>Введите цену входа:</b>", InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]]))
        return SingleTradeState.ENTRY.value if is_single else MultiTradeState.ENTRY.value

    @staticmethod
    async def handle_entry(update: Update, context: ContextTypes.DEFAULT_TYPE, is_single: bool = True) -> int:
        try:
            entry_price = float(update.message.text.replace(',', '.'))
            if entry_price <= 0:
                await SafeMessageSender.send_message(update.message.chat_id, "❌ Цена должна быть больше 0\nПопробуйте еще раз:", context)
                return SingleTradeState.ENTRY.value if is_single else MultiTradeState.ENTRY.value
            
            context.user_data['current_trade']['entry_price'] = entry_price
            
            await SafeMessageSender.send_message(update.message.chat_id, f"✅ Цена входа: {entry_price}\n\n<b>Введите уровень стоп-лосса:</b>", context, InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]]))
            return SingleTradeState.STOP_LOSS.value if is_single else MultiTradeState.STOP_LOSS.value
        except ValueError:
            await SafeMessageSender.send_message(update.message.chat_id, "❌ Введите число (например: 50000)\nПопробуйте еще раз:", context)
            return SingleTradeState.ENTRY.value if is_single else MultiTradeState.ENTRY.value

    @staticmethod
    async def handle_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE, is_single: bool = True) -> int:
        try:
            stop_loss = float(update.message.text.replace(',', '.'))
            current_trade = context.user_data['current_trade']
            entry_price = current_trade['entry_price']
            direction = current_trade['direction']
            
            if direction == 'LONG' and stop_loss >= entry_price:
                await SafeMessageSender.send_message(update.message.chat_id, "❌ Для LONG стоп-лосс должен быть НИЖЕ цены входа\nПопробуйте еще раз:", context)
                return SingleTradeState.STOP_LOSS.value if is_single else MultiTradeState.STOP_LOSS.value
            elif direction == 'SHORT' and stop_loss <= entry_price:
                await SafeMessageSender.send_message(update.message.chat_id, "❌ Для SHORT стоп-лосс должен быть ВЫШЕ цены входа\nПопробуйте еще раз:", context)
                return SingleTradeState.STOP_LOSS.value if is_single else MultiTradeState.STOP_LOSS.value
            
            current_trade['stop_loss'] = stop_loss
            
            keyboard = [[InlineKeyboardButton(level, callback_data=f"risk_{level}")] for level in RISK_LEVELS]
            keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
            
            await SafeMessageSender.send_message(update.message.chat_id, f"✅ Стоп-лосс: {stop_loss}\n\n<b>Выберите уровень риска:</b>", context, InlineKeyboardMarkup(keyboard))
            return SingleTradeState.RISK_LEVEL.value if is_single else MultiTradeState.RISK_LEVEL.value
        except ValueError:
            await SafeMessageSender.send_message(update.message.chat_id, "❌ Введите число (например: 48000)\nПопробуйте еще раз:", context)
            return SingleTradeState.STOP_LOSS.value if is_single else MultiTradeState.STOP_LOSS.value

    @staticmethod
    async def handle_risk_level(update: Update, context: ContextTypes.DEFAULT_TYPE, is_single: bool = True) -> int:
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        risk_level = query.data.replace('risk_', '')
        context.user_data['current_trade']['risk_level'] = risk_level
        
        await SafeMessageSender.edit_message_text(query, f"✅ Уровень риска: {risk_level}\n\n<b>Введите уровень тейк-профита:</b>", InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]]))
        return SingleTradeState.TAKE_PROFIT.value if is_single else MultiTradeState.TAKE_PROFIT.value

    @staticmethod
    async def handle_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE, is_single: bool = True) -> int:
        try:
            take_profit = float(update.message.text.replace(',', '.'))
            current_trade = context.user_data['current_trade']
            entry_price = current_trade['entry_price']
            direction = current_trade['direction']
            
            if direction == 'LONG' and take_profit <= entry_price:
                await SafeMessageSender.send_message(update.message.chat_id, "❌ Для LONG тейк-профит должен быть ВЫШЕ цены входа\nПопробуйте еще раз:", context)
                return SingleTradeState.TAKE_PROFIT.value if is_single else MultiTradeState.TAKE_PROFIT.value
            elif direction == 'SHORT' and take_profit >= entry_price:
                await SafeMessageSender.send_message(update.message.chat_id, "❌ Для SHORT тейк-профит должен быть НИЖЕ цены входа\nПопробуйте еще раз:", context)
                return SingleTradeState.TAKE_PROFIT.value if is_single else MultiTradeState.TAKE_PROFIT.value
            
            current_trade['take_profit'] = take_profit
            
            # Здесь рассчитать метрики и добавить в портфель
            user_id = update.effective_user.id
            PortfolioManager.add_trade(user_id, current_trade)
            PortfolioManager.set_deposit_leverage(user_id, context.user_data['deposit'], context.user_data['leverage'])
            
            context.user_data.clear()
            
            await SafeMessageSender.send_message(update.message.chat_id, "✅ Сделка добавлена в портфель!", context)
            return ConversationHandler.END
        except ValueError:
            await SafeMessageSender.send_message(update.message.chat_id, "❌ Введите число (например: 52000)\nПопробуйте еще раз:", context)
            return SingleTradeState.TAKE_PROFIT.value if is_single else MultiTradeState.TAKE_PROFIT.value

    @staticmethod
    async def handle_add_another(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        if query.data == "multi_finish":
            return await StateHandlers.finish_multi_trade(update, context)
        
        return await StateHandlers.start_asset_input(update, context, is_single=False)

    @staticmethod
    async def finish_multi_trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_id = update.effective_user.id if update.message else update.callback_query.from_user.id
        PortfolioManager.set_deposit_leverage(user_id, context.user_data['deposit'], context.user_data['leverage'])
        
        context.user_data.clear()
        
        await SafeMessageSender.send_message(user_id, "✅ Мультипозиция завершена!", context)
        return ConversationHandler.END

    @staticmethod
    async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE, is_single: bool = True) -> int:
        context.user_data.clear()
        await SafeMessageSender.send_message(update.effective_user.id, "❌ Расчет отменен", context)
        return ConversationHandler.END

# --- Функции для ConversationHandler (Используют StateHandlers) ---
async def single_trade_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_deposit(update, context, is_single=True)

async def single_trade_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_leverage(update, context, is_single=True)

async def single_trade_asset_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_asset_category(update, context, is_single=True)

async def single_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_asset(update, context, is_single=True)

async def single_trade_asset_manual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_asset_manual(update, context, is_single=True)

async def single_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_direction(update, context, is_single=True)

async def single_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_entry(update, context, is_single=True)

async def single_trade_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_stop_loss(update, context, is_single=True)

async def single_trade_risk_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_risk_level(update, context, is_single=True)

async def single_trade_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_take_profit(update, context, is_single=True)

async def single_trade_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_cancel(update, context, is_single=True)

async def multi_trade_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_deposit(update, context, is_single=False)

async def multi_trade_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_leverage(update, context, is_single=False)

async def multi_trade_asset_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_asset_category(update, context, is_single=False)

async def multi_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_asset(update, context, is_single=False)

async def multi_trade_asset_manual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_asset_manual(update, context, is_single=False)

async def multi_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_direction(update, context, is_single=False)

async def multi_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_entry(update, context, is_single=False)

async def multi_trade_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_stop_loss(update, context, is_single=False)

async def multi_trade_risk_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_risk_level(update, context, is_single=False)

async def multi_trade_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_take_profit(update, context, is_single=False)

async def multi_trade_add_another(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_add_another(update, context)

async def multi_trade_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await StateHandlers.handle_cancel(update, context, is_single=False)

# --- Команда /start ---
@retry_on_timeout(max_retries=2, delay=1.0)
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🎯 <b>PRO Risk Calculator v3.0</b>\n\n"
        "Профессиональный инструмент для расчета рисков в трейдинге с реальными котировками!\n\n"
        "Выберите действие:"
    )
    
    keyboard = [
        [InlineKeyboardButton("🎯 Профессиональные сделки", callback_data="pro_calculation")],
        [InlineKeyboardButton("📊 Мой портфель", callback_data="portfolio")],
        [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")],
        [InlineKeyboardButton("🚀 Будущие разработки", callback_data="future_features")],
        [InlineKeyboardButton("💖 Поддержать разработчика", callback_data="donate_start")]
    ]
    
    await SafeMessageSender.send_message(update.effective_user.id, text, context, InlineKeyboardMarkup(keyboard))

# --- Главное меню ---
@retry_on_timeout(max_retries=2, delay=1.0)
async def main_menu_save_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = "🎯 <b>PRO Risk Calculator v3.0</b>\n\nВыберите действие:"
    
    keyboard = [
        [InlineKeyboardButton("🎯 Профессиональные сделки", callback_data="pro_calculation")],
        [InlineKeyboardButton("📊 Мой портфель", callback_data="portfolio")],
        [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")],
        [InlineKeyboardButton("🚀 Будущие разработки", callback_data="future_features")],
        [InlineKeyboardButton("💖 Поддержать разработчика", callback_data="donate_start")]
    ]
    
    await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(keyboard))

# --- Показ портфеля ---
@retry_on_timeout(max_retries=2, delay=1.0)
async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    PortfolioManager.ensure_user(user_id)
    
    user_portfolio = user_data[user_id]
    trades = user_portfolio['trades']
    
    if not trades:
        text = "📊 Ваш портфель пуст"
    else:
        text = f"📊 Ваш портфель:\nДепозит: ${user_portfolio['deposit']}\nПлечо: {user_portfolio['leverage']}\nСделок: {len(trades)}"
    
    keyboard = [
        [InlineKeyboardButton("🗑 Очистить портфель", callback_data="clear_portfolio")],
        [InlineKeyboardButton("📤 Экспорт отчета", callback_data="export_portfolio")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(keyboard))

# --- Callback Router ---
@retry_on_timeout(max_retries=2, delay=1.0)
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    data = query.data
    
    if data == "main_menu" or data == "main_menu_save":
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
    else:
        await SafeMessageSender.answer_callback_query(query, "Команда не распознана")

# --- Pro Calculation Handler ---
@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_calculation_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = "🎯 <b>ПРОФЕССИОНАЛЬНЫЕ СДЕЛКИ v3.0</b>\n\nВыберите тип расчета:"
    
    keyboard = [
        [InlineKeyboardButton("🎯 Одна сделка", callback_data="single_trade")],
        [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")],
        [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
    ]
    
    await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(keyboard))

# --- Pro Info ---
@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "📚 <b>PRO ИНСТРУКЦИИ v3.0</b>\n\n[Здесь полный текст инструкций как в оригинале]"
    keyboard = [
        [InlineKeyboardButton("🎯 Начать расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("📊 Мой портфель", callback_data="portfolio")],
        [InlineKeyboardButton("💖 Поддержать разработчика", callback_data="donate_start")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    if update.callback_query:
        await SafeMessageSender.edit_message_text(update.callback_query, text, InlineKeyboardMarkup(keyboard))
    else:
        await SafeMessageSender.send_message(update.effective_user.id, text, context, InlineKeyboardMarkup(keyboard))

# --- Future Features ---
@retry_on_timeout(max_retries=2, delay=1.0)
async def future_features_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "<b>🚀 БУДУЩИЕ РАЗРАБОТКИ v4.0</b>\n\n[Здесь текст о будущих функциях]"
    keyboard = [
        [InlineKeyboardButton("💖 Поддержать разработку", callback_data="donate_start")],
        [InlineKeyboardButton("🎯 Начать расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    if update.callback_query:
        await SafeMessageSender.edit_message_text(update.callback_query, text, InlineKeyboardMarkup(keyboard))
    else:
        await SafeMessageSender.send_message(update.effective_user.id, text, context, InlineKeyboardMarkup(keyboard))

# --- Clear Portfolio ---
@retry_on_timeout(max_retries=2, delay=1.0)
async def clear_portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

# --- Export Portfolio ---
@retry_on_timeout(max_retries=2, delay=1.0)
async def export_portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    user_id = query.from_user.id
    PortfolioManager.ensure_user(user_id)
    
    user_portfolio = user_data[user_id]
    trades = user_portfolio['trades']
    
    if not trades:
        await SafeMessageSender.answer_callback_query(query, "❌ Портфель пуст")
        return
    
    report = f"📊 ОТЧЕТ ПОРТФЕЛЯ v3.0\nДата: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    report += f"Депозит: ${user_portfolio['deposit']:,.2f}\n"
    report += f"Плечо: {user_portfolio['leverage']}\n"
    report += f"Всего сделок: {len(trades)}\n\n"
    
    for i, trade in enumerate(trades, 1):
        report += f"СДЕЛКА #{i}:\nАктив: {trade['asset']}\nНаправление: {trade['direction']}\nВход: {trade['entry_price']}\nSL: {trade['stop_loss']}\nTP: {trade['take_profit']}\n\n"
    
    bio = io.BytesIO(report.encode('utf-8'))
    bio.seek(0)
    
    await context.bot.send_document(
        chat_id=query.message.chat_id,
        document=InputFile(bio, filename=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"),
        caption="📊 Отчет вашего портфеля"
    )

# --- Single Trade Start ---
@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = "<b>Введите ваш депозит в USD:</b>"
    keyboard = [[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]]
    
    await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(keyboard))
    return SingleTradeState.DEPOSIT.value

# --- Multi Trade Start ---
@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = "<b>Введите ваш депозит в USD для мультипозиции:</b>"
    keyboard = [[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]]
    
    await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(keyboard))
    return MultiTradeState.DEPOSIT.value

# --- Setup Conversation Handlers ---
def setup_conversation_handlers(application: Application):
    single_trade_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(single_trade_start, pattern="^single_trade$")],
        states={
            SingleTradeState.DEPOSIT.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_deposit)],
            SingleTradeState.LEVERAGE.value: [CallbackQueryHandler(single_trade_leverage, pattern="^lev_")],
            SingleTradeState.ASSET_CATEGORY.value: [CallbackQueryHandler(single_trade_asset_category, pattern="^(cat_|asset_manual)")],
            SingleTradeState.ASSET.value: [CallbackQueryHandler(single_trade_asset, pattern="^(asset_|back_to_categories)"), MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_asset_manual)],
            SingleTradeState.DIRECTION.value: [CallbackQueryHandler(single_trade_direction, pattern="^dir_")],
            SingleTradeState.ENTRY.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_entry)],
            SingleTradeState.STOP_LOSS.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_stop_loss)],
            SingleTradeState.RISK_LEVEL.value: [CallbackQueryHandler(single_trade_risk_level, pattern="^risk_")],
            SingleTradeState.TAKE_PROFIT.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_take_profit)]
        },
        fallbacks=[CommandHandler("cancel", single_trade_cancel), CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")],
    )
    
    multi_trade_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(multi_trade_start, pattern="^multi_trade_start$")],
        states={
            MultiTradeState.DEPOSIT.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_deposit)],
            MultiTradeState.LEVERAGE.value: [CallbackQueryHandler(multi_trade_leverage, pattern="^lev_")],
            MultiTradeState.ASSET_CATEGORY.value: [CallbackQueryHandler(multi_trade_asset_category, pattern="^(cat_|asset_manual|multi_finish)")],
            MultiTradeState.ASSET.value: [CallbackQueryHandler(multi_trade_asset, pattern="^(asset_|back_to_categories)"), MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_asset_manual)],
            MultiTradeState.DIRECTION.value: [CallbackQueryHandler(multi_trade_direction, pattern="^dir_")],
            MultiTradeState.ENTRY.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_entry)],
            MultiTradeState.STOP_LOSS.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_stop_loss)],
            MultiTradeState.RISK_LEVEL.value: [CallbackQueryHandler(multi_trade_risk_level, pattern="^risk_")],
            MultiTradeState.TAKE_PROFIT.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_take_profit)],
            MultiTradeState.ADD_MORE.value: [CallbackQueryHandler(multi_trade_add_another, pattern="^(add_another|multi_finish)$")]
        },
        fallbacks=[CommandHandler("cancel", multi_trade_cancel), CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")],
    )
    
    application.add_handler(single_trade_conv)
    application.add_handler(multi_trade_conv)

# --- Main Function ---
async def main():
    application = RobustApplicationBuilder.create_application(TOKEN)
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(callback_router))
    
    setup_conversation_handlers(application)
    
    await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
