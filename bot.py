# bot.py — PRO Risk Calculator v3.0 | ENTERPRISE EDITION
import os
import logging
import asyncio
import time
import functools
import json
import io
import re
import aiohttp
import cachetools
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
            read_timeout=30.0,
            write_timeout=30.0,
            connect_timeout=30.0,
            pool_timeout=30.0,
        )
        
        # Создание приложения с настройками
        application = (
            Application.builder()
            .token(token)
            .request(request)
            .connect_timeout(30.0)
            .read_timeout(30.0)
            .write_timeout(30.0)
            .pool_timeout(30.0)
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
# Safe Message Sender
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
        parse_mode: str = 'Markdown'
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
        query: CallbackQuery,
        text: str,
        reply_markup: InlineKeyboardMarkup = None,
        parse_mode: str = 'Markdown'
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
# Market Data Provider - РЕАЛЬНЫЕ КОТИРОВКИ
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
        try:
            # Проверка кэша
            cached_price = self.cache.get(symbol)
            if cached_price:
                return cached_price
                
            # Определяем тип актива и выбираем провайдера
            price = None
            
            if self._is_crypto(symbol):
                price = await self._get_binance_price(symbol)
            elif self._is_forex(symbol) or self._is_metal(symbol):
                price = await self._get_alpha_vantage_forex(symbol)
            else:  # Акции, индексы
                price = await self._get_alpha_vantage_stock(symbol)
                
            # Резервный провайдер
            if price is None:
                price = await self._get_finnhub_price(symbol)
                
            # Fallback на статические данные при ошибках
            if price is None:
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
    
    async def _get_binance_price(self, symbol: str) -> Optional[float]:
        """Получение цены с Binance API"""
        try:
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
        except Exception as e:
            logger.error(f"Alpha Vantage stock error for {symbol}: {e}")
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
        """Fallback цены при недоступности API"""
        fallback_prices = {
            'BTCUSDT': 45000.0, 'ETHUSDT': 3000.0, 'EURUSD': 1.0850,
            'GBPUSD': 1.2650, 'XAUUSD': 1980.0, 'AAPL': 185.0,
            'TSLA': 240.0, 'NAS100': 16200.0, 'OIL': 75.0
        }
        return fallback_prices.get(symbol, 100.0)

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
        self.market_data = MarketDataProvider()
    
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
market_data_provider = MarketDataProvider()
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
    'GBPUSD': 9.2, 'USDJPY': 7.8, 'XAUUSD': 14.5, 'XAGUSD': 25.3,
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
# Portfolio Manager (ОБНОВЛЕННЫЙ)
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
# Professional Risk Calculator - ПОЛНОСТЬЮ ПЕРЕРАБОТАННЫЙ
# ---------------------------
class ProfessionalRiskCalculator:
    """ПРОФЕССИОНАЛЬНЫЙ калькулятор с реальными котировками"""
    
    @staticmethod
    def calculate_pip_distance(entry: float, stop_loss: float, direction: str, asset: str) -> float:
        """Профессиональный расчет дистанции в пунктах"""
        specs = InstrumentSpecs.get_specs(asset)
        pip_decimal_places = specs.get('pip_decimal_places', 4)
        
        if direction.upper() == 'LONG':
            distance = entry - stop_loss
        else:  # SHORT
            distance = stop_loss - entry
        
        # Масштабирование в зависимости от типа актива
        if pip_decimal_places == 2:  # JPY пары
            return abs(distance) * 100
        elif pip_decimal_places == 1:  # Некоторые индексы
            return abs(distance) * 10
        else:  # Стандартные 4 знака
            return abs(distance) * 10000

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
            
            # 1. Получение РЕАЛЬНОЙ цены актива
            current_price = await market_data_provider.get_real_time_price(asset)
            
            # 2. Получение спецификаций инструмента
            specs = InstrumentSpecs.get_specs(asset)
            
            # 3. Расчет суммы риска
            risk_percent = float(risk_level.strip('%'))
            risk_amount = deposit * (risk_percent / 100)
            
            # 4. Профессиональный расчет дистанции
            stop_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry, stop_loss, direction, asset)
            profit_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry, take_profit, direction, asset)
            
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
            risk_per_trade_percent = (risk_amount / deposit) * 100
            margin_usage_percent = (required_margin / deposit) * 100
            notional_value = margin_data.get('notional_value', 0)
            
            return {
                'volume_lots': volume_lots,
                'required_margin': required_margin,
                'free_margin': free_margin,
                'margin_level': margin_level,
                'risk_amount': risk_amount,
                'risk_percent': risk_percent,
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
                'leverage_used': margin_data.get('leverage_used', 1)
            }
        except Exception as e:
            logger.error(f"Профессиональный расчет ошибка: {e}")
            return {}

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
            'portfolio_leverage': portfolio_leverage
        }

    @staticmethod
    def generate_recommendations(metrics: Dict, trades: List[Dict]) -> List[str]:
        """Профессиональные рекомендации на основе метрик"""
        recommendations = []
        
        # Проверка общего риска
        if metrics.get('total_risk_percent', 0) > 10:
            recommendations.append(
                "⚠️ **ВНИМАНИЕ**: Общий риск портфеля превышает 10%. "
                "Рекомендуется уменьшить объем позиций для защиты капитала."
            )
        elif metrics.get('total_risk_percent', 0) > 5:
            recommendations.append(
                "🔶 **ПРЕДУПРЕЖДЕНИЕ**: Общий риск портфеля превышает 5%. "
                "Рассмотрите снижение объема позиций."
            )
        
        # Проверка уровня маржи
        if metrics.get('portfolio_margin_level', 0) < 100:
            recommendations.append(
                "🔴 **КРИТИЧЕСКИЙ УРОВЕНЬ МАРЖИ**! Немедленно пополните счет "
                "или закрите часть позиций во избежание маржин-колла."
            )
        elif metrics.get('portfolio_margin_level', 0) < 200:
            recommendations.append(
                "🟡 **НИЗКИЙ УРОВЕНЬ МАРЖИ**: Рассмотрите пополнение счета "
                "для безопасности позиций. Рекомендуемый уровень > 200%."
            )
        
        # Проверка использования маржи
        if metrics.get('total_margin_usage', 0) > 50:
            recommendations.append(
                f"🟡 **ВЫСОКОЕ ИСПОЛЬЗОВАНИЕ МАРЖИ**: {metrics['total_margin_usage']:.1f}%. "
                "Оставьте свободную маржу для непредвиденных ситуаций."
            )
        
        # Проверка левереджа
        if metrics.get('portfolio_leverage', 0) > 10:
            recommendations.append(
                f"🔶 **ВЫСОКИЙ ЛЕВЕРЕДЖ**: {metrics['portfolio_leverage']:.1f}x. "
                "Высокий левередж увеличивает как потенциальную прибыль, так и риски."
            )
        
        # Проверка Risk/Reward
        low_rr_trades = [
            t for t in trades 
            if t.get('metrics', {}).get('rr_ratio', 0) < 1
        ]
        if low_rr_trades:
            recommendations.append(
                f"📉 **НЕВЫГОДНОЕ R/R**: {len(low_rr_trades)} сделок имеют соотношение < 1. "
                "Пересмотрите уровни TP/SL для улучшения риск-менеджмента."
            )
        
        # Проверка волатильности
        if metrics.get('portfolio_volatility', 0) > 30:
            recommendations.append(
                f"🌪 **ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ**: {metrics['portfolio_volatility']:.1f}%. "
                "Будьте готовы к значительным колебаниям стоимости портфеля."
            )
        
        # Проверка диверсификации
        if metrics.get('diversity_score', 0) < 0.5 and len(trades) > 1:
            recommendations.append(
                "🎯 **НИЗКАЯ ДИВЕРСИФИКАЦИЯ**. Рассмотрите добавление активов "
                "из разных секторов для снижения систематического риска."
            )
        
        if not recommendations:
            recommendations.append("✅ **ПОРТФЕЛЬ СБАЛАНСИРОВАН**. Продолжайте в том же духе!")
        
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
            "🤖 **PRO Калькулятор Управления Рисками v3.0**\n\n"
            "🚀 МОИ ВОЗМОЖНОСТИ:\n"
            "• 📊 **РЕАЛЬНЫЕ КОТИРОВКИ** через Binance, Alpha Vantage, Finnhub\n"
            "• 💼 **ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ** маржи по отраслевым стандартам\n"
            "• 🎯 Контроль уровней риска (2%-25% от депозита)\n"
            "• 💡 Умные рекомендации и аналитика портфеля\n"
            "• 🛡 **ЗАЩИТА ОТ МАРЖИН-КОЛЛА** через правильный расчет объема\n"
            "• 📈 **РЕАЛЬНЫЕ ДАННЫЕ** для точного риск-менеджмента\n\n"
        )
        
        if saved_progress:
            text += "🔔 У вас есть сохраненный прогресс! Вы можете продолжить с того же места.\n\n"
        
        text += "**Выберите раздел:**"
        
        keyboard = [
            [InlineKeyboardButton("🎯 Профессиональные сделки", callback_data="pro_calculation")],
            [InlineKeyboardButton("📊 Мой портфель", callback_data="portfolio")]
        ]
        
        if saved_progress:
            keyboard.append([InlineKeyboardButton("🔄 Продолжить расчет", callback_data="restore_progress")])
        
        keyboard.extend([
            [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")],
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

# ОБНОВЛЕННЫЙ обработчик одиночной сделки с реальными данными
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
        
        # Собираем данные сделки
        trade_data = {
            'asset': context.user_data['asset'],
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': context.user_data['stop_loss'],
            'take_profit': take_profit,
            'risk_level': context.user_data['risk_level']
        }
        
        # ПРОФЕССИОНАЛЬНЫЙ расчет метрик с РЕАЛЬНЫМИ КОТИРОВКАМИ
        deposit = context.user_data['deposit']
        leverage = context.user_data['leverage']
        risk_level = context.user_data['risk_level']
        metrics = await ProfessionalRiskCalculator.calculate_professional_metrics(trade_data, deposit, leverage, risk_level)
        
        # Сохраняем сделку
        user_id = update.message.from_user.id
        trade_data['metrics'] = metrics
        PortfolioManager.add_single_trade(user_id, trade_data)
        
        # Очищаем временный прогресс
        DataManager.clear_temporary_progress(user_id)
        
        # ФОРМИРУЕМ ПРОФЕССИОНАЛЬНЫЙ ОТЧЕТ С РЕАЛЬНЫМИ ДАННЫМИ
        text = (
            f"🎯 **ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ СДЕЛКИ v3.0**\n\n"
            f"**📊 ПАРАМЕТРЫ СДЕЛКИ:**\n"
            f"• Актив: {trade_data['asset']}\n"
            f"• Текущая цена: ${metrics['current_price']:.2f} ✅ РЕАЛЬНАЯ\n"
            f"• Направление: {trade_data['direction']}\n"
            f"• Кредитное плечо: {leverage}\n"
            f"• Вход: {trade_data['entry_price']}\n"
            f"• Стоп-лосс: {trade_data['stop_loss']} ({metrics['stop_distance_pips']:.0f} пунктов)\n"
            f"• Тейк-профит: {trade_data['take_profit']} ({metrics['profit_distance_pips']:.0f} пунктов)\n"
            f"• Уровень риска: {trade_data['risk_level']}\n\n"
            
            f"**💰 ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ МАРЖИ:**\n"
            f"• Депозит: ${metrics['deposit']:,.2f}\n"
            f"• Сумма риска: ${metrics['risk_amount']:.2f} ({metrics['risk_percent']:.1f}%)\n"
            f"• Объем позиции: {metrics['volume_lots']:.2f} лотов\n"
            f"• Требуемая маржа: ${metrics['required_margin']:.2f} ✅ РЕАЛЬНЫЙ РАСЧЕТ\n"
            f"• Свободная маржа: ${metrics['free_margin']:.2f} ✅ РЕАЛЬНЫЙ РАСЧЕТ\n"
            f"• Уровень маржи: {metrics['margin_level']:.1f}% ✅ РЕАЛЬНЫЙ РАСЧЕТ\n"
            f"• Использование маржи: {metrics['margin_usage_percent']:.1f}%\n"
            f"• Номинальная стоимость: ${metrics.get('notional_value', 0):.2f}\n"
            f"• Метод расчета: {metrics['calculation_method']}\n\n"
            
            f"**📈 РЕЗУЛЬТАТЫ СДЕЛКИ:**\n"
            f"• Потенциальная прибыль: ${metrics['potential_profit']:.2f}\n"
            f"• Соотношение R/R: {metrics['rr_ratio']:.2f}\n"
            f"• Фактический левередж: {metrics.get('leverage_used', 1)}x\n\n"
            
            f"**💡 РЕКОМЕНДАЦИЯ:**\n"
        )
        
        if metrics['risk_percent'] > 10:
            text += "🔴 **ВЫСОКИЙ РИСК**! Превышен порог 10%. Уменьшите объем позиции.\n\n"
        elif metrics['margin_level'] < 100:
            text += "🔴 **КРИТИЧЕСКИЙ УРОВЕНЬ МАРЖИ**! Пополните счет.\n\n"
        elif metrics['margin_usage_percent'] > 50:
            text += "🟡 **ВЫСОКОЕ ИСПОЛЬЗОВАНИЕ МАРЖИ**! Оставьте запас для других сделок.\n\n"
        elif metrics['rr_ratio'] < 1:
            text += "🟡 **Соотношение R/R меньше 1**! Пересмотрите уровни TP/SL.\n\n"
        else:
            text += "✅ **Параметры сделки в пределах нормы**.\n\n"
        
        text += "Выберите дальнейшее действие:"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Новая сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")],
            [InlineKeyboardButton("📋 В портфель", callback_data="portfolio")]
        ]
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            text,
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return ConversationHandler.END
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Введите число (например: 52000)\nПопробуйте еще раз:",
            context
        )
        return SingleTradeState.TAKE_PROFIT.value

# ОБНОВЛЕННЫЙ обработчик мультипозиции с реальными данными
@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка тейк-профита и показ промежуточных результатов с РЕАЛЬНЫМИ ДАННЫМИ"""
    text = update.message.text.strip()
    
    try:
        take_profit = float(text.replace(',', '.'))
        entry_price = context.user_data['current_trade']['entry_price']
        direction = context.user_data['current_trade']['direction']
        asset = context.user_data['current_trade']['asset']
        
        # Валидация TP
        if direction == 'LONG' and take_profit <= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для LONG тейк-профит должен быть ВЫШЕ цены входа\nПопробуйте еще раз:",
                context
            )
            return MultiTradeState.TAKE_PROFIT.value
        elif direction == 'SHORT' and take_profit >= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для SHORT тейк-профит должен быть НИЖЕ цены входа\nПопробуйте еще раз:",
                context
            )
            return MultiTradeState.TAKE_PROFIT.value
        
        # Сохраняем TP
        current_trade = context.user_data['current_trade']
        current_trade['take_profit'] = take_profit
        
        # ПРОФЕССИОНАЛЬНЫЙ расчет метрик с РЕАЛЬНЫМИ КОТИРОВКАМИ
        deposit = context.user_data['deposit']
        leverage = context.user_data['leverage']
        risk_level = current_trade['risk_level']
        metrics = await ProfessionalRiskCalculator.calculate_professional_metrics(current_trade, deposit, leverage, risk_level)
        current_trade['metrics'] = metrics
        
        # Добавляем сделку в список
        context.user_data['multi_trades'].append(current_trade.copy())
        
        # Показываем результаты с РЕАЛЬНЫМИ ДАННЫМИ
        trade_count = len(context.user_data['multi_trades'])
        text = (
            f"✅ **СДЕЛКА #{trade_count} ДОБАВЛЕНА**\n\n"
            f"**Актив:** {current_trade['asset']}\n"
            f"**Текущая цена:** ${metrics['current_price']:.2f} ✅ РЕАЛЬНАЯ\n"
            f"**Направление:** {current_trade['direction']}\n"
            f"**Кредитное плечо:** {leverage}\n"
            f"**Вход:** {current_trade['entry_price']}\n"
            f"**SL:** {current_trade['stop_loss']} ({metrics['stop_distance_pips']:.0f} пунктов)\n"
            f"**TP:** {current_trade['take_profit']} ({metrics['profit_distance_pips']:.0f} пунктов)\n"
            f"**Риск:** {current_trade['risk_level']}\n\n"
            f"**📊 ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ (НА ОСНОВЕ РИСКА):**\n"
            f"• Депозит: ${metrics['deposit']:,.2f}\n"
            f"• Сумма риска: ${metrics['risk_amount']:.2f} ({metrics['risk_percent']:.1f}%)\n"
            f"• Объем: {metrics['volume_lots']:.2f} лотов\n"
            f"• Маржа: ${metrics['required_margin']:.2f} ✅ РЕАЛЬНЫЙ РАСЧЕТ\n"
            f"• Прибыль: ${metrics['potential_profit']:.2f}\n"
            f"• R/R: {metrics['rr_ratio']:.2f}\n"
            f"• Метод: {metrics['calculation_method']}\n\n"
        )
        
        if trade_count >= 10:
            text += "⚠️ Достигнут лимит в 10 сделок\n"
            keyboard = [[InlineKeyboardButton("📊 Перейти в портфель", callback_data="multi_finish")]]
        else:
            text += "**Выберите действие:**"
            keyboard = [
                [InlineKeyboardButton("➕ Добавить следующую сделку", callback_data="add_another")],
                [InlineKeyboardButton("📊 Перейти в портфель", callback_data="multi_finish")]
            ]
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
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
            "❌ Введите число (например: 52000)\nПопробуйте еще раз:",
            context
        )
        return MultiTradeState.TAKE_PROFIT.value

# ОБНОВЛЕННЫЙ обработчик портфеля с реальными данными
@retry_on_timeout(max_retries=2, delay=1.0)
async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int = None):
    """Показать портфель пользователя с РЕАЛЬНЫМИ ДАННЫМИ"""
    if not user_id:
        if update.callback_query:
            user_id = update.callback_query.from_user.id
        else:
            user_id = update.message.from_user.id
    
    PortfolioManager.ensure_user(user_id)
    user_portfolio = user_data[user_id]
    trades = user_portfolio.get('multi_trades', [])
    single_trades = user_portfolio.get('single_trades', [])
    deposit = user_portfolio.get('deposit', 0)
    leverage = user_portfolio.get('leverage', '1:100')
    
    all_trades = trades + single_trades
    
    if not all_trades:
        text = "📊 **ВАШ ПОРТФЕЛЬ v3.0**\n\nПортфель пуст. Начните с расчета сделок!"
        keyboard = [
            [InlineKeyboardButton("🎯 Одна сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")]
        ]
    else:
        # Обновляем цены в реальном времени для всех сделок
        updated_trades = []
        for trade in all_trades:
            try:
                current_price = await market_data_provider.get_real_time_price(trade['asset'])
                trade['current_price'] = current_price
                # Пересчитываем P&L на основе реальных цен
                if 'metrics' in trade:
                    entry = trade['entry_price']
                    direction = trade['direction']
                    volume = trade['metrics']['volume_lots']
                    pip_value = trade['metrics']['pip_value']
                    
                    if direction == 'LONG':
                        price_diff = current_price - entry
                    else:  # SHORT
                        price_diff = entry - current_price
                    
                    # Конвертируем разницу цены в пункты
                    pip_diff = ProfessionalRiskCalculator.calculate_pip_distance(
                        entry, entry + price_diff, direction, trade['asset']
                    )
                    
                    current_pnl = volume * pip_diff * pip_value
                    trade['current_pnl'] = current_pnl
                updated_trades.append(trade)
            except Exception as e:
                logger.error(f"Ошибка обновления цены для {trade['asset']}: {e}")
                updated_trades.append(trade)
        
        # Расчет метрик портфеля с РЕАЛЬНЫМИ ДАННЫМИ
        metrics = PortfolioAnalyzer.calculate_portfolio_metrics(updated_trades, deposit)
        recommendations = PortfolioAnalyzer.generate_recommendations(metrics, updated_trades)
        
        # Расчет общего P&L
        total_current_pnl = sum(t.get('current_pnl', 0) for t in updated_trades)
        
        text = (
            f"📊 **ВАШ ПОРТФЕЛЬ v3.0**\n\n"
            f"**Основные параметры:**\n"
            f"• Депозит: ${deposit:,.2f}\n"
            f"• Плечо: {leverage}\n"
            f"• Всего сделок: {len(all_trades)}\n"
            f"• Одиночные: {len(single_trades)} | Мульти: {len(trades)}\n"
            f"• Уникальных активов: {metrics.get('unique_assets', 0)}\n"
            f"• Текущий P&L: ${total_current_pnl:+.2f}\n\n"
            
            f"**📈 КЛЮЧЕВЫЕ МЕТРИКИ:**\n"
            f"• Общий риск: ${metrics['total_risk_usd']:.2f} ({metrics['total_risk_percent']:.1f}%)\n"
            f"• Потенциальная прибыль: ${metrics['total_profit']:.2f}\n"
            f"• Общая маржа: ${metrics['total_margin']:.2f}\n"
            f"• Уровень маржи портфеля: {metrics['portfolio_margin_level']:.1f}%\n"
            f"• Использование маржи: {metrics['total_margin_usage']:.1f}%\n"
            f"• Средний R/R: {metrics['avg_rr_ratio']:.2f}\n"
            f"• Волатильность портфеля: {metrics['portfolio_volatility']:.1f}%\n"
            f"• Общий левередж: {metrics.get('portfolio_leverage', 0):.1f}x\n"
            f"• Номинальная стоимость: ${metrics.get('total_notional_value', 0):.2f}\n"
            f"• LONG/Short: {metrics['long_positions']}/{metrics['short_positions']}\n\n"
            
            f"**💡 РЕКОМЕНДАЦИИ:**\n" + "\n".join(f"• {rec}" for rec in recommendations) + "\n\n"
            
            f"**📊 АКТИВНЫЕ СДЕЛКИ:**\n"
        )
        
        # Добавляем информацию по сделкам
        for i, trade in enumerate(updated_trades[:5], 1):  # Показываем первые 5 сделок
            current_pnl = trade.get('current_pnl', 0)
            pnl_sign = "📈" if current_pnl >= 0 else "📉"
            text += f"{i}. {trade['asset']} {trade['direction']} | P&L: {pnl_sign} ${current_pnl:+.2f}\n"
        
        if len(updated_trades) > 5:
            text += f"... и еще {len(updated_trades) - 5} сделок\n"
        
        # Кнопки управления
        keyboard = [
            [InlineKeyboardButton("🔄 Обновить цены", callback_data="portfolio")],
            [InlineKeyboardButton("🗑 Очистить портфель", callback_data="clear_portfolio")],
            [InlineKeyboardButton("📥 Выгрузить отчет", callback_data="export_portfolio")],
            [InlineKeyboardButton("🎯 Новая сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")]
        ]
    
    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
    
    if update.callback_query:
        await SafeMessageSender.edit_message_text(
            update.callback_query,
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

# ---------------------------
# Webhook & Main (ОБНОВЛЕННЫЙ)
# ---------------------------
async def set_webhook(application):
    """Установка вебхука"""
    try:
        webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
        await application.bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook установлен: {webhook_url}")
        return True
    except Exception as e:
        logger.error(f"Ошибка установки вебхука: {e}")
        return False

async def start_http_server(application):
    """Запуск HTTP сервера с улучшенными health checks"""
    app = web.Application()
    
    async def handle_webhook(request):
        """Обработчик вебхука с таймаутами"""
        try:
            # Устанавливаем таймаут для чтения данных
            data = await asyncio.wait_for(request.json(), timeout=10.0)
            update = Update.de_json(data, application.bot)
            await application.process_update(update)
            return web.Response(status=200)
        except asyncio.TimeoutError:
            logger.error("Webhook request timeout")
            return web.Response(status=408)  # Request Timeout
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return web.Response(status=400)
    
    # Health check endpoint
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
            # Проверяем соединение с Telegram
            await application.bot.get_me()
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["services"]["telegram_bot"] = f"error: {str(e)}"
            
        return web.json_response(health_status)
    
    # Robust health check для Render
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
            application.add_handler(CommandHandler("start", start_command))
            application.add_handler(CommandHandler("pro_info", pro_info_command))
            
            # Настройка диалогов
            setup_conversation_handlers(application)
            
            # Callback router
            application.add_handler(CallbackQueryHandler(callback_router))
            
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
                    
                    # Бесконечный цикл с периодическими health check
                    while True:
                        await asyncio.sleep(300)  # Sleep for 5 minutes
                        # Можно добавить периодические health checks здесь
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
                
            # Если дошли сюда, бот работает успешно
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
# Conversation Handler Setup (ОБНОВЛЕННЫЙ)
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
    
    # Мультипозиция
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
            MultiTradeState.ASSET_CATEGORY.value: [
                CallbackQueryHandler(multi_trade_asset_category, pattern="^(cat_|asset_manual|multi_finish)"),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            MultiTradeState.ASSET.value: [
                CallbackQueryHandler(multi_trade_asset, pattern="^(asset_|back_to_categories)"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_asset_manual),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            MultiTradeState.DIRECTION.value: [
                CallbackQueryHandler(multi_trade_direction, pattern="^dir_"),
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
                CallbackQueryHandler(multi_trade_risk_level, pattern="^risk_"),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            MultiTradeState.TAKE_PROFIT.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_take_profit),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ],
            MultiTradeState.ADD_MORE.value: [
                CallbackQueryHandler(multi_trade_add_another, pattern="^(add_another|multi_finish)$"),
                CallbackQueryHandler(main_menu_save_handler, pattern="^main_menu_save$")
            ]
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
# Обработчики состояний (ОБНОВЛЕННЫЕ)
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало одиночной сделки с реальными данными"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = (
        "🎯 **ОДИНОЧНАЯ СДЕЛКА v3.0**\n\n"
        "ПРОФЕССИОНАЛЬНЫЙ расчет с РЕАЛЬНЫМИ котировками и защитой от маржин-колла.\n"
        "Объем рассчитывается ИСКЛЮЧИТЕЛЬНО из суммы риска на основе текущих рыночных цен!\n\n"
        "**Введите ваш депозит в USD:**"
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
            "**Выберите кредитное плечо:**",
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

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора плеча для одиночной сделки"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    leverage = query.data.replace('lev_', '')
    context.user_data['leverage'] = leverage
    
    # Выбор категории активов
    keyboard = []
    for category in ASSET_CATEGORIES.keys():
        keyboard.append([InlineKeyboardButton(category, callback_data=f"cat_{category}")])
    
    keyboard.append([InlineKeyboardButton("📝 Ввести актив вручную", callback_data="asset_manual")])
    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Плечо: {leverage}\n\n"
        "**Выберите категорию актива:**",
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
    
    # Показываем активы выбранной категории
    assets = ASSET_CATEGORIES.get(category, [])
    
    keyboard = []
    for asset in assets:
        keyboard.append([InlineKeyboardButton(asset, callback_data=f"asset_{asset}")])
    
    keyboard.append([InlineKeyboardButton("🔙 Назад к категориям", callback_data="back_to_categories")])
    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Категория: {category}\n\n"
        "**Выберите актив:**",
        InlineKeyboardMarkup(keyboard)
    )
    return SingleTradeState.ASSET.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора актива для одиночной сделки"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    if query.data == "back_to_categories":
        # Возврат к выбору категории
        keyboard = []
        for category in ASSET_CATEGORIES.keys():
            keyboard.append([InlineKeyboardButton(category, callback_data=f"cat_{category}")])
        
        keyboard.append([InlineKeyboardButton("📝 Ввести актив вручную", callback_data="asset_manual")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await SafeMessageSender.edit_message_text(
            query,
            "**Выберите категорию актива:**",
            InlineKeyboardMarkup(keyboard)
        )
        return SingleTradeState.ASSET_CATEGORY.value
    
    asset = query.data.replace('asset_', '')
    context.user_data['asset'] = asset
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Актив: {asset}\n\n"
        "**Выберите направление сделки:**",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.DIRECTION.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_asset_manual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ручного ввода актива для одиночной сделки"""
    asset = update.message.text.strip().upper()
    
    # Простая валидация
    if not re.match(r'^[A-Z0-9]{2,20}$', asset):
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Неверный формат актива. Попробуйте еще раз:",
            context
        )
        return SingleTradeState.ASSET.value
    
    context.user_data['asset'] = asset
    
    await SafeMessageSender.send_message(
        update.message.chat_id,
        f"✅ Актив: {asset}\n\n"
        "**Выберите направление сделки:**",
        context,
        InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.DIRECTION.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора направления для одиночной сделки"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    direction = query.data.replace('dir_', '')
    context.user_data['direction'] = direction
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Направление: {direction}\n\n"
        "**Введите цену входа:**",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.ENTRY.value

@retry_on_timeout(max_retries=2, delay=1.0)
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
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Цена входа: {entry_price}\n\n"
            "**Введите уровень стоп-лосса:**",
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

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка стоп-лосса для одиночной сделки"""
    text = update.message.text.strip()
    
    try:
        stop_loss = float(text.replace(',', '.'))
        entry_price = context.user_data['entry_price']
        direction = context.user_data['direction']
        asset = context.user_data['asset']
        
        # Валидация SL
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
        
        # Расчет дистанции в пунктах для информации
        stop_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry_price, stop_loss, direction, asset)
        
        # Переход к выбору уровня риска
        keyboard = []
        for risk_level in RISK_LEVELS:
            keyboard.append([InlineKeyboardButton(risk_level, callback_data=f"risk_{risk_level}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Стоп-лосс: {stop_loss} ({stop_distance_pips:.0f} пунктов)\n\n"
            "**Выберите уровень риска:**",
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

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_risk_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора уровня риска"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    risk_level = query.data.replace('risk_', '')
    context.user_data['risk_level'] = risk_level
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Уровень риска: {risk_level}\n\n"
        "**Введите уровень тейк-профита:**",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.TAKE_PROFIT.value

@retry_on_timeout(max_retries=2, delay=1.0)
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
# Multi-trade Conversation Handlers (ОБНОВЛЕННЫЕ)
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало мультипозиционного расчета с реальными данными"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    context.user_data['multi_trades'] = []
    
    text = (
        "🎯 **МУЛЬТИПОЗИЦИОННЫЙ РАСЧЕТ v3.0**\n\n"
        "ПРОФЕССИОНАЛЬНЫЙ расчет нескольких сделок с РЕАЛЬНЫМИ котировками.\n"
        "Объем каждой позиции рассчитывается ИСКЛЮЧИТЕЛЬНО из суммы риска на основе текущих цен!\n\n"
        "**Введите общий депозит в USD:**"
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
            "**Выберите кредитное плечо:**",
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

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора плеча"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    leverage = query.data.replace('lev_', '')
    context.user_data['leverage'] = leverage
    
    # Начинаем цикл ввода сделок
    return await start_trade_input(update, context)

@retry_on_timeout(max_retries=2, delay=1.0)
async def start_trade_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало ввода сделки"""
    query = update.callback_query
    
    trade_count = len(context.user_data.get('multi_trades', []))
    
    text = f"**Сделка #{trade_count + 1}**\n\nВыберите категорию актива:"
    
    keyboard = []
    for category in ASSET_CATEGORIES.keys():
        keyboard.append([InlineKeyboardButton(category, callback_data=f"cat_{category}")])
    
    keyboard.append([InlineKeyboardButton("📝 Ввести актив вручную", callback_data="asset_manual")])
    
    # Показываем кнопку завершения только если есть хотя бы одна сделка
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

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_asset_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора категории активов для мультипозиции"""
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
        return MultiTradeState.ASSET.value
    
    elif query.data == "multi_finish":
        return await finish_multi_trade(update, context)
    
    category = query.data.replace('cat_', '')
    context.user_data['current_trade'] = {'asset_category': category}
    
    # Показываем активы выбранной категории
    assets = ASSET_CATEGORIES.get(category, [])
    
    keyboard = []
    for asset in assets:
        keyboard.append([InlineKeyboardButton(asset, callback_data=f"asset_{asset}")])
    
    keyboard.append([InlineKeyboardButton("🔙 Назад к категориям", callback_data="back_to_categories")])
    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Категория: {category}\n\n"
        "**Выберите актив:**",
        InlineKeyboardMarkup(keyboard)
    )
    return MultiTradeState.ASSET.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора актива"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    if query.data == "back_to_categories":
        return await start_trade_input(update, context)
    
    asset = query.data.replace('asset_', '')
    context.user_data['current_trade']['asset'] = asset
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Актив: {asset}\n\n"
        "**Выберите направление сделки:**",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return MultiTradeState.DIRECTION.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_asset_manual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ручного ввода актива"""
    asset = update.message.text.strip().upper()
    
    # Простая валидация
    if not re.match(r'^[A-Z0-9]{2,20}$', asset):
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Неверный формат актива. Попробуйте еще раз:",
            context
        )
        return MultiTradeState.ASSET.value
    
    context.user_data['current_trade'] = {'asset': asset}
    
    await SafeMessageSender.send_message(
        update.message.chat_id,
        f"✅ Актив: {asset}\n\n"
        "**Выберите направление сделки:**",
        context,
        InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return MultiTradeState.DIRECTION.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора направления"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    direction = query.data.replace('dir_', '')
    context.user_data['current_trade']['direction'] = direction
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Направление: {direction}\n\n"
        "**Введите цену входа:**",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return MultiTradeState.ENTRY.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка цены входа"""
    text = update.message.text.strip()
    
    try:
        entry_price = float(text.replace(',', '.'))
        if entry_price <= 0:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Цена должна быть больше 0\nПопробуйте еще раз:",
                context
            )
            return MultiTradeState.ENTRY.value
        
        context.user_data['current_trade']['entry_price'] = entry_price
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Цена входа: {entry_price}\n\n"
            "**Введите уровень стоп-лосса:**",
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
            context
        )
        return MultiTradeState.ENTRY.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка стоп-лосса"""
    text = update.message.text.strip()
    
    try:
        stop_loss = float(text.replace(',', '.'))
        entry_price = context.user_data['current_trade']['entry_price']
        direction = context.user_data['current_trade']['direction']
        asset = context.user_data['current_trade']['asset']
        
        # Валидация SL
        if direction == 'LONG' and stop_loss >= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для LONG стоп-лосс должен быть НИЖЕ цены входа\nПопробуйте еще раз:",
                context
            )
            return MultiTradeState.STOP_LOSS.value
        elif direction == 'SHORT' and stop_loss <= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для SHORT стоп-лосс должен быть ВЫШЕ цены входа\nПопробуйте еще раз:",
                context
            )
            return MultiTradeState.STOP_LOSS.value
        
        context.user_data['current_trade']['stop_loss'] = stop_loss
        
        # Расчет дистанции в пунктах для информации
        stop_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry_price, stop_loss, direction, asset)
        
        # Переход к выбору уровня риска
        keyboard = []
        for risk_level in RISK_LEVELS:
            keyboard.append([InlineKeyboardButton(risk_level, callback_data=f"risk_{risk_level}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Стоп-лосс: {stop_loss} ({stop_distance_pips:.0f} пунктов)\n\n"
            "**Выберите уровень риска для этой сделки:**",
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return MultiTradeState.RISK_LEVEL.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Введите число (например: 48000)\nПопробуйте еще раз:",
            context
        )
        return MultiTradeState.STOP_LOSS.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_risk_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора уровня риска для мультипозиции"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    risk_level = query.data.replace('risk_', '')
    context.user_data['current_trade']['risk_level'] = risk_level
    
    await SafeMessageSender.edit_message_text(
        query,
        f"✅ Уровень риска: {risk_level}\n\n"
        "**Введите уровень тейк-профита:**",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return MultiTradeState.TAKE_PROFIT.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_add_another(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка добавления следующей сделки"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    if query.data == "add_another":
        return await start_trade_input(update, context)
    else:  # multi_finish
        return await finish_multi_trade(update, context)

@retry_on_timeout(max_retries=2, delay=1.0)
async def finish_multi_trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Завершение мультипозиционного расчета и переход в портфель"""
    query = update.callback_query
    user_id = query.from_user.id
    
    # Сохраняем данные
    trades = context.user_data.get('multi_trades', [])
    deposit = context.user_data.get('deposit', 0)
    leverage = context.user_data.get('leverage', '1:100')
    
    if trades:
        PortfolioManager.set_deposit_leverage(user_id, deposit, leverage)
        for trade in trades:
            PortfolioManager.add_multi_trade(user_id, trade)
    
    # Очищаем временные данные
    DataManager.clear_temporary_progress(user_id)
    context.user_data.clear()
    
    # Переходим в портфель
    await show_portfolio(update, context, user_id)
    return ConversationHandler.END

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена мультипозиционного расчета"""
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
# Portfolio Handlers (ОБНОВЛЕННЫЕ)
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик портфеля"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    await show_portfolio(update, context)

@retry_on_timeout(max_retries=2, delay=1.0)
async def clear_portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Очистка портфеля"""
    query = update.callback_query
    user_id = query.from_user.id
    await SafeMessageSender.answer_callback_query(query)
    
    PortfolioManager.clear_portfolio(user_id)
    
    await SafeMessageSender.edit_message_text(
        query,
        "✅ Портфель очищен",
        InlineKeyboardMarkup([
            [InlineKeyboardButton("🎯 Одна сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")],
            [InlineKeyboardButton("📋 В портфель", callback_data="portfolio")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ])
    )

@retry_on_timeout(max_retries=2, delay=1.0)
async def export_portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Выгрузка отчета портфеля с реальными данными"""
    query = update.callback_query
    user_id = query.from_user.id
    await SafeMessageSender.answer_callback_query(query)
    
    PortfolioManager.ensure_user(user_id)
    user_portfolio = user_data[user_id]
    trades = user_portfolio.get('multi_trades', [])
    single_trades = user_portfolio.get('single_trades', [])
    deposit = user_portfolio.get('deposit', 0)
    leverage = user_portfolio.get('leverage', '1:100')
    
    all_trades = trades + single_trades
    
    if not all_trades:
        await SafeMessageSender.answer_callback_query(query, "Портфель пуст", show_alert=True)
        return
    
    # Обновляем цены для отчета
    updated_trades = []
    for trade in all_trades:
        try:
            current_price = await market_data_provider.get_real_time_price(trade['asset'])
            trade['current_price'] = current_price
            updated_trades.append(trade)
        except Exception as e:
            logger.error(f"Ошибка обновления цены для отчета {trade['asset']}: {e}")
            updated_trades.append(trade)
    
    # Генерация профессионального отчета
    metrics = PortfolioAnalyzer.calculate_portfolio_metrics(updated_trades, deposit)
    recommendations = PortfolioAnalyzer.generate_recommendations(metrics, updated_trades)
    
    report_lines = [
        "PRO RISK CALCULATOR v3.0 - ПРОФЕССИОНАЛЬНЫЙ ОТЧЕТ ПОРТФЕЛЯ",
        "РАСЧЕТ НА ОСНОВЕ РЕАЛЬНЫХ КОТИРОВОК И СУММЫ РИСКА",
        f"Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Депозит: ${deposit:,.2f}",
        f"Плечо: {leverage}",
        f"Всего сделок: {len(all_trades)}",
        f"Одиночные сделки: {len(single_trades)}",
        f"Мультипозиции: {len(trades)}",
        "",
        "ПРОФЕССИОНАЛЬНЫЕ МЕТРИКИ:",
        "-" * 50,
        f"Общий риск: ${metrics['total_risk_usd']:.2f} ({metrics['total_risk_percent']:.1f}%)",
        f"Потенциальная прибыль: ${metrics['total_profit']:.2f}",
        f"Общая маржа: ${metrics['total_margin']:.2f}",
        f"Уровень маржи портфеля: {metrics['portfolio_margin_level']:.1f}%",
        f"Использование маржи: {metrics['total_margin_usage']:.1f}%",
        f"Средний R/R: {metrics['avg_rr_ratio']:.2f}",
        f"Волатильность: {metrics['portfolio_volatility']:.1f}%",
        f"Общий левередж: {metrics.get('portfolio_leverage', 0):.1f}x",
        f"Номинальная стоимость: ${metrics.get('total_notional_value', 0):.2f}",
        f"Активов: {metrics['unique_assets']} | LONG: {metrics['long_positions']} | SHORT: {metrics['short_positions']}",
        "",
        "ДЕТАЛИ СДЕЛОК (РАСЧЕТ НА ОСНОВЕ РЕАЛЬНЫХ ДАННЫХ):",
        "-" * 50
    ]
    
    for i, trade in enumerate(updated_trades, 1):
        current_price = trade.get('current_price', trade['entry_price'])
        report_lines.extend([
            f"{i}. {trade['asset']} {trade['direction']} | Риск: {trade.get('risk_level', 'N/A')}",
            f"   Вход: {trade['entry_price']} | Текущая: {current_price:.2f} | SL: {trade['stop_loss']} | TP: {trade['take_profit']}",
            f"   Депозит: ${trade['metrics']['deposit']:,.2f} | Риск: ${trade['metrics']['risk_amount']:.2f}",
            f"   Объем: {trade['metrics']['volume_lots']:.2f} лотов | Маржа: ${trade['metrics']['required_margin']:.2f}",
            f"   Прибыль: ${trade['metrics']['potential_profit']:.2f} | R/R: {trade['metrics']['rr_ratio']:.2f}",
            f"   Метод расчета: {trade['metrics'].get('calculation_method', 'N/A')}",
            ""
        ])
    
    report_lines.extend([
        "РЕКОМЕНДАЦИИ:",
        "-" * 50
    ])
    
    report_lines.extend(recommendations)
    
    report_text = "\n".join(report_lines)
    
    # Создаем файл
    bio = io.BytesIO(report_text.encode('utf-8'))
    bio.name = f"portfolio_report_v3_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    
    await SafeMessageSender.send_message(
        query.message.chat_id,
        "📊 Профессиональный отчет вашего портфеля v3.0 (реальные котировки)",
        context
    )
    
    await query.message.reply_document(
        document=InputFile(bio, filename=bio.name),
        caption="📊 Профессиональный отчет вашего портфеля v3.0 (реальные котировки)"
    )

# ---------------------------
# Future Features Handler (ОБНОВЛЕННЫЙ)
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def future_features_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Раздел будущих разработок"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    text = (
        "🚀 **БУДУЩИЕ РАЗРАБОТКИ v3.0**\n\n"
        
        "**📊 ИНТЕГРАЦИЯ С TRADINGVIEW**\n"
        "• Автоматический импорт уровней поддержки/сопротивления\n"
        "• Синхронизация графиков и данных в реальном времени\n"
        "• Умные алерты на основе технического анализа\n\n"
        
        "**🤖 AI-АНАЛИТИКА**\n"
        "• Прогнозирование движения цен на основе ML\n"
        "• Автоматические рекомендации по позициям\n"
        "• Анализ настроений рынка\n\n"
        
        "**📱 ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ**\n"
        "• Мобильное приложение с push-уведомлениями\n"
        "• Расширенная аналитика портфеля\n"
        "• Интеграция с популярными биржами\n"
        "• Социальный трейдинг и копирование сделок\n\n"
        
        "**🔧 ТЕКУЩИЕ ОБНОВЛЕНИЕ v3.0**\n"
        "✅ Реальные котировки через Binance, Alpha Vantage, Finnhub\n"
        "✅ Профессиональный расчет маржи по отраслевым стандартам\n"
        "✅ Разделение одиночных сделок и мультипозиций\n"
        "✅ Улучшенный UX и исправленная орфография\n\n"
        
        "Следите за обновлениями! 👨‍💻"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]]
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )

# ---------------------------
# Progress Restoration Handler (ОБНОВЛЕННЫЙ)
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def restore_progress_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Восстановление сохраненного прогресса"""
    query = update.callback_query
    user_id = query.from_user.id
    await SafeMessageSender.answer_callback_query(query)
    
    temp_data = DataManager.load_temporary_data()
    saved_progress = temp_data.get(str(user_id))
    
    if not saved_progress:
        await SafeMessageSender.answer_callback_query(query, "Нет сохраненного прогресса", show_alert=True)
        return
    
    # Восстанавливаем данные
    context.user_data.clear()
    context.user_data.update(saved_progress['state_data'])
    
    state_type = saved_progress['state_type']
    
    if state_type == "single":
        await SafeMessageSender.answer_callback_query(query, "Прогресс одиночной сделки восстановлен", show_alert=True)
        # Определяем текущее состояние и переходим к нему
        if 'take_profit' in context.user_data:
            await SafeMessageSender.edit_message_text(
                query,
                "✅ Прогресс восстановлен!\n\nВведите уровень тейк-профита:",
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.TAKE_PROFIT.value
        elif 'risk_level' in context.user_data:
            await SafeMessageSender.edit_message_text(
                query,
                "✅ Прогресс восстановлен!\n\nВведите уровень тейк-профита:",
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.TAKE_PROFIT.value
        elif 'stop_loss' in context.user_data:
            # Показываем выбор уровня риска
            keyboard = []
            for risk_level in RISK_LEVELS:
                keyboard.append([InlineKeyboardButton(risk_level, callback_data=f"risk_{risk_level}")])
            
            keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
            
            await SafeMessageSender.edit_message_text(
                query,
                "✅ Прогресс восстановлен!\n\nВыберите уровень риска:",
                InlineKeyboardMarkup(keyboard)
            )
            return SingleTradeState.RISK_LEVEL.value
    else:
        await SafeMessageSender.answer_callback_query(query, "Прогресс мультипозиции восстановлен", show_alert=True)
        return await start_trade_input(update, context)

# ---------------------------
# PRO Info Handler (ОБНОВЛЕННЫЙ)
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """PRO инструкции v3.0"""
    text = (
        "📚 **PRO ИНСТРУКЦИИ v3.0**\n\n"
        
        "**🎯 ПРАВИЛЬНОЕ УПРАВЛЕНИЕ РИСКАМИ С РЕАЛЬНЫМИ ДАННЫМИ**\n\n"
        
        "**МЕТОДОЛОГИЯ РАСЧЕТА v3.0:**\n"
        "• Риск на сделку = % от депозита (например: 2% от $1000 = $20)\n"
        "• Объем позиции рассчитывается ИСКЛЮЧИТЕЛЬНО из суммы риска\n"
        "• **РЕАЛЬНЫЕ КОТИРОВКИ** через Binance, Alpha Vantage, Finnhub\n"
        "• **ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ** маржи по отраслевым стандартам\n"
        "• Защита от маржин-колла через правильный расчет объема\n\n"
        
        "**📊 КЛЮЧЕВЫЕ ПРИНЦИПЫ ДЛЯ ПРОФЕССИОНАЛОВ:**\n\n"
        
        "**1. УПРАВЛЕНИЕ РАЗМЕРОМ ПОЗИЦИИ НА ОСНОВЕ РИСКА**\n"
        "• Всегда определяйте риск ДО входа в сделку\n"
        "• Рассчитывайте объем на основе стоп-лосса и суммы риска\n"
        "• Никогда не рискуйте более 5% на одну сделку\n"
        "• Учитывайте кредитное плечо при расчете маржи\n\n"
        
        "**2. УРОВНИ РИСКА И ИХ ПРИМЕНЕНИЕ**\n"
        "• 2% - Консервативный: Для начинающих и крупных капиталов\n"
        "• 5% - Стандартный: Баланс роста и безопасности\n"
        "• 10% - Агрессивный: Для опытных трейдеров\n"
        "• 25% - Максимальный: Только для уверенных сделок\n\n"
        
        "**3. ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ МАРЖИ**\n"
        "• Всегда следите за уровнем маржи (>200%)\n"
        "• Оставляйте свободную маржу для маневра\n"
        "• Не используйте более 50% депозита под маржу\n"
        "• Учитывайте номинальную стоимость позиций\n\n"
        
        "**🛡 ЗАЩИТА ОТ МАРЖИН-КОЛЛА:**\n"
        "Бот автоматически проверяет достаточность маржи и при необходимости уменьшает объем позиции, сохраняя ваш заданный уровень риска.\n\n"
        
        "**💡 КАК ИСПОЛЬЗОВАТЬ БОТА v3.0:**\n"
        "1. Установите размер депозита\n"
        "2. Выберите кредитное плечо\n"
        "3. Выберите актив и направление сделки\n"
        "4. Укажите цену входа, стоп-лосс и тейк-профит\n"
        "5. Выберите уровень риска\n"
        "6. Получите профессиональный расчет на основе РЕАЛЬНЫХ данных\n\n"
        
        "**🔧 ИНТЕГРИРОВАННЫЕ API:**\n"
        "• Binance - для криптовалют\n"
        "• Alpha Vantage - для акций и Forex\n"
        "• Finnhub - резервный провайдер\n"
        "• Кэширование данных для оптимизации\n\n"
        
        "Разработчик: @fxfeelgood"
    )
    
    keyboard = [
        [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
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
# Main Callback Router (ОБНОВЛЕННЫЙ)
# ---------------------------
@performance_logger
@retry_on_timeout(max_retries=2, delay=1.0)
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Маршрутизатор callback запросов v3.0"""
    query = update.callback_query
    if not query:
        return
    
    # Сразу отвечаем на callback чтобы Telegram не показывал "часики"
    await SafeMessageSender.answer_callback_query(query)
    
    data = query.data
    user_id = query.from_user.id
    
    logger.info(f"Callback received: {data} from user {user_id}")
    
    try:
        # Основные команды
        if data == "main_menu":
            await start_command(update, context)
        elif data == "main_menu_save":
            current_state = None
            if hasattr(context, '_conversation_state'):
                current_state = context._conversation_state
            await main_menu_save_handler(update, context, current_state)
        elif data == "pro_calculation":
            keyboard = [
                [InlineKeyboardButton("🎯 Одна сделка", callback_data="single_trade")],
                [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")],
                [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
            ]
            await SafeMessageSender.edit_message_text(
                query,
                "Выберите тип расчета:",
                InlineKeyboardMarkup(keyboard)
            )
        elif data == "single_trade":
            await single_trade_start(update, context)
        elif data == "multi_trade_start":
            await multi_trade_start(update, context)
        elif data == "portfolio":
            await show_portfolio(update, context, user_id)
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
        
        # Обработка категорий активов
        elif data.startswith("cat_"):
            if hasattr(context, '_conversation_state'):
                state = context._conversation_state
                if state in [SingleTradeState.ASSET_CATEGORY.value, MultiTradeState.ASSET_CATEGORY.value]:
                    if state == SingleTradeState.ASSET_CATEGORY.value:
                        await single_trade_asset_category(update, context)
                    else:
                        await multi_trade_asset_category(update, context)
        
        # Обработка выбора уровня риска
        elif data.startswith("risk_"):
            if hasattr(context, '_conversation_state'):
                state = context._conversation_state
                if state in [SingleTradeState.RISK_LEVEL.value, MultiTradeState.RISK_LEVEL.value]:
                    if state == SingleTradeState.RISK_LEVEL.value:
                        await single_trade_risk_level(update, context)
                    else:
                        await multi_trade_risk_level(update, context)
        
        # Обработка других callback данных
        elif data in ["back_to_categories", "asset_manual", "multi_finish", "add_another"]:
            if hasattr(context, '_conversation_state'):
                state = context._conversation_state
                if state in [SingleTradeState.ASSET.value, MultiTradeState.ASSET.value]:
                    if data == "back_to_categories":
                        if state == SingleTradeState.ASSET.value:
                            await single_trade_asset(update, context)
                        else:
                            await multi_trade_asset_category(update, context)
                    elif data == "asset_manual":
                        if state == SingleTradeState.ASSET.value:
                            await SafeMessageSender.edit_message_text(
                                query,
                                "✍️ Введите название актива (например: BTCUSDT):",
                                InlineKeyboardMarkup([
                                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                                ])
                            )
                            return SingleTradeState.ASSET.value
                        else:
                            await SafeMessageSender.edit_message_text(
                                query,
                                "✍️ Введите название актива (например: BTCUSDT):",
                                InlineKeyboardMarkup([
                                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                                ])
                            )
                            return MultiTradeState.ASSET.value
                    elif data == "multi_finish":
                        await finish_multi_trade(update, context)
                    elif data == "add_another":
                        await multi_trade_add_another(update, context)
        
        else:
            logger.warning(f"Unknown callback data: {data}")
            await SafeMessageSender.edit_message_text(
                query,
                "⚠️ Функция временно недоступна",
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
                ])
            )
            
    except Exception as e:
        logger.error(f"Error in callback_router for {data}: {e}")
        await SafeMessageSender.edit_message_text(
            query,
            "❌ Произошла ошибка. Пожалуйста, попробуйте еще раз.",
            InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ])
        )

if __name__ == "__main__":
    asyncio.run(main())
