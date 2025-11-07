# bot.py — PRO Risk Calculator v3.0 | ENTERPRISE EDITION - ИСПРАВЛЕННАЯ ВЕРСИЯ
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
# Критически важный глобальный error handler для Application
# ---------------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)
    try:
        if update and hasattr(update, 'effective_user') and update.effective_user:
            await context.bot.send_message(
                chat_id=update.effective_user.id,
                text="Произошла критическая ошибка, попробуйте заново. Если ошибка повторяется, обратитесь к разработчику.",
            )
    except Exception as e:
        logger.error(f"Failed to send error message: {e}")

# ---------------------------
# RobustApplicationBuilder – создание приложения с error handler
# ---------------------------
class RobustApplicationBuilder:
    """Строитель приложения с улучшенной обработкой ошибок"""
    @staticmethod
    def create_application(token: str) -> Application:
        request = telegram.request.HTTPXRequest(connection_pool_size=8)
        application = (
            Application.builder()
            .token(token)
            .request(request)
            .build()
        )
        # Критическое добавление глобального error handler
        application.add_error_handler(error_handler)
        return application

# Retry Decorator для обработки таймаутов
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
# Safe Message Sender - ПРОФЕССИОНАЛЬНАЯ ОБРАБОТКА HTML
# ---------------------------
class SafeMessageSender:
    """Безопасная отправка сообщений с обработкой ошибок, защищённая от некорректных HTML/Markdown тегов"""
    @staticmethod
    @retry_on_timeout(max_retries=3, delay=1.0)
    async def send_message(
        chat_id: int,
        text: str,
        context: ContextTypes.DEFAULT_TYPE = None,
        reply_markup: InlineKeyboardMarkup = None,
        parse_mode: str = 'HTML'
    ) -> bool:
        # Критическое исправление — защита HTML
        try:
            text = SafeMessageSender.sanitize_html(text, parse_mode)
            if context and hasattr(context, 'bot'):
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode
                )
            else:
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
        try:
            text = SafeMessageSender.sanitize_html(text, parse_mode)
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
    def sanitize_html(text: str, parse_mode: str = 'HTML') -> str:
        if parse_mode == 'HTML':
            # Исправляем незакрытые или неправильные HTML-теги
            text = re.sub(r'<[^>]+>', lambda m: html.escape(m.group(0)), text)
            # Дополнительно — можно добавить строгий html.escape
            text = html.escape(text, quote=True)
        # Для Markdown можно добавить markdown.escape или аналогично.
        return text

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

# ============================================================================
# КЛАССЫ ДАННЫХ И СОСТОЯНИЯ
# ============================================================================

class TradeState(Enum):
    """Состояния тейдов"""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"

class ConvergenceState(Enum):
    """Состояния заказа"""
    SELECTING_PAIR = range(1)[0]
    ENTERING_ENTRY = range(2)[0]
    ENTERING_EXIT = range(3)[0]
    ENTERING_STOP = range(4)[0]
    ENTERING_VOLUME = range(5)[0]
    ENTERING_LEVERAGE = range(6)[0]
    ENTERING_COMMISSION = range(7)[0]
    CONFIRMING = range(8)[0]

SELECT_OPERATION = 100
SELECT_TRADE_MODE = 101
SELECT_PAIR = 102
SELECT_TIMEFRAME = 103
ENTER_ENTRY_PRICE = 104
ENTER_EXIT_PRICE = 105
ENTER_STOP_LOSS = 106
ENTER_VOLUME = 107
ENTER_LEVERAGE = 108
ENTER_COMMISSION = 109
CONFIRM_TRADE = 110

# ============================================================================
# DATA CLASSES И СТРУКТУРЫ ДАННЫХ
# ============================================================================

class TradeMetrics:
    """Метрики тейда"""
    def __init__(self, trade_data: Dict[str, Any]):
        self.entry_price = float(trade_data.get('entry_price', 0))
        self.exit_price = float(trade_data.get('exit_price', 0))
        self.stop_loss = float(trade_data.get('stop_loss', 0))
        self.volume = float(trade_data.get('volume', 0))
        self.leverage = float(trade_data.get('leverage', 1))
        self.commission = float(trade_data.get('commission', 0))
        self.asset = trade_data.get('asset', 'UNKNOWN')
        self.trade_type = trade_data.get('trade_type', 'unknown')
        self.timestamp = trade_data.get('timestamp', datetime.now().isoformat())
        self.currency = trade_data.get('currency', 'USD')

    @property
    def profit_loss(self) -> float:
        if self.trade_type == 'long':
            return (self.exit_price - self.entry_price) * self.volume * self.leverage - (self.commission * 2)
        else:  # short
            return (self.entry_price - self.exit_price) * self.volume * self.leverage - (self.commission * 2)

    @property
    def profit_loss_percent(self) -> float:
        cost = self.entry_price * self.volume
        if cost == 0:
            return 0
        return (self.profit_loss / cost) * 100

    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.stop_loss) * self.volume * self.leverage

    @property
    def reward_amount(self) -> float:
        if self.trade_type == 'long':
            return abs(self.exit_price - self.entry_price) * self.volume * self.leverage
        else:
            return abs(self.entry_price - self.exit_price) * self.volume * self.leverage

    @property
    def risk_reward_ratio(self) -> float:
        if self.risk_amount == 0:
            return 0
        return self.reward_amount / self.risk_amount

    @property
    def max_loss(self) -> float:
        return self.risk_amount + (self.commission * 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_loss': self.stop_loss,
            'volume': self.volume,
            'leverage': self.leverage,
            'commission': self.commission,
            'asset': self.asset,
            'trade_type': self.trade_type,
            'timestamp': self.timestamp,
            'currency': self.currency,
            'profit_loss': self.profit_loss,
            'profit_loss_percent': self.profit_loss_percent,
            'risk_amount': self.risk_amount,
            'reward_amount': self.reward_amount,
            'risk_reward_ratio': self.risk_reward_ratio,
        }

# ============================================================================
# МЕНЕДЖЕР ДАННЫХ И ПОРТФОЛИО
# ============================================================================

class PortfolioManager:
    """Управление портфолио и аналитика по тейдам"""
    
    def __init__(self):
        self.trades: Dict[str, List[TradeMetrics]] = {}
        self.portfolio_cache = cachetools.TTLCache(maxsize=1000, ttl=300)

    def add_trade(self, user_id: int, trade_data: Dict[str, Any]) -> TradeMetrics:
        key = str(user_id)
        metrics = TradeMetrics(trade_data)
        if key not in self.trades:
            self.trades[key] = []
        self.trades[key].append(metrics)
        self._invalidate_cache(user_id)
        return metrics

    def get_user_trades(self, user_id: int) -> List[TradeMetrics]:
        return self.trades.get(str(user_id), [])

    def calculate_portfolio_metrics(self, user_id: int) -> Dict[str, Any]:
        cache_key = f"portfolio_{user_id}"
        if cache_key in self.portfolio_cache:
            return self.portfolio_cache[cache_key]

        trades = self.get_user_trades(user_id)
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit_loss': 0,
                'total_profit_loss_percent': 0,
                'win_rate': 0,
                'average_rr_ratio': 0,
                'total_risk': 0,
                'total_margin_usage': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'consecutive_wins': 0,
                'consecutive_losses': 0,
            }

        total_profit_loss = sum(trade.profit_loss for trade in trades)
        winning_trades = len([t for t in trades if t.profit_loss > 0])
        losing_trades = len([t for t in trades if t.profit_loss < 0])
        win_rate = (winning_trades / len(trades) * 100) if trades else 0

        total_cost = sum(trade.entry_price * trade.volume for trade in trades)
        total_profit_loss_percent = (total_profit_loss / total_cost * 100) if total_cost > 0 else 0

        rr_ratios = [t.risk_reward_ratio for t in trades if t.risk_reward_ratio > 0]
        average_rr = sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0

        total_risk = sum(trade.max_loss for trade in trades)
        total_margin_usage = sum(trade.volume * trade.leverage for trade in trades)

        largest_win = max([t.profit_loss for t in trades], default=0)
        largest_loss = min([t.profit_loss for t in trades], default=0)

        consecutive_wins = self._calculate_consecutive(trades, True)
        consecutive_losses = self._calculate_consecutive(trades, False)

        result = {
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_profit_loss': total_profit_loss,
            'total_profit_loss_percent': total_profit_loss_percent,
            'win_rate': win_rate,
            'average_rr_ratio': average_rr,
            'total_risk': total_risk,
            'total_margin_usage': total_margin_usage,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
        }

        self.portfolio_cache[cache_key] = result
        return result

    def _calculate_consecutive(self, trades: List[TradeMetrics], winning: bool) -> int:
        consecutive = 0
        for trade in reversed(trades):
            if (winning and trade.profit_loss > 0) or (not winning and trade.profit_loss < 0):
                consecutive += 1
            else:
                break
        return consecutive

    def _invalidate_cache(self, user_id: int):
        cache_key = f"portfolio_{user_id}"
        if cache_key in self.portfolio_cache:
            del self.portfolio_cache[cache_key]

    def export_trades_csv(self, user_id: int) -> str:
        trades = self.get_user_trades(user_id)
        if not trades:
            return "No trades to export"

        output = io.StringIO()
        output.write("Asset,Type,Entry Price,Exit Price,Stop Loss,Volume,Leverage,Commission,P&L,P&L %,RR Ratio,Risk,Timestamp\n")
        for trade in trades:
            output.write(
                f"{trade.asset},{trade.trade_type},{trade.entry_price},{trade.exit_price},"
                f"{trade.stop_loss},{trade.volume},{trade.leverage},{trade.commission},"
                f"{trade.profit_loss:.2f},{trade.profit_loss_percent:.2f}%,"
                f"{trade.risk_reward_ratio:.2f},{trade.risk_amount:.2f},{trade.timestamp}\n"
            )
        return output.getvalue()

# Глобальный менеджер портфолио
portfolio_manager = PortfolioManager()

# ============================================================================
# MARKET DATA PROVIDER - REAL-TIME ДАННЫЕ
# ============================================================================

class MarketDataProvider:
    """Провайдер рыночных данных"""
    
    def __init__(self):
        self.cache = cachetools.TTLCache(maxsize=500, ttl=60)
        self.session = None

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()

    async def get_crypto_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        cache_key = f"crypto_{symbol}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            await self.init_session()
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if symbol.lower() in data:
                        result = {
                            'price': data[symbol.lower()].get('usd', 0),
                            'market_cap': data[symbol.lower()].get('usd_market_cap', 0),
                            'volume_24h': data[symbol.lower()].get('usd_24h_vol', 0),
                            'change_24h': data[symbol.lower()].get('usd_24h_change', 0),
                        }
                        self.cache[cache_key] = result
                        return result
        except Exception as e:
            logger.error(f"Error fetching crypto price for {symbol}: {e}")
        return None

    async def get_forex_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        cache_key = f"forex_{from_currency}_{to_currency}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            await self.init_session()
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    rate = data['rates'].get(to_currency, 0)
                    self.cache[cache_key] = rate
                    return rate
        except Exception as e:
            logger.error(f"Error fetching forex rate {from_currency}/{to_currency}: {e}")
        return None

market_data_provider = MarketDataProvider()

# ============================================================================
# ANALYTICS ENGINE - ПРОФЕССИОНАЛЬНЫЙ АНАЛИЗ
# ============================================================================

class AdvancedAnalyticsEngine:
    """Продвинутый аналитический движок для профессиональной оценки портфолио"""
    
    @staticmethod
    def calculate_sharpe_ratio(trades: List[TradeMetrics], risk_free_rate: float = 0.02) -> float:
        if not trades or len(trades) < 2:
            return 0
        returns = [trade.profit_loss_percent / 100 for trade in trades]
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        if std_dev == 0:
            return 0
        return (mean_return - risk_free_rate) / std_dev

    @staticmethod
    def calculate_sortino_ratio(trades: List[TradeMetrics], risk_free_rate: float = 0.02) -> float:
        if not trades or len(trades) < 2:
            return 0
        returns = [trade.profit_loss_percent / 100 for trade in trades]
        mean_return = sum(returns) / len(returns)
        downside_returns = [r for r in returns if r < mean_return]
        if not downside_returns:
            return 0
        downside_variance = sum((r - mean_return) ** 2 for r in downside_returns) / len(downside_returns)
        downside_std_dev = downside_variance ** 0.5
        if downside_std_dev == 0:
            return 0
        return (mean_return - risk_free_rate) / downside_std_dev

    @staticmethod
    def calculate_calmar_ratio(trades: List[TradeMetrics]) -> float:
        if not trades:
            return 0
        total_return = sum(trade.profit_loss_percent for trade in trades)
        max_drawdown = AdvancedAnalyticsEngine.calculate_max_drawdown(trades)
        if max_drawdown == 0:
            return 0
        return total_return / abs(max_drawdown)

    @staticmethod
    def calculate_max_drawdown(trades: List[TradeMetrics]) -> float:
        if not trades:
            return 0
        peak = 0
        max_drawdown = 0
        cumulative_profit = 0
        for trade in trades:
            cumulative_profit += trade.profit_loss
            if cumulative_profit > peak:
                peak = cumulative_profit
            drawdown = peak - cumulative_profit
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return -max_drawdown if max_drawdown > 0 else 0

    @staticmethod
    def generate_recommendations(trades: List[TradeMetrics], metrics: Dict[str, Any]) -> List[str]:
        recommendations = []

        if metrics['total_trades'] < 10:
            recommendations.append("📊 Недостаточно данных для полного анализа. Рекомендуется провести минимум 10 тейдов.")

        if metrics['win_rate'] > 0 and metrics['win_rate'] < 40:
            recommendations.append("⚠️ Win Rate ниже 40%. Необходимо пересмотреть стратегию входа.")

        if metrics['average_rr_ratio'] > 0 and metrics['average_rr_ratio'] < 1.5:
            recommendations.append("⚠️ Средний RR Ratio < 1.5. Рекомендуется расширить Take Profit зоны.")

        if metrics['total_profit_loss'] < 0:
            recommendations.append("🔴 Портфолио в минусе. Требуется немедленный пересмотр торговой стратегии.")

        if metrics['total_margin_usage'] > 0 and metrics['total_margin_usage'] * 100 / metrics['total_risk'] > 80:
            recommendations.append("⚠️ Маржа более 80%. Снизьте leverage или объём позиций.")

        high_vol_assets = [t for t in trades if t.volume > 100]
        if len(high_vol_assets) > 2:
            recommendations.append("💡 Обнаружены позиции с высоким объёмом. Проверьте диверсификацию.")

        low_rr_trades = [t for t in trades if t.risk_reward_ratio > 0 and t.risk_reward_ratio < 1]
        if len(low_rr_trades) > 0:
            recommendations.append(f"💡 {len(low_rr_trades)} тейдов с RR < 1.0. Пересмотрите точки выхода.")

        if metrics['consecutive_losses'] > 3:
            recommendations.append(f"⚠️ Серия из {metrics['consecutive_losses']} убыточных тейдов. Сделайте перерыв для анализа.")

        if len(recommendations) == 0:
            recommendations.append("✅ Портфолио в хорошем состоянии. Продолжайте соблюдать вашу стратегию.")

        return recommendations

# ============================================================================
# CONVERSATION HANDLERS - ОСНОВНЫЕ ОБРАБОТЧИКИ ДИАЛОГА
# ============================================================================

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Главный старт хендлер"""
    user = update.effective_user
    welcome_text = (
        f"🤖 <b>Добро пожаловать в PRO Risk Calculator v3.0!</b>\n\n"
        f"👋 Привет, {user.first_name}!\n\n"
        f"Это <b>профессиональный инструмент</b> для управления рисками в трейдинге.\n\n"
        f"<b>Основные возможности:</b>\n"
        f"• 📊 Анализ портфолио в реальном времени\n"
        f"• 📈 Расчёт Risk/Reward Ratio\n"
        f"• 💹 Профессиональные метрики (Sharpe, Sortino, Calmar)\n"
        f"• 🎯 Рекомендации по оптимизации\n"
        f"• 📥 Экспорт данных в CSV\n\n"
        f"<b>Выберите операцию:</b>"
    )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Добавить тейд", callback_data="add_trade")],
        [InlineKeyboardButton("📊 Просмотр портфолио", callback_data="view_portfolio")],
        [InlineKeyboardButton("📈 Аналитика", callback_data="analytics")],
        [InlineKeyboardButton("💾 Экспорт CSV", callback_data="export_csv")],
        [InlineKeyboardButton("❓ Справка", callback_data="help")],
        [InlineKeyboardButton("💰 Поддержать", callback_data="donate")],
    ])

    await SafeMessageSender.send_message(
        chat_id=user.id,
        text=welcome_text,
        context=context,
        reply_markup=keyboard,
        parse_mode='HTML'
    )
    return SELECT_OPERATION

async def add_trade_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Добавление тейда"""
    query = update.callback_query
    await query.answer()

    mode_text = (
        "<b>🎯 Выберите тип тейда:</b>\n\n"
        "Какой тип позиции вы открываете?"
    )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔼 Long", callback_data="mode_long")],
        [InlineKeyboardButton("🔽 Short", callback_data="mode_short")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_main")],
    ])

    await SafeMessageSender.edit_message_text(
        query=query,
        text=mode_text,
        reply_markup=keyboard,
        parse_mode='HTML'
    )
    return SELECT_TRADE_MODE

async def select_pair_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор торговой пары"""
    query = update.callback_query
    await query.answer()

    trade_type = query.data.split("_")[1]
    context.user_data['trade_type'] = trade_type

    pair_text = "<b>📍 Выберите торговую пару или введите пользовательскую:</b>"

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("BTC/USD", callback_data="pair_BTC/USD")],
        [InlineKeyboardButton("ETH/USD", callback_data="pair_ETH/USD")],
        [InlineKeyboardButton("EUR/USD", callback_data="pair_EUR/USD")],
        [InlineKeyboardButton("GBP/USD", callback_data="pair_GBP/USD")],
        [InlineKeyboardButton("XAU/USD", callback_data="pair_XAU/USD")],
        [InlineKeyboardButton("🔧 Другое", callback_data="pair_custom")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_trade_type")],
    ])

    await SafeMessageSender.edit_message_text(
        query=query,
        text=pair_text,
        reply_markup=keyboard,
        parse_mode='HTML'
    )
    return SELECT_PAIR

async def pair_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора пары"""
    query = update.callback_query
    await query.answer()

    pair = query.data.split("_", 1)[1]
    
    if pair == "custom":
        instruction_text = "<b>📝 Введите пару (например, BTC/USD):</b>"
        await SafeMessageSender.edit_message_text(
            query=query,
            text=instruction_text,
            parse_mode='HTML'
        )
        context.user_data['expecting_input'] = 'pair_custom'
        return SELECT_PAIR
    
    context.user_data['pair'] = pair

    entry_text = "<b>💰 Введите цену входа (Entry Price):</b>"
    await SafeMessageSender.edit_message_text(
        query=query,
        text=entry_text,
        parse_mode='HTML'
    )
    context.user_data['expecting_input'] = 'entry_price'
    return ENTER_ENTRY_PRICE

async def text_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка текстовых вводов"""
    user = update.effective_user
    text = update.message.text.strip()

    expecting = context.user_data.get('expecting_input')

    if expecting == 'pair_custom':
        context.user_data['pair'] = text
        entry_text = "<b>💰 Введите цену входа (Entry Price):</b>"
        await SafeMessageSender.send_message(
            chat_id=user.id,
            text=entry_text,
            context=context,
            parse_mode='HTML'
        )
        context.user_data['expecting_input'] = 'entry_price'
        return ENTER_ENTRY_PRICE

    elif expecting == 'entry_price':
        try:
            entry_price = float(text)
            context.user_data['entry_price'] = entry_price
            exit_text = "<b>🎯 Введите цену выхода (Exit Price / Take Profit):</b>"
            await SafeMessageSender.send_message(
                chat_id=user.id,
                text=exit_text,
                context=context,
                parse_mode='HTML'
            )
            context.user_data['expecting_input'] = 'exit_price'
            return ENTER_EXIT_PRICE
        except ValueError:
            error_text = "❌ Ошибка: Введите корректное число для цены входа"
            await SafeMessageSender.send_message(
                chat_id=user.id,
                text=error_text,
                context=context,
                parse_mode='HTML'
            )
            return ENTER_ENTRY_PRICE

    elif expecting == 'exit_price':
        try:
            exit_price = float(text)
            context.user_data['exit_price'] = exit_price
            sl_text = "<b>🛑 Введите Stop Loss цену:</b>"
            await SafeMessageSender.send_message(
                chat_id=user.id,
                text=sl_text,
                context=context,
                parse_mode='HTML'
            )
            context.user_data['expecting_input'] = 'stop_loss'
            return ENTER_STOP_LOSS
        except ValueError:
            error_text = "❌ Ошибка: Введите корректное число для цены выхода"
            await SafeMessageSender.send_message(
                chat_id=user.id,
                text=error_text,
                context=context,
                parse_mode='HTML'
            )
            return ENTER_EXIT_PRICE

    elif expecting == 'stop_loss':
        try:
            stop_loss = float(text)
            context.user_data['stop_loss'] = stop_loss
            volume_text = "<b>📦 Введите объём позиции (Volume):</b>"
            await SafeMessageSender.send_message(
                chat_id=user.id,
                text=volume_text,
                context=context,
                parse_mode='HTML'
            )
            context.user_data['expecting_input'] = 'volume'
            return ENTER_VOLUME
        except ValueError:
            error_text = "❌ Ошибка: Введите корректное число для Stop Loss"
            await SafeMessageSender.send_message(
                chat_id=user.id,
                text=error_text,
                context=context,
                parse_mode='HTML'
            )
            return ENTER_STOP_LOSS

    elif expecting == 'volume':
        try:
            volume = float(text)
            context.user_data['volume'] = volume
            leverage_text = "<b>⚡ Введите leverage (по умолчанию 1):</b>"
            await SafeMessageSender.send_message(
                chat_id=user.id,
                text=leverage_text,
                context=context,
                parse_mode='HTML'
            )
            context.user_data['expecting_input'] = 'leverage'
            return ENTER_LEVERAGE
        except ValueError:
            error_text = "❌ Ошибка: Введите корректное число для объёма"
            await SafeMessageSender.send_message(
                chat_id=user.id,
                text=error_text,
                context=context,
                parse_mode='HTML'
            )
            return ENTER_VOLUME

    elif expecting == 'leverage':
        try:
            leverage = float(text) if text else 1.0
            context.user_data['leverage'] = leverage
            commission_text = "<b>💸 Введите комиссию (по умолчанию 0):</b>"
            await SafeMessageSender.send_message(
                chat_id=user.id,
                text=commission_text,
                context=context,
                parse_mode='HTML'
            )
            context.user_data['expecting_input'] = 'commission'
            return ENTER_COMMISSION
        except ValueError:
            error_text = "❌ Ошибка: Введите корректное число для leverage"
            await SafeMessageSender.send_message(
                chat_id=user.id,
                text=error_text,
                context=context,
                parse_mode='HTML'
            )
            return ENTER_LEVERAGE

    elif expecting == 'commission':
        try:
            commission = float(text) if text else 0.0
            context.user_data['commission'] = commission

            # Добавляем тейд в портфолио
            trade_data = {
                'entry_price': context.user_data['entry_price'],
                'exit_price': context.user_data['exit_price'],
                'stop_loss': context.user_data['stop_loss'],
                'volume': context.user_data['volume'],
                'leverage': context.user_data['leverage'],
                'commission': commission,
                'asset': context.user_data['pair'],
                'trade_type': context.user_data['trade_type'],
                'currency': 'USD',
            }

            trade_metrics = portfolio_manager.add_trade(user.id, trade_data)

            confirmation_text = (
                f"<b>✅ Тейд успешно добавлен!</b>\n\n"
                f"<b>Параметры:</b>\n"
                f"Пара: <code>{trade_metrics.asset}</code>\n"
                f"Тип: {trade_metrics.trade_type.upper()}\n"
                f"Вход: {trade_metrics.entry_price}\n"
                f"Выход: {trade_metrics.exit_price}\n"
                f"SL: {trade_metrics.stop_loss}\n"
                f"Объём: {trade_metrics.volume}\n"
                f"Leverage: {trade_metrics.leverage}x\n"
                f"Комиссия: {commission}\n\n"
                f"<b>Метрики:</b>\n"
                f"P&L: ${trade_metrics.profit_loss:.2f}\n"
                f"P&L %: {trade_metrics.profit_loss_percent:.2f}%\n"
                f"RR Ratio: {trade_metrics.risk_reward_ratio:.2f}\n"
                f"Риск: ${trade_metrics.risk_amount:.2f}"
            )

            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("➕ Ещё тейд", callback_data="add_trade")],
                [InlineKeyboardButton("📊 Портфолио", callback_data="view_portfolio")],
                [InlineKeyboardButton("📈 Анализ", callback_data="analytics")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="back_main")],
            ])

            await SafeMessageSender.send_message(
                chat_id=user.id,
                text=confirmation_text,
                context=context,
                reply_markup=keyboard,
                parse_mode='HTML'
            )

            context.user_data.clear()
            return SELECT_OPERATION

        except ValueError:
            error_text = "❌ Ошибка: Введите корректное число для комиссии"
            await SafeMessageSender.send_message(
                chat_id=user.id,
                text=error_text,
                context=context,
                parse_mode='HTML'
            )
            return ENTER_COMMISSION

    return SELECT_OPERATION

async def view_portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Просмотр портфолио и метрик"""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    metrics = portfolio_manager.calculate_portfolio_metrics(user_id)

    if metrics['total_trades'] == 0:
        empty_text = "📊 Портфолио пусто. Добавьте свой первый тейд!"
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("➕ Добавить тейд", callback_data="add_trade")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="back_main")],
        ])
        await SafeMessageSender.edit_message_text(
            query=query,
            text=empty_text,
            reply_markup=keyboard,
            parse_mode='HTML'
        )
        return SELECT_OPERATION

    portfolio_text = (
        f"<b>📊 ПОРТФОЛИО АНАЛИЗ</b>\n\n"
        f"<b>Общая статистика:</b>\n"
        f"Всего тейдов: {metrics['total_trades']}\n"
        f"Прибыльных: {metrics['winning_trades']} ✅\n"
        f"Убыточных: {metrics['losing_trades']} ❌\n"
        f"Win Rate: {metrics['win_rate']:.1f}%\n\n"
        f"<b>Финансовые показатели:</b>\n"
        f"Общая P&L: ${metrics['total_profit_loss']:.2f}\n"
        f"P&L %: {metrics['total_profit_loss_percent']:.2f}%\n"
        f"Макс выигрыш: ${metrics['largest_win']:.2f}\n"
        f"Макс убыток: ${metrics['largest_loss']:.2f}\n\n"
        f"<b>Рисч-менеджмент:</b>\n"
        f"Ср. RR Ratio: {metrics['average_rr_ratio']:.2f}\n"
        f"Общий Риск: ${metrics['total_risk']:.2f}\n"
        f"Маржа (Notional): ${metrics['total_margin_usage']:.2f}\n"
        f"Посл. победы: {metrics['consecutive_wins']}\n"
        f"Посл. поражения: {metrics['consecutive_losses']}"
    )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Анализ", callback_data="analytics")],
        [InlineKeyboardButton("💾 Экспорт", callback_data="export_csv")],
        [InlineKeyboardButton("➕ Тейд", callback_data="add_trade")],
        [InlineKeyboardButton("🏠 Меню", callback_data="back_main")],
    ])

    await SafeMessageSender.edit_message_text(
        query=query,
        text=portfolio_text,
        reply_markup=keyboard,
        parse_mode='HTML'
    )
    return SELECT_OPERATION

async def analytics_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Профессиональная аналитика"""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    trades = portfolio_manager.get_user_trades(user_id)
    metrics = portfolio_manager.calculate_portfolio_metrics(user_id)

    if not trades:
        empty_text = "📈 Для аналитики требуется минимум 1 тейд"
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("➕ Добавить тейд", callback_data="add_trade")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="back_main")],
        ])
        await SafeMessageSender.edit_message_text(
            query=query,
            text=empty_text,
            reply_markup=keyboard,
            parse_mode='HTML'
        )
        return SELECT_OPERATION

    sharpe = AdvancedAnalyticsEngine.calculate_sharpe_ratio(trades)
    sortino = AdvancedAnalyticsEngine.calculate_sortino_ratio(trades)
    calmar = AdvancedAnalyticsEngine.calculate_calmar_ratio(trades)
    max_dd = AdvancedAnalyticsEngine.calculate_max_drawdown(trades)
    recommendations = AdvancedAnalyticsEngine.generate_recommendations(trades, metrics)

    analytics_text = (
        f"<b>📈 ПРОДВИНУТАЯ АНАЛИТИКА</b>\n\n"
        f"<b>Профессиональные метрики:</b>\n"
        f"🎯 Sharpe Ratio: {sharpe:.2f}\n"
        f"🔻 Sortino Ratio: {sortino:.2f}\n"
        f"📉 Calmar Ratio: {calmar:.2f}\n"
        f"💥 Max Drawdown: {max_dd:.2f}%\n\n"
        f"<b>Рекомендации:</b>\n"
    )

    for i, rec in enumerate(recommendations[:5], 1):
        analytics_text += f"{i}. {rec}\n"

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Портфолио", callback_data="view_portfolio")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="back_main")],
    ])

    await SafeMessageSender.edit_message_text(
        query=query,
        text=analytics_text,
        reply_markup=keyboard,
        parse_mode='HTML'
    )
    return SELECT_OPERATION

async def export_csv_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Экспорт портфолио в CSV"""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    csv_data = portfolio_manager.export_trades_csv(user_id)

    try:
        csv_file = InputFile(
            io.BytesIO(csv_data.encode()),
            filename=f"trades_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        await context.bot.send_document(
            chat_id=user_id,
            document=csv_file,
            caption="📥 Ваше портфолио в CSV формате"
        )
        success_text = "✅ CSV файл успешно отправлен!"
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        success_text = f"❌ Ошибка при экспорте: {str(e)[:100]}"

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Портфолио", callback_data="view_portfolio")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="back_main")],
    ])

    await SafeMessageSender.send_message(
        chat_id=user_id,
        text=success_text,
        context=context,
        reply_markup=keyboard,
        parse_mode='HTML'
    )
    return SELECT_OPERATION

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Справочная информация"""
    query = update.callback_query
    await query.answer()

    help_text = (
        "<b>❓ СПРАВКА</b>\n\n"
        "<b>Как использовать PRO Risk Calculator:</b>\n\n"
        "<b>1️⃣ Добавить тейд</b>\n"
        "Вводите данные: Entry, Exit, Stop Loss, Volume, Leverage, Комиссия\n\n"
        "<b>2️⃣ Просмотр портфолио</b>\n"
        "Видите все метрики: P&L, Win Rate, RR Ratio, риск\n\n"
        "<b>3️⃣ Аналитика</b>\n"
        "Sharpe, Sortino, Calmar Ratios + умные рекомендации\n\n"
        "<b>4️⃣ Экспорт</b>\n"
        "Скачайте портфолио в CSV для дальнейшего анализа\n\n"
        "<b>Что такое RR Ratio?</b>\n"
        "Risk/Reward — соотношение риска к потенциальной прибыли\n"
        "Идеал: ≥ 2:1 (риск $10 → прибыль $20)\n\n"
        "<b>Leverage:</b> По умолчанию 1x (без плеча)\n"
        "<b>Комиссия:</b> Добавляется дважды (вход + выход)"
    )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 Главное меню", callback_data="back_main")],
    ])

    await SafeMessageSender.edit_message_text(
        query=query,
        text=help_text,
        reply_markup=keyboard,
        parse_mode='HTML'
    )
    return SELECT_OPERATION

async def donate_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Поддержка проекта"""
    query = update.callback_query
    await query.answer()

    donate_text = (
        "<b>💰 ПОДДЕРЖКА ПРОЕКТА</b>\n\n"
        "Спасибо за интерес к PRO Risk Calculator!\n\n"
        "<b>USDT (TRX/TON):</b>\n"
        f"<code>{USDT_WALLET_ADDRESS}</code>\n\n"
        "<b>TON:</b>\n"
        f"<code>{TON_WALLET_ADDRESS}</code>\n\n"
        "Любая сумма приветствуется! 🙏\n"
        "Ваша поддержка помогает улучшать бот!"
    )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 Главное меню", callback_data="back_main")],
    ])

    await SafeMessageSender.edit_message_text(
        query=query,
        text=donate_text,
        reply_markup=keyboard,
        parse_mode='HTML'
    )
    return SELECT_OPERATION

async def back_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Возврат на главное меню"""
    query = update.callback_query
    await query.answer()

    user = query.from_user
    welcome_text = (
        f"🤖 <b>PRO Risk Calculator v3.0</b>\n\n"
        f"👋 Привет, {user.first_name}!\n\n"
        f"<b>Выберите операцию:</b>"
    )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Добавить тейд", callback_data="add_trade")],
        [InlineKeyboardButton("📊 Портфолио", callback_data="view_portfolio")],
        [InlineKeyboardButton("📈 Аналитика", callback_data="analytics")],
        [InlineKeyboardButton("💾 Экспорт CSV", callback_data="export_csv")],
        [InlineKeyboardButton("❓ Справка", callback_data="help")],
        [InlineKeyboardButton("💰 Поддержать", callback_data="donate")],
    ])

    await SafeMessageSender.edit_message_text(
        query=query,
        text=welcome_text,
        reply_markup=keyboard,
        parse_mode='HTML'
    )
    return SELECT_OPERATION

async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Маршрутизация callback queries"""
    query = update.callback_query
    data = query.data

    try:
        if data == "add_trade":
            return await add_trade_handler(update, context)
        elif data.startswith("mode_"):
            return await select_pair_handler(update, context)
        elif data.startswith("pair_"):
            return await pair_handler(update, context)
        elif data == "view_portfolio":
            return await view_portfolio_handler(update, context)
        elif data == "analytics":
            return await analytics_handler(update, context)
        elif data == "export_csv":
            return await export_csv_handler(update, context)
        elif data == "help":
            return await help_handler(update, context)
        elif data == "donate":
            return await donate_handler(update, context)
        elif data == "back_main" or data == "back_trade_type":
            return await back_handler(update, context)
    except Exception as e:
        logger.error(f"Error in callback handler: {e}")
        await SafeMessageSender.answer_callback_query(query, "Произошла ошибка", show_alert=True)

    return SELECT_OPERATION

# ============================================================================
# WEBHOOK SETUP И MAIN ФУНКЦИИ
# ============================================================================

async def post_init(app: Application) -> None:
    """Инициализация при старте приложения"""
    logger.info("🚀 Bot initializing...")
    await market_data_provider.init_session()
    logger.info("✅ Market data provider initialized")

async def post_stop(app: Application) -> None:
    """Очистка при остановке приложения"""
    logger.info("🛑 Bot stopping...")
    await market_data_provider.close_session()
    logger.info("✅ Cleanup complete")

async def health_check(request: web.Request) -> web.Response:
    """Health check для Render/Heroku"""
    return web.json_response({'status': 'ok', 'timestamp': datetime.now().isoformat()})

async def main():
    """Главная функция запуска приложения"""
    # Создаём приложение с error handler
    application = RobustApplicationBuilder.create_application(TOKEN)

    # Регистрируем handlers
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CallbackQueryHandler(callback_query_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_input_handler))

    # Инициализация/очистка
    application.post_init = post_init
    application.post_stop = post_stop

    logger.info("📋 All handlers registered successfully")

    # Webhook setup (для Render Free)
    if WEBHOOK_URL:
        logger.info(f"🌐 Setting up webhook at {WEBHOOK_URL}{WEBHOOK_PATH}")

        async def webhook_handler(request: web.Request):
            data = await request.json()
            update = telegram.Update.de_json(data, application.bot)
            if update:
                await application.update_queue.put(update)
            return web.Response()

        app = web.Application()
        app.router.add_post(WEBHOOK_PATH, webhook_handler)
        app.router.add_get("/health", health_check)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", PORT)
        await site.start()

        await application.bot.set_webhook(f"{WEBHOOK_URL}{WEBHOOK_PATH}")
        logger.info(f"✅ Webhook set successfully at port {PORT}")

        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down webhook...")
            await runner.cleanup()
            await application.stop()
    else:
        logger.info("🚀 Starting polling mode...")
        await application.run_polling()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
