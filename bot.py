import os
import logging
import asyncio
import re
import time
import functools
from datetime import datetime
from typing import Dict, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    ConversationHandler,
    CallbackQueryHandler
)
from pymongo import MongoClient
from flask import Flask
import threading
import requests
import schedule

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
(
    MAIN_MENU, INSTRUMENT_TYPE, CUSTOM_INSTRUMENT, DIRECTION,
    RISK_PERCENT, DEPOSIT, LEVERAGE, CURRENCY, ENTRY,
    STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION,
    PORTFOLIO_MENU
) = range(13)

# Temporary storage
user_ Dict[int, Dict[str, Any]] = {}

# MongoDB Connection
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb+srv://felixalseny_db_user:kontraktaciA22@felix22.3nx1ibi.mongodb.net/risk_bot_pro?retryWrites=true&w=majority&appName=Felix22')
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.server_info()
    db = mongo_client.risk_bot
    users_collection = db.users
    portfolio_collection = db.portfolios
    logger.info("✅ MongoDB connected successfully!")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    users_collection = None
    portfolio_collection = None

# MongoDB Manager
class MongoDBManager:
    @staticmethod
    def get_user_data(user_id: int) -> Dict[str, Any]:
        if not users_collection:
            return {}
        try:
            user_data = users_collection.find_one({"user_id": user_id})
            return user_data.get('data', {}) if user_data else {}
        except Exception as e:
            logger.error(f"Error getting user  {e}")
            return {}

    @staticmethod
    def update_user_data(user_id: int, data: Dict[str, Any]):
        if not users_collection:
            return
        try:
            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"data": data, "last_updated": datetime.now()}},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error updating user  {e}")

    @staticmethod
    def get_portfolio(user_id: int) -> Dict[str, Any]:
        if not portfolio_collection:
            return {}
        try:
            portfolio = portfolio_collection.find_one({"user_id": user_id})
            return portfolio.get('portfolio_data', {}) if portfolio else {}
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return {}

    @staticmethod
    def update_portfolio(user_id: int, portfolio_ Dict[str, Any]):
        if not portfolio_collection:
            return
        try:
            portfolio_collection.update_one(
                {"user_id": user_id},
                {"$set": {"portfolio_data": portfolio_data, "last_updated": datetime.now()}},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")

# Portfolio Data Management
class PortfolioManager:
    @staticmethod
    def initialize_user_portfolio(user_id: int):
        portfolio_data = MongoDBManager.get_portfolio(user_id)
        if not portfolio_
            portfolio_data = {
                'initial_balance': 10000,
                'current_balance': 10000,
                'trades': [],
                'performance': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0,
                    'total_loss': 0,
                    'win_rate': 0,
                    'average_profit': 0,
                    'average_loss': 0
                },
                'allocation': {},
                'history': []
            }
            MongoDBManager.update_portfolio(user_id, portfolio_data)
        return portfolio_data

    @staticmethod
    def add_trade(user_id: int, trade_ Dict):
        portfolio_data = PortfolioManager.initialize_user_portfolio(user_id)
        trade_id = len(portfolio_data['trades']) + 1
        trade_data['id'] = trade_id
        trade_data['timestamp'] = datetime.now().isoformat()
        portfolio_data['trades'].append(trade_data)

        # Update performance metrics
        closed_trades = [t for t in portfolio_data['trades'] if t.get('status') == 'closed']
        winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('profit', 0) <= 0]

        perf = portfolio_data['performance']
        perf['total_trades'] = len(closed_trades)
        perf['winning_trades'] = len(winning_trades)
        perf['losing_trades'] = len(losing_trades)
        perf['total_profit'] = sum(t.get('profit', 0) for t in winning_trades)
        perf['total_loss'] = sum(t.get('profit', 0) for t in losing_trades)
        if closed_trades:
            perf['win_rate'] = (len(winning_trades) / len(closed_trades)) * 100
            perf['average_profit'] = perf['total_profit'] / len(winning_trades) if winning_trades else 0
            perf['average_loss'] = perf['total_loss'] / len(losing_trades) if losing_trades else 0

        # Update allocation
        instrument = trade_data.get('instrument', 'Unknown')
        portfolio_data['allocation'][instrument] = portfolio_data['allocation'].get(instrument, 0) + 1

        # Add to history
        portfolio_data['history'].append({
            'type': 'trade',
            'action': 'open' if trade_data.get('status') == 'open' else 'close',
            'instrument': instrument,
            'profit': trade_data.get('profit', 0),
            'timestamp': trade_data['timestamp']
        })
        MongoDBManager.update_portfolio(user_id, portfolio_data)

    @staticmethod
    def add_balance_operation(user_id: int, operation_type: str, amount: float, description: str = ""):
        portfolio_data = PortfolioManager.initialize_user_portfolio(user_id)
        portfolio_data['history'].append({
            'type': 'balance',
            'action': operation_type,
            'amount': amount,
            'description': description,
            'timestamp': datetime.now().isoformat()
        })
        if operation_type == 'deposit':
            portfolio_data['current_balance'] += amount
            if portfolio_data['initial_balance'] == 0:
                portfolio_data['initial_balance'] = amount
        MongoDBManager.update_portfolio(user_id, portfolio_data)

# Flask server for health checks
app = Flask(__name__)

@app.route('/')
def home():
    return "🤖 PRO Risk Management Bot is ALIVE and RUNNING!"

@app.route('/health')
def health():
    try:
        if users_collection:
            users_collection.find_one({})
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "mongodb": "connected" if users_collection else "disconnected",
                "telegram_bot": "running"
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 500

def run_flask():
    port = int(os.environ.get('FLASK_PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
logger.info("✅ Flask health check server started!")

# Keep-alive system
def keep_alive():
    try:
        bot_url = os.getenv('RENDER_EXTERNAL_URL', 'https://risk-management-bot-pro.onrender.com')
        if bot_url:
            response = requests.get(f"{bot_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Keep-alive ping successful")
            else:
                logger.warning(f"⚠️ Keep-alive ping failed: {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Keep-alive failed: {e}")

schedule.every(5).minutes.do(keep_alive)

def schedule_runner():
    while True:
        schedule.run_pending()
        time.sleep(60)

schedule_thread = threading.Thread(target=schedule_runner, daemon=True)
schedule_thread.start()
logger.info("✅ Keep-alive scheduler started!")

# Fast cache
class FastCache:
    def __init__(self, max_size=500, ttl=300):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None

    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            self.cache.clear()
        self.cache[key] = (value, time.time())

fast_cache = FastCache()

# Performance logging decorator
def log_performance(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        if execution_time > 1.0:
            logger.warning(f"Slow operation: {func.__name__} took {execution_time:.2f}s")
        return result
    return wrapper

# Constants
INSTRUMENT_TYPES = {
    'forex': 'Forex',
    'crypto': 'Криптовалюты',
    'indices': 'Индексы',
    'commodities': 'Товары',
    'metals': 'Металлы'
}

INSTRUMENT_PRESETS = {
    'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'NZDUSD', 'EURGBP'],
    'crypto': ['BTCUSD', 'ETHUSD', 'XRPUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD'],
    'indices': ['US30', 'NAS100', 'SPX500', 'DAX40', 'FTSE100'],
    'commodities': ['OIL', 'NATGAS', 'COPPER', 'GOLD'],
    'metals': ['XAUUSD', 'XAGUSD', 'XPTUSD']
}

PIP_VALUES = {
    'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9, 'USDCHF': 10,
    'USDCAD': 10, 'AUDUSD': 10, 'NZDUSD': 10, 'EURGBP': 10,
    'BTCUSD': 1, 'ETHUSD': 1, 'XRPUSD': 10, 'ADAUSD': 10,
    'DOTUSD': 1, 'SOLUSD': 1,
    'US30': 1, 'NAS100': 1, 'SPX500': 1, 'DAX40': 1,
    'FTSE100': 1,
    'OIL': 10, 'NATGAS': 10, 'COPPER': 10, 'GOLD': 10,
    'XAUUSD': 10, 'XAGUSD': 50, 'XPTUSD': 10
}

CONTRACT_SIZES = {
    'forex': 100000,
    'crypto': 1,
    'indices': 1,
    'commodities': 100,
    'metals': 100
}

LEVERAGES = ['1:10', '1:20', '1:50', '1:100', '1:200', '1:500', '1:1000']
RISK_LEVELS = ['2%', '5%', '10%', '15%', '20%', '25%']
TRADE_DIRECTIONS = ['BUY', 'SELL']

# Fast Risk Calculator
class FastRiskCalculator:
    @staticmethod
    def calculate_pip_value_fast(instrument_type: str, currency_pair: str, lot_size: float) -> float:
        base_pip_value = PIP_VALUES.get(currency_pair, 10)
        if instrument_type == 'crypto':
            return base_pip_value * lot_size * 0.1
        elif instrument_type == 'indices':
            return base_pip_value * lot_size * 0.01
        else:
            return base_pip_value * lot_size

    @staticmethod
    def calculate_position_size_fast(
        deposit: float,
        leverage: str,
        instrument_type: str,
        currency_pair: str,
        entry_price: float,
        stop_loss: float,
        direction: str,
        risk_percent: float = 0.02
    ) -> Dict[str, float]:
        try:
            cache_key = f"pos_{deposit}_{leverage}_{instrument_type}_{currency_pair}_{entry_price}_{stop_loss}_{direction}_{risk_percent}"
            cached = fast_cache.get(cache_key)
            if cached:
                return cached

            lev_value = int(leverage.split(':')[1])
            risk_amount = deposit * risk_percent

            if instrument_type == 'forex':
                stop_pips = abs(entry_price - stop_loss) * 10000
            elif instrument_type == 'crypto':
                stop_pips = abs(entry_price - stop_loss) * 100
            elif instrument_type in ['indices', 'commodities', 'metals']:
                stop_pips = abs(entry_price - stop_loss) * 10
            else:
                stop_pips = abs(entry_price - stop_loss) * 10000

            pip_value_per_lot = FastRiskCalculator.calculate_pip_value_fast(instrument_type, currency_pair, 1.0)

            max_lots_by_risk = risk_amount / (stop_pips * pip_value_per_lot) if stop_pips > 0 and pip_value_per_lot > 0 else 0
            contract_size = CONTRACT_SIZES.get(instrument_type, 100000)
            max_lots_by_margin = (deposit * lev_value) / (contract_size * entry_price) if entry_price > 0 else 0

            position_size = min(max_lots_by_risk, max_lots_by_margin, 50.0)
            position_size = max(0.01, round(position_size * 100) / 100)

            required_margin = (position_size * contract_size * entry_price) / lev_value if lev_value > 0 else 0

            result = {
                'position_size': position_size,
                'risk_amount': risk_amount,
                'stop_pips': stop_pips,
                'required_margin': required_margin,
                'risk_percent': (risk_amount / deposit) * 100 if deposit > 0 else 0,
                'free_margin': deposit - required_margin
            }
            fast_cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error in position calc: {e}")
            return {
                'position_size': 0.01,
                'risk_amount': 0,
                'stop_pips': 0,
                'required_margin': 0,
                'risk_percent': 0,
                'free_margin': deposit
            }

    @staticmethod
    def calculate_profits_fast(
        instrument_type: str,
        currency_pair: str,
        entry_price: float,
        take_profits: List[float],
        position_size: float,
        volume_distribution: List[float],
        direction: str
    ) -> List[Dict[str, Any]]:
        profits = []
        total_profit = 0
        for i, (tp, vol_pct) in enumerate(zip(take_profits, volume_distribution)):
            if instrument_type == 'forex':
                tp_pips = abs(entry_price - tp) * 10000
            elif instrument_type == 'crypto':
                tp_pips = abs(entry_price - tp) * 100
            elif instrument_type in ['indices', 'commodities', 'metals']:
                tp_pips = abs(entry_price - tp) * 10
            else:
                tp_pips = abs(entry_price - tp) * 10000

            volume_lots = position_size * (vol_pct / 100)
            pip_value = FastRiskCalculator.calculate_pip_value_fast(instrument_type, currency_pair, volume_lots)
            profit = tp_pips * pip_value
            total_profit += profit
            contract_size = CONTRACT_SIZES.get(instrument_type, 100000)
            position_value = position_size * contract_size * entry_price

            profits.append({
                'level': i + 1,
                'price': tp,
                'volume_percent': vol_pct,
                'volume_lots': volume_lots,
                'profit': profit,
                'cumulative_profit': total_profit,
                'pips': tp_pips,
                'roi_percent': (profit / position_value) * 100 if position_value > 0 else 0
            })
        return profits

# Main handlers
@log_performance
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message:
        user = update.message.from_user
    elif update.callback_query:
        user = update.callback_query.from_user
    else:
        return ConversationHandler.END

    user_name = user.first_name or "Трейдер"
    welcome_text = f"""
👋 *Привет, {user_name}!*
🎯 *PRO Калькулятор Риск-Менеджмента v3.0*
⚡ *Выберите опцию:*
"""
    user_id = user.id
    user_data_dict = MongoDBManager.get_user_data(user_id)
    updated_data = {
        'start_time': datetime.now().isoformat(),
        'last_activity': time.time(),
        'presets': user_data_dict.get('presets', []),
        'username': user.username,
        'first_name': user.first_name
    }
    MongoDBManager.update_user_data(user_id, updated_data)

    keyboard = [
        [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("⚡ Быстрый расчет", callback_data="quick_calculation")],
        [InlineKeyboardButton("💼 Мой портфель", callback_data="portfolio")],
        [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")]
    ]

    if update.message:
        await update.message.reply_text(welcome_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.callback_query.edit_message_text(welcome_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
    return MAIN_MENU

@log_performance
async def quick_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await start_quick_calculation(update, context)

@log_performance
async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = query.from_user.id if query else update.message.from_user.id
    PortfolioManager.initialize_user_portfolio(user_id)

    portfolio_text = """
💼 *Мой Портфель*
📊 *Доступные функции:*
• 📈 Обзор сделок
• 💰 Баланс и распределение
• 📊 Анализ эффективности
• 🔄 История операций
• ➕ Добавить сделку/депозит
Выберите опцию:
"""
    keyboard = [
        [InlineKeyboardButton("📈 Мои сделки", callback_data="portfolio_trades")],
        [InlineKeyboardButton("💰 Баланс", callback_data="portfolio_balance")],
        [InlineKeyboardButton("📊 Эффективность", callback_data="portfolio_performance")],
        [InlineKeyboardButton("🔄 История", callback_data="portfolio_history")],
        [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
        [InlineKeyboardButton("💸 Внести депозит", callback_data="portfolio_deposit")],
        [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
    ]
    if update.message:
        await update.message.reply_text(portfolio_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await query.edit_message_text(portfolio_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
    return PORTFOLIO_MENU

@log_performance
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info_text = """
📚 *PRO ИНСТРУКЦИИ v3.0*
🎯 *РАСШИРЕННЫЕ ВОЗМОЖНОСТИ:*
⚡ *ВСЕ ТИПЫ ИНСТРУМЕНТОВ:*
• 🌐 Forex (50+ валютных пар)
• ₿ Криптовалюты (15+ пар)
• 📈 Индексы (12+ индексов)
• ⚡ Товары (8+ типов)
• 🏅 Металлы (6+ типов)
📋 *КАК ИСПОЛЬЗОВАТЬ:*
*Профессиональный расчет:*
1. Выберите тип инструмента
2. Выберите конкретный инструмент или введите свой
3. Укажите направление сделки (BUY/SELL)
4. Выберите уровень риска
5. Введите основные параметры
6. Получите детальный анализ
*Быстрый расчет:*
1. Введите инструмент
2. Укажите базовые параметры
3. Получите мгновенный результат
👨‍💻 *РАЗРАБОТЧИК:* [@fxfeelgood](https://t.me/fxfeelgood)
*PRO v3.0 | Быстро • Умно • Точно* 🚀
"""
    if update.message:
        await update.message.reply_text(info_text, parse_mode='Markdown', disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))
    else:
        await update.callback_query.edit_message_text(info_text, parse_mode='Markdown', disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))

# Portfolio handlers
@log_performance
async def portfolio_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    portfolio_data = MongoDBManager.get_portfolio(user_id)
    trades = portfolio_data.get('trades', [])
    if not trades:
        await query.edit_message_text(
            "📭 *У вас пока нет сделок*\n"
            "Используйте кнопку '➕ Добавить сделку' для начала торговли.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
                [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
            ])
        )
        return

    recent_trades = trades[-5:]
    trades_text = "📈 *Последние сделки:*\n"
    for trade in reversed(recent_trades):
        status_emoji = "🟢" if trade.get('profit', 0) > 0 else "🔴" if trade.get('profit', 0) < 0 else "⚪"
        trades_text += (
            f"{status_emoji} *{trade.get('instrument', 'N/A')}* | "
            f"{trade.get('direction', 'N/A')} | "
            f"Прибыль: ${trade.get('profit', 0):.2f}\n"
            f"📅 {trade.get('timestamp', '')[:16]}\n"
        )
    trades_text += f"📊 Всего сделок: {len(trades)}"

    await query.edit_message_text(
        trades_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📋 Вся история", callback_data="portfolio_full_history")],
            [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
            [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    portfolio_data = MongoDBManager.get_portfolio(user_id)
    allocation = portfolio_data.get('allocation', {})
    performance = portfolio_data.get('performance', {})

    initial_balance = portfolio_data.get('initial_balance', 0)
    current_balance = portfolio_data.get('current_balance', 0)
    total_profit = performance.get('total_profit', 0)
    total_loss = performance.get('total_loss', 0)
    net_profit = total_profit + total_loss

    balance_text = "💰 *Баланс и распределение*\n"
    balance_text += f"💳 Начальный депозит: ${initial_balance:,.2f}\n"
    balance_text += f"💵 Текущий баланс: ${current_balance:,.2f}\n"
    balance_text += f"📈 Чистая прибыль: ${net_profit:.2f}\n"

    if allocation:
        balance_text += "🌐 *Распределение по инструментам:*\n"
        for instrument, count in list(allocation.items())[:5]:
            balance_text += f"• {instrument}: {count} сделок\n"
    else:
        balance_text += "🌐 *Распределение:* Нет данных\n"

    await query.edit_message_text(
        balance_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("💸 Внести депозит", callback_data="portfolio_deposit")],
            [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    portfolio_data = MongoDBManager.get_portfolio(user_id)
    performance = portfolio_data.get('performance', {})

    total_trades = performance.get('total_trades', 0)
    win_rate = performance.get('win_rate', 0)
    avg_profit = performance.get('average_profit', 0)
    avg_loss = performance.get('average_loss', 0)

    perf_text = "📊 *Анализ эффективности*\n"
    perf_text += f"📈 Всего сделок: {total_trades}\n"
    perf_text += f"🎯 Процент прибыльных: {win_rate:.1f}%\n"
    perf_text += f"💰 Средняя прибыль: ${avg_profit:.2f}\n"
    perf_text += f"📉 Средний убыток: ${avg_loss:.2f}\n"

    # Simple recommendations
    recommendations = []
    if win_rate < 40:
        recommendations.append("📉 Рассмотрите снижение риска на сделку до 1-2%")
    if avg_profit < abs(avg_loss) and win_rate > 50:
        recommendations.append("⚡ Улучшите управление позицией - фиксируйте прибыль раньше")
    if len(portfolio_data.get('allocation', {})) < 3:
        recommendations.append("🌐 Диверсифицируйте портфель - торгуйте разные инструменты")
    if not recommendations:
        recommendations.append("✅ Ваша стратегия показывает хорошие результаты! Продолжайте в том же духе")

    if recommendations:
        perf_text += "💡 *Рекомендации:*\n"
        for rec in recommendations[:3]:
            perf_text += f"• {rec}\n"

    await query.edit_message_text(
        perf_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    portfolio_data = MongoDBManager.get_portfolio(user_id)
    history = portfolio_data.get('history', [])
    if not history:
        await query.edit_message_text(
            "📭 *История операций пуста*\n"
            "Здесь будут отображаться все ваши операции по счету.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💸 Внести депозит", callback_data="portfolio_deposit")],
                [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
            ])
        )
        return

    history_text = "🔄 *История операций*\n"
    for op in reversed(history[-10:]):
        emoji = "💳" if op['type'] == 'balance' else "📈"
        action_emoji = "⬆️" if op.get('amount', 0) > 0 else "⬇️"
        history_text += f"{emoji} {op['type'].title()} | {op['action']} {action_emoji}\n"
        if op['type'] == 'balance':
            history_text += f"💵 Сумма: ${op.get('amount', 0):.2f}\n"
        else:
            history_text += f"💰 Прибыль: ${op.get('profit', 0):.2f}\n"
        history_text += f"📅 {op.get('timestamp', '')[:16]}\n"

    await query.edit_message_text(
        history_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_add_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    sample_trade = {
        'instrument': 'EURUSD',
        'direction': 'BUY',
        'volume': 0.1,
        'entry_price': 1.0850,
        'exit_price': 1.0900,
        'profit': 50.0,
        'risk_amount': 20.0,
        'status': 'closed',
        'strategy': 'Breakout'
    }
    PortfolioManager.add_trade(user_id, sample_trade)
    await query.edit_message_text(
        "✅ *Сделка добавлена в портфель!*\n"
        f"📈 {sample_trade['instrument']} {sample_trade['direction']}\n"
        f"💰 Прибыль: ${sample_trade['profit']:.2f}\n"
        "Обновите анализ эффективности для просмотра новой статистики.",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Обновить аналитику", callback_data="portfolio_performance")],
            [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    PortfolioManager.add_balance_operation(user_id, 'deposit', 1000.0, "Начальный депозит")
    await query.edit_message_text(
        "✅ *Депозит добавлен!*\n"
        "💵 Сумма: $1,000.00\n"
        "Теперь вы можете отслеживать баланс и эффективность вашего портфеля.",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("💰 Проверить баланс", callback_data="portfolio_balance")],
            [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
        ])
    )

# Navigation
@log_performance
async def portfolio_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await portfolio_command(update, context)

@log_performance
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if not query:
        return MAIN_MENU
    await query.answer()
    choice = query.data
    user_id = query.from_user.id
    if user_id in user_data:
        user_data[user_id]['last_activity'] = time.time()

    if choice == "pro_calculation":
        return await start_pro_calculation(update, context)
    elif choice == "quick_calculation":
        return await start_quick_calculation(update, context)
    elif choice == "portfolio":
        return await portfolio_command(update, context)
    elif choice == "pro_info":
        await pro_info_command(update, context)
        return MAIN_MENU
    elif choice == "main_menu":
        return await start(update, context)
    elif choice == "portfolio_back":
        return await portfolio_command(update, context)
    return MAIN_MENU

# Professional calculation flow
@log_performance
async def start_pro_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if query:
        await query.edit_message_text(
            "🎯 *Профессиональный расчет*\n"
            "📊 *Выберите тип инструмента:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 Forex", callback_data="inst_type_forex")],
                [InlineKeyboardButton("₿ Криптовалюты", callback_data="inst_type_crypto")],
                [InlineKeyboardButton("📈 Индексы", callback_data="inst_type_indices")],
                [InlineKeyboardButton("⚡ Товары", callback_data="inst_type_commodities")],
                [InlineKeyboardButton("🏅 Металлы", callback_data="inst_type_metals")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
    return INSTRUMENT_TYPE

@log_performance
async def start_quick_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = "⚡ *Быстрый расчет*\n" "📊 *Введите тикер инструмента* (например: EURUSD, BTCUSD, NAS100):"
    markup = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
    if update.message:
        await update.message.reply_text(text, parse_mode='Markdown', reply_markup=markup)
    else:
        await update.callback_query.edit_message_text(text, parse_mode='Markdown', reply_markup=markup)
    return CUSTOM_INSTRUMENT

@log_performance
async def process_instrument_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if not query: return INSTRUMENT_TYPE
    await query.answer()
    user_id = query.from_user.id
    instrument_type = query.data.replace('inst_type_', '')
    user_data[user_id] = user_data.get(user_id, {})
    user_data[user_id]['instrument_type'] = instrument_type
    user_data[user_id]['last_activity'] = time.time()

    display_type = INSTRUMENT_TYPES.get(instrument_type, instrument_type)
    keyboard = []
    presets = INSTRUMENT_PRESETS.get(instrument_type, [])
    for i in range(0, len(presets), 2):
        row = []
        if i < len(presets):
            row.append(InlineKeyboardButton(presets[i], callback_data=f"currency_{presets[i]}"))
        if i + 1 < len(presets):
            row.append(InlineKeyboardButton(presets[i + 1], callback_data=f"currency_{presets[i + 1]}"))
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("✏️ Ввести свой тикер", callback_data="custom_currency")])
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="pro_calculation")])

    await query.edit_message_text(
        f"✅ *Тип инструмента:* {display_type}\n"
        "📊 *Выберите инструмент из списка или введите свой:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return CURRENCY

@log_performance
async def process_currency_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if not query: return CURRENCY
    await query.answer()
    user_id = query.from_user.id
    currency = query.data.replace('currency_', '')
    user_data[user_id]['currency'] = currency
    await query.edit_message_text(
        f"✅ *Инструмент:* {currency}\n"
        "🎯 *Выберите направление сделки:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 BUY", callback_data="direction_BUY")],
            [InlineKeyboardButton("📉 SELL", callback_data="direction_SELL")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_instrument")]
        ])
    )
    return DIRECTION

@log_performance
async def process_custom_currency(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if not query: return CURRENCY
    await query.answer()
    await query.edit_message_text(
        "✏️ *Введите тикер инструмента* (например: EURUSD, BTCUSD, NAS100):",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_instrument")]
        ])
    )
    return CUSTOM_INSTRUMENT

@log_performance
async def process_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if not query: return DIRECTION
    await query.answer()
    user_id = query.from_user.id
    direction = query.data.replace('direction_', '')
    user_data[user_id]['direction'] = direction
    keyboard = []
    for i in range(0, len(RISK_LEVELS), 3):
        row = []
        for j in range(3):
            if i + j < len(RISK_LEVELS):
                risk = RISK_LEVELS[i + j]
                row.append(InlineKeyboardButton(risk, callback_data=f"risk_{risk}"))
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_currency")])
    await query.edit_message_text(
        f"✅ *Направление:* {direction}\n"
        "⚠️ *Выберите уровень риска (% от депозита):*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return RISK_PERCENT

@log_performance
async def process_risk_percent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if not query: return RISK_PERCENT
    await query.answer()
    user_id = query.from_user.id
    risk_percent = query.data.replace('risk_', '').replace('%', '')
    user_data[user_id]['risk_percent'] = float(risk_percent) / 100
    await query.edit_message_text(
        f"✅ *Уровень риска:* {risk_percent}%\n"
        "💰 *Введите сумму депозита* (в USD):\n"
        "Пример: 10000",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_direction")]
        ])
    )
    return DEPOSIT

@log_performance
async def process_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message: return DEPOSIT
    user_id = update.message.from_user.id
    try:
        deposit = float(update.message.text.strip())
        if deposit <= 0: raise ValueError()
        user_data[user_id]['deposit'] = deposit
        keyboard = []
        for i in range(0, len(LEVERAGES), 3):
            row = []
            for j in range(3):
                if i + j < len(LEVERAGES):
                    leverage = LEVERAGES[i + j]
                    row.append(InlineKeyboardButton(leverage, callback_data=f"leverage_{leverage}"))
            keyboard.append(row)
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_risk")])
        await update.message.reply_text(
            f"✅ *Депозит:* ${deposit:,.2f}\n"
            "⚖️ *Выберите кредитное плечо:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return LEVERAGE
    except:
        await update.message.reply_text(
            "❌ *Неверная сумма депозита!*\n"
            "💰 Пожалуйста, введите корректную сумму (только числа):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_risk")]
            ])
        )
        return DEPOSIT

@log_performance
async def process_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if not query: return LEVERAGE
    await query.answer()
    user_id = query.from_user.id
    leverage = query.data.replace('leverage_', '')
    user_data[user_id]['leverage'] = leverage
    currency = user_data[user_id].get('currency', 'инструмент')
    await query.edit_message_text(
        f"✅ *Плечо:* {leverage}\n"
        f"📈 *Введите цену входа для {currency}:*\n"
        "Пример: 1.0850",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_deposit")]
        ])
    )
    return ENTRY

@log_performance
async def process_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message: return ENTRY
    user_id = update.message.from_user.id
    try:
        entry_price = float(update.message.text.strip())
        if entry_price <= 0: raise ValueError()
        user_data[user_id]['entry'] = entry_price
        currency = user_data[user_id].get('currency', 'инструмент')
        await update.message.reply_text(
            f"✅ *Цена входа:* {entry_price}\n"
            f"🛑 *Введите цену стоп-лосса для {currency}:*\n"
            "Пример: 1.0800",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_leverage")]
            ])
        )
        return STOP_LOSS
    except:
        await update.message.reply_text(
            "❌ *Неверная цена входа!*\n"
            "📈 Пожалуйста, введите корректную цену (только числа):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_leverage")]
            ])
        )
        return ENTRY

@log_performance
async def process_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message: return STOP_LOSS
    user_id = update.message.from_user.id
    try:
        stop_loss = float(update.message.text.strip())
        entry_price = user_data[user_id].get('entry', 0)
        direction = user_data[user_id].get('direction', 'BUY')
        if direction == 'BUY' and stop_loss >= entry_price:
            await update.message.reply_text(
                "❌ *Для BUY сделки стоп-лосс должен быть ниже цены входа!*\n"
                "🛑 Пожалуйста, введите корректную цену стоп-лосса:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="back_to_entry")]
                ])
            )
            return STOP_LOSS
        elif direction == 'SELL' and stop_loss <= entry_price:
            await update.message.reply_text(
                "❌ *Для SELL сделки стоп-лосс должен быть выше цены входа!*\n"
                "🛑 Пожалуйста, введите корректную цену стоп-лосса:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="back_to_entry")]
                ])
            )
            return STOP_LOSS
        user_data[user_id]['stop_loss'] = stop_loss
        currency = user_data[user_id].get('currency', 'инструмент')
        await update.message.reply_text(
            f"✅ *Стоп-лосс:* {stop_loss}\n"
            f"🎯 *Введите цены тейк-профитов для {currency} (через запятую):*\n"
            "Пример: 1.0900, 1.0950, 1.1000\n"
            "Можно указать до 3 уровней",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_entry")]
            ])
        )
        return TAKE_PROFITS
    except:
        await update.message.reply_text(
            "❌ *Неверная цена стоп-лосса!*\n"
            "🛑 Пожалуйста, введите корректную цену (только числа):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_entry")]
            ])
        )
        return STOP_LOSS

@log_performance
async def process_take_profits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message: return TAKE_PROFITS
    user_id = update.message.from_user.id
    try:
        tp_list = [float(tp.strip()) for tp in update.message.text.split(',')][:3]
        entry_price = user_data[user_id].get('entry', 0)
        direction = user_data[user_id].get('direction', 'BUY')
        for tp in tp_list:
            if (direction == 'BUY' and tp <= entry_price) or (direction == 'SELL' and tp >= entry_price):
                await update.message.reply_text(
                    f"❌ *Нарушение логики направления для {direction}!*\n"
                    "🎯 Пожалуйста, введите корректные цены тейк-профитов:",
                    parse_mode='Markdown',
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_stop_loss")]
                    ])
                )
                return TAKE_PROFITS
        user_data[user_id]['take_profits'] = tp_list
        await update.message.reply_text(
            f"✅ *Тейк-профиты:* {', '.join(str(tp) for tp in tp_list)}\n"
            "📊 *Введите распределение объема между тейк-профитами (% через запятую):*\n"
            f"Пример для {len(tp_list)} уровней: {', '.join([str(100//len(tp_list)) for _ in tp_list])}\n"
            "Сумма должна равняться 100%",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_stop_loss")]
            ])
        )
        return VOLUME_DISTRIBUTION
    except:
        await update.message.reply_text(
            "❌ *Неверный формат тейк-профитов!*\n"
            "🎯 Пожалуйста, введите корректные цены (только числа, через запятую):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_stop_loss")]
            ])
        )
        return TAKE_PROFITS

@log_performance
async def process_volume_distribution(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message: return VOLUME_DISTRIBUTION
    user_id = update.message.from_user.id
    try:
        volume_list = [float(vol.strip()) for vol in update.message.text.split(',')]
        take_profits = user_data[user_id].get('take_profits', [])
        if len(volume_list) != len(take_profits):
            await update.message.reply_text(
                f"❌ *Количество уровней ({len(volume_list)}) не совпадает с TP ({len(take_profits)})!*\n"
                "📊 Пожалуйста, введите корректное распределение:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="back_to_take_profits")]
                ])
            )
            return VOLUME_DISTRIBUTION
        if abs(sum(volume_list) - 100) > 1e-5:
            await update.message.reply_text(
                f"❌ *Сумма распределения ({sum(volume_list):.1f}%) ≠ 100%!*\n"
                "📊 Пожалуйста, введите корректное распределение:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="back_to_take_profits")]
                ])
            )
            return VOLUME_DISTRIBUTION
        user_data[user_id]['volume_distribution'] = volume_list
        result = await perform_pro_calculation(user_id)
        await update.message.reply_text(
            result,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💼 Добавить в портфель", callback_data="portfolio_add_trade")],
                [InlineKeyboardButton("📊 Новый расчет", callback_data="pro_calculation")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return MAIN_MENU
    except:
        await update.message.reply_text(
            "❌ *Неверный формат распределения!*\n"
            "📊 Пожалуйста, введите корректные проценты (только числа, через запятую):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_take_profits")]
            ])
        )
        return VOLUME_DISTRIBUTION

@log_performance
async def perform_pro_calculation(user_id: int) -> str:
    if user_id not in user_
        return "❌ Ошибка: данные пользователя не найдены"
    u = user_data[user_id]
    instrument_type = u.get('instrument_type', 'forex')
    currency = u.get('currency', 'EURUSD')
    direction = u.get('direction', 'BUY')
    risk_percent = u.get('risk_percent', 0.02)
    deposit = u.get('deposit', 10000)
    leverage = u.get('leverage', '1:100')
    entry_price = u.get('entry', 1.0850)
    stop_loss = u.get('stop_loss', 1.0800)
    take_profits = u.get('take_profits', [1.0900, 1.0950])
    volume_distribution = u.get('volume_distribution', [70, 30])

    pos = FastRiskCalculator.calculate_position_size_fast(
        deposit, leverage, instrument_type, currency, entry_price, stop_loss, direction, risk_percent
    )
    profits = FastRiskCalculator.calculate_profits_fast(
        instrument_type, currency, entry_price, take_profits, pos['position_size'], volume_distribution, direction
    )

    result = f"""
🎯 *ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ ЗАВЕРШЕН*
📊 *Параметры сделки:*
🌐 Инструмент: {currency} ({INSTRUMENT_TYPES.get(instrument_type, instrument_type)})
🎯 Направление: {direction}
💵 Депозит: ${deposit:,.2f}
⚖️ Плечо: {leverage}
📈 Вход: {entry_price}
🛑 Стоп-лосс: {stop_loss}
⚠️ *Управление рисками:*
📦 Размер позиции: *{pos['position_size']:.2f} лотов*
💰 Риск на сделку: ${pos['risk_amount']:.2f} ({risk_percent*100}%)
📉 Стоп-лосс: {pos['stop_pips']:.0f} пунктов
🏦 Залог: ${pos['required_margin']:.2f}
💳 Свободные средства: ${pos['free_margin']:.2f}
🎯 *Уровни тейк-профита:*
"""
    for i, p in enumerate(profits):
        result += f"\n📈 Уровень {i+1}: {p['price']}"
        result += f"\n   📊 Объем: {volume_distribution[i]}% ({p['volume_lots']:.2f} лотов)"
        result += f"\n   💰 Прибыль: ${p['profit']:.2f}"
        result += f"\n   📈 Пункты: {p['pips']:.0f}"
        result += f"\n   📊 ROI: {p['roi_percent']:.2f}%"
    result += f"\n💰 *Общая прибыль:* ${profits[-1]['cumulative_profit']:.2f}"

    rr = abs(profits[0]['profit'] / pos['risk_amount']) if pos['risk_amount'] > 0 else 0
    result += f"\n⚡ *Соотношение R/R:* {rr:.2f}"
    if rr < 1:
        result += "\n🔻 *Внимание:* Соотношение риск/вознаграждение менее 1:1"
    elif rr >= 3:
        result += "\n✅ *Отлично:* Соотношение риск/вознаграждение 1:3 или выше"
    if pos['risk_percent'] > 5:
        result += "\n🔻 *Внимание:* Риск на сделку превышает 5%"
    result += "\n💡 *PRO рекомендация:* Всегда используйте стоп-лосс и управляйте рисками!"
    return result

# Back handlers
@log_performance
async def back_to_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await start_pro_calculation(update, context)

@log_performance
async def back_to_currency(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = query.from_user.id
    instrument_type = user_data.get(user_id, {}).get('instrument_type', 'forex')
    display_type = INSTRUMENT_TYPES.get(instrument_type, instrument_type)
    keyboard = []
    presets = INSTRUMENT_PRESETS.get(instrument_type, [])
    for i in range(0, len(presets), 2):
        row = []
        if i < len(presets):
            row.append(InlineKeyboardButton(presets[i], callback_data=f"currency_{presets[i]}"))
        if i + 1 < len(presets):
            row.append(InlineKeyboardButton(presets[i + 1], callback_data=f"currency_{presets[i + 1]}"))
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("✏️ Ввести свой тикер", callback_data="custom_currency")])
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="pro_calculation")])
    await query.edit_message_text(
        f"✅ *Тип инструмента:* {display_type}\n"
        "📊 *Выберите инструмент из списка или введите свой:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return CURRENCY

@log_performance
async def back_to_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = query.from_user.id
    currency = user_data.get(user_id, {}).get('currency', 'EURUSD')
    await query.edit_message_text(
        f"✅ *Инструмент:* {currency}\n"
        "🎯 *Выберите направление сделки:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 BUY", callback_data="direction_BUY")],
            [InlineKeyboardButton("📉 SELL", callback_data="direction_SELL")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_instrument")]
        ])
    )
    return DIRECTION

@log_performance
async def back_to_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = query.from_user.id
    direction = user_data.get(user_id, {}).get('direction', 'BUY')
    keyboard = []
    for i in range(0, len(RISK_LEVELS), 3):
        row = []
        for j in range(3):
            if i + j < len(RISK_LEVELS):
                risk = RISK_LEVELS[i + j]
                row.append(InlineKeyboardButton(risk, callback_data=f"risk_{risk}"))
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_currency")])
    await query.edit_message_text(
        f"✅ *Направление:* {direction}\n"
        "⚠️ *Выберите уровень риска (% от депозита):*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return RISK_PERCENT

@log_performance
async def back_to_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = query.from_user.id
    risk_percent = user_data.get(user_id, {}).get('risk_percent', 0.02) * 100
    await query.edit_message_text(
        f"✅ *Уровень риска:* {risk_percent:.0f}%\n"
        "💰 *Введите сумму депозита* (в USD):\n"
        "Пример: 10000",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_direction")]
        ])
    )
    return DEPOSIT

@log_performance
async def back_to_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = query.from_user.id
    deposit = user_data.get(user_id, {}).get('deposit', 10000)
    keyboard = []
    for i in range(0, len(LEVERAGES), 3):
        row = []
        for j in range(3):
            if i + j < len(LEVERAGES):
                leverage = LEVERAGES[i + j]
                row.append(InlineKeyboardButton(leverage, callback_data=f"leverage_{leverage}"))
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_risk")])
    await query.edit_message_text(
        f"✅ *Депозит:* ${deposit:,.2f}\n"
        "⚖️ *Выберите кредитное плечо:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return LEVERAGE

@log_performance
async def back_to_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = query.from_user.id
    leverage = user_data.get(user_id, {}).get('leverage', '1:100')
    currency = user_data.get(user_id, {}).get('currency', 'инструмент')
    await query.edit_message_text(
        f"✅ *Плечо:* {leverage}\n"
        f"📈 *Введите цену входа для {currency}:*\n"
        "Пример: 1.0850",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_deposit")]
        ])
    )
    return ENTRY

@log_performance
async def back_to_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = query.from_user.id
    entry_price = user_data.get(user_id, {}).get('entry', 1.0850)
    currency = user_data.get(user_id, {}).get('currency', 'инструмент')
    await query.edit_message_text(
        f"✅ *Цена входа:* {entry_price}\n"
        f"🛑 *Введите цену стоп-лосса для {currency}:*\n"
        "Пример: 1.0800",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_leverage")]
        ])
    )
    return STOP_LOSS

@log_performance
async def back_to_take_profits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = query.from_user.id
    stop_loss = user_data.get(user_id, {}).get('stop_loss', 1.0800)
    currency = user_data.get(user_id, {}).get('currency', 'инструмент')
    await query.edit_message_text(
        f"✅ *Стоп-лосс:* {stop_loss}\n"
        f"🎯 *Введите цены тейк-профитов для {currency} (через запятую):*\n"
        "Пример: 1.0900, 1.0950, 1.1000\n"
        "Можно указать до 3 уровней",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_entry")]
        ])
    )
    return TAKE_PROFITS

@log_performance
async def process_currency_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message: return CUSTOM_INSTRUMENT
    user_id = update.message.from_user.id
    currency = update.message.text.upper().strip()
    if not re.match(r'^[A-Z0-9]{2,10}$', currency):
        await update.message.reply_text(
            "❌ *Неверный формат тикера!*\n"
            "Пожалуйста, введите правильный тикер (только буквы и цифры, 2-10 символов):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return CUSTOM_INSTRUMENT

    user_data[user_id] = {
        'currency': currency,
        'instrument_type': 'forex',
        'direction': 'BUY',
        'risk_percent': 0.02,
        'deposit': 10000,
        'leverage': '1:100',
        'entry': 1.0850,
        'stop_loss': 1.0800,
        'take_profits': [1.0900, 1.0950],
        'volume_distribution': [70, 30]
    }

    pos = FastRiskCalculator.calculate_position_size_fast(
        10000, '1:100', 'forex', currency, 1.0850, 1.0800, 'BUY'
    )
    result = f"""
🎯 *РЕЗУЛЬТАТ БЫСТРОГО РАСЧЕТА*
📊 *Основные параметры:*
🌐 Инструмент: {currency}
💵 Депозит: $10,000
⚖️ Плечо: 1:100
📈 Вход: 1.0850
🛑 Стоп-лосс: 1.0800
⚠️ *Управление рисками:*
📦 Размер позиции: *{pos['position_size']:.2f} лотов*
💰 Риск на сделку: ${pos['risk_amount']:.2f}
📉 Стоп-лосс: {pos['stop_pips']:.0f} пунктов
💡 *Рекомендация:* Используйте профессиональный расчет для детального анализа
"""
    await update.message.reply_text(
        result,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("💼 Добавить в портфель", callback_data="portfolio_add_trade")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ])
    )
    return MAIN_MENU

@log_performance
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message:
        await update.message.reply_text(
            "❌ *PRO расчет отменен.*\n"
            "🚀 Используйте /start для нового PRO расчета\n"
            "📚 Используйте /info для PRO инструкций\n"
            "👨‍💻 *PRO Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)",
            parse_mode='Markdown',
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
    return ConversationHandler.END

def main():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ PRO Bot token not found!")
        return

    logger.info("🚀 Starting ULTRA-FAST PRO Risk Management Bot v3.0 (Analytics REMOVED)...")

    application = Application.builder().token(token).build()

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', start),
            CommandHandler('quick', quick_command),
            CommandHandler('portfolio', portfolio_command),
            CommandHandler('info', pro_info_command),
            CallbackQueryHandler(handle_main_menu, pattern='^(pro_calculation|quick_calculation|portfolio|pro_info|main_menu)$')
        ],
        states={
            MAIN_MENU: [
                CallbackQueryHandler(handle_main_menu, pattern='^(pro_calculation|quick_calculation|portfolio|pro_info|main_menu|portfolio_back)$')
            ],
            INSTRUMENT_TYPE: [CallbackQueryHandler(process_instrument_type, pattern='^inst_type_')],
            CURRENCY: [
                CallbackQueryHandler(process_currency_selection, pattern='^currency_'),
                CallbackQueryHandler(process_custom_currency, pattern='^custom_currency$'),
                CallbackQueryHandler(back_to_instrument, pattern='^back_to_instrument$'),
                CallbackQueryHandler(back_to_instrument, pattern='^pro_calculation$')
            ],
            CUSTOM_INSTRUMENT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_currency_input),
                CallbackQueryHandler(back_to_currency, pattern='^back_to_currency$')
            ],
            DIRECTION: [
                CallbackQueryHandler(process_direction, pattern='^direction_'),
                CallbackQueryHandler(back_to_currency, pattern='^back_to_currency$')
            ],
            RISK_PERCENT: [
                CallbackQueryHandler(process_risk_percent, pattern='^risk_'),
                CallbackQueryHandler(back_to_direction, pattern='^back_to_direction$')
            ],
            DEPOSIT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_deposit),
                CallbackQueryHandler(back_to_risk, pattern='^back_to_risk$')
            ],
            LEVERAGE: [
                CallbackQueryHandler(process_leverage, pattern='^leverage_'),
                CallbackQueryHandler(back_to_deposit, pattern='^back_to_deposit$')
            ],
            ENTRY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_entry),
                CallbackQueryHandler(back_to_leverage, pattern='^back_to_leverage$')
            ],
            STOP_LOSS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_stop_loss),
                CallbackQueryHandler(back_to_entry, pattern='^back_to_entry$')
            ],
            TAKE_PROFITS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_take_profits),
                CallbackQueryHandler(back_to_stop_loss, pattern='^back_to_stop_loss$')
            ],
            VOLUME_DISTRIBUTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_volume_distribution),
                CallbackQueryHandler(back_to_take_profits, pattern='^back_to_take_profits$')
            ],
            PORTFOLIO_MENU: [
                CallbackQueryHandler(portfolio_trades, pattern='^portfolio_trades$'),
                CallbackQueryHandler(portfolio_balance, pattern='^portfolio_balance$'),
                CallbackQueryHandler(portfolio_performance, pattern='^portfolio_performance$'),
                CallbackQueryHandler(portfolio_history, pattern='^portfolio_history$'),
                CallbackQueryHandler(portfolio_add_trade, pattern='^portfolio_add_trade$'),
                CallbackQueryHandler(portfolio_deposit, pattern='^portfolio_deposit$'),
                CallbackQueryHandler(handle_main_menu, pattern='^main_menu$'),
                CallbackQueryHandler(portfolio_back, pattern='^portfolio_back$')
            ],
        },
        fallbacks=[
            CommandHandler('cancel', cancel),
            CommandHandler('start', start)
        ],
        allow_reentry=True
    )

    application.add_handler(conv_handler)

    # Back navigation
    application.add_handler(CallbackQueryHandler(back_to_currency, pattern='^back_to_currency$'))
    application.add_handler(CallbackQueryHandler(back_to_direction, pattern='^back_to_direction$'))
    application.add_handler(CallbackQueryHandler(back_to_risk, pattern='^back_to_risk$'))
    application.add_handler(CallbackQueryHandler(back_to_deposit, pattern='^back_to_deposit$'))
    application.add_handler(CallbackQueryHandler(back_to_leverage, pattern='^back_to_leverage$'))
    application.add_handler(CallbackQueryHandler(back_to_entry, pattern='^back_to_entry$'))
    application.add_handler(CallbackQueryHandler(back_to_stop_loss, pattern='^back_to_stop_loss$'))
    application.add_handler(CallbackQueryHandler(back_to_take_profits, pattern='^back_to_take_profits$'))

    port = int(os.environ.get('PORT', 10000))
    webhook_url = os.getenv('RENDER_EXTERNAL_URL', 'https://risk-management-bot-pro.onrender.com')

    try:
        if webhook_url and "render.com" in webhook_url:
            application.run_webhook(
                listen="0.0.0.0",
                port=port,
                url_path="/webhook",
                webhook_url=webhook_url + "/webhook"
            )
        else:
            application.run_polling()
    except Exception as e:
        logger.error(f"❌ Error starting PRO bot: {e}")

if __name__ == '__main__':
    main()
