import os
import logging
import asyncio
import re
import time
import functools
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
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
logging.getLogger("httpx").setLevel(logging.WARNING)

# Conversation states
(
    MAIN_MENU, INSTRUMENT_TYPE, CUSTOM_INSTRUMENT, DIRECTION, 
    RISK_PERCENT, DEPOSIT, LEVERAGE, CURRENCY, ENTRY, 
    STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION,
    PORTFOLIO_MENU, ANALYTICS_MENU, TRADE_HISTORY, PERFORMANCE_ANALYSIS
) = range(16)

# Temporary storage
user_data: Dict[int, Dict[str, Any]] = {}

# MongoDB Connection
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb+srv://felixalseny_db_user:kontraktaciA22@felix22.3nx1ibi.mongodb.net/risk_bot_pro?retryWrites=true&w=majority&appName=Felix22')
mongo_client = None
db = None
users_collection = None
portfolio_collection = None

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.server_info()  # Test connection
    db = mongo_client.risk_bot_pro
    users_collection = db.users
    portfolio_collection = db.portfolios
    logger.info("✅ MongoDB connected successfully!")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    users_collection = None
    portfolio_collection = None

class MongoDBManager:
    @staticmethod
    def get_user_data(user_id: int) -> Dict[str, Any]:
        """Get user data from MongoDB"""
        if users_collection is None:
            return {}
        try:
            user_data = users_collection.find_one({"user_id": user_id})
            return user_data.get('data', {}) if user_data else {}
        except Exception as e:
            logger.error(f"Error getting user data: {e}")
            return {}

    @staticmethod
    def update_user_data(user_id: int, data: Dict[str, Any]):
        """Update user data in MongoDB"""
        if users_collection is None:
            return
        try:
            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"data": data, "last_updated": datetime.now()}},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error updating user data: {e}")

    @staticmethod
    def get_portfolio(user_id: int) -> Dict[str, Any]:
        """Get portfolio from MongoDB"""
        if portfolio_collection is None:
            return {}
        try:
            portfolio = portfolio_collection.find_one({"user_id": user_id})
            return portfolio.get('portfolio_data', {}) if portfolio else {}
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return {}

    @staticmethod
    def update_portfolio(user_id: int, portfolio_data: Dict[str, Any]):
        """Update portfolio in MongoDB"""
        if portfolio_collection is None:
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
        
        if not portfolio_data:
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
    def add_trade(user_id: int, trade_data: Dict):
        portfolio_data = PortfolioManager.initialize_user_portfolio(user_id)
        
        trade_id = len(portfolio_data['trades']) + 1
        trade_data['id'] = trade_id
        trade_data['timestamp'] = datetime.now().isoformat()
        
        portfolio_data['trades'].append(trade_data)
        
        # Update performance metrics
        PortfolioManager.update_performance_metrics(user_id, portfolio_data)
        
        # Update allocation
        instrument = trade_data.get('instrument', 'Unknown')
        if instrument not in portfolio_data['allocation']:
            portfolio_data['allocation'][instrument] = 0
        portfolio_data['allocation'][instrument] += 1
        
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
    def update_performance_metrics(user_id: int, portfolio_data: Dict):
        trades = portfolio_data['trades']
        
        if not trades:
            return
        
        closed_trades = [t for t in trades if t.get('status') == 'closed']
        winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('profit', 0) <= 0]
        
        portfolio_data['performance']['total_trades'] = len(closed_trades)
        portfolio_data['performance']['winning_trades'] = len(winning_trades)
        portfolio_data['performance']['losing_trades'] = len(losing_trades)
        portfolio_data['performance']['total_profit'] = sum(t.get('profit', 0) for t in winning_trades)
        portfolio_data['performance']['total_loss'] = sum(t.get('profit', 0) for t in losing_trades)
        
        if closed_trades:
            portfolio_data['performance']['win_rate'] = (len(winning_trades) / len(closed_trades)) * 100
            portfolio_data['performance']['average_profit'] = (
                portfolio_data['performance']['total_profit'] / len(winning_trades) 
                if winning_trades else 0
            )
            portfolio_data['performance']['average_loss'] = (
                portfolio_data['performance']['total_loss'] / len(losing_trades) 
                if losing_trades else 0
            )
        
        MongoDBManager.update_portfolio(user_id, portfolio_data)

# Flask server for health checks
app = Flask(__name__)

@app.route('/')
def home():
    return "🤖 PRO Risk Management Bot is ALIVE and RUNNING!"

@app.route('/health')
def health():
    try:
        # Test MongoDB connection
        mongo_ok = False
        if users_collection is not None:
            users_collection.find_one({})
            mongo_ok = True
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "mongodb": "connected" if mongo_ok else "disconnected",
                "telegram_bot": "running"
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook endpoint for Telegram"""
    return "OK", 200

def run_flask():
    port = int(os.environ.get('FLASK_PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

# Start Flask in background
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
logger.info("✅ Flask health check server started!")

# Keep-alive to prevent Render sleep
def keep_alive():
    try:
        bot_url = os.getenv('RENDER_EXTERNAL_URL', 'https://risk-management-bot-pro.onrender.com')
        if bot_url:
            response = requests.get(f"{bot_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Keep-alive ping successful")
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

# === ОСНОВНЫЕ ОБРАБОТЧИКИ ===

@log_performance
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Main menu"""
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
    
    # Get user data from MongoDB
    user_data_dict = MongoDBManager.get_user_data(user_id)
    old_presets = user_data_dict.get('presets', [])
    
    # Update user data in MongoDB
    updated_data = {
        'start_time': datetime.now().isoformat(),
        'last_activity': time.time(),
        'presets': old_presets,
        'username': user.username,
        'first_name': user.first_name
    }
    MongoDBManager.update_user_data(user_id, updated_data)
    
    keyboard = [
        [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("⚡ Быстрый расчет", callback_data="quick_calculation")],
        [InlineKeyboardButton("💼 Мой портфель", callback_data="portfolio")],
        [InlineKeyboardButton("📈 Аналитика", callback_data="analytics")],
        [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")]
    ]
    
    if update.message:
        await update.message.reply_text(
            welcome_text, 
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    else:
        await update.callback_query.edit_message_text(
            welcome_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    return MAIN_MENU

@log_performance
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle main menu callbacks"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == "pro_calculation":
        return await start_pro_calculation(update, context)
    elif data == "quick_calculation":
        return await start_quick_calculation(update, context)
    elif data == "portfolio":
        return await show_portfolio_menu(update, context)
    elif data == "analytics":
        return await show_analytics_menu(update, context)
    elif data == "pro_info":
        return await show_pro_info(update, context)
    elif data == "main_menu":
        return await start(update, context)
    
    return MAIN_MENU

@log_performance
async def start_pro_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start professional calculation"""
    keyboard = []
    for key, value in INSTRUMENT_TYPES.items():
        keyboard.append([InlineKeyboardButton(value, callback_data=f"inst_type_{key}")])
    
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="main_menu")])
    
    await update.callback_query.edit_message_text(
        "📊 *Профессиональный расчет*\n\n"
        "Выберите тип инструмента:",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return INSTRUMENT_TYPE

@log_performance
async def start_quick_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start quick calculation"""
    # For now, we'll use the same as pro calculation but with default parameters
    # In a real bot, you might simplify the steps
    await update.callback_query.edit_message_text(
        "⚡ *Быстрый расчет*\n\n"
        "Эта функция в разработке. Используйте профессиональный расчет для полного анализа.",
        parse_mode='Markdown'
    )
    return MAIN_MENU

@log_performance
async def show_portfolio_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show portfolio menu"""
    user_id = update.callback_query.from_user.id
    portfolio_data = PortfolioManager.initialize_user_portfolio(user_id)
    
    performance = portfolio_data['performance']
    
    text = f"""
💼 *Мой Портфель*

*Баланс:* ${portfolio_data['current_balance']:,.2f}
*Сделок:* {performance['total_trades']}
*Винрейт:* {performance['win_rate']:.1f}%
*Общий профит:* ${performance['total_profit']:,.2f}
"""
    
    keyboard = [
        [InlineKeyboardButton("📊 Мои сделки", callback_data="portfolio_trades")],
        [InlineKeyboardButton("💰 Баланс", callback_data="portfolio_balance")],
        [InlineKeyboardButton("📈 Производительность", callback_data="portfolio_performance")],
        [InlineKeyboardButton("📋 История", callback_data="portfolio_history")],
        [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
        [InlineKeyboardButton("💳 Пополнить депозит", callback_data="portfolio_deposit")],
        [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
    ]
    
    await update.callback_query.edit_message_text(
        text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return PORTFOLIO_MENU

@log_performance
async def show_analytics_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show analytics menu"""
    text = """
📈 *Аналитика и Отчеты*

Здесь вы можете получить детальную аналитику по вашим сделкам и стратегии.
"""
    
    keyboard = [
        [InlineKeyboardButton("⚖️ Анализ риск/вознаграждение", callback_data="analytics_risk_reward")],
        [InlineKeyboardButton("📊 Эффективность стратегии", callback_data="analytics_strategy_perf")],
        [InlineKeyboardButton("📉 Статистика сделок", callback_data="analytics_trade_stats")],
        [InlineKeyboardButton("🔍 Оптимизация рисков", callback_data="analytics_optimization")],
        [InlineKeyboardButton("💡 Рекомендации", callback_data="analytics_recommendations")],
        [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
    ]
    
    await update.callback_query.edit_message_text(
        text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return ANALYTICS_MENU

@log_performance
async def show_pro_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show PRO instructions"""
    text = """
📚 *PRO Инструкции*

*🎯 Калькулятор Риск-Менеджмента v3.0*

*Основные функции:*
• 📊 Профессиональный расчет - полный анализ позиции с учетом риска, плеча, объема
• ⚡ Быстрый расчет - упрощенный расчет для быстрых решений
• 💼 Портфель - отслеживание сделок и баланса
• 📈 Аналитика - детальная аналитика и рекомендации

*Как использовать:*
1. Выберите тип инструмента (Forex, Крипто, Индексы и т.д.)
2. Укажите параметры сделки: направление, риск, депозит, плечо
3. Введите цены: вход, стоп-лосс, тейк-профиты
4. Распределите объем между тейк-профитами
5. Получите детальный расчет позиции

*Рекомендации по рискам:*
• Риск на сделку: 1-2% от депозита
• Соотношение риск/вознаграждение: минимум 1:2
• Диверсифицируйте портфель
"""
    
    keyboard = [
        [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
    ]
    
    await update.callback_query.edit_message_text(
        text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return MAIN_MENU

@log_performance
async def quick_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /quick command"""
    return await start(update, context)

@log_performance
async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /portfolio command"""
    return await show_portfolio_menu(update, context)

@log_performance
async def analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /analytics command"""
    return await show_analytics_menu(update, context)

@log_performance
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /info command"""
    return await show_pro_info(update, context)

@log_performance
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel conversation"""
    if update.message:
        await update.message.reply_text(
            "❌ Операция отменена.",
            reply_markup=ReplyKeyboardRemove()
        )
    elif update.callback_query:
        await update.callback_query.edit_message_text(
            "❌ Операция отменена."
        )
    
    return ConversationHandler.END

# === PORTFOLIO HANDLERS ===

@log_performance
async def portfolio_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show portfolio trades"""
    user_id = update.callback_query.from_user.id
    portfolio_data = PortfolioManager.initialize_user_portfolio(user_id)
    
    trades = portfolio_data['trades']
    
    if not trades:
        text = "📊 *Мои Сделки*\n\nУ вас пока нет сделок."
    else:
        text = "📊 *Мои Сделки*\n\n"
        for trade in trades[-10:]:  # Show last 10 trades
            status = "🟢 Открыта" if trade.get('status') == 'open' else "🔴 Закрыта"
            profit = trade.get('profit', 0)
            profit_text = f"+${profit:.2f}" if profit > 0 else f"-${abs(profit):.2f}"
            text += f"• {trade.get('instrument', 'Unknown')} - {status} - {profit_text}\n"
    
    keyboard = [[InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio")]]
    await update.callback_query.edit_message_text(
        text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return PORTFOLIO_MENU

@log_performance
async def portfolio_balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show portfolio balance"""
    user_id = update.callback_query.from_user.id
    portfolio_data = PortfolioManager.initialize_user_portfolio(user_id)
    
    text = f"""
💰 *Баланс Портфеля*

*Начальный депозит:* ${portfolio_data['initial_balance']:,.2f}
*Текущий баланс:* ${portfolio_data['current_balance']:,.2f}
*Общий PnL:* ${portfolio_data['current_balance'] - portfolio_data['initial_balance']:,.2f}
*ROI:* {((portfolio_data['current_balance'] / portfolio_data['initial_balance'] - 1) * 100):.2f}%
"""
    
    keyboard = [[InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio")]]
    await update.callback_query.edit_message_text(
        text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return PORTFOLIO_MENU

@log_performance
async def portfolio_performance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show portfolio performance"""
    user_id = update.callback_query.from_user.id
    portfolio_data = PortfolioManager.initialize_user_portfolio(user_id)
    
    perf = portfolio_data['performance']
    
    text = f"""
📈 *Производительность*

*Всего сделок:* {perf['total_trades']}
*Выигрышных:* {perf['winning_trades']}
*Проигрышных:* {perf['losing_trades']}
*Винрейт:* {perf['win_rate']:.1f}%
*Средний профит:* ${perf['average_profit']:.2f}
*Средний убыток:* ${perf['average_loss']:.2f}
*Общий профит:* ${perf['total_profit']:.2f}
*Общий убыток:* ${perf['total_loss']:.2f}
"""
    
    keyboard = [[InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio")]]
    await update.callback_query.edit_message_text(
        text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return PORTFOLIO_MENU

@log_performance
async def portfolio_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show portfolio history"""
    user_id = update.callback_query.from_user.id
    portfolio_data = PortfolioManager.initialize_user_portfolio(user_id)
    
    history = portfolio_data['history']
    
    if not history:
        text = "📋 *История*\n\nИстория операций пуста."
    else:
        text = "📋 *История Операций*\n\n"
        for event in history[-10:]:  # Show last 10 events
            if event['type'] == 'trade':
                action = "📈 Открыта" if event['action'] == 'open' else "📉 Закрыта"
                profit_text = f" | PnL: ${event['profit']:.2f}" if event.get('profit') else ""
                text += f"• {action} {event['instrument']}{profit_text}\n"
    
    keyboard = [[InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio")]]
    await update.callback_query.edit_message_text(
        text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return PORTFOLIO_MENU

@log_performance
async def portfolio_add_trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Add trade to portfolio"""
    # This would typically open a conversation to add trade details
    # For now, we'll add a sample trade
    user_id = update.callback_query.from_user.id
    
    sample_trade = {
        'instrument': 'EURUSD',
        'direction': 'BUY',
        'entry_price': 1.0850,
        'exit_price': 1.0900,
        'volume': 0.1,
        'profit': 50.0,
        'status': 'closed',
        'risk_percent': 2.0
    }
    
    PortfolioManager.add_trade(user_id, sample_trade)
    
    await update.callback_query.edit_message_text(
        "✅ *Сделка добавлена!*\n\nПример сделки EURUSD добавлен в ваш портфель.",
        parse_mode='Markdown'
    )
    return await show_portfolio_menu(update, context)

@log_performance
async def portfolio_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle portfolio deposit"""
    # This would typically open a conversation to add deposit
    # For now, we'll show a message
    await update.callback_query.edit_message_text(
        "💳 *Пополнение Депозита*\n\nЭта функция в разработке. В будущем вы сможете пополнять депозит через различные платежные системы.",
        parse_mode='Markdown'
    )
    return PORTFOLIO_MENU

@log_performance
async def portfolio_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Back to portfolio menu"""
    return await show_portfolio_menu(update, context)

# === ANALYTICS HANDLERS ===

@log_performance
async def analytics_risk_reward(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show risk/reward analysis"""
    user_id = update.callback_query.from_user.id
    portfolio_data = PortfolioManager.initialize_user_portfolio(user_id)
    
    trades = portfolio_data['trades']
    analysis = AnalyticsEngine.calculate_risk_reward_analysis(trades)
    
    text = f"""
⚖️ *Анализ Риск/Вознаграждение*

*Среднее соотношение R/R:* {analysis['average_risk_reward']:.2f}:1
*Лучшая сделка:* ${analysis['best_trade']:.2f}
*Худшая сделка:* ${analysis['worst_trade']:.2f}
*Консистентность:* {analysis['consistency_score']:.1f}%
*Оценка риска:* {analysis['risk_score']:.1f}/100

*Рекомендации:*
• Целевое соотношение R/R: 1:3
• Риск на сделку: 1-2%
• Диверсификация инструментов
"""
    
    keyboard = [[InlineKeyboardButton("🔙 Назад к аналитике", callback_data="analytics")]]
    await update.callback_query.edit_message_text(
        text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return ANALYTICS_MENU

@log_performance
async def analytics_strategy_perf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show strategy performance"""
    user_id = update.callback_query.from_user.id
    portfolio_data = PortfolioManager.initialize_user_portfolio(user_id)
    
    perf = portfolio_data['performance']
    allocation = portfolio_data.get('allocation', {})
    
    text = f"""
📊 *Эффективность Стратегии*

*Общая статистика:*
• Винрейт: {perf['win_rate']:.1f}%
• Всего сделок: {perf['total_trades']}
• Средний профит: ${perf['average_profit']:.2f}
• Средний убыток: ${perf['average_loss']:.2f}

*Распределение по инструментам:*
"""
    
    for instrument, count in list(allocation.items())[:5]:  # Show top 5
        text += f"• {instrument}: {count} сделок\n"
    
    if perf['win_rate'] > 60:
        rating = "⭐️⭐️⭐️⭐️⭐️ (Отлично)"
    elif perf['win_rate'] > 50:
        rating = "⭐️⭐️⭐️⭐️ (Хорошо)"
    elif perf['win_rate'] > 40:
        rating = "⭐️⭐️⭐️ (Удовлетворительно)"
    else:
        rating = "⭐️⭐️ (Требует улучшения)"
    
    text += f"\n*Оценка стратегии:* {rating}"
    
    keyboard = [[InlineKeyboardButton("🔙 Назад к аналитике", callback_data="analytics")]]
    await update.callback_query.edit_message_text(
        text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return ANALYTICS_MENU

@log_performance
async def analytics_trade_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show trade statistics"""
    user_id = update.callback_query.from_user.id
    portfolio_data = PortfolioManager.initialize_user_portfolio(user_id)
    
    trades = portfolio_data['trades']
    closed_trades = [t for t in trades if t.get('status') == 'closed']
    
    if not closed_trades:
        text = "📉 *Статистика Сделок*\n\nНет закрытых сделок для анализа."
    else:
        profits = [t.get('profit', 0) for t in closed_trades]
        total_profit = sum(profits)
        avg_profit = total_profit / len(profits)
        max_profit = max(profits)
        max_loss = min(profits)
        
        text = f"""
📉 *Детальная Статистика Сделок*

*Основные метрики:*
• Всего сделок: {len(closed_trades)}
• Общий PnL: ${total_profit:.2f}
• Средний PnL: ${avg_profit:.2f}
• Макс. профит: ${max_profit:.2f}
• Макс. убыток: ${max_loss:.2f}

*Распределение:*
• Профитных: {len([p for p in profits if p > 0])}
• Убыточных: {len([p for p in profits if p <= 0])}
"""
    
    keyboard = [[InlineKeyboardButton("🔙 Назад к аналитике", callback_data="analytics")]]
    await update.callback_query.edit_message_text(
        text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return ANALYTICS_MENU

@log_performance
async def analytics_optimization(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show risk optimization suggestions"""
    user_id = update.callback_query.from_user.id
    portfolio_data = PortfolioManager.initialize_user_portfolio(user_id)
    
    trades = portfolio_data['trades']
    closed_trades = [t for t in trades if t.get('status') == 'closed']
    
    text = """
🔍 *Оптимизация Рисков*

*Рекомендации по улучшению:*

1. *Управление рисками:*
   • Риск на сделку: 1-2% от депозита
   • Соотношение R/R: минимум 1:2
   • Стоп-лосс обязателен для каждой сделки

2. *Психология трейдинга:*
   • Следуйте торговому плану
   • Избегайте эмоциональных решений
   • Регулярно анализируйте результаты

3. *Управление капиталом:*
   • Диверсификация по инструментам
   • Корреляционный анализ
   • Постепенное увеличение объема
"""
    
    if closed_trades:
        risk_levels = [t.get('risk_percent', 0) for t in closed_trades]
        avg_risk = sum(risk_levels) / len(risk_levels) if risk_levels else 0
        
        if avg_risk > 3:
            text += f"\n*Внимание!* Средний риск ({avg_risk:.1f}%) слишком высок. Рекомендуется снизить до 1-2%."
        elif avg_risk < 0.5:
            text += f"\n*Возможность!* Средний риск ({avg_risk:.1f}%) низкий. Можно рассмотреть увеличение до 1-2%."
    
    keyboard = [[InlineKeyboardButton("🔙 Назад к аналитике", callback_data="analytics")]]
    await update.callback_query.edit_message_text(
        text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return ANALYTICS_MENU

@log_performance
async def analytics_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show personalized recommendations"""
    user_id = update.callback_query.from_user.id
    portfolio_data = PortfolioManager.initialize_user_portfolio(user_id)
    
    recommendations = AnalyticsEngine.generate_strategy_recommendations(portfolio_data)
    
    text = """
💡 *Персонализированные Рекомендации*

*Для улучшения ваших результатов:*
"""
    
    for i, rec in enumerate(recommendations, 1):
        text += f"{i}. {rec}\n"
    
    text += "\n*Общие советы:*"
    text += """
• Регулярно обновляйте торговый журнал
• Анализируйте ошибки и успехи
• Следите за экономическим календарем
• Используйте технический и фундаментальный анализ
"""
    
    keyboard = [[InlineKeyboardButton("🔙 Назад к аналитике", callback_data="analytics")]]
    await update.callback_query.edit_message_text(
        text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return ANALYTICS_MENU

@log_performance
async def analytics_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Back to analytics menu"""
    return await show_analytics_menu(update, context)

# === CALCULATION PROCESS HANDLERS ===

@log_performance
async def process_instrument_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process instrument type selection"""
    query = update.callback_query
    await query.answer()
    
    instrument_type = query.data.replace('inst_type_', '')
    context.user_data['instrument_type'] = instrument_type
    
    # Get presets for this instrument type
    presets = INSTRUMENT_PRESETS.get(instrument_type, [])
    
    keyboard = []
    for preset in presets:
        keyboard.append([InlineKeyboardButton(preset, callback_data=f"currency_{preset}")])
    
    keyboard.append([InlineKeyboardButton("✏️ Ввести вручную", callback_data="custom_currency")])
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_instrument")])
    
    await query.edit_message_text(
        f"📊 *{INSTRUMENT_TYPES[instrument_type]}*\n\n"
        "Выберите инструмент из списка или введите вручную:",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return CURRENCY

@log_performance
async def process_currency_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process currency selection from presets"""
    query = update.callback_query
    await query.answer()
    
    currency_pair = query.data.replace('currency_', '')
    context.user_data['currency_pair'] = currency_pair
    
    keyboard = []
    for direction in TRADE_DIRECTIONS:
        keyboard.append([InlineKeyboardButton(direction, callback_data=f"direction_{direction}")])
    
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_instrument")])
    
    await query.edit_message_text(
        f"💱 *Инструмент:* {currency_pair}\n\n"
        "Выберите направление сделки:",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return DIRECTION

@log_performance
async def process_custom_currency(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process custom currency input request"""
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text(
        "💱 *Ввод инструмента*\n\n"
        "Введите название инструмента вручную (например: EURUSD, BTCUSD, GOLD):",
        parse_mode='Markdown'
    )
    return CUSTOM_INSTRUMENT

@log_performance
async def process_currency_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process custom currency input"""
    currency_pair = update.message.text.upper().strip()
    context.user_data['currency_pair'] = currency_pair
    
    keyboard = []
    for direction in TRADE_DIRECTIONS:
        keyboard.append([InlineKeyboardButton(direction, callback_data=f"direction_{direction}")])
    
    await update.message.reply_text(
        f"💱 *Инструмент:* {currency_pair}\n\n"
        "Выберите направление сделки:",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return DIRECTION

@log_performance
async def back_to_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Go back to instrument selection"""
    return await start_pro_calculation(update, context)

@log_performance
async def process_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process trade direction"""
    query = update.callback_query
    await query.answer()
    
    direction = query.data.replace('direction_', '')
    context.user_data['direction'] = direction
    
    keyboard = []
    for risk_level in RISK_LEVELS:
        keyboard.append([InlineKeyboardButton(risk_level, callback_data=f"risk_{risk_level}")])
    
    await query.edit_message_text(
        f"📊 *Направление:* {direction}\n\n"
        "Выберите уровень риска (% от депозита):",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return RISK_PERCENT

@log_performance
async def process_risk_percent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process risk percentage"""
    query = update.callback_query
    await query.answer()
    
    risk_percent = float(query.data.replace('risk_', '').replace('%', '')) / 100
    context.user_data['risk_percent'] = risk_percent
    
    await query.edit_message_text(
        f"⚡ *Риск:* {risk_percent*100}%\n\n"
        "💵 Введите размер депозита (в USD):\n\n"
        "Пример: 1000",
        parse_mode='Markdown'
    )
    return DEPOSIT

@log_performance
async def process_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process deposit amount"""
    try:
        deposit = float(update.message.text.replace(',', '.'))
        if deposit <= 0:
            await update.message.reply_text("❌ Депозит должен быть положительным числом. Попробуйте снова:")
            return DEPOSIT
        
        context.user_data['deposit'] = deposit
        
        keyboard = []
        for leverage in LEVERAGES:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"leverage_{leverage}")])
        
        await update.message.reply_text(
            f"💰 *Депозит:* ${deposit:,.2f}\n\n"
            "Выберите плечо:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return LEVERAGE
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректное число для депозита:")
        return DEPOSIT

@log_performance
async def process_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process leverage selection"""
    query = update.callback_query
    await query.answer()
    
    leverage = query.data.replace('leverage_', '')
    context.user_data['leverage'] = leverage
    
    await query.edit_message_text(
        f"📈 *Плечо:* {leverage}\n\n"
        "💹 Введите цену входа:\n\n"
        "Пример: 1.0850",
        parse_mode='Markdown'
    )
    return ENTRY

@log_performance
async def process_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process entry price"""
    try:
        entry_price = float(update.message.text.replace(',', '.'))
        if entry_price <= 0:
            await update.message.reply_text("❌ Цена входа должна быть положительной. Попробуйте снова:")
            return ENTRY
        
        context.user_data['entry_price'] = entry_price
        
        await update.message.reply_text(
            f"🎯 *Цена входа:* {entry_price}\n\n"
            "🛑 Введите цену стоп-лосса:\n\n"
            "Пример: 1.0800",
            parse_mode='Markdown'
        )
        return STOP_LOSS
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректное число для цены входа:")
        return ENTRY

@log_performance
async def process_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process stop loss price"""
    try:
        stop_loss = float(update.message.text.replace(',', '.'))
        entry_price = context.user_data.get('entry_price', 0)
        
        if stop_loss <= 0:
            await update.message.reply_text("❌ Цена стоп-лосса должна быть положительной. Попробуйте снова:")
            return STOP_LOSS
        
        direction = context.user_data.get('direction', 'BUY')
        if direction == 'BUY' and stop_loss >= entry_price:
            await update.message.reply_text("❌ Для BUY стоп-лосс должен быть ниже цены входа. Попробуйте снова:")
            return STOP_LOSS
        elif direction == 'SELL' and stop_loss <= entry_price:
            await update.message.reply_text("❌ Для SELL стоп-лосс должен быть выше цены входа. Попробуйте снова:")
            return STOP_LOSS
        
        context.user_data['stop_loss'] = stop_loss
        
        await update.message.reply_text(
            f"🛑 *Стоп-лосс:* {stop_loss}\n\n"
            "🎯 Введите цены тейк-профитов через запятую:\n\n"
            "Пример: 1.0900, 1.0950, 1.1000",
            parse_mode='Markdown'
        )
        return TAKE_PROFITS
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректное число для стоп-лосса:")
        return STOP_LOSS

@log_performance
async def process_take_profits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process take profit prices"""
    try:
        take_profits_text = update.message.text.replace(',', '.').split(',')
        take_profits = [float(tp.strip()) for tp in take_profits_text if tp.strip()]
        
        if not take_profits:
            await update.message.reply_text("❌ Пожалуйста, введите хотя бы один тейк-профит:")
            return TAKE_PROFITS
        
        entry_price = context.user_data.get('entry_price', 0)
        direction = context.user_data.get('direction', 'BUY')
        
        # Validate take profit prices based on direction
        for tp in take_profits:
            if direction == 'BUY' and tp <= entry_price:
                await update.message.reply_text("❌ Для BUY тейк-профиты должны быть выше цены входа. Попробуйте снова:")
                return TAKE_PROFITS
            elif direction == 'SELL' and tp >= entry_price:
                await update.message.reply_text("❌ Для SELL тейк-профиты должны быть ниже цены входа. Попробуйте снова:")
                return TAKE_PROFITS
        
        context.user_data['take_profits'] = take_profits
        
        await update.message.reply_text(
            f"🎯 *Тейк-профиты:* {', '.join(map(str, take_profits))}\n\n"
            "📊 Введите распределение объема между тейк-профитами в % через запятую (сумма должна быть 100%):\n\n"
            "Пример для 3 тейк-профитов: 50, 30, 20",
            parse_mode='Markdown'
        )
        return VOLUME_DISTRIBUTION
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректные числа для тейк-профитов:")
        return TAKE_PROFITS

@log_performance
async def process_volume_distribution(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process volume distribution"""
    try:
        volume_text = update.message.text.replace(',', '.').split(',')
        volume_distribution = [float(vol.strip()) for vol in volume_text if vol.strip()]
        
        if not volume_distribution:
            await update.message.reply_text("❌ Пожалуйста, введите распределение объема:")
            return VOLUME_DISTRIBUTION
        
        # Check if sum is approximately 100%
        total_volume = sum(volume_distribution)
        if abs(total_volume - 100) > 0.1:
            await update.message.reply_text(f"❌ Сумма распределения объема должна быть 100% (сейчас: {total_volume}%). Попробуйте снова:")
            return VOLUME_DISTRIBUTION
        
        take_profits = context.user_data.get('take_profits', [])
        if len(volume_distribution) != len(take_profits):
            await update.message.reply_text(f"❌ Количество элементов распределения объема ({len(volume_distribution)}) должно совпадать с количеством тейк-профитов ({len(take_profits)}). Попробуйте снова:")
            return VOLUME_DISTRIBUTION
        
        context.user_data['volume_distribution'] = volume_distribution
        
        # Perform final calculation and show results
        return await show_calculation_results(update, context)
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректные числа для распределения объема:")
        return VOLUME_DISTRIBUTION

@log_performance
async def show_calculation_results(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show final calculation results"""
    user_data = context.user_data
    
    # Extract all parameters
    instrument_type = user_data.get('instrument_type', 'forex')
    currency_pair = user_data.get('currency_pair', 'EURUSD')
    direction = user_data.get('direction', 'BUY')
    risk_percent = user_data.get('risk_percent', 0.02)
    deposit = user_data.get('deposit', 1000)
    leverage = user_data.get('leverage', '1:100')
    entry_price = user_data.get('entry_price', 1.0)
    stop_loss = user_data.get('stop_loss', 0.9)
    take_profits = user_data.get('take_profits', [1.1])
    volume_distribution = user_data.get('volume_distribution', [100])
    
    # Calculate position size and risk
    calculation_result = FastRiskCalculator.calculate_position_size_fast(
        deposit=deposit,
        leverage=leverage,
        instrument_type=instrument_type,
        currency_pair=currency_pair,
        entry_price=entry_price,
        stop_loss=stop_loss,
        direction=direction,
        risk_percent=risk_percent
    )
    
    position_size = calculation_result['position_size']
    risk_amount = calculation_result['risk_amount']
    stop_pips = calculation_result['stop_pips']
    required_margin = calculation_result['required_margin']
    free_margin = calculation_result['free_margin']
    
    # Calculate profits for each take profit level
    profit_calculations = FastRiskCalculator.calculate_profits_fast(
        instrument_type=instrument_type,
        currency_pair=currency_pair,
        entry_price=entry_price,
        take_profits=take_profits,
        position_size=position_size,
        volume_distribution=volume_distribution,
        direction=direction
    )
    
    # Build results text
    results_text = f"""
🎯 *РЕЗУЛЬТАТЫ РАСЧЕТА*

*Основные параметры:*
• Инструмент: {currency_pair} ({INSTRUMENT_TYPES.get(instrument_type, instrument_type)})
• Направление: {direction}
• Депозит: ${deposit:,.2f}
• Плечо: {leverage}
• Риск: {risk_percent*100}% (${risk_amount:.2f})

*Цены:*
• Вход: {entry_price}
• Стоп-лосс: {stop_loss}
• Расстояние: {stop_pips:.1f} пунктов

*Позиция:*
• Размер позиции: {position_size:.2f} лотов
• Требуемая маржа: ${required_margin:.2f}
• Свободная маржа: ${free_margin:.2f}

*Тейк-профиты:*
"""
    
    for i, profit_calc in enumerate(profit_calculations):
        results_text += f"""
TP{profit_calc['level']}:
  • Цена: {profit_calc['price']}
  • Объем: {profit_calc['volume_percent']}% ({profit_calc['volume_lots']:.2f} лотов)
  • Профит: ${profit_calc['profit']:.2f}
  • ROI: {profit_calc['roi_percent']:.1f}%
"""
    
    total_profit = profit_calculations[-1]['cumulative_profit'] if profit_calculations else 0
    risk_reward_ratio = total_profit / risk_amount if risk_amount > 0 else 0
    
    results_text += f"""
*Итоги:*
• Общий потенциальный профит: ${total_profit:.2f}
• Соотношение R/R: {risk_reward_ratio:.2f}:1
• Риск на сделку: {risk_percent*100}% от депозита
"""
    
    # Add recommendation based on risk/reward
    if risk_reward_ratio >= 2:
        results_text += "\n✅ *Отличное соотношение риск/вознаграждение!*"
    elif risk_reward_ratio >= 1:
        results_text += "\n⚠️ *Соотношение приемлемое, но можно улучшить.*"
    else:
        results_text += "\n❌ *Соотношение риск/вознаграждение неблагоприятное!*"
    
    keyboard = [
        [InlineKeyboardButton("💾 Сохранить в портфель", callback_data="save_to_portfolio")],
        [InlineKeyboardButton("🔄 Новый расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
    ]
    
    if update.message:
        await update.message.reply_text(
            results_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    else:
        await update.callback_query.edit_message_text(
            results_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    return MAIN_MENU

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

# Analytics Engine
class AnalyticsEngine:
    @staticmethod
    def calculate_risk_reward_analysis(trades: List[Dict]) -> Dict[str, Any]:
        closed_trades = [t for t in trades if t.get('status') == 'closed']
        if not closed_trades:
            return {
                'average_risk_reward': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'consistency_score': 0,
                'risk_score': 0
            }
        risk_reward_ratios = []
        profits = []
        for trade in closed_trades:
            risk = trade.get('risk_amount', 0)
            profit = trade.get('profit', 0)
            if risk > 0:
                risk_reward_ratios.append(abs(profit / risk))
            profits.append(profit)
        return {
            'average_risk_reward': sum(risk_reward_ratios) / len(risk_reward_ratios) if risk_reward_ratios else 0,
            'best_trade': max(profits) if profits else 0,
            'worst_trade': min(profits) if profits else 0,
            'consistency_score': AnalyticsEngine.calculate_consistency(profits),
            'risk_score': AnalyticsEngine.calculate_risk_score(profits)
        }

    @staticmethod
    def calculate_consistency(profits: List[float]) -> float:
        if len(profits) < 2:
            return 0
        positive_profits = [p for p in profits if p > 0]
        if not positive_profits:
            return 0
        return (len(positive_profits) / len(profits)) * 100

    @staticmethod
    def calculate_risk_score(profits: List[float]) -> float:
        if not profits:
            return 0
        avg_profit = sum(profits) / len(profits)
        if avg_profit == 0:
            return 0
        positive_count = len([p for p in profits if p > 0])
        return (positive_count / len(profits)) * 100

    @staticmethod
    def generate_strategy_recommendations(portfolio: Dict) -> List[str]:
        recommendations = []
        performance = portfolio.get('performance', {})
        win_rate = performance.get('win_rate', 0)
        avg_profit = performance.get('average_profit', 0)
        avg_loss = performance.get('average_loss', 0)
        if win_rate < 40:
            recommendations.append("📉 Рассмотрите снижение риска на сделку до 1-2%")
            recommendations.append("🎯 Увеличьте соотношение риск/вознаграждение до 1:3")
        if avg_profit < abs(avg_loss) and win_rate > 50:
            recommendations.append("⚡ Улучшите управление позицией - фиксируйте прибыль раньше")
        if len(portfolio.get('allocation', {})) < 3:
            recommendations.append("🌐 Диверсифицируйте портфель - торгуйте разные инструменты")
        if not recommendations:
            recommendations.append("✅ Ваша стратегия показывает хорошие результаты! Продолжайте в том же духе")
        return recommendations

# Ultra-fast risk calculator
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
            cached_result = fast_cache.get(cache_key)
            if cached_result:
                return cached_result

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

            pip_value_per_lot = FastRiskCalculator.calculate_pip_value_fast(
                instrument_type, currency_pair, 1.0
            )

            if stop_pips > 0 and pip_value_per_lot > 0:
                max_lots_by_risk = risk_amount / (stop_pips * pip_value_per_lot)
            else:
                max_lots_by_risk = 0

            contract_size = CONTRACT_SIZES.get(instrument_type, 100000)
            if entry_price > 0:
                max_lots_by_margin = (deposit * lev_value) / (contract_size * entry_price)
            else:
                max_lots_by_margin = 0

            position_size = min(max_lots_by_risk, max_lots_by_margin, 50.0)
            if position_size < 0.01:
                position_size = 0.01
            else:
                position_size = round(position_size * 100) / 100

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
            logger.error(f"Error in fast position size calculation: {e}")
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
            pip_value = FastRiskCalculator.calculate_pip_value_fast(
                instrument_type, currency_pair, volume_lots
            )
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

# Ultra-fast cache manager
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

# Global cache
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

# Главная функция запуска
def main():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ PRO Bot token not found!")
        return

    logger.info("🚀 Starting ULTRA-FAST PRO Risk Management Bot v3.0 with Enhanced Portfolio & Analytics...")

    application = Application.builder().token(token).build()

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', start),
            CommandHandler('quick', quick_command),
            CommandHandler('portfolio', portfolio_command),
            CommandHandler('analytics', analytics_command),
            CommandHandler('info', pro_info_command),
            CallbackQueryHandler(handle_main_menu, pattern='^(pro_calculation|quick_calculation|portfolio|analytics|pro_info|main_menu)$')
        ],
        states={
            MAIN_MENU: [CallbackQueryHandler(handle_main_menu)],
            INSTRUMENT_TYPE: [CallbackQueryHandler(process_instrument_type, pattern='^inst_type_')],
            CURRENCY: [
                CallbackQueryHandler(process_currency_selection, pattern='^currency_'),
                CallbackQueryHandler(process_custom_currency, pattern='^custom_currency$'),
                CallbackQueryHandler(back_to_instrument, pattern='^back_to_instrument$')
            ],
            CUSTOM_INSTRUMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_currency_input)],
            DIRECTION: [CallbackQueryHandler(process_direction, pattern='^direction_')],
            RISK_PERCENT: [CallbackQueryHandler(process_risk_percent, pattern='^risk_')],
            DEPOSIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_deposit)],
            LEVERAGE: [CallbackQueryHandler(process_leverage, pattern='^leverage_')],
            ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_entry)],
            STOP_LOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_stop_loss)],
            TAKE_PROFITS: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_take_profits)],
            VOLUME_DISTRIBUTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_volume_distribution)],
            PORTFOLIO_MENU: [
                CallbackQueryHandler(portfolio_trades, pattern='^portfolio_trades$'),
                CallbackQueryHandler(portfolio_balance, pattern='^portfolio_balance$'),
                CallbackQueryHandler(portfolio_performance, pattern='^portfolio_performance$'),
                CallbackQueryHandler(portfolio_history, pattern='^portfolio_history$'),
                CallbackQueryHandler(portfolio_add_trade, pattern='^portfolio_add_trade$'),
                CallbackQueryHandler(portfolio_deposit, pattern='^portfolio_deposit$'),
                CallbackQueryHandler(portfolio_back, pattern='^portfolio_back$')
            ],
            ANALYTICS_MENU: [
                CallbackQueryHandler(analytics_risk_reward, pattern='^analytics_risk_reward$'),
                CallbackQueryHandler(analytics_strategy_perf, pattern='^analytics_strategy_perf$'),
                CallbackQueryHandler(analytics_trade_stats, pattern='^analytics_trade_stats$'),
                CallbackQueryHandler(analytics_optimization, pattern='^analytics_optimization$'),
                CallbackQueryHandler(analytics_recommendations, pattern='^analytics_recommendations$'),
                CallbackQueryHandler(analytics_back, pattern='^analytics_back$')
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel), CommandHandler('start', start)],
        allow_reentry=True
    )

    application.add_handler(conv_handler)

    # Запуск в webhook-режиме
    port = int(os.environ.get('PORT', 10000))
    webhook_url = os.getenv('RENDER_EXTERNAL_URL', 'https://risk-management-bot-pro.onrender.com')
    logger.info(f"🌐 PRO Starting on port {port}")
    try:
        if webhook_url and "render.com" in webhook_url:
            logger.info(f"🔗 PRO Webhook URL: {webhook_url}/webhook")
            application.run_webhook(
                listen="0.0.0.0",
                port=port,
                webhook_url=webhook_url + "/webhook"
            )
        else:
            application.run_polling()
    except Exception as e:
        logger.error(f"❌ Error starting PRO bot: {e}")

if __name__ == '__main__':
    main()
