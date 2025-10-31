import os
import logging
import asyncio
import re
import time
import functools
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
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

# MongoDB Connection — ИСПРАВЛЕНО: используем risk_bot_pro
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb+srv://felixalseny_db_user:kontraktaciA22@felix22.3nx1ibi.mongodb.net/risk_bot_pro?retryWrites=true&w=majority&appName=Felix22')
mongo_client = None
db = None
users_collection = None
portfolio_collection = None

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.server_info()  # Test connection
    db = mongo_client.risk_bot_pro  # 🔥 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
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
        if users_collection is None:  # 🔥 ИСПРАВЛЕНО: не используем `if not`
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
        if users_collection is None:  # 🔥 ИСПРАВЛЕНО
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
        if portfolio_collection is None:  # 🔥 ИСПРАВЛЕНО
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
        if portfolio_collection is None:  # 🔥 ИСПРАВЛЕНО
            return
        try:
            portfolio_collection.update_one(
                {"user_id": user_id},
                {"$set": {"portfolio_data": portfolio_data, "last_updated": datetime.now()}},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")

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

# === ОСНОВНЫЕ ОБРАБОТЧИКИ (start, portfolio, analytics и т.д.) ===
# ... [все обработчики остаются без изменений — они уже корректны] ...

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
