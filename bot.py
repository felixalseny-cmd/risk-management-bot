import os
import logging
import asyncio
import re
import time
import functools
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove
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
from flask import Flask, request
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

# Global application instance
application = None

# Flask server for health checks and webhook
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
    if request.method == 'POST':
        update = Update.de_json(request.get_json(force=True), application.bot)
        application.update_queue.put(update)
    return 'OK', 200

def run_flask():
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🌐 Flask starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# Keep-alive to prevent Render sleep
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

# Schedule keep-alive every 14 minutes (Render free tier sleeps after 15 minutes of inactivity)
schedule.every(14).minutes.do(keep_alive)

def schedule_runner():
    while True:
        schedule.run_pending()
        time.sleep(60)

# Start schedule runner
schedule_thread = threading.Thread(target=schedule_runner, daemon=True)
schedule_thread.start()
logger.info("✅ Keep-alive scheduler started!")

# Главная функция запуска
def main():
    global application
    
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ PRO Bot token not found!")
        return

    logger.info("🚀 Starting ULTRA-FAST PRO Risk Management Bot v3.0 with Enhanced Portfolio & Analytics...")

    # Create application
    application = Application.builder().token(token).build()

    # Add conversation handler
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

    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("✅ Flask health check server started!")

    # Set up webhook
    port = int(os.environ.get('PORT', 10000))
    webhook_url = os.getenv('RENDER_EXTERNAL_URL', 'https://risk-management-bot-pro.onrender.com')
    
    logger.info(f"🌐 PRO Starting on port {port}")
    logger.info(f"🔗 PRO Webhook URL: {webhook_url}/webhook")
    
    try:
        # Set webhook
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            webhook_url=f"{webhook_url}/webhook",
            cert=None,
            key=None,
            secret_token=None
        )
    except Exception as e:
        logger.error(f"❌ Error starting PRO bot: {e}")
        # Fallback to polling
        logger.info("🔄 Falling back to polling mode...")
        application.run_polling()

if __name__ == '__main__':
    main()
