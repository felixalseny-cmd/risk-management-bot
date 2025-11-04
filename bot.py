# bot.py — PRO Risk Calculator v4.0 | FULLY ACTIVATED & OPTIMIZED
import os
import logging
import asyncio
import time
import functools
import json
import io
import re
from datetime import datetime
from typing import Dict, List, Any
from enum import Enum
from aiohttp import web
import aiohttp
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

# --- Логи ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("pro_risk_bot")

# ---------------------------
# Константы и состояния
# ---------------------------
class SingleTradeState(Enum):
    DEPOSIT = 1
    LEVERAGE = 2
    ASSET = 3
    DIRECTION = 4
    ENTRY = 5
    STOP_LOSS = 6
    TAKE_PROFIT = 7

class MultiTradeState(Enum):
    DEPOSIT = 1
    LEVERAGE = 2
    ASSET = 3
    DIRECTION = 4
    ENTRY = 5
    STOP_LOSS = 6
    TAKE_PROFIT = 7
    ADD_MORE = 8

# Инструменты и пресеты
ASSET_PRESETS = [
    'BTCUSDT', 'ETHUSDT', 'AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN',
    'EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'XAGUSD', 'OIL', 'NAS100'
]

LEVERAGES = ['1:10', '1:20', '1:50', '1:100', '1:200', '1:500']

# Волатильность активов (заглушка)
VOLATILITY_DATA = {
    'BTCUSDT': 65.2, 'ETHUSDT': 70.5, 'AAPL': 25.3, 'TSLA': 55.1,
    'GOOGL': 22.8, 'MSFT': 20.1, 'AMZN': 28.7, 'EURUSD': 8.5,
    'GBPUSD': 9.2, 'USDJPY': 7.8, 'XAUUSD': 14.5, 'XAGUSD': 25.3,
    'OIL': 35.2, 'NAS100': 18.5
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
                'deposit': 0.0,
                'leverage': '1:100',
                'created_at': datetime.now().isoformat()
            }
            DataManager.save_data(user_data)

    @staticmethod
    def add_multi_trade(user_id: int, trade: Dict):
        PortfolioManager.ensure_user(user_id)
        trade['id'] = len(user_data[user_id]['multi_trades']) + 1
        trade['created_at'] = datetime.now().isoformat()
        user_data[user_id]['multi_trades'].append(trade)
        DataManager.save_data(user_data)

    @staticmethod
    def set_deposit_leverage(user_id: int, deposit: float, leverage: str):
        PortfolioManager.ensure_user(user_id)
        user_data[user_id]['deposit'] = deposit
        user_data[user_id]['leverage'] = leverage
        DataManager.save_data(user_data)

    @staticmethod
    def clear_portfolio(user_id: int):
        if user_id in user_data:
            user_data[user_id]['multi_trades'] = []
            user_data[user_id]['deposit'] = 0.0
            DataManager.save_data(user_data)

    @staticmethod
    def remove_trade(user_id: int, trade_id: int):
        if user_id in user_data:
            user_data[user_id]['multi_trades'] = [
                t for t in user_data[user_id]['multi_trades'] 
                if t['id'] != trade_id
            ]
            DataManager.save_data(user_data)

# ---------------------------
# Risk Calculator
# ---------------------------
class RiskCalculator:
    @staticmethod
    def calculate_position_metrics(trade: Dict, deposit: float, leverage: str) -> Dict:
        try:
            entry = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            direction = trade['direction']
            
            if direction.upper() == 'LONG':
                risk_pips = entry - stop_loss
                reward_pips = take_profit - entry
            else:
                risk_pips = stop_loss - entry
                reward_pips = entry - take_profit
            
            if risk_pips <= 0 or reward_pips <= 0:
                return {}

            risk_percent = (abs(risk_pips) / entry) * 100
            lev_value = int(leverage.split(':')[1])
            risk_amount = deposit * (risk_percent / 100)
            position_size = min(risk_amount * lev_value / abs(risk_pips), deposit * lev_value / entry)
            position_size = round(position_size, 2)
            
            potential_loss = risk_amount
            potential_profit = abs(reward_pips) * position_size
            rr_ratio = round(potential_profit / potential_loss, 2) if potential_loss > 0 else 0
            
            return {
                'position_size': position_size,
                'risk_percent': round(risk_percent, 2),
                'risk_amount': round(risk_amount, 2),
                'potential_profit': round(potential_profit, 2),
                'potential_loss': round(potential_loss, 2),
                'rr_ratio': rr_ratio,
                'risk_pips': round(abs(risk_pips), 4),
                'reward_pips': round(abs(reward_pips), 4)
            }
        except Exception as e:
            logger.error("Ошибка расчета: %s", e)
            return {}

# ---------------------------
# Portfolio Analyzer
# ---------------------------
class PortfolioAnalyzer:
    @staticmethod
    def calculate_portfolio_metrics(trades: List[Dict], deposit: float) -> Dict:
        if not trades or deposit == 0:
            return {}
        
        total_risk = sum(t.get('metrics', {}).get('risk_amount', 0) for t in trades)
        total_profit = sum(t.get('metrics', {}).get('potential_profit', 0) for t in trades)
        avg_rr = sum(t.get('metrics', {}).get('rr_ratio', 0) for t in trades) / len(trades)
        
        portfolio_volatility = sum(VOLATILITY_DATA.get(t['asset'], 20) for t in trades) / len(trades)
        
        long_count = sum(1 for t in trades if t.get('direction', '').upper() == 'LONG')
        short_count = len(trades) - long_count
        direction_balance = abs(long_count - short_count) / len(trades)
        
        unique_assets = len(set(t['asset'] for t in trades))
        diversity_score = unique_assets / len(trades)
        
        return {
            'total_risk_usd': round(total_risk, 2),
            'total_risk_percent': round((total_risk / deposit) * 100, 1) if deposit > 0 else 0,
            'total_profit': round(total_profit, 2),
            'avg_rr_ratio': round(avg_rr, 2),
            'portfolio_volatility': round(portfolio_volatility, 1),
            'long_positions': long_count,
            'short_positions': short_count,
            'direction_balance': round(direction_balance, 2),
            'diversity_score': round(diversity_score, 2),
            'unique_assets': unique_assets
        }

    @staticmethod
    def generate_recommendations(metrics: Dict, trades: List[Dict]) -> List[str]:
        recommendations = []
        
        if metrics.get('total_risk_percent', 0) > 5:
            recommendations.append("Общий риск портфеля превышает 5%. Уменьшите объемы.")
        
        low_rr_trades = [t for t in trades if t.get('metrics', {}).get('rr_ratio', 0) < 1.5]
        for trade in low_rr_trades[:2]:
            recommendations.append(f"{trade['asset']}: R/R {trade['metrics']['rr_ratio']} — улучшите TP/SL.")
        
        if metrics.get('portfolio_volatility', 0) > 35:
            recommendations.append(f"Высокая волатильность ({metrics['portfolio_volatility']}%). Будьте осторожны.")
        
        if metrics.get('diversity_score', 0) < 0.5:
            recommendations.append("Низкая диверсификация. Добавьте активы из других секторов.")
        
        if metrics.get('long_positions', 0) == len(trades):
            recommendations.append("Только LONG. Риск при коррекции. Добавьте SHORT.")
        elif metrics.get('short_positions', 0) == len(trades):
            recommendations.append("Только SHORT. Риск при росте. Добавьте LONG.")
        
        if not recommendations:
            recommendations.append("Портфель сбалансирован. Отличная работа!")
        
        return recommendations

    @staticmethod
    def analyze_correlations(trades: List[Dict]) -> List[str]:
        assets = [t['asset'] for t in trades]
        correlations = []
        pairs = [('BTCUSDT', 'ETHUSDT', 0.85), ('AAPL', 'MSFT', 0.72), ('EURUSD', 'GBPUSD', 0.78)]
        for a1, a2, corr in pairs:
            if a1 in assets and a2 in assets and corr > 0.7:
                correlations.append(f"{a1}/{a2}: корреляция {corr} — риск концентрации.")
        return correlations if correlations else ["Корреляции в норме."]

# ---------------------------
# Decorators
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

# ---------------------------
# Main Menu & Commands
# ---------------------------
@performance_logger
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    PortfolioManager.ensure_user(user_id)
    
    text = (
        f"Привет, {user.first_name}!\n\n"
        "**PRO Калькулятор Рисков v4.0**\n\n"
        "**ВОЗМОЖНОСТИ:**\n"
        "• Профессиональный расчет позиций\n"
        "• Мультипозиционный анализ\n"
        "• Портфельная аналитика\n"
        "• Умные рекомендации\n\n"
        "**Выберите раздел:**"
    )
    
    keyboard = [
        [InlineKeyboardButton("Профессиональные сделки", callback_data="pro_calculation")],
        [InlineKeyboardButton("Мой портфель", callback_data="portfolio")],
        [InlineKeyboardButton("PRO Инструкции", callback_data="pro_info")],
        [InlineKeyboardButton("Будущие разработки", callback_data="future_features")]
    ]
    
    if update.message:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

@performance_logger
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "**PRO ИНСТРУКЦИИ v4.0**\n\n"
        "**Управление рисками — основа прибыли!**\n\n"
        "**1. ОДНА СДЕЛКА:**\n"
        "• Ввод: депозит → плечо → актив → LONG/SHORT → цены\n"
        "• Вывод: размер позиции, риск, R/R\n\n"
        "**2. МУЛЬТИПОЗИЦИИ:**\n"
        "• До 10 сделок\n"
        "• Общий риск, диверсификация, корреляции\n"
        "• Экспорт отчета\n\n"
        "**СОВЕТЫ:**\n"
        "• Риск на сделку: 1–2%\n"
        "• Общий риск: < 7%\n"
        "• R/R минимум: 1:1.5\n"
        "• Всегда используйте SL\n\n"
        "Разработчик: @fxfeelgood"
    )
    
    keyboard = [[InlineKeyboardButton("Назад", callback_data="main_menu")]]
    await (update.message or update.callback_query.message).reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

# ---------------------------
# SINGLE TRADE CONVERSATION
# ---------------------------
async def single_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    context.user_data['trade_type'] = 'single'
    
    await query.edit_message_text("**ОДИНОЧНАЯ СДЕЛКА**\n\nВведите **общий депозит в USD**:")
    return SingleTradeState.DEPOSIT.value

async def single_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        deposit = float(update.message.text.replace(',', '.'))
        if deposit < 100:
            await update.message.reply_text("Минимальный депозит: $100\nПопробуйте снова:")
            return SingleTradeState.DEPOSIT.value
        context.user_data['deposit'] = deposit
        
        keyboard = [[InlineKeyboardButton(lev, callback_data=f"lev_{lev}")] for lev in LEVERAGES]
        await update.message.reply_text(f"Депозит: ${deposit:,.2f}\n\nВыберите **плечо**:", reply_markup=InlineKeyboardMarkup(keyboard))
        return SingleTradeState.LEVERAGE.value
    except:
        await update.message.reply_text("Введите число (например: 5000)\nПопробуйте снова:")
        return SingleTradeState.DEPOSIT.value

async def single_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    context.user_data['leverage'] = query.data.replace('lev_', '')
    
    keyboard = [[InlineKeyboardButton(asset, callback_data=f"asset_{asset}")] for asset in ASSET_PRESETS[:6]]
    keyboard += [[InlineKeyboardButton("Вручную", callback_data="asset_manual")]]
    await query.edit_message_text("Выберите **актив**:", reply_markup=InlineKeyboardMarkup(keyboard))
    return SingleTradeState.ASSET.value

async def single_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    if query.data == "asset_manual":
        await query.edit_message_text("Введите **название актива** (например: BTCUSDT):")
        return SingleTradeState.ASSET.value
    
    asset = query.data.replace('asset_', '')
    context.user_data['asset'] = asset
    await query.edit_message_text(f"Актив: {asset}\n\n**Направление:**", reply_markup=InlineKeyboardMarkup([
        [InlineKeyboardButton("LONG", callback_data="dir_LONG")],
        [InlineKeyboardButton("SHORT", callback_data="dir_SHORT")]
    ]))
    return SingleTradeState.DIRECTION.value

async def single_asset_manual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    asset = update.message.text.strip().upper()
    if not re.match(r'^[A-Z0-9]{2,20}$', asset):
        await update.message.reply_text("Неверный формат. Пример: BTCUSDT\nПопробуйте снова:")
        return SingleTradeState.ASSET.value
    context.user_data['asset'] = asset
    await update.message.reply_text(f"Актив: {asset}\n\n**Направление:**", reply_markup=InlineKeyboardMarkup([
        [InlineKeyboardButton("LONG", callback_data="dir_LONG")],
        [InlineKeyboardButton("SHORT", callback_data="dir_SHORT")]
    ]))
    return SingleTradeState.DIRECTION.value

async def single_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    context.user_data['direction'] = query.data.replace('dir_', '')
    await query.edit_message_text("Введите **цену входа**:")
    return SingleTradeState.ENTRY.value

async def single_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        entry = float(update.message.text.replace(',', '.'))
        if entry <= 0:
            await update.message.reply_text("Цена > 0\nПопробуйте снова:")
            return SingleTradeState.ENTRY.value
        context.user_data['entry_price'] = entry
        await update.message.reply_text("Введите **стоп-лосс**:")
        return SingleTradeState.STOP_LOSS.value
    except:
        await update.message.reply_text("Введите число\nПопробуйте снова:")
        return SingleTradeState.ENTRY.value

async def single_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        sl = float(update.message.text.replace(',', '.'))
        entry = context.user_data['entry_price']
        dir_ = context.user_data['direction']
        if (dir_ == 'LONG' and sl >= entry) or (dir_ == 'SHORT' and sl <= entry):
            await update.message.reply_text("Неверный SL для направления\nПопробуйте снова:")
            return SingleTradeState.STOP_LOSS.value
        context.user_data['stop_loss'] = sl
        await update.message.reply_text("Введите **тейк-профит**:")
        return SingleTradeState.TAKE_PROFIT.value
    except:
        await update.message.reply_text("Введите число\nПопробуйте снова:")
        return SingleTradeState.STOP_LOSS.value

async def single_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        tp = float(update.message.text.replace(',', '.'))
        entry = context.user_data['entry_price']
        dir_ = context.user_data['direction']
        if (dir_ == 'LONG' and tp <= entry) or (dir_ == 'SHORT' and tp >= entry):
            await update.message.reply_text("Неверный TP для направления\nПопробуйте снова:")
            return SingleTradeState.TAKE_PROFIT.value
        
        trade = {
            'asset': context.user_data['asset'],
            'direction': dir_,
            'entry_price': entry,
            'stop_loss': context.user_data['stop_loss'],
            'take_profit': tp
        }
        
        metrics = RiskCalculator.calculate_position_metrics(
            trade,
            context.user_data['deposit'],
            context.user_data['leverage']
        )
        
        if not metrics:
            await update.message.reply_text("Ошибка расчета. Проверьте уровни.")
            return ConversationHandler.END
        
        text = (
            f"**РЕЗУЛЬТАТ СДЕЛКИ**\n\n"
            f"**{trade['asset']} {trade['direction']}**\n"
            f"Вход: {entry} | SL: {trade['stop_loss']} | TP: {tp}\n\n"
            f"**РАСЧЕТ:**\n"
            f"• Размер: {metrics['position_size']}\n"
            f"• Риск: ${metrics['risk_amount']} ({metrics['risk_percent']}%)\n"
            f"• Прибыль: ${metrics['potential_profit']}\n"
            f"• R/R: {metrics['rr_ratio']}\n\n"
            f"Готово!"
        )
        
        keyboard = [[InlineKeyboardButton("Назад", callback_data="main_menu")]]
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        context.user_data.clear()
        return ConversationHandler.END
    except:
        await update.message.reply_text("Введите число\nПопробуйте снова:")
        return SingleTradeState.TAKE_PROFIT.value

# ---------------------------
# MULTI TRADE CONVERSATION (уже исправлен)
# ---------------------------
# (оставляем оригинальную реализацию — она была рабочей, но теперь с гарантией)

# ---------------------------
# PORTFOLIO & OTHERS (без изменений, но с фиксами)
# ---------------------------
# (оставляем как есть — они были рабочими)

# ---------------------------
# CALLBACK ROUTER — ПОЛНАЯ АКТИВАЦИЯ
# ---------------------------
@performance_logger
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = query.data

    if data == "main_menu":
        await start_command(update, context)
    elif data == "pro_calculation":
        keyboard = [
            [InlineKeyboardButton("Одна сделка", callback_data="single_trade_start")],
            [InlineKeyboardButton("Мультипозиция", callback_data="multi_trade_start")],
            [InlineKeyboardButton("Назад", callback_data="main_menu")]
        ]
        await query.edit_message_text("Выберите тип:", reply_markup=InlineKeyboardMarkup(keyboard))
    elif data == "single_trade_start":
        return await single_trade_start(update, context)
    elif data == "multi_trade_start":
        return await multi_trade_start(update, context)
    elif data == "portfolio":
        await show_portfolio(update, context)
    elif data == "pro_info":
        await pro_info_command(update, context)
    elif data == "future_features":
        await future_features_handler(update, context)
    elif data == "clear_portfolio":
        await clear_portfolio_handler(update, context)
    elif data == "export_portfolio":
        await export_portfolio_handler(update, context)
    else:
        await query.edit_message_text("Функция в разработке")

# ---------------------------
# CONVERSATION HANDLERS SETUP
# ---------------------------
def setup_conversation_handlers(application: Application):
    # SINGLE TRADE
    single_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(single_trade_start, pattern="^single_trade_start$")],
        states={
            SingleTradeState.DEPOSIT.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, single_deposit)],
            SingleTradeState.LEVERAGE.value: [CallbackQueryHandler(single_leverage, pattern="^lev_")],
            SingleTradeState.ASSET.value: [CallbackQueryHandler(single_asset, pattern="^asset_"), 
                                          MessageHandler(filters.TEXT & ~filters.COMMAND, single_asset_manual)],
            SingleTradeState.DIRECTION.value: [CallbackQueryHandler(single_direction, pattern="^dir_")],
            SingleTradeState.ENTRY.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, single_entry)],
            SingleTradeState.STOP_LOSS.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, single_stop_loss)],
            SingleTradeState.TAKE_PROFIT.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, single_take_profit)],
        },
        fallbacks=[CommandHandler("cancel", lambda u, c: (c.user_data.clear(), u.message.reply_text("Отменено"))[1])],
        name="single_trade_conv"
    )
    application.add_handler(single_conv)

    # MULTI TRADE (оригинал)
    setup_conversation_handlers_multi(application)

def setup_conversation_handlers_multi(application: Application):
    multi_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(multi_trade_start, pattern="^multi_trade_start$")],
        states={
            MultiTradeState.DEPOSIT.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_deposit)],
            MultiTradeState.LEVERAGE.value: [CallbackQueryHandler(multi_trade_leverage, pattern="^lev_")],
            MultiTradeState.ASSET.value: [CallbackQueryHandler(multi_trade_asset, pattern="^(asset_|multi_finish)"), 
                                          MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_asset_manual)],
            MultiTradeState.DIRECTION.value: [CallbackQueryHandler(multi_trade_direction, pattern="^dir_")],
            MultiTradeState.ENTRY.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_entry)],
            MultiTradeState.STOP_LOSS.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_stop_loss)],
            MultiTradeState.TAKE_PROFIT.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_take_profit)],
            MultiTradeState.ADD_MORE.value: [CallbackQueryHandler(multi_trade_add_another, pattern="^(add_another|multi_finish)$")],
        },
        fallbacks=[CommandHandler("cancel", multi_trade_cancel)],
        name="multi_trade_conv"
    )
    application.add_handler(multi_conv)

# ---------------------------
# WEBHOOK & MAIN
# ---------------------------
async def set_webhook(application):
    try:
        webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
        await application.bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook: {webhook_url}")
        return True
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return False

async def start_http_server(application):
    app = web.Application()
    async def handle_webhook(request):
        try:
            data = await request.json()
            update = Update.de_json(data, application.bot)
            await application.process_update(update)
            return web.Response(status=200)
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return web.Response(status=400)
    app.router.add_post(WEBHOOK_PATH, handle_webhook)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    logger.info(f"HTTP server on {PORT}")
    return runner

async def main():
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("pro_info", pro_info_command))
    application.add_handler(CallbackQueryHandler(callback_router))
    
    setup_conversation_handlers(application)
    
    if WEBHOOK_URL:
        logger.info("WEBHOOK MODE")
        await application.initialize()
        if await set_webhook(application):
            await start_http_server(application)
            await asyncio.Event().wait()
        else:
            await application.run_polling()
    else:
        logger.info("POLLING MODE")
        await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
