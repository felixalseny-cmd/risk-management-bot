# bot.py — PRO Risk Calculator v4.0 | COMPLETE FIX & ENHANCEMENT
import os
import logging
import asyncio
import time
import functools
import json
import io
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
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
RISK_LEVELS = ['1%', '2%', '3%', '5%', '7%', '10%']

# Волатильность активов
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
                'single_trades': [],
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
    def add_single_trade(user_id: int, trade: Dict):
        PortfolioManager.ensure_user(user_id)
        trade['id'] = len(user_data[user_id]['single_trades']) + 1
        trade['created_at'] = datetime.now().isoformat()
        user_data[user_id]['single_trades'].append(trade)
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
            user_data[user_id]['single_trades'] = []
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
        """Расчет метрик для одной сделки"""
        try:
            entry = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            direction = trade['direction']
            
            # Расчет риска в пунктах
            if direction.upper() == 'LONG':
                risk_pips = entry - stop_loss
                reward_pips = take_profit - entry
            else:  # SHORT
                risk_pips = stop_loss - entry
                reward_pips = entry - take_profit
            
            # Процент риска
            risk_percent = (abs(risk_pips) / entry) * 100
            
            # Размер позиции (упрощенный расчет)
            lev_value = int(leverage.split(':')[1])
            risk_amount = deposit * (risk_percent / 100)
            position_size = min(risk_amount * lev_value / abs(risk_pips), deposit * lev_value / entry)
            position_size = round(position_size, 2)
            
            # Потенциальная прибыль/убыток
            potential_loss = risk_amount
            potential_profit = abs(reward_pips) * position_size
            
            # Risk/Reward ratio
            rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
            
            return {
                'position_size': position_size,
                'risk_percent': risk_percent,
                'risk_amount': risk_amount,
                'potential_profit': potential_profit,
                'potential_loss': potential_loss,
                'rr_ratio': rr_ratio,
                'risk_pips': abs(risk_pips),
                'reward_pips': abs(reward_pips)
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
        """Расчет метрик портфеля"""
        if not trades:
            return {}
        
        total_risk = sum(t.get('metrics', {}).get('risk_amount', 0) for t in trades)
        total_profit = sum(t.get('metrics', {}).get('potential_profit', 0) for t in trades)
        total_loss = sum(t.get('metrics', {}).get('potential_loss', 0) for t in trades)
        
        avg_rr = sum(t.get('metrics', {}).get('rr_ratio', 0) for t in trades) / len(trades)
        
        # Волатильность портфеля
        portfolio_volatility = sum(VOLATILITY_DATA.get(t['asset'], 20) for t in trades) / len(trades)
        
        # Анализ направлений
        long_count = sum(1 for t in trades if t.get('direction', '').upper() == 'LONG')
        short_count = len(trades) - long_count
        direction_balance = abs(long_count - short_count) / len(trades)
        
        # Диверсификация
        unique_assets = len(set(t['asset'] for t in trades))
        diversity_score = unique_assets / len(trades)
        
        return {
            'total_risk_usd': total_risk,
            'total_risk_percent': (total_risk / deposit) * 100 if deposit > 0 else 0,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'avg_rr_ratio': avg_rr,
            'portfolio_volatility': portfolio_volatility,
            'long_positions': long_count,
            'short_positions': short_count,
            'direction_balance': direction_balance,
            'diversity_score': diversity_score,
            'unique_assets': unique_assets
        }

    @staticmethod
    def generate_recommendations(metrics: Dict, trades: List[Dict]) -> List[str]:
        """Генерация рекомендаций на основе метрик портфеля"""
        recommendations = []
        
        # Проверка общего риска
        if metrics.get('total_risk_percent', 0) > 5:
            recommendations.append(
                "⚠️ ВНИМАНИЕ: Общий риск портфеля превышает 5%. "
                "Рекомендуется уменьшить объем позиций."
            )
        
        # Проверка Risk/Reward
        low_rr_trades = [
            t for t in trades 
            if t.get('metrics', {}).get('rr_ratio', 0) < 1
        ]
        for trade in low_rr_trades:
            recommendations.append(
                f"📉 Невыгодное R/R: {trade['asset']} имеет соотношение "
                f"{trade['metrics']['rr_ratio']:.2f}. Пересмотрите уровни TP/SL."
            )
        
        # Проверка волатильности
        if metrics.get('portfolio_volatility', 0) > 30:
            recommendations.append(
                f"🌪 Высокая волатильность портфеля ({metrics['portfolio_volatility']:.1f}%). "
                "Будьте готовы к значительным колебаниям."
            )
        
        # Проверка диверсификации
        if metrics.get('diversity_score', 0) < 0.5:
            recommendations.append(
                "🎯 Низкая диверсификация. Рассмотрите добавление активов "
                "из разных секторов для снижения риска."
            )
        
        # Проверка направлений
        if metrics.get('long_positions', 0) == len(trades):
            recommendations.append(
                "📈 Портфель состоит только из LONG позиций. Уязвим к "
                "рыночным коррекциям. Рассмотрите хеджирование."
            )
        elif metrics.get('short_positions', 0) == len(trades):
            recommendations.append(
                "📉 Портфель состоит только из SHORT позиций. Рискованно "
                "при росте рынка. Добавьте LONG позиции."
            )
        
        if not recommendations:
            recommendations.append("✅ Портфель сбалансирован. Продолжайте в том же духе!")
        
        return recommendations

    @staticmethod
    def analyze_correlations(trades: List[Dict]) -> List[str]:
        """Анализ корреляций между активами"""
        correlations = []
        asset_pairs = [
            ('BTCUSDT', 'ETHUSDT', 0.85),
            ('AAPL', 'MSFT', 0.72),
            ('EURUSD', 'GBPUSD', 0.78),
            ('XAUUSD', 'XAGUSD', 0.65)
        ]
        
        assets = [t['asset'] for t in trades]
        for asset1, asset2, corr in asset_pairs:
            if asset1 in assets and asset2 in assets and abs(corr) > 0.7:
                correlations.append(
                    f"🔗 {asset1} и {asset2} имеют высокую корреляцию ({corr:.2f}). "
                    "Рассмотрите диверсификацию."
                )
        
        return correlations if correlations else ["✅ Корреляции в пределах нормы"]

# ---------------------------
# Handlers
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

@performance_logger
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    user_id = user.id
    PortfolioManager.ensure_user(user_id)
    
    text = (
        f"👋 Привет, {user.first_name}!\n\n"
        "🤖 **PRO Калькулятор Управления Рисками v4.0**\n\n"
        "**МОИ ВОЗМОЖНОСТИ:**\n"
        "• 📊 Профессиональный расчет позиций\n"
        "• 🎯 Мультипозиционный анализ\n"
        "• 📈 Портфельная аналитика\n"
        "• 💡 Умные рекомендации\n\n"
        "**Выберите раздел:**"
    )
    
    keyboard = [
        [InlineKeyboardButton("🎯 Профессиональные сделки", callback_data="pro_calculation")],
        [InlineKeyboardButton("📊 Мой портфель", callback_data="portfolio")],
        [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")],
        [InlineKeyboardButton("🚀 Будущие разработки", callback_data="future_features")]
    ]
    
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

@performance_logger
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ОБНОВЛЕННЫЕ PRO инструкции для опытных трейдеров"""
    text = (
        "📚 **PRO ИНСТРУКЦИИ v4.0**\n\n"
        
        "**🎯 ПРОДВИНУТОЕ УПРАВЛЕНИЕ РИСКАМИ**\n\n"
        
        "**📊 КЛЮЧЕВЫЕ МЕТРИКИ ДЛЯ ПРОФЕССИОНАЛОВ:**\n\n"
        
        "**1. КАЛЬКУЛЯЦИЯ ПОЗИЦИЙ С УЧЕТОМ ВОЛАТИЛЬНОСТИ**\n"
        "• ATR-адоптированный размер позиции\n"
        "• Учет корреляции между активами\n"
        "• Динамический расчет стоп-лоссов\n"
        "• Взвешивание позиций по волатильности\n\n"
        
        "**2. ПОРТФЕЛЬНАЯ ТЕОРИЯ В ТРЕЙДИНГЕ**\n"
        "• Эффективная диверсификация по секторам\n"
        "• Балансировка beta-коэффициентов\n"
        "• Управление ковариационной матрицей\n"
        "• Оптимизация Sharpe Ratio портфеля\n\n"
        
        "**3. РИСК-МЕНЕДЖМЕНТ ПРОФЕССИОНАЛЬНОГО УРОВНЯ**\n"
        "• Kelly Criterion для оптимизации размера\n"
        "• Maximum Drawdown контроль\n"
        "• Value at Risk (VaR) расчеты\n"
        "• Stress-testing сценарии\n\n"
        
        "**4. ПСИХОЛОГИЯ И ДИСЦИПЛИНА**\n"
        "• Emotional Intelligence в трейдинге\n"
        "• Протоколы принятия решений\n"
        "• Управление когнитивными искажениями\n"
        "• Журналирование и анализ ошибок\n\n"
        
        "**💡 ПРОДВИНУТЫЕ СТРАТЕГИИ:**\n"
        "• Хеджирование валютных рисков\n"
        "• Опционные стратегии защиты\n"
        "• Тактическое распределение активов\n"
        "• Квантитативные модели оценки\n\n"
        
        "**📈 ФОРМУЛЫ И РАСЧЕТЫ:**\n"
        "• Position Size = (Account Risk %) / (Stop Loss %)\n"
        "• Risk/Reward = (Target Price - Entry) / (Entry - Stop Loss)\n"
        "• Portfolio Volatility = √(wᵀ × Σ × w)\n"
        "• Sharpe Ratio = (Return - Risk Free) / Volatility\n\n"
        
        "Разработчик: @fxfeelgood"
    )
    
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
        ]))
    else:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
        ]))

# ---------------------------
# Single Trade Conversation Handler
# ---------------------------
async def single_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало одиночной сделки"""
    query = update.callback_query
    await query.answer()
    
    text = (
        "🎯 **ОДИНОЧНАЯ СДЕЛКА**\n\n"
        "Я помогу вам рассчитать оптимальный размер позиции "
        "с учетом управления рисками.\n\n"
        "**Введите ваш депозит в USD:**"
    )
    
    await query.edit_message_text(text)
    return SingleTradeState.DEPOSIT.value

async def single_trade_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода депозита для одиночной сделки"""
    text = update.message.text.strip()
    
    try:
        deposit = float(text.replace(',', '.'))
        if deposit < 100:
            await update.message.reply_text("❌ Минимальный депозит: $100\nПопробуйте еще раз:")
            return SingleTradeState.DEPOSIT.value
        
        context.user_data['deposit'] = deposit
        
        keyboard = []
        for leverage in LEVERAGES:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"lev_{leverage}")])
        
        await update.message.reply_text(
            f"✅ Депозит: ${deposit:,.2f}\n\n"
            "**Выберите кредитное плечо:**",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return SingleTradeState.LEVERAGE.value
        
    except ValueError:
        await update.message.reply_text("❌ Введите число (например: 1000)\nПопробуйте еще раз:")
        return SingleTradeState.DEPOSIT.value

async def single_trade_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора плеча для одиночной сделки"""
    query = update.callback_query
    await query.answer()
    
    leverage = query.data.replace('lev_', '')
    context.user_data['leverage'] = leverage
    
    await query.edit_message_text(
        f"✅ Плечо: {leverage}\n\n"
        "**Выберите актив из списка или введите свой:**",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton(asset, callback_data=f"asset_{asset}")] 
            for asset in ASSET_PRESETS[:7]
        ] + [
            [InlineKeyboardButton("📝 Ввести вручную", callback_data="asset_manual")]
        ])
    )
    return SingleTradeState.ASSET.value

async def single_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора актива для одиночной сделки"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "asset_manual":
        await query.edit_message_text("✍️ Введите название актива (например: BTCUSDT):")
        return SingleTradeState.ASSET.value
    
    else:
        asset = query.data.replace('asset_', '')
        context.user_data['asset'] = asset
        
        await query.edit_message_text(
            f"✅ Актив: {asset}\n\n"
            "**Выберите направление сделки:**",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
                [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")]
            ])
        )
        return SingleTradeState.DIRECTION.value

async def single_trade_asset_manual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ручного ввода актива для одиночной сделки"""
    asset = update.message.text.strip().upper()
    
    # Простая валидация
    if not re.match(r'^[A-Z0-9]{2,20}$', asset):
        await update.message.reply_text("❌ Неверный формат актива. Попробуйте еще раз:")
        return SingleTradeState.ASSET.value
    
    context.user_data['asset'] = asset
    
    await update.message.reply_text(
        f"✅ Актив: {asset}\n\n"
        "**Выберите направление сделки:**",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")]
        ])
    )
    return SingleTradeState.DIRECTION.value

async def single_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора направления для одиночной сделки"""
    query = update.callback_query
    await query.answer()
    
    direction = query.data.replace('dir_', '')
    context.user_data['direction'] = direction
    
    await query.edit_message_text(
        f"✅ Направление: {direction}\n\n"
        "**Введите цену входа:**"
    )
    return SingleTradeState.ENTRY.value

async def single_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка цены входа для одиночной сделки"""
    text = update.message.text.strip()
    
    try:
        entry_price = float(text.replace(',', '.'))
        if entry_price <= 0:
            await update.message.reply_text("❌ Цена должна быть больше 0\nПопробуйте еще раз:")
            return SingleTradeState.ENTRY.value
        
        context.user_data['entry_price'] = entry_price
        
        await update.message.reply_text(
            f"✅ Цена входа: {entry_price}\n\n"
            "**Введите уровень стоп-лосса:**"
        )
        return SingleTradeState.STOP_LOSS.value
        
    except ValueError:
        await update.message.reply_text("❌ Введите число (например: 50000)\nПопробуйте еще раз:")
        return SingleTradeState.ENTRY.value

async def single_trade_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка стоп-лосса для одиночной сделки"""
    text = update.message.text.strip()
    
    try:
        stop_loss = float(text.replace(',', '.'))
        entry_price = context.user_data['entry_price']
        direction = context.user_data['direction']
        
        # Валидация SL
        if direction == 'LONG' and stop_loss >= entry_price:
            await update.message.reply_text(
                "❌ Для LONG стоп-лосс должен быть НИЖЕ цены входа\nПопробуйте еще раз:"
            )
            return SingleTradeState.STOP_LOSS.value
        elif direction == 'SHORT' and stop_loss <= entry_price:
            await update.message.reply_text(
                "❌ Для SHORT стоп-лосс должен быть ВЫШЕ цены входа\nПопробуйте еще раз:"
            )
            return SingleTradeState.STOP_LOSS.value
        
        context.user_data['stop_loss'] = stop_loss
        
        await update.message.reply_text(
            f"✅ Стоп-лосс: {stop_loss}\n\n"
            "**Введите уровень тейк-профита:**"
        )
        return SingleTradeState.TAKE_PROFIT.value
        
    except ValueError:
        await update.message.reply_text("❌ Введите число (например: 48000)\nПопробуйте еще раз:")
        return SingleTradeState.STOP_LOSS.value

async def single_trade_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка тейк-профита и показ результатов одиночной сделки"""
    text = update.message.text.strip()
    
    try:
        take_profit = float(text.replace(',', '.'))
        entry_price = context.user_data['entry_price']
        direction = context.user_data['direction']
        
        # Валидация TP
        if direction == 'LONG' and take_profit <= entry_price:
            await update.message.reply_text(
                "❌ Для LONG тейк-профит должен быть ВЫШЕ цены входа\nПопробуйте еще раз:"
            )
            return SingleTradeState.TAKE_PROFIT.value
        elif direction == 'SHORT' and take_profit >= entry_price:
            await update.message.reply_text(
                "❌ Для SHORT тейк-профит должен быть НИЖЕ цены входа\nПопробуйте еще раз:"
            )
            return SingleTradeState.TAKE_PROFIT.value
        
        # Собираем данные сделки
        trade_data = {
            'asset': context.user_data['asset'],
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': context.user_data['stop_loss'],
            'take_profit': take_profit
        }
        
        # Расчет метрик
        deposit = context.user_data['deposit']
        leverage = context.user_data['leverage']
        metrics = RiskCalculator.calculate_position_metrics(trade_data, deposit, leverage)
        
        # Сохраняем сделку
        user_id = update.message.from_user.id
        trade_data['metrics'] = metrics
        PortfolioManager.add_single_trade(user_id, trade_data)
        
        # Формируем результат
        text = (
            f"🎯 **РАСЧЕТ ОДИНОЧНОЙ СДЕЛКИ**\n\n"
            f"**📊 ПАРАМЕТРЫ СДЕЛКИ:**\n"
            f"• Актив: {trade_data['asset']}\n"
            f"• Направление: {trade_data['direction']}\n"
            f"• Вход: {trade_data['entry_price']}\n"
            f"• Стоп-лосс: {trade_data['stop_loss']}\n"
            f"• Тейк-профит: {trade_data['take_profit']}\n\n"
            
            f"**💰 РАСЧЕТ РИСКОВ:**\n"
            f"• Размер позиции: {metrics['position_size']:.2f}\n"
            f"• Риск на сделку: ${metrics['risk_amount']:.2f} ({metrics['risk_percent']:.2f}%)\n"
            f"• Потенциальная прибыль: ${metrics['potential_profit']:.2f}\n"
            f"• Потенциальный убыток: ${metrics['potential_loss']:.2f}\n"
            f"• Соотношение R/R: {metrics['rr_ratio']:.2f}\n\n"
            
            f"**💡 РЕКОМЕНДАЦИЯ:**\n"
        )
        
        if metrics['risk_percent'] > 2:
            text += "⚠️ Риск превышает 2%! Рекомендуется уменьшить размер позиции.\n\n"
        elif metrics['rr_ratio'] < 1:
            text += "⚠️ Соотношение R/R меньше 1! Пересмотрите уровни TP/SL.\n\n"
        else:
            text += "✅ Параметры сделки в пределах нормы.\n\n"
        
        text += "Выберите дальнейшее действие:"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Новая сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")],
            [InlineKeyboardButton("📋 В портфель", callback_data="portfolio")]
        ]
        
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        return ConversationHandler.END
        
    except ValueError:
        await update.message.reply_text("❌ Введите число (например: 52000)\nПопробуйте еще раз:")
        return SingleTradeState.TAKE_PROFIT.value

async def single_trade_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена одиночной сделки"""
    context.user_data.clear()
    await update.message.reply_text("❌ Расчет отменен")
    return ConversationHandler.END

# ---------------------------
# Multi-trade Conversation Handler
# ---------------------------
async def multi_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало мультипозиционного расчета"""
    query = update.callback_query
    await query.answer()
    
    context.user_data['multi_trades'] = []
    
    text = (
        "🎯 **МУЛЬТИПОЗИЦИОННЫЙ РАСЧЕТ**\n\n"
        "Я помогу вам рассчитать оптимальные размеры позиций "
        "для нескольких сделок с учетом общего риска.\n\n"
        "**Введите общий депозит в USD:**"
    )
    
    await query.edit_message_text(text)
    return MultiTradeState.DEPOSIT.value

async def multi_trade_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода депозита"""
    text = update.message.text.strip()
    
    try:
        deposit = float(text.replace(',', '.'))
        if deposit < 100:
            await update.message.reply_text("❌ Минимальный депозит: $100\nПопробуйте еще раз:")
            return MultiTradeState.DEPOSIT.value
        
        context.user_data['deposit'] = deposit
        
        keyboard = []
        for leverage in LEVERAGES:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"lev_{leverage}")])
        
        await update.message.reply_text(
            f"✅ Депозит: ${deposit:,.2f}\n\n"
            "**Выберите кредитное плечо:**",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return MultiTradeState.LEVERAGE.value
        
    except ValueError:
        await update.message.reply_text("❌ Введите число (например: 1000)\nПопробуйте еще раз:")
        return MultiTradeState.DEPOSIT.value

async def multi_trade_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора плеча"""
    query = update.callback_query
    await query.answer()
    
    leverage = query.data.replace('lev_', '')
    context.user_data['leverage'] = leverage
    
    # Начинаем цикл ввода сделок
    return await start_trade_input(update, context)

async def start_trade_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало ввода сделки"""
    query = update.callback_query
    
    trade_count = len(context.user_data.get('multi_trades', []))
    
    text = f"**Сделка #{trade_count + 1}**\n\nВыберите актив из списка или введите свой:"
    
    if query:
        await query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton(asset, callback_data=f"asset_{asset}")] 
                for asset in ASSET_PRESETS[:6]
            ] + [
                [InlineKeyboardButton("📝 Ввести вручную", callback_data="asset_manual")],
                [InlineKeyboardButton("🚀 Завершить ввод", callback_data="multi_finish")]
            ])
        )
    else:
        await update.message.reply_text(
            text,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton(asset, callback_data=f"asset_{asset}")] 
                for asset in ASSET_PRESETS[:6]
            ] + [
                [InlineKeyboardButton("📝 Ввести вручную", callback_data="asset_manual")],
                [InlineKeyboardButton("🚀 Завершить ввод", callback_data="multi_finish")]
            ])
        )
    
    return MultiTradeState.ASSET.value

async def multi_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора актива"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "asset_manual":
        await query.edit_message_text("✍️ Введите название актива (например: BTCUSDT):")
        return MultiTradeState.ASSET.value
    
    elif query.data == "multi_finish":
        return await finish_multi_trade(update, context)
    
    else:
        asset = query.data.replace('asset_', '')
        context.user_data['current_trade'] = {'asset': asset}
        
        await query.edit_message_text(
            f"✅ Актив: {asset}\n\n"
            "**Выберите направление сделки:**",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
                [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")]
            ])
        )
        return MultiTradeState.DIRECTION.value

async def multi_trade_asset_manual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ручного ввода актива"""
    asset = update.message.text.strip().upper()
    
    # Простая валидация
    if not re.match(r'^[A-Z0-9]{2,20}$', asset):
        await update.message.reply_text("❌ Неверный формат актива. Попробуйте еще раз:")
        return MultiTradeState.ASSET.value
    
    context.user_data['current_trade'] = {'asset': asset}
    
    await update.message.reply_text(
        f"✅ Актив: {asset}\n\n"
        "**Выберите направление сделки:**",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")]
        ])
    )
    return MultiTradeState.DIRECTION.value

async def multi_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора направления"""
    query = update.callback_query
    await query.answer()
    
    direction = query.data.replace('dir_', '')
    context.user_data['current_trade']['direction'] = direction
    
    await query.edit_message_text(
        f"✅ Направление: {direction}\n\n"
        "**Введите цену входа:**"
    )
    return MultiTradeState.ENTRY.value

async def multi_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка цены входа"""
    text = update.message.text.strip()
    
    try:
        entry_price = float(text.replace(',', '.'))
        if entry_price <= 0:
            await update.message.reply_text("❌ Цена должна быть больше 0\nПопробуйте еще раз:")
            return MultiTradeState.ENTRY.value
        
        context.user_data['current_trade']['entry_price'] = entry_price
        
        await update.message.reply_text(
            f"✅ Цена входа: {entry_price}\n\n"
            "**Введите уровень стоп-лосса:**"
        )
        return MultiTradeState.STOP_LOSS.value
        
    except ValueError:
        await update.message.reply_text("❌ Введите число (например: 50000)\nПопробуйте еще раз:")
        return MultiTradeState.ENTRY.value

async def multi_trade_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка стоп-лосса"""
    text = update.message.text.strip()
    
    try:
        stop_loss = float(text.replace(',', '.'))
        entry_price = context.user_data['current_trade']['entry_price']
        direction = context.user_data['current_trade']['direction']
        
        # Валидация SL
        if direction == 'LONG' and stop_loss >= entry_price:
            await update.message.reply_text(
                "❌ Для LONG стоп-лосс должен быть НИЖЕ цены входа\nПопробуйте еще раз:"
            )
            return MultiTradeState.STOP_LOSS.value
        elif direction == 'SHORT' and stop_loss <= entry_price:
            await update.message.reply_text(
                "❌ Для SHORT стоп-лосс должен быть ВЫШЕ цены входа\nПопробуйте еще раз:"
            )
            return MultiTradeState.STOP_LOSS.value
        
        context.user_data['current_trade']['stop_loss'] = stop_loss
        
        await update.message.reply_text(
            f"✅ Стоп-лосс: {stop_loss}\n\n"
            "**Введите уровень тейк-профита:**"
        )
        return MultiTradeState.TAKE_PROFIT.value
        
    except ValueError:
        await update.message.reply_text("❌ Введите число (например: 48000)\nПопробуйте еще раз:")
        return MultiTradeState.STOP_LOSS.value

async def multi_trade_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка тейк-профита и показ промежуточных результатов"""
    text = update.message.text.strip()
    
    try:
        take_profit = float(text.replace(',', '.'))
        entry_price = context.user_data['current_trade']['entry_price']
        direction = context.user_data['current_trade']['direction']
        
        # Валидация TP
        if direction == 'LONG' and take_profit <= entry_price:
            await update.message.reply_text(
                "❌ Для LONG тейк-профит должен быть ВЫШЕ цены входа\nПопробуйте еще раз:"
            )
            return MultiTradeState.TAKE_PROFIT.value
        elif direction == 'SHORT' and take_profit >= entry_price:
            await update.message.reply_text(
                "❌ Для SHORT тейк-профит должен быть НИЖЕ цены входа\nПопробуйте еще раз:"
            )
            return MultiTradeState.TAKE_PROFIT.value
        
        # Сохраняем TP
        current_trade = context.user_data['current_trade']
        current_trade['take_profit'] = take_profit
        
        # Расчет метрик
        deposit = context.user_data['deposit']
        leverage = context.user_data['leverage']
        metrics = RiskCalculator.calculate_position_metrics(current_trade, deposit, leverage)
        current_trade['metrics'] = metrics
        
        # Добавляем сделку в список
        context.user_data['multi_trades'].append(current_trade.copy())
        
        # Показываем результаты
        trade_count = len(context.user_data['multi_trades'])
        text = (
            f"✅ **СДЕЛКА #{trade_count} ДОБАВЛЕНА**\n\n"
            f"**Актив:** {current_trade['asset']}\n"
            f"**Направление:** {current_trade['direction']}\n"
            f"**Вход:** {current_trade['entry_price']}\n"
            f"**SL:** {current_trade['stop_loss']}\n"
            f"**TP:** {current_trade['take_profit']}\n\n"
            f"**📊 РАСЧЕТ:**\n"
            f"• Размер позиции: {metrics['position_size']:.2f}\n"
            f"• Риск на сделку: ${metrics['risk_amount']:.2f} ({metrics['risk_percent']:.2f}%)\n"
            f"• Потенциальная прибыль: ${metrics['potential_profit']:.2f}\n"
            f"• Risk/Reward: {metrics['rr_ratio']:.2f}\n\n"
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
        
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        return MultiTradeState.ADD_MORE.value
        
    except ValueError:
        await update.message.reply_text("❌ Введите число (например: 52000)\nПопробуйте еще раз:")
        return MultiTradeState.TAKE_PROFIT.value

async def multi_trade_add_another(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка добавления следующей сделки"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "add_another":
        return await start_trade_input(update, context)
    else:  # multi_finish
        return await finish_multi_trade(update, context)

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
    context.user_data.clear()
    
    # Переходим в портфель
    await show_portfolio(update, context, user_id)
    return ConversationHandler.END

async def multi_trade_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена мультипозиционного расчета"""
    context.user_data.clear()
    await update.message.reply_text("❌ Расчет отменен")
    return ConversationHandler.END

# ---------------------------
# Portfolio Handlers
# ---------------------------
async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int = None):
    """Показать портфель пользователя"""
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
        text = "📊 **ВАШ ПОРТФЕЛЬ**\n\nПортфель пуст. Начните с расчета сделок!"
        keyboard = [
            [InlineKeyboardButton("🎯 Одна сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")]
        ]
    else:
        # Расчет метрик портфеля
        metrics = PortfolioAnalyzer.calculate_portfolio_metrics(all_trades, deposit)
        recommendations = PortfolioAnalyzer.generate_recommendations(metrics, all_trades)
        correlations = PortfolioAnalyzer.analyze_correlations(all_trades)
        
        text = (
            f"📊 **ВАШ ПОРТФЕЛЬ**\n\n"
            f"**Основные параметры:**\n"
            f"• Депозит: ${deposit:,.2f}\n"
            f"• Плечо: {leverage}\n"
            f"• Всего сделок: {len(all_trades)}\n"
            f"• Одиночные: {len(single_trades)} | Мульти: {len(trades)}\n"
            f"• Уникальных активов: {metrics.get('unique_assets', 0)}\n\n"
            
            f"**📈 КЛЮЧЕВЫЕ МЕТРИКИ:**\n"
            f"• Общий риск: ${metrics['total_risk_usd']:.2f} ({metrics['total_risk_percent']:.1f}%)\n"
            f"• Потенциальная прибыль: ${metrics['total_profit']:.2f}\n"
            f"• Средний R/R: {metrics['avg_rr_ratio']:.2f}\n"
            f"• Волатильность портфеля: {metrics['portfolio_volatility']:.1f}%\n"
            f"• LONG/Short: {metrics['long_positions']}/{metrics['short_positions']}\n\n"
            
            f"**💡 РЕКОМЕНДАЦИИ:**\n" + "\n".join(f"• {rec}" for rec in recommendations) + "\n\n"
            
            f"**🔗 КОРРЕЛЯЦИИ:**\n" + "\n".join(f"• {corr}" for corr in correlations)
        )
        
        # Кнопки управления
        keyboard = [
            [InlineKeyboardButton("🗑 Очистить портфель", callback_data="clear_portfolio")],
            [InlineKeyboardButton("📥 Выгрузить отчет", callback_data="export_portfolio")],
            [InlineKeyboardButton("🎯 Новая сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")]
        ]
    
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

async def clear_portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Очистка портфеля"""
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    
    PortfolioManager.clear_portfolio(user_id)
    
    await query.edit_message_text(
        "✅ Портфель очищен",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🎯 Одна сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")],
            [InlineKeyboardButton("📋 В портфель", callback_data="portfolio")]
        ])
    )

async def export_portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Выгрузка отчета портфеля"""
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    
    PortfolioManager.ensure_user(user_id)
    user_portfolio = user_data[user_id]
    trades = user_portfolio.get('multi_trades', [])
    single_trades = user_portfolio.get('single_trades', [])
    deposit = user_portfolio.get('deposit', 0)
    leverage = user_portfolio.get('leverage', '1:100')
    
    all_trades = trades + single_trades
    
    if not all_trades:
        await query.answer("Портфель пуст", show_alert=True)
        return
    
    # Генерация текстового отчета
    metrics = PortfolioAnalyzer.calculate_portfolio_metrics(all_trades, deposit)
    recommendations = PortfolioAnalyzer.generate_recommendations(metrics, all_trades)
    
    report_lines = [
        "PRO RISK CALCULATOR - ОТЧЕТ ПОРТФЕЛЯ",
        f"Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Депозит: ${deposit:,.2f}",
        f"Плечо: {leverage}",
        f"Всего сделок: {len(all_trades)}",
        f"Одиночные сделки: {len(single_trades)}",
        f"Мультипозиции: {len(trades)}",
        "",
        "ДЕТАЛИ СДЕЛОК:",
        "-" * 50
    ]
    
    for i, trade in enumerate(all_trades, 1):
        report_lines.extend([
            f"{i}. {trade['asset']} {trade['direction']}",
            f"   Вход: {trade['entry_price']} | SL: {trade['stop_loss']} | TP: {trade['take_profit']}",
            f"   Размер: {trade['metrics']['position_size']:.2f} | Риск: ${trade['metrics']['risk_amount']:.2f}",
            f"   R/R: {trade['metrics']['rr_ratio']:.2f}",
            ""
        ])
    
    report_lines.extend([
        "МЕТРИКИ ПОРТФЕЛЯ:",
        "-" * 50,
        f"Общий риск: ${metrics['total_risk_usd']:.2f} ({metrics['total_risk_percent']:.1f}%)",
        f"Потенциальная прибыль: ${metrics['total_profit']:.2f}",
        f"Средний R/R: {metrics['avg_rr_ratio']:.2f}",
        f"Волатильность: {metrics['portfolio_volatility']:.1f}%",
        f"Активов: {metrics['unique_assets']} | LONG: {metrics['long_positions']} | SHORT: {metrics['short_positions']}",
        "",
        "РЕКОМЕНДАЦИИ:",
        "-" * 50
    ])
    
    report_lines.extend(recommendations)
    
    report_text = "\n".join(report_lines)
    
    # Создаем файл
    bio = io.BytesIO(report_text.encode('utf-8'))
    bio.name = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    
    await query.message.reply_document(
        document=InputFile(bio, filename=bio.name),
        caption="📊 Отчет вашего портфеля"
    )

# ---------------------------
# Future Features Handler
# ---------------------------
async def future_features_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Раздел будущих разработок"""
    query = update.callback_query
    await query.answer()
    
    text = (
        "🚀 **БУДУЩИЕ РАЗРАБОТКИ**\n\n"
        
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
        
        "**🛠 ТЕХНИЧЕСКИЕ УЛУЧШЕНИЯ**\n"
        "• Еще более быстрые расчеты\n"
        "• Расширенные API интеграции\n"
        "• Кастомные стратегии и скрипты\n\n"
        
        "Следите за обновлениями! 👨‍💻"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]]
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

# ---------------------------
# Main Callback Router
# ---------------------------
@performance_logger
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Маршрутизатор callback запросов"""
    query = update.callback_query
    if not query:
        return
    
    await query.answer()
    data = query.data
    user_id = query.from_user.id
    
    logger.info(f"Callback received: {data} from user {user_id}")
    
    if data == "main_menu":
        await start_command(update, context)
    elif data == "pro_calculation":
        keyboard = [
            [InlineKeyboardButton("🎯 Одна сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
        ]
        await query.edit_message_text("Выберите тип расчета:", reply_markup=InlineKeyboardMarkup(keyboard))
    elif data == "single_trade":
        # Запуск одиночной сделки через ConversationHandler
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
    else:
        logger.warning(f"Unknown callback data: {data}")
        await query.edit_message_text("⚠️ Функция временно недоступна")

# ---------------------------
# Conversation Handler Setup
# ---------------------------
def setup_conversation_handlers(application: Application):
    """Настройка обработчиков диалогов"""
    
    # Одиночная сделка
    single_trade_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(single_trade_start, pattern="^single_trade$")],
        states={
            SingleTradeState.DEPOSIT.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_deposit)
            ],
            SingleTradeState.LEVERAGE.value: [
                CallbackQueryHandler(single_trade_leverage, pattern="^lev_")
            ],
            SingleTradeState.ASSET.value: [
                CallbackQueryHandler(single_trade_asset, pattern="^(asset_|asset_manual)"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_asset_manual)
            ],
            SingleTradeState.DIRECTION.value: [
                CallbackQueryHandler(single_trade_direction, pattern="^dir_")
            ],
            SingleTradeState.ENTRY.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_entry)
            ],
            SingleTradeState.STOP_LOSS.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_stop_loss)
            ],
            SingleTradeState.TAKE_PROFIT.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_take_profit)
            ]
        },
        fallbacks=[
            CommandHandler("cancel", single_trade_cancel),
            MessageHandler(filters.TEXT, single_trade_cancel)
        ],
        name="single_trade_conversation"
    )
    
    # Мультипозиция
    multi_trade_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(multi_trade_start, pattern="^multi_trade_start$")],
        states={
            MultiTradeState.DEPOSIT.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_deposit)
            ],
            MultiTradeState.LEVERAGE.value: [
                CallbackQueryHandler(multi_trade_leverage, pattern="^lev_")
            ],
            MultiTradeState.ASSET.value: [
                CallbackQueryHandler(multi_trade_asset, pattern="^(asset_|asset_manual|multi_finish)"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_asset_manual)
            ],
            MultiTradeState.DIRECTION.value: [
                CallbackQueryHandler(multi_trade_direction, pattern="^dir_")
            ],
            MultiTradeState.ENTRY.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_entry)
            ],
            MultiTradeState.STOP_LOSS.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_stop_loss)
            ],
            MultiTradeState.TAKE_PROFIT.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, multi_trade_take_profit)
            ],
            MultiTradeState.ADD_MORE.value: [
                CallbackQueryHandler(multi_trade_add_another, pattern="^(add_another|multi_finish)$")
            ]
        },
        fallbacks=[
            CommandHandler("cancel", multi_trade_cancel),
            MessageHandler(filters.TEXT, multi_trade_cancel)
        ],
        name="multi_trade_conversation"
    )
    
    application.add_handler(single_trade_conv)
    application.add_handler(multi_trade_conv)

# ---------------------------
# Webhook & Main
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
    """Запуск HTTP сервера"""
    app = web.Application()
    
    async def handle_webhook(request):
        """Обработчик вебхука"""
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
    
    logger.info(f"HTTP сервер запущен на порту {PORT}")
    return runner

async def main():
    """Основная функция"""
    application = Application.builder().token(TOKEN).build()
    
    # Регистрация обработчиков - ВАЖНО: СНАЧАЛА ConversationHandler, ПОТОМ остальные
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("pro_info", pro_info_command))
    
    # Настройка диалогов - ДОЛЖНЫ БЫТЬ ПЕРВЫМИ
    setup_conversation_handlers(application)
    
    # Callback router - ПОСЛЕ ConversationHandler
    application.add_handler(CallbackQueryHandler(callback_router))
    
    # Обработчик для любых сообщений (fallback) - ДОЛЖЕН БЫТЬ ПОСЛЕДНИМ
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, 
        lambda update, context: update.message.reply_text(
            "Используйте меню для навигации или /start для начала работы"
        )
    ))
    
    # Режим запуска
    if WEBHOOK_URL and WEBHOOK_URL.strip():
        logger.info("Запуск в режиме WEBHOOK")
        await application.initialize()
        
        if await set_webhook(application):
            await start_http_server(application)
            logger.info("Бот запущен в режиме WEBHOOK")
            await asyncio.Event().wait()
        else:
            logger.error("Не удалось установить вебхук, запуск в режиме polling")
            await application.run_polling()
    else:
        logger.info("Запуск в режиме POLLING")
        await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
