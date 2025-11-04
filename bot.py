# bot.py — PRO Risk Calculator v2.0 | ENTERPRISE EDITION
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
    ASSET_CATEGORY = 3  # НОВОЕ: Категория активов
    ASSET = 4
    DIRECTION = 5
    ENTRY = 6
    STOP_LOSS = 7
    RISK_LEVEL = 8
    TAKE_PROFIT = 9

class MultiTradeState(Enum):
    DEPOSIT = 1
    LEVERAGE = 2
    ASSET_CATEGORY = 3  # НОВОЕ: Категория активов
    ASSET = 4
    DIRECTION = 5
    ENTRY = 6
    STOP_LOSS = 7
    RISK_LEVEL = 8
    TAKE_PROFIT = 9
    ADD_MORE = 10

# Инструменты и пресеты - ОБНОВЛЕНО: Группировка по категориям
ASSET_CATEGORIES = {
    "FOREX": ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'],
    "CRYPTO": ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'ADAUSDT', 'DOTUSDT'],
    "INDICES": ['NAS100', 'SPX500', 'DJ30', 'FTSE100', 'DAX40', 'NIKKEI225', 'ASX200'],
    "METALS": ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD'],
    "ENERGY": ['OIL', 'NATURALGAS', 'BRENT'],
    "STOCKS": ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX']
}

LEVERAGES = ['1:10', '1:20', '1:50', '1:100', '1:200', '1:500', '1:1000']  # ОБНОВЛЕНО: Добавлено 1:1000
RISK_LEVELS = ['2%', '5%', '7%', '10%', '15%', '20%', '25%']

# Волатильность активов
VOLATILITY_DATA = {
    'BTCUSDT': 65.2, 'ETHUSDT': 70.5, 'AAPL': 25.3, 'TSLA': 55.1,
    'GOOGL': 22.8, 'MSFT': 20.1, 'AMZN': 28.7, 'EURUSD': 8.5,
    'GBPUSD': 9.2, 'USDJPY': 7.8, 'XAUUSD': 14.5, 'XAGUSD': 25.3,
    'OIL': 35.2, 'NAS100': 18.5
}

# ОБНОВЛЕНО: Размеры контрактов и стоимость пункта
CONTRACT_SIZES = {
    'BTCUSDT': 1, 'ETHUSDT': 1, 'XRPUSDT': 1, 'LTCUSDT': 1, 'BCHUSDT': 1,
    'ADAUSDT': 1, 'DOTUSDT': 1, 'AAPL': 100, 'TSLA': 100, 'GOOGL': 100,
    'MSFT': 100, 'AMZN': 100, 'META': 100, 'NFLX': 100, 'EURUSD': 100000,
    'GBPUSD': 100000, 'USDJPY': 100000, 'USDCHF': 100000, 'AUDUSD': 100000,
    'USDCAD': 100000, 'NZDUSD': 100000, 'XAUUSD': 100, 'XAGUSD': 5000,
    'XPTUSD': 100, 'XPDUSD': 100, 'OIL': 1000, 'NATURALGAS': 10000,
    'BRENT': 1000, 'NAS100': 10, 'SPX500': 50, 'DJ30': 5, 'FTSE100': 10,
    'DAX40': 25, 'NIKKEI225': 5, 'ASX200': 1
}

# НОВОЕ: Стоимость пункта за лот (в валюте депозита - USD)
PIP_VALUES = {
    'EURUSD': 10.0, 'GBPUSD': 10.0, 'USDJPY': 9.09, 'USDCHF': 10.0,
    'AUDUSD': 10.0, 'USDCAD': 10.0, 'NZDUSD': 10.0, 'XAUUSD': 10.0,
    'XAGUSD': 50.0, 'OIL': 10.0, 'NAS100': 1.0, 'SPX500': 0.5,
    'DJ30': 1.0, 'FTSE100': 1.0, 'DAX40': 0.25
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
# Risk Calculator - ПОЛНОСТЬЮ ПЕРЕРАБОТАН
# ---------------------------
class RiskCalculator:
    @staticmethod
    def calculate_margin_metrics(trade: Dict, deposit: float, leverage: str, risk_level: str) -> Dict:
        """
        ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ МАРЖИ И ЛОТОВ - ИСПРАВЛЕННЫЙ
        """
        try:
            entry = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            direction = trade['direction']
            asset = trade['asset']
            
            # Расчет дистанции стоп-лосса в пунктах
            if direction.upper() == 'LONG':
                stop_distance_pips = entry - stop_loss
                profit_distance_pips = take_profit - entry
            else:  # SHORT
                stop_distance_pips = stop_loss - entry
                profit_distance_pips = entry - take_profit
            
            # Получаем стоимость пункта для актива
            pip_value = PIP_VALUES.get(asset, 10.0)  # По умолчанию $10 за лот
            
            # Расчет суммы риска
            risk_percent = float(risk_level.strip('%'))
            risk_amount = deposit * (risk_percent / 100)
            
            # Расчет объема в лотах на основе риска
            # Lots = Risk Amount / (Stop Loss Distance in Pips × Pip Value)
            if stop_distance_pips > 0 and pip_value > 0:
                volume_lots = risk_amount / (stop_distance_pips * pip_value)
                volume_lots = round(volume_lots, 2)
            else:
                volume_lots = 0
            
            # Расчет требуемой маржи
            lev_value = int(leverage.split(':')[1])
            contract_size = CONTRACT_SIZES.get(asset, 1)
            
            # Required Margin = (Volume × Contract Size) / Leverage
            required_margin = (volume_lots * contract_size) / lev_value
            required_margin = round(required_margin, 2)
            
            # Проверка на достаточность депозита
            if required_margin > deposit:
                volume_lots = (deposit * lev_value) / contract_size
                volume_lots = round(volume_lots, 2)
                required_margin = deposit
                # Пересчет risk_amount на основе нового объема
                risk_amount = volume_lots * stop_distance_pips * pip_value
            
            # Свободная маржа и уровень маржи
            free_margin = deposit - required_margin
            free_margin = round(free_margin, 2)
            
            margin_level = (deposit / required_margin) * 100 if required_margin > 0 else 0
            margin_level = round(margin_level, 1)
            
            # Расчет потенциальной прибыли
            potential_profit = volume_lots * profit_distance_pips * pip_value
            potential_profit = round(potential_profit, 2)
            
            # Risk/Reward ratio
            rr_ratio = potential_profit / risk_amount if risk_amount > 0 else 0
            rr_ratio = round(rr_ratio, 2)
            
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
                'contract_size': contract_size
            }
        except Exception as e:
            logger.error("Ошибка расчета маржи: %s", e)
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
        total_margin = sum(t.get('metrics', {}).get('required_margin', 0) for t in trades)
        
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
        
        # Уровень маржи портфеля
        portfolio_margin_level = (deposit / total_margin) * 100 if total_margin > 0 else 0
        
        return {
            'total_risk_usd': total_risk,
            'total_risk_percent': (total_risk / deposit) * 100 if deposit > 0 else 0,
            'total_profit': total_profit,
            'total_margin': total_margin,
            'portfolio_margin_level': portfolio_margin_level,
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
        
        # Проверка уровня маржи
        if metrics.get('portfolio_margin_level', 0) < 100:
            recommendations.append(
                "🔴 КРИТИЧЕСКИЙ УРОВЕНЬ МАРЖИ! Немедленно пополните счет "
                "или закрите часть позиций."
            )
        elif metrics.get('portfolio_margin_level', 0) < 200:
            recommendations.append(
                "🟡 НИЗКИЙ УРОВЕНЬ МАРЖИ: Рассмотрите пополнение счета "
                "для безопасности позиций."
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
# Handlers - ПОЛНОСТЬЮ ПЕРЕРАБОТАНЫ
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
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    user_id = user.id
    PortfolioManager.ensure_user(user_id)
    
    # Проверяем есть ли сохраненный прогресс
    temp_data = DataManager.load_temporary_data()
    saved_progress = temp_data.get(str(user_id))
    
    text = (
        f"👋 Привет, {user.first_name}!\n\n"
        "🤖 **PRO Калькулятор Управления Рисками v2.0**\n\n"
        "**МОИ ВОЗМОЖНОСТИ:**\n"
        "• 📊 Профессиональный расчет позиций с маржой и лотами\n"
        "• 🎯 Контроль уровней риска (2%-25%)\n"
        "• 💼 Мультипозиционный анализ портфеля\n"
        "• 💡 Умные рекомендации и аналитика\n\n"
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
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

@performance_logger
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """PRO инструкции"""
    text = (
        "📚 **PRO ИНСТРУКЦИИ v2.0**\n\n"
        
        "**🎯 ПРАКТИЧЕСКОЕ УПРАВЛЕНИЕ РИСКАМИ**\n\n"
        
        "**📊 КЛЮЧЕВЫЕ ПРИНЦИПЫ ДЛЯ ПРОФЕССИОНАЛОВ:**\n\n"
        
        "**1. УПРАВЛЕНИЕ РАЗМЕРОМ ПОЗИЦИИ**\n"
        "• Всегда определяйте риск ДО входа в сделку\n"
        "• Используйте фиксированный % от депозита (2-5%)\n"
        "• Рассчитывайте объем позиции на основе стоп-лосса\n"
        "• Учитывайте кредитное плечо при расчете маржи\n\n"
        
        "**2. УРОВНИ РИСКА И ИХ ПРИМЕНЕНИЕ**\n"
        "• 2% - Консервативный: Для начинающих и крупных капиталов\n"
        "• 5% - Стандартный: Баланс роста и безопасности\n"
        "• 10% - Агрессивный: Для опытных трейдеров\n"
        "• 25% - Максимальный: Только для уверенных сделок\n\n"
        
        "**3. РАСЧЕТ МАРЖИ И ЛОТОВ**\n"
        "• Volume = Risk Amount / (Stop Distance × Pip Value)\n"
        "• Required Margin = (Volume × Contract Size) / Leverage\n"
        "• Всегда следите за уровнем маржи (>200%)\n"
        "• Оставляйте свободную маржу для маневра\n\n"
        
        "Разработчик: @fxfeelgood"
    )
    
    keyboard = [
        [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
    ]
    
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

# ---------------------------
# Single Trade Conversation Handler - ПОЛНОСТЬЮ ПЕРЕРАБОТАН
# ---------------------------
async def single_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало одиночной сделки"""
    query = update.callback_query
    await query.answer()
    
    text = (
        "🎯 **ОДИНОЧНАЯ СДЕЛКА v2.0**\n\n"
        "Профессиональный расчет с контролем риска и расчетом маржи.\n\n"
        "**Введите ваш депозит в USD:**"
    )
    
    keyboard = [
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
    ]
    
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
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
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
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
    
    # НОВОЕ: Выбор категории активов
    keyboard = []
    for category in ASSET_CATEGORIES.keys():
        keyboard.append([InlineKeyboardButton(category, callback_data=f"cat_{category}")])
    
    keyboard.append([InlineKeyboardButton("📝 Ввести актив вручную", callback_data="asset_manual")])
    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
    
    await query.edit_message_text(
        f"✅ Плечо: {leverage}\n\n"
        "**Выберите категорию актива:**",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return SingleTradeState.ASSET_CATEGORY.value

async def single_trade_asset_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """НОВОЕ: Обработка выбора категории активов"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "asset_manual":
        await query.edit_message_text(
            "✍️ Введите название актива (например: BTCUSDT):",
            reply_markup=InlineKeyboardMarkup([
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
    
    await query.edit_message_text(
        f"✅ Категория: {category}\n\n"
        "**Выберите актив:**",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return SingleTradeState.ASSET.value

async def single_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора актива для одиночной сделки"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "back_to_categories":
        # Возврат к выбору категории
        keyboard = []
        for category in ASSET_CATEGORIES.keys():
            keyboard.append([InlineKeyboardButton(category, callback_data=f"cat_{category}")])
        
        keyboard.append([InlineKeyboardButton("📝 Ввести актив вручную", callback_data="asset_manual")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await query.edit_message_text(
            "**Выберите категорию актива:**",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return SingleTradeState.ASSET_CATEGORY.value
    
    asset = query.data.replace('asset_', '')
    context.user_data['asset'] = asset
    
    await query.edit_message_text(
        f"✅ Актив: {asset}\n\n"
        "**Выберите направление сделки:**",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
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
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
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
        "**Введите цену входа:**",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
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
            "**Введите уровень стоп-лосса:**",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
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
        
        # Переход к выбору уровня риска
        keyboard = []
        for risk_level in RISK_LEVELS:
            keyboard.append([InlineKeyboardButton(risk_level, callback_data=f"risk_{risk_level}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await update.message.reply_text(
            f"✅ Стоп-лосс: {stop_loss}\n\n"
            "**Выберите уровень риска:**",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return SingleTradeState.RISK_LEVEL.value
        
    except ValueError:
        await update.message.reply_text("❌ Введите число (например: 48000)\nПопробуйте еще раз:")
        return SingleTradeState.STOP_LOSS.value

async def single_trade_risk_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора уровня риска"""
    query = update.callback_query
    await query.answer()
    
    risk_level = query.data.replace('risk_', '')
    context.user_data['risk_level'] = risk_level
    
    await query.edit_message_text(
        f"✅ Уровень риска: {risk_level}\n\n"
        "**Введите уровень тейк-профита:**",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return SingleTradeState.TAKE_PROFIT.value

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
            'take_profit': take_profit,
            'risk_level': context.user_data['risk_level']
        }
        
        # Расчет метрик с ИСПРАВЛЕННЫМ калькулятором
        deposit = context.user_data['deposit']
        leverage = context.user_data['leverage']
        risk_level = context.user_data['risk_level']
        metrics = RiskCalculator.calculate_margin_metrics(trade_data, deposit, leverage, risk_level)
        
        # Сохраняем сделку
        user_id = update.message.from_user.id
        trade_data['metrics'] = metrics
        PortfolioManager.add_single_trade(user_id, trade_data)
        
        # Очищаем временный прогресс
        DataManager.clear_temporary_progress(user_id)
        
        # Формируем результат с ИСПРАВЛЕННЫМ ВЫВОДОМ
        text = (
            f"🎯 **ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ СДЕЛКИ v2.0**\n\n"
            f"**📊 ПАРАМЕТРЫ СДЕЛКИ:**\n"
            f"• Актив: {trade_data['asset']}\n"
            f"• Направление: {trade_data['direction']}\n"
            f"• Вход: {trade_data['entry_price']}\n"
            f"• Стоп-лосс: {trade_data['stop_loss']}\n"
            f"• Тейк-профит: {trade_data['take_profit']}\n"
            f"• Уровень риска: {trade_data['risk_level']}\n\n"
            
            f"**💰 ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ:**\n"
            f"• Объем позиции: {metrics['volume_lots']:.2f} лотов\n"
            f"• Требуемая маржа: ${metrics['required_margin']:.2f}\n"
            f"• Свободная маржа: ${metrics['free_margin']:.2f}\n"
            f"• Уровень маржи: {metrics['margin_level']:.1f}%\n"
            f"• Сумма риска: ${metrics['risk_amount']:.2f} ({metrics['risk_percent']}%)\n"
            f"• Потенциальная прибыль: ${metrics['potential_profit']:.2f}\n"
            f"• Соотношение R/R: {metrics['rr_ratio']:.2f}\n\n"
            
            f"**💡 РЕКОМЕНДАЦИЯ:**\n"
        )
        
        if metrics['risk_percent'] > 10:
            text += "🔴 ВЫСОКИЙ РИСК! Превышен порог 10%. Уменьшите объем позиции.\n\n"
        elif metrics['margin_level'] < 100:
            text += "🔴 КРИТИЧЕСКИЙ УРОВЕНЬ МАРЖИ! Пополните счет.\n\n"
        elif metrics['rr_ratio'] < 1:
            text += "🟡 Соотношение R/R меньше 1! Пересмотрите уровни TP/SL.\n\n"
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
    user_id = update.message.from_user.id
    DataManager.clear_temporary_progress(user_id)
    context.user_data.clear()
    await update.message.reply_text("❌ Расчет отменен")
    return ConversationHandler.END

# ---------------------------
# Multi-trade Conversation Handler - ПОЛНОСТЬЮ ПЕРЕРАБОТАН
# ---------------------------
async def multi_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало мультипозиционного расчета"""
    query = update.callback_query
    await query.answer()
    
    context.user_data['multi_trades'] = []
    
    text = (
        "🎯 **МУЛЬТИПОЗИЦИОННЫЙ РАСЧЕТ v2.0**\n\n"
        "Профессиональный расчет нескольких сделок с контролем общего риска.\n\n"
        "**Введите общий депозит в USD:**"
    )
    
    keyboard = [
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
    ]
    
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
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
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
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
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    return MultiTradeState.ASSET_CATEGORY.value

async def multi_trade_asset_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """НОВОЕ: Обработка выбора категории активов для мультипозиции"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "asset_manual":
        await query.edit_message_text(
            "✍️ Введите название актива (например: BTCUSDT):",
            reply_markup=InlineKeyboardMarkup([
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
    
    await query.edit_message_text(
        f"✅ Категория: {category}\n\n"
        "**Выберите актив:**",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return MultiTradeState.ASSET.value

async def multi_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора актива"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "back_to_categories":
        return await start_trade_input(update, context)
    
    asset = query.data.replace('asset_', '')
    context.user_data['current_trade']['asset'] = asset
    
    await query.edit_message_text(
        f"✅ Актив: {asset}\n\n"
        "**Выберите направление сделки:**",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 LONG", callback_data="dir_LONG")],
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
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
            [InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
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
        "**Введите цену входа:**",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
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
            "**Введите уровень стоп-лосса:**",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
            ])
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
        
        # Переход к выбору уровня риска
        keyboard = []
        for risk_level in RISK_LEVELS:
            keyboard.append([InlineKeyboardButton(risk_level, callback_data=f"risk_{risk_level}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await update.message.reply_text(
            f"✅ Стоп-лосс: {stop_loss}\n\n"
            "**Выберите уровень риска для этой сделки:**",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return MultiTradeState.RISK_LEVEL.value
        
    except ValueError:
        await update.message.reply_text("❌ Введите число (например: 48000)\nПопробуйте еще раз:")
        return MultiTradeState.STOP_LOSS.value

async def multi_trade_risk_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора уровня риска для мультипозиции"""
    query = update.callback_query
    await query.answer()
    
    risk_level = query.data.replace('risk_', '')
    context.user_data['current_trade']['risk_level'] = risk_level
    
    await query.edit_message_text(
        f"✅ Уровень риска: {risk_level}\n\n"
        "**Введите уровень тейк-профита:**",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
        ])
    )
    return MultiTradeState.TAKE_PROFIT.value

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
        
        # Расчет метрик с ИСПРАВЛЕННЫМ калькулятором
        deposit = context.user_data['deposit']
        leverage = context.user_data['leverage']
        risk_level = current_trade['risk_level']
        metrics = RiskCalculator.calculate_margin_metrics(current_trade, deposit, leverage, risk_level)
        current_trade['metrics'] = metrics
        
        # Добавляем сделку в список
        context.user_data['multi_trades'].append(current_trade.copy())
        
        # Показываем результаты с ИСПРАВЛЕННЫМ ВЫВОДОМ
        trade_count = len(context.user_data['multi_trades'])
        text = (
            f"✅ **СДЕЛКА #{trade_count} ДОБАВЛЕНА**\n\n"
            f"**Актив:** {current_trade['asset']}\n"
            f"**Направление:** {current_trade['direction']}\n"
            f"**Вход:** {current_trade['entry_price']}\n"
            f"**SL:** {current_trade['stop_loss']}\n"
            f"**TP:** {current_trade['take_profit']}\n"
            f"**Риск:** {current_trade['risk_level']}\n\n"
            f"**📊 ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ:**\n"
            f"• Объем: {metrics['volume_lots']:.2f} лотов\n"
            f"• Маржа: ${metrics['required_margin']:.2f}\n"
            f"• Риск: ${metrics['risk_amount']:.2f}\n"
            f"• Прибыль: ${metrics['potential_profit']:.2f}\n"
            f"• R/R: {metrics['rr_ratio']:.2f}\n\n"
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
    DataManager.clear_temporary_progress(user_id)
    context.user_data.clear()
    
    # Переходим в портфель
    await show_portfolio(update, context, user_id)
    return ConversationHandler.END

async def multi_trade_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена мультипозиционного расчета"""
    user_id = update.message.from_user.id
    DataManager.clear_temporary_progress(user_id)
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
        text = "📊 **ВАШ ПОРТФЕЛЬ v2.0**\n\nПортфель пуст. Начните с расчета сделок!"
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
            f"📊 **ВАШ ПОРТФЕЛЬ v2.0**\n\n"
            f"**Основные параметры:**\n"
            f"• Депозит: ${deposit:,.2f}\n"
            f"• Плечо: {leverage}\n"
            f"• Всего сделок: {len(all_trades)}\n"
            f"• Одиночные: {len(single_trades)} | Мульти: {len(trades)}\n"
            f"• Уникальных активов: {metrics.get('unique_assets', 0)}\n\n"
            
            f"**📈 КЛЮЧЕВЫЕ МЕТРИКИ:**\n"
            f"• Общий риск: ${metrics['total_risk_usd']:.2f} ({metrics['total_risk_percent']:.1f}%)\n"
            f"• Потенциальная прибыль: ${metrics['total_profit']:.2f}\n"
            f"• Общая маржа: ${metrics['total_margin']:.2f}\n"
            f"• Уровень маржи портфеля: {metrics['portfolio_margin_level']:.1f}%\n"
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
    
    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
    
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
            [InlineKeyboardButton("📋 В портфель", callback_data="portfolio")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
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
        "PRO RISK CALCULATOR v2.0 - ОТЧЕТ ПОРТФЕЛЯ",
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
        f"Средний R/R: {metrics['avg_rr_ratio']:.2f}",
        f"Волатильность: {metrics['portfolio_volatility']:.1f}%",
        f"Активов: {metrics['unique_assets']} | LONG: {metrics['long_positions']} | SHORT: {metrics['short_positions']}",
        "",
        "ДЕТАЛИ СДЕЛОК:",
        "-" * 50
    ]
    
    for i, trade in enumerate(all_trades, 1):
        report_lines.extend([
            f"{i}. {trade['asset']} {trade['direction']} | Риск: {trade.get('risk_level', 'N/A')}",
            f"   Вход: {trade['entry_price']} | SL: {trade['stop_loss']} | TP: {trade['take_profit']}",
            f"   Объем: {trade['metrics']['volume_lots']:.2f} лотов | Маржа: ${trade['metrics']['required_margin']:.2f}",
            f"   Риск: ${trade['metrics']['risk_amount']:.2f} | Прибыль: ${trade['metrics']['potential_profit']:.2f}",
            f"   R/R: {trade['metrics']['rr_ratio']:.2f}",
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
    bio.name = f"portfolio_report_v2_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    
    await query.message.reply_document(
        document=InputFile(bio, filename=bio.name),
        caption="📊 Отчет вашего портфеля v2.0"
    )

# ---------------------------
# Future Features Handler
# ---------------------------
async def future_features_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Раздел будущих разработок"""
    query = update.callback_query
    await query.answer()
    
    text = (
        "🚀 **БУДУЩИЕ РАЗРАБОТКИ v2.0**\n\n"
        
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
        
        "Следите за обновлениями! 👨‍💻"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]]
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

# ---------------------------
# Progress Restoration Handler
# ---------------------------
async def restore_progress_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Восстановление сохраненного прогресса"""
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    
    temp_data = DataManager.load_temporary_data()
    saved_progress = temp_data.get(str(user_id))
    
    if not saved_progress:
        await query.answer("Нет сохраненного прогресса", show_alert=True)
        return
    
    # Восстанавливаем данные
    context.user_data.clear()
    context.user_data.update(saved_progress['state_data'])
    
    state_type = saved_progress['state_type']
    
    if state_type == "single":
        await query.answer("Прогресс одиночной сделки восстановлен", show_alert=True)
        # Определяем текущее состояние и переходим к нему
        if 'take_profit' in context.user_data:
            await query.edit_message_text(
                "✅ Прогресс восстановлен!\n\nВведите уровень тейк-профита:",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                ])
            )
            return SingleTradeState.TAKE_PROFIT.value
        elif 'risk_level' in context.user_data:
            await query.edit_message_text(
                "✅ Прогресс восстановлен!\n\nВведите уровень тейк-профита:",
                reply_markup=InlineKeyboardMarkup([
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
            
            await query.edit_message_text(
                "✅ Прогресс восстановлен!\n\nВыберите уровень риска:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return SingleTradeState.RISK_LEVEL.value
    else:
        await query.answer("Прогресс мультипозиции восстановлен", show_alert=True)
        return await start_trade_input(update, context)

# ---------------------------
# Main Callback Router - ПОЛНОСТЬЮ ПЕРЕРАБОТАН
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
    
    # Основные команды
    if data == "main_menu":
        await start_command(update, context)
    elif data == "main_menu_save":
        # Определяем текущее состояние
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
        await query.edit_message_text("Выберите тип расчета:", reply_markup=InlineKeyboardMarkup(keyboard))
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
                        await query.edit_message_text(
                            "✍️ Введите название актива (например: BTCUSDT):",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
                            ])
                        )
                        return SingleTradeState.ASSET.value
                    else:
                        await query.edit_message_text(
                            "✍️ Введите название актива (например: BTCUSDT):",
                            reply_markup=InlineKeyboardMarkup([
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
        await query.edit_message_text("⚠️ Функция временно недоступна")

# ---------------------------
# Conversation Handler Setup - ПОЛНОСТЬЮ ПЕРЕРАБОТАН
# ---------------------------
def setup_conversation_handlers(application: Application):
    """Настройка обработчиков диалогов"""
    
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
            SingleTradeState.ASSET_CATEGORY.value: [  # НОВОЕ: состояние категории
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
            MultiTradeState.ASSET_CATEGORY.value: [  # НОВОЕ: состояние категории
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
