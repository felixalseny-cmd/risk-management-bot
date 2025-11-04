import os
import logging
import asyncio
import re
import time
import functools
import json
import math
import aiohttp
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
from aiohttp import web

# === НАСТРОЙКИ ДЛЯ RENDER ===
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден! Добавь в переменные окружения.")

PORT = int(os.getenv("PORT", 5000))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_PATH = f"/webhook/{TOKEN}"

# Оптимизированное логирование для Render
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Декоратор для логирования производительности
def log_performance(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            if execution_time > 1.0:
                logger.warning(f"Медленная операция: {func.__name__} заняла {execution_time:.2f}с")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Ошибка в {func.__name__}: {e} (время: {execution_time:.2f}с)")
            raise
    return wrapper

# Состояния диалога
(
    MAIN_MENU, INSTRUMENT_TYPE, CUSTOM_INSTRUMENT, DIRECTION, 
    RISK_PERCENT, DEPOSIT, LEVERAGE, ENTRY, 
    STOP_LOSS, TAKE_PROFIT_SINGLE, SINGLE_OR_MULTI,
    PORTFOLIO_MENU, ADD_TRADE_INSTRUMENT, ADD_TRADE_DIRECTION,
    ADD_TRADE_ENTRY, ADD_TRADE_EXIT, ADD_TRADE_VOLUME, ADD_TRADE_PROFIT,
    DEPOSIT_AMOUNT, WITHDRAW_AMOUNT, ANALYTICS_MENU
) = range(21)

# Константы
INSTRUMENT_TYPES = {
    'forex': 'Форекс',
    'crypto': 'Криптовалюты', 
    'indices': 'Индексы',
    'commodities': 'Сырьевые товары',
    'metals': 'Металлы'
}

INSTRUMENT_PRESETS = {
    'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'NZDUSD', 'EURGBP'],
    'crypto': ['BTCUSD', 'ETHUSD', 'XRPUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD'],
    'indices': ['US30', 'NAS100', 'SPX500', 'DAX40', 'FTSE100'],
    'commodities': ['OIL', 'NATGAS', 'COPPER', 'GOLD'],
    'metals': ['XAUUSD', 'XAGUSD', 'XPTUSD']
}

# Данные по корреляции
CORRELATION_MATRIX = {
    'EURUSD': {'GBPUSD': 0.8, 'USDJPY': -0.7, 'USDCAD': -0.8, 'AUDUSD': 0.6, 'XAUUSD': 0.3},
    'GBPUSD': {'EURUSD': 0.8, 'USDJPY': -0.6, 'USDCAD': -0.7, 'AUDUSD': 0.5, 'XAUUSD': 0.2},
    'USDJPY': {'EURUSD': -0.7, 'GBPUSD': -0.6, 'USDCAD': 0.9, 'AUDUSD': -0.5, 'XAUUSD': -0.4},
    'USDCAD': {'EURUSD': -0.8, 'GBPUSD': -0.7, 'USDJPY': 0.9, 'AUDUSD': -0.6, 'XAUUSD': -0.3},
    'AUDUSD': {'EURUSD': 0.6, 'GBPUSD': 0.5, 'USDJPY': -0.5, 'USDCAD': -0.6, 'XAUUSD': 0.4},
    'XAUUSD': {'EURUSD': 0.3, 'GBPUSD': 0.2, 'USDJPY': -0.4, 'USDCAD': -0.3, 'AUDUSD': 0.4}
}

# Данные по волатильности (среднегодовая в %)
VOLATILITY_DATA = {
    'EURUSD': 8.5, 'GBPUSD': 9.2, 'USDJPY': 7.8, 'USDCAD': 7.5, 
    'AUDUSD': 10.1, 'NZDUSD': 9.8, 'EURGBP': 6.5,
    'BTCUSD': 65.2, 'ETHUSD': 70.5, 'XRPUSD': 85.3,
    'US30': 15.2, 'NAS100': 18.5, 'SPX500': 16.1,
    'XAUUSD': 14.5, 'XAGUSD': 25.3, 'OIL': 35.2
}

PIP_VALUES = {
    # Forex
    'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9, 'USDCHF': 10,
    'USDCAD': 10, 'AUDUSD': 10, 'NZDUSD': 10, 'EURGBP': 10,
    # Криптовалюты
    'BTCUSD': 1, 'ETHUSD': 1, 'XRPUSD': 10, 'ADAUSD': 10,
    # Индексы
    'US30': 1, 'NAS100': 1, 'SPX500': 1, 'DAX40': 1,
    # Сырьевые товары
    'OIL': 10, 'NATGAS': 10, 'COPPER': 10,
    # Металлы
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
RISK_LEVELS = ['1%', '2%', '3%', '5%', '7%', '10%', '15%']
TRADE_DIRECTIONS = ['BUY', 'SELL']

# Файл для сохранения данных
DATA_FILE = "user_data.json"

# Менеджер данных
class DataManager:
    @staticmethod
    def load_data():
        """Загрузка данных из файла"""
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return {}

    @staticmethod
    def save_data():
        """Сохранение данных в файл"""
        try:
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения данных: {e}")

# Глобальное хранилище данных пользователей
user_data: Dict[int, Dict[str, Any]] = DataManager.load_data()

# Упрощенный кэш
class FastCache:
    def __init__(self, max_size=100, ttl=300):
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
            oldest_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])[:10]
            for old_key in oldest_keys:
                del self.cache[old_key]
        self.cache[key] = (value, time.time())

fast_cache = FastCache()

# Анализатор портфеля
class PortfolioAnalyzer:
    @staticmethod
    def analyze_correlations(trades: List[Dict]) -> List[str]:
        """Анализ корреляций между позициями"""
        if len(trades) < 2:
            return ["ℹ️ Для анализа корреляций нужно минимум 2 позиции"]
        
        analysis = []
        for i, trade1 in enumerate(trades):
            for j, trade2 in enumerate(trades[i+1:], i+1):
                inst1, dir1 = trade1['instrument'], trade1['direction']
                inst2, dir2 = trade2['instrument'], trade2['direction']
                
                if inst1 in CORRELATION_MATRIX and inst2 in CORRELATION_MATRIX[inst1]:
                    corr = CORRELATION_MATRIX[inst1][inst2]
                    
                    if abs(corr) > 0.7:
                        if dir1 == dir2:
                            if corr > 0:
                                analysis.append(f"⚠️ Высокая позитивная корреляция ({corr:.2f}) между {inst1} {dir1} и {inst2} {dir2}")
                            else:
                                analysis.append(f"🔄 Высокая негативная корреляция ({corr:.2f}) между {inst1} {dir1} и {inst2} {dir2}")
        
        return analysis if analysis else ["✅ Корреляционный риск под контролем"]

    @staticmethod
    def analyze_volatility(trades: List[Dict]) -> List[str]:
        """Анализ волатильности позиций"""
        analysis = []
        high_vol_count = 0
        
        for trade in trades:
            instrument = trade['instrument']
            if instrument in VOLATILITY_DATA:
                vol = VOLATILITY_DATA[instrument]
                
                if vol > 20:
                    high_vol_count += 1
                    analysis.append(f"⚡ Высокая волатильность {instrument}: {vol}%")
        
        if high_vol_count >= 3:
            analysis.append("🚨 ВНИМАНИЕ: Много высоковолатильных инструментов")
        
        return analysis

# Менеджер портфеля
class PortfolioManager:
    @staticmethod
    def initialize_user_portfolio(user_id: int):
        if user_id not in user_data:
            user_data[user_id] = {}
        
        if 'portfolio' not in user_data[user_id]:
            user_data[user_id]['portfolio'] = {
                'initial_balance': 0,
                'current_balance': 0,
                'trades': [],
                'performance': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0,
                    'total_loss': 0,
                    'win_rate': 0,
                    'average_profit': 0,
                    'average_loss': 0,
                    'profit_factor': 0,
                    'max_drawdown': 0
                },
                'allocation': {},
                'history': [],
                'settings': {
                    'default_risk': 0.02,
                    'currency': 'USD',
                    'leverage': '1:100'
                }
            }
        DataManager.save_data()
    
    @staticmethod
    def add_trade(user_id: int, trade_data: Dict):
        PortfolioManager.initialize_user_portfolio(user_id)
        
        if len(user_data[user_id]['portfolio']['trades']) >= 10:
            raise ValueError("❌ Достигнут лимит в 10 сделок")
        
        trade_id = len(user_data[user_id]['portfolio']['trades']) + 1
        trade_data['id'] = trade_id
        trade_data['timestamp'] = datetime.now().isoformat()
        
        user_data[user_id]['portfolio']['trades'].append(trade_data)
        
        profit = trade_data.get('profit', 0)
        user_data[user_id]['portfolio']['current_balance'] += profit
        
        PortfolioManager.update_performance_metrics(user_id)
        
        instrument = trade_data.get('instrument', 'Unknown')
        if instrument not in user_data[user_id]['portfolio']['allocation']:
            user_data[user_id]['portfolio']['allocation'][instrument] = 0
        user_data[user_id]['portfolio']['allocation'][instrument] += 1
        
        DataManager.save_data()
        return trade_id
    
    @staticmethod
    def update_performance_metrics(user_id: int):
        portfolio = user_data[user_id]['portfolio']
        trades = portfolio['trades']
        
        if not trades:
            return
        
        closed_trades = [t for t in trades if t.get('status') == 'closed']
        if not closed_trades:
            return
            
        winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('profit', 0) < 0]
        
        portfolio['performance']['total_trades'] = len(closed_trades)
        portfolio['performance']['winning_trades'] = len(winning_trades)
        portfolio['performance']['losing_trades'] = len(losing_trades)
        portfolio['performance']['total_profit'] = sum(t.get('profit', 0) for t in winning_trades)
        portfolio['performance']['total_loss'] = abs(sum(t.get('profit', 0) for t in losing_trades))
        
        if closed_trades:
            portfolio['performance']['win_rate'] = (len(winning_trades) / len(closed_trades)) * 100
            
            portfolio['performance']['average_profit'] = (
                portfolio['performance']['total_profit'] / len(winning_trades) 
                if winning_trades else 0
            )
            portfolio['performance']['average_loss'] = (
                portfolio['performance']['total_loss'] / len(losing_trades) 
                if losing_trades else 0
            )
            
            if portfolio['performance']['total_loss'] > 0:
                portfolio['performance']['profit_factor'] = (
                    portfolio['performance']['total_profit'] / portfolio['performance']['total_loss']
                )
            else:
                portfolio['performance']['profit_factor'] = float('inf') if portfolio['performance']['total_profit'] > 0 else 0
        
        DataManager.save_data()

# Калькулятор рисков
class FastRiskCalculator:
    @staticmethod
    def calculate_pip_value_fast(instrument_type: str, currency_pair: str, lot_size: float) -> float:
        """Быстрый расчет стоимости пипса"""
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
        take_profit: float,
        direction: str,
        risk_percent: float = 0.02
    ) -> Dict[str, float]:
        """Ультра-быстрый расчет размера позиции"""
        try:
            cache_key = f"pos_{deposit}_{leverage}_{instrument_type}_{currency_pair}_{entry_price}_{stop_loss}_{take_profit}_{direction}_{risk_percent}"
            cached_result = fast_cache.get(cache_key)
            if cached_result:
                return cached_result
            
            lev_value = int(leverage.split(':')[1])
            risk_amount = deposit * risk_percent
            
            # Расчет стоп-лосса в пунктах
            if instrument_type == 'forex':
                stop_pips = abs(entry_price - stop_loss) * 10000
                take_profit_pips = abs(entry_price - take_profit) * 10000
            elif instrument_type == 'crypto':
                stop_pips = abs(entry_price - stop_loss) * 100
                take_profit_pips = abs(entry_price - take_profit) * 100
            elif instrument_type in ['indices', 'commodities', 'metals']:
                stop_pips = abs(entry_price - stop_loss) * 10
                take_profit_pips = abs(entry_price - take_profit) * 10
            else:
                stop_pips = abs(entry_price - stop_loss) * 10000
                take_profit_pips = abs(entry_price - take_profit) * 10000

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
            
            # Расчет прибыли/убытка
            if direction == 'BUY':
                potential_profit = (take_profit - entry_price) * pip_value_per_lot * position_size
                potential_loss = (stop_loss - entry_price) * pip_value_per_lot * position_size
            else:  # SELL
                potential_profit = (entry_price - take_profit) * pip_value_per_lot * position_size
                potential_loss = (entry_price - stop_loss) * pip_value_per_lot * position_size
            
            if potential_profit < 0:
                potential_profit = 0
                reward_risk_ratio = 0
            else:
                reward_risk_ratio = potential_profit / risk_amount if risk_amount > 0 else 0
            
            result = {
                'position_size': position_size,
                'risk_amount': risk_amount,
                'stop_pips': stop_pips,
                'take_profit_pips': take_profit_pips,
                'potential_profit': potential_profit,
                'potential_loss': abs(potential_loss),
                'reward_risk_ratio': reward_risk_ratio,
                'required_margin': required_margin,
                'risk_percent': (risk_amount / deposit) * 100 if deposit > 0 else 0,
                'free_margin': deposit - required_margin,
                'is_profitable': potential_profit > 0
            }
            
            fast_cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Ошибка в расчете размера позиции: {e}")
            return {
                'position_size': 0.01,
                'risk_amount': 0,
                'stop_pips': 0,
                'take_profit_pips': 0,
                'potential_profit': 0,
                'potential_loss': 0,
                'reward_risk_ratio': 0,
                'required_margin': 0,
                'risk_percent': 0,
                'free_margin': deposit,
                'is_profitable': False
            }

# Валидатор ввода
class InputValidator:
    @staticmethod
    def validate_number(text: str, min_val: float = 0, max_val: float = None) -> Tuple[bool, float, str]:
        """Валидация числового значения"""
        try:
            value = float(text.replace(',', '.'))
            if value < min_val:
                return False, value, f"❌ Значение не может быть меньше {min_val}"
            if max_val and value > max_val:
                return False, value, f"❌ Значение не может быть больше {max_val}"
            return True, value, "✅ Корректное значение"
        except ValueError:
            return False, 0, "❌ Введите корректное числовое значение"
    
    @staticmethod
    def validate_instrument(instrument: str) -> Tuple[bool, str]:
        """Валидация названия инструмента"""
        instrument = instrument.upper().strip()
        if not instrument:
            return False, "❌ Введите название инструмента"
        if len(instrument) > 20:
            return False, "❌ Название инструмента слишком длинное"
        return True, instrument
    
    @staticmethod
    def validate_price(price: str) -> Tuple[bool, float, str]:
        """Валидация цены"""
        return InputValidator.validate_number(price, 0.0001, 1000000)

# Генератор отчетов
class ReportGenerator:
    @staticmethod
    def get_professional_recommendations(calculation_data: Dict, user_data_context: Dict) -> str:
        """Генерация профессиональных рекомендаций"""
        recommendations = []
        
        rr_ratio = calculation_data.get('reward_risk_ratio', 0)
        risk_percent = calculation_data.get('risk_percent', 0)
        is_profitable = calculation_data.get('is_profitable', True)
        
        if not is_profitable:
            recommendations.append("🔴 УБЫТОЧНАЯ СДЕЛКА: Пересмотрите уровни тейк-профита")
        elif rr_ratio < 1:
            recommendations.append("🔴 КРИТИЧЕСКИЙ УРОВЕНЬ: Соотношение прибыль/риск меньше 1")
        elif rr_ratio < 1.5:
            recommendations.append("🟡 НИЗКИЙ УРОВЕНЬ: Стремитесь к соотношению не менее 1:2")
        elif rr_ratio >= 2:
            recommendations.append("🟢 ОТЛИЧНО: Соотношение прибыль/риск более 2:1")
        
        if risk_percent > 5:
            recommendations.append("🔴 ВЫСОКИЙ РИСК: Уменьшите риск до 1-2%")
        elif risk_percent < 1:
            recommendations.append("🟡 НИЗКИЙ РИСК: Можно увеличить риск до 2-3%")
        else:
            recommendations.append("🟢 ОПТИМАЛЬНЫЙ РИСК: 1-5% на сделку")
        
        return "\n".join(recommendations)

# ========== ОСНОВНЫЕ ОБРАБОТЧИКИ ==========

@log_performance
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Главное меню"""
    try:
        logger.info(f"Команда /start от пользователя {update.effective_user.id}")
        
        if context.user_data:
            context.user_data.clear()
        
        user = update.message.from_user if update.message else update.callback_query.from_user
        user_name = user.first_name or "Трейдер"
        
        welcome_text = f"""
👋 *Привет, {user_name}!*

🎯 PRO Калькулятор Управления Рисками v4.0

⚡ *ВОЗМОЖНОСТИ:*
• ✅ Профессиональный расчет позиций
• ✅ Управление портфелем
• ✅ Анализ корреляций и волатильности
• ✅ Рекомендации по риск-менеджменту

*Выберите опцию:*
"""
        
        user_id = user.id
        PortfolioManager.initialize_user_portfolio(user_id)
        
        keyboard = [
            [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("💼 Мой портфель", callback_data="portfolio")],
            [InlineKeyboardButton("🔮 Аналитика", callback_data="analytics")],
            [InlineKeyboardButton("📚 Инструкции", callback_data="pro_info")]
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
    except Exception as e:
        logger.error(f"Ошибка в start: {e}")
        if update.message:
            await update.message.reply_text("🔄 Произошла ошибка. Попробуйте еще раз /start")
        return MAIN_MENU

@log_performance
async def start_pro_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало профессионального расчета"""
    try:
        query = update.callback_query
        await query.answer()
        
        keyboard = [
            [InlineKeyboardButton("📈 Одна сделка", callback_data="single_trade")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            "📊 *ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ*\n\nВыберите тип расчета:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return SINGLE_OR_MULTI
    except Exception as e:
        logger.error(f"Ошибка в start_pro_calculation: {e}")
        await handle_error(update, context, e)

@log_performance
async def pro_select_instrument_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор типа инструмента"""
    try:
        query = update.callback_query
        await query.answer()
        
        instrument_type = query.data.replace("pro_type_", "")
        context.user_data['instrument_type'] = instrument_type
        
        presets = INSTRUMENT_PRESETS.get(instrument_type, [])
        
        keyboard = []
        for preset in presets:
            keyboard.append([InlineKeyboardButton(preset, callback_data=f"pro_preset_{preset}")])
        keyboard.append([InlineKeyboardButton("✏️ Ввести свой инструмент", callback_data="pro_custom")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="pro_calculation")])
        
        await query.edit_message_text(
            f"📊 *{INSTRUMENT_TYPES[instrument_type]}*\n\nВыберите инструмент:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return CUSTOM_INSTRUMENT
    except Exception as e:
        logger.error(f"Ошибка в pro_select_instrument_type: {e}")
        await handle_error(update, context, e)

@log_performance
async def pro_select_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор инструмента"""
    try:
        query = update.callback_query
        await query.answer()
        
        if query.data == "pro_custom":
            await query.edit_message_text("✏️ *Введите название инструмента:*\n\nПример: EURUSD, BTCUSD, XAUUSD", parse_mode='Markdown')
            return CUSTOM_INSTRUMENT
        else:
            instrument = query.data.replace("pro_preset_", "")
            context.user_data['instrument'] = instrument
            
            keyboard = [
                [InlineKeyboardButton("📈 BUY", callback_data="BUY"), InlineKeyboardButton("📉 SELL", callback_data="SELL")],
                [InlineKeyboardButton("🔙 Назад", callback_data=f"pro_type_{context.user_data['instrument_type']}")]
            ]
            
            await query.edit_message_text(f"🎯 *Инструмент:* {instrument}\n\nВыберите направление:", parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
            return DIRECTION
    except Exception as e:
        logger.error(f"Ошибка в pro_select_instrument: {e}")
        await handle_error(update, context, e)

@log_performance
async def pro_handle_custom_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка пользовательского инструмента"""
    try:
        instrument = update.message.text.upper().strip()
        is_valid, validated_instrument, message = InputValidator.validate_instrument(instrument)
        
        if not is_valid:
            await update.message.reply_text(f"{message}\n\n✏️ Введите название инструмента:", parse_mode='Markdown')
            return CUSTOM_INSTRUMENT
        
        context.user_data['instrument'] = validated_instrument
        
        keyboard = [
            [InlineKeyboardButton("📈 BUY", callback_data="BUY"), InlineKeyboardButton("📉 SELL", callback_data="SELL")],
            [InlineKeyboardButton("🔙 Назад", callback_data=f"pro_type_{context.user_data['instrument_type']}")]
        ]
        
        await update.message.reply_text(f"🎯 *Инструмент:* {validated_instrument}\n\nВыберите направление:", parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        return DIRECTION
        
    except Exception as e:
        logger.error(f"Ошибка в pro_handle_custom_instrument: {e}")
        await handle_error(update, context, e)

@log_performance
async def pro_select_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор направления сделки"""
    try:
        query = update.callback_query
        await query.answer()
        
        direction = query.data
        context.user_data['direction'] = direction
        
        keyboard = []
        for risk in RISK_LEVELS:
            keyboard.append([InlineKeyboardButton(risk, callback_data=f"pro_risk_{risk.replace('%', '')}")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="pro_custom" if 'custom' in context.user_data else f"pro_preset_{context.user_data['instrument']}")])
        
        await query.edit_message_text(f"🎯 *{context.user_data['instrument']}* | *{direction}*\n\nВыберите уровень риска:", parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        return RISK_PERCENT
    except Exception as e:
        logger.error(f"Ошибка в pro_select_direction: {e}")
        await handle_error(update, context, e)

@log_performance
async def pro_select_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор уровня риска"""
    try:
        query = update.callback_query
        await query.answer()
        
        risk_percent = float(query.data.replace("pro_risk_", "")) / 100
        context.user_data['risk_percent'] = risk_percent
        
        await query.edit_message_text(f"💰 *Уровень риска:* {risk_percent*100}%\n\n💵 Введите размер депозита:", parse_mode='Markdown')
        return DEPOSIT
    except Exception as e:
        logger.error(f"Ошибка в pro_select_risk: {e}")
        await handle_error(update, context, e)

@log_performance
async def pro_handle_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода депозита"""
    try:
        text = update.message.text
        is_valid, deposit, message = InputValidator.validate_number(text, 1, 1000000)
        
        if not is_valid:
            await update.message.reply_text(f"{message}\n\n💵 Введите размер депозита:", parse_mode='Markdown')
            return DEPOSIT
        
        context.user_data['deposit'] = deposit
        
        keyboard = []
        for leverage in LEVERAGES:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"pro_leverage_{leverage}")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data=f"pro_risk_{int(context.user_data['risk_percent']*100)}")])
        
        await update.message.reply_text(f"💰 *Депозит:* ${deposit:,.2f}\n\n⚖️ Выберите кредитное плечо:", parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        return LEVERAGE
        
    except Exception as e:
        logger.error(f"Ошибка в pro_handle_deposit: {e}")
        await handle_error(update, context, e)

@log_performance
async def pro_select_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор плеча"""
    try:
        query = update.callback_query
        await query.answer()
        
        leverage = query.data.replace("pro_leverage_", "")
        context.user_data['leverage'] = leverage
        
        await query.edit_message_text(f"⚖️ *Плечо:* {leverage}\n\n💎 Введите цену входа:", parse_mode='Markdown')
        return ENTRY
    except Exception as e:
        logger.error(f"Ошибка в pro_select_leverage: {e}")
        await handle_error(update, context, e)

@log_performance
async def pro_handle_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка цены входа"""
    try:
        text = update.message.text
        is_valid, entry_price, message = InputValidator.validate_price(text)
        
        if not is_valid:
            await update.message.reply_text(f"{message}\n\n💎 Введите цену входа:", parse_mode='Markdown')
            return ENTRY
        
        context.user_data['entry_price'] = entry_price
        
        direction = context.user_data.get('direction', 'BUY')
        direction_text = "ниже" if direction == "BUY" else "выше"
        
        await update.message.reply_text(f"💎 *Цена входа:* {entry_price}\n\n🛑 Введите цену стоп-лосса ({direction_text} цены входа):", parse_mode='Markdown')
        return STOP_LOSS
        
    except Exception as e:
        logger.error(f"Ошибка в pro_handle_entry: {e}")
        await handle_error(update, context, e)

@log_performance
async def pro_handle_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка стоп-лосса"""
    try:
        text = update.message.text
        is_valid, stop_loss, message = InputValidator.validate_price(text)
        
        if not is_valid:
            await update.message.reply_text(f"{message}\n\n🛑 Введите цену стоп-лосса:", parse_mode='Markdown')
            return STOP_LOSS
        
        context.user_data['stop_loss'] = stop_loss
        
        await update.message.reply_text(f"🛑 *Стоп-лосс:* {stop_loss}\n\n🎯 Введите цену тейк-профита:", parse_mode='Markdown')
        return TAKE_PROFIT_SINGLE
        
    except Exception as e:
        logger.error(f"Ошибка в pro_handle_stop_loss: {e}")
        await handle_error(update, context, e)

@log_performance
async def pro_handle_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка тейк-профита"""
    try:
        text = update.message.text
        is_valid, take_profit, message = InputValidator.validate_price(text)
        
        if not is_valid:
            await update.message.reply_text(f"{message}\n\n🎯 Введите цену тейк-профита:", parse_mode='Markdown')
            return TAKE_PROFIT_SINGLE
        
        context.user_data['take_profit'] = take_profit
        
        return await pro_calculate_and_show_results(update, context)
        
    except Exception as e:
        logger.error(f"Ошибка в pro_handle_take_profit: {e}")
        await handle_error(update, context, e)

@log_performance
async def pro_calculate_and_show_results(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Расчет и показ результатов"""
    try:
        user_data_context = context.user_data
        
        deposit = user_data_context['deposit']
        leverage = user_data_context['leverage']
        instrument_type = user_data_context['instrument_type']
        instrument = user_data_context['instrument']
        entry_price = user_data_context['entry_price']
        stop_loss = user_data_context['stop_loss']
        take_profit = user_data_context['take_profit']
        direction = user_data_context['direction']
        risk_percent = user_data_context['risk_percent']
        
        calculation = FastRiskCalculator.calculate_position_size_fast(
            deposit=deposit,
            leverage=leverage,
            instrument_type=instrument_type,
            currency_pair=instrument,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            direction=direction,
            risk_percent=risk_percent
        )
        
        is_profitable = calculation.get('is_profitable', True)
        status_emoji = "🟢" if is_profitable else "🔴"
        status_text = "ПРИБЫЛЬНАЯ" if is_profitable else "УБЫТОЧНАЯ"
        
        result_text = f"""
🎯 *РЕЗУЛЬТАТЫ РАСЧЕТА*
{status_emoji} *СТАТУС: {status_text}*

📊 *Параметры:*
• Инструмент: {instrument}
• Направление: {direction}
• Депозит: ${deposit:,.2f}
• Плечо: {leverage}
• Риск: {risk_percent*100}%

💎 *Цены:*
• Вход: {entry_price}
• Стоп-лосс: {stop_loss}
• Тейк-профит: {take_profit}

📈 *Результаты:*
• Размер позиции: {calculation['position_size']:.2f} лотов
• Сумма риска: ${calculation['risk_amount']:.2f}
• Потенциальная прибыль: ${calculation['potential_profit']:.2f}
• Потенциальный убыток: ${calculation['potential_loss']:.2f}
• Соотношение прибыль/риск: {calculation['reward_risk_ratio']:.2f}
• Требуемая маржа: ${calculation['required_margin']:.2f}

💡 *Рекомендации:*
{ReportGenerator.get_professional_recommendations(calculation, user_data_context)}
"""
        
        keyboard = [
            [InlineKeyboardButton("📊 Новый расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ]
        
        if hasattr(update, 'message'):
            await update.message.reply_text(result_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            await update.callback_query.edit_message_text(result_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        
        return ConversationHandler.END
        
    except Exception as e:
        logger.error(f"Ошибка в pro_calculate_and_show_results: {e}")
        error_msg = "❌ Произошла ошибка при расчете. Попробуйте еще раз."
        if hasattr(update, 'message'):
            await update.message.reply_text(error_msg)
        else:
            await update.callback_query.edit_message_text(error_msg)
        return ConversationHandler.END

@log_performance
async def handle_single_or_multi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора типа расчета"""
    try:
        query = update.callback_query
        await query.answer()
        
        choice = query.data
        
        if choice == "single_trade":
            keyboard = []
            for key, value in INSTRUMENT_TYPES.items():
                keyboard.append([InlineKeyboardButton(value, callback_data=f"pro_type_{key}")])
            keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="pro_calculation")])
            
            await query.edit_message_text("📊 *ОДНА СДЕЛКА*\n\n🎯 Выберите тип инструмента:", parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
            return INSTRUMENT_TYPE
            
    except Exception as e:
        logger.error(f"Ошибка в handle_single_or_multi: {e}")
        await handle_error(update, context, e)

@log_performance
async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Главное меню портфеля"""
    try:
        if update.message:
            user_id = update.message.from_user.id
        else:
            user_id = update.callback_query.from_user.id
            await update.callback_query.answer()
        
        PortfolioManager.initialize_user_portfolio(user_id)
        portfolio = user_data[user_id]['portfolio']
        
        portfolio_text = f"""
💼 *PRO ПОРТФЕЛЬ*

💰 *Баланс:* ${portfolio['current_balance']:,.2f}
📊 *Сделки:* {len(portfolio['trades'])}/10
🎯 *Win Rate:* {portfolio['performance']['win_rate']:.1f}%
"""
        
        keyboard = [
            [InlineKeyboardButton("📈 Обзор сделок", callback_data="portfolio_trades")],
            [InlineKeyboardButton("📊 Анализ эффективности", callback_data="portfolio_performance")],
            [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ]
        
        if update.message:
            await update.message.reply_text(portfolio_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            await update.callback_query.edit_message_text(portfolio_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        return PORTFOLIO_MENU
    except Exception as e:
        logger.error(f"Ошибка в portfolio_command: {e}")
        await handle_error(update, context, e)

@log_performance
async def portfolio_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать обзор сделок"""
    await update.callback_query.edit_message_text("📈 Раздел сделок в разработке", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]]))

@log_performance
async def portfolio_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать анализ эффективности"""
    await update.callback_query.edit_message_text("📊 Раздел аналитики в разработке", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]]))

@log_performance
async def portfolio_add_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало добавления сделки"""
    await update.callback_query.edit_message_text("➕ Раздел добавления сделок в разработке", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]]))
    return ConversationHandler.END

@log_performance
async def analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Раздел аналитики"""
    await update.callback_query.edit_message_text("🔮 Раздел аналитики в разработке", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))
    return ANALYTICS_MENU

@log_performance
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """PRO Инструкции"""
    await update.callback_query.edit_message_text("📚 PRO инструкции в разработке", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))

@log_performance
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена операции"""
    await update.message.reply_text("Операция отменена.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))
    return ConversationHandler.END

@log_performance
async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка неизвестных команд"""
    await update.message.reply_text("❌ Неизвестная команда. Используйте кнопки меню.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))

@log_performance
async def handle_error(update: Update, context: ContextTypes.DEFAULT_TYPE, error: Exception = None):
    """Обработка ошибок"""
    try:
        error_msg = "❌ Произошла ошибка. Попробуйте еще раз."
        
        if update.callback_query:
            await update.callback_query.edit_message_text(error_msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))
        elif update.message:
            await update.message.reply_text(error_msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))
    except Exception as e:
        logger.error(f"Ошибка в обработчике ошибок: {e}")

@log_performance
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка главного меню"""
    try:
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
        elif choice == "portfolio":
            return await portfolio_command(update, context)
        elif choice == "analytics":
            return await analytics_command(update, context)
        elif choice == "pro_info":
            await pro_info_command(update, context)
            return MAIN_MENU
        elif choice == "main_menu":
            return await start(update, context)
        elif choice == "portfolio_trades":
            await portfolio_trades(update, context)
            return PORTFOLIO_MENU
        elif choice == "portfolio_performance":
            await portfolio_performance(update, context)
            return PORTFOLIO_MENU
        elif choice == "portfolio_add_trade":
            return await portfolio_add_trade_start(update, context)
        elif choice == "single_trade":
            return await handle_single_or_multi(update, context)
        
        return MAIN_MENU
        
    except Exception as e:
        logger.error(f"Ошибка в handle_main_menu: {e}")
        await handle_error(update, context, e)
        return await start(update, context)

# ========== HTTP СЕРВЕР И WEBHOOKS ==========

async def health_check(request):
    """Health check endpoint для Render"""
    return web.Response(text="OK", status=200)

async def handle_webhook(request, application):
    """Обработчик вебхуков от Telegram"""
    try:
        data = await request.json()
        update = Update.de_json(data, application.bot)
        await application.process_update(update)
        return web.Response(status=200)
    except Exception as e:
        logger.error(f"Ошибка обработки вебхука: {e}")
        return web.Response(status=500)

async def set_webhook(application):
    """Установка вебхука"""
    if not WEBHOOK_URL:
        logger.warning("WEBHOOK_URL не установлен, используем polling")
        return False
    
    try:
        webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
        await application.bot.set_webhook(url=webhook_url, drop_pending_updates=True)
        logger.info(f"Webhook установлен: {webhook_url}")
        return True
    except Exception as e:
        logger.error(f"Ошибка установки webhook: {e}")
        return False

async def start_http_server(application):
    """Запуск HTTP сервера"""
    app = web.Application()
    app.router.add_get('/', health_check)
    app.router.add_get('/health', health_check)
    app.router.add_post(WEBHOOK_PATH, lambda request: handle_webhook(request, application))
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    
    logger.info(f"HTTP сервер запущен на порту {PORT}")
    return runner

async def start_webhook_mode(application):
    """Запуск бота в режиме webhook"""
    try:
        webhook_set = await set_webhook(application)
        if not webhook_set:
            return False
        
        runner = await start_http_server(application)
        logger.info("✅ Бот запущен в режиме Webhook!")
        
        while True:
            await asyncio.sleep(3600)
            
    except Exception as e:
        logger.error(f"Ошибка в режиме webhook: {e}")
        return False

def create_application():
    """Создание и настройка приложения"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ Токен бота не найден!")
        return None

    logger.info("🚀 Запуск ПРОФЕССИОНАЛЬНОГО калькулятора рисков v4.0...")
    
    application = Application.builder().token(token).build()

    # Обработчики для профессионального расчета
    pro_calc_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(start_pro_calculation, pattern='^pro_calculation$')],
        states={
            SINGLE_OR_MULTI: [CallbackQueryHandler(handle_single_or_multi)],
            INSTRUMENT_TYPE: [CallbackQueryHandler(pro_select_instrument_type)],
            CUSTOM_INSTRUMENT: [
                CallbackQueryHandler(pro_select_instrument),
                MessageHandler(filters.TEXT & ~filters.COMMAND, pro_handle_custom_instrument)
            ],
            DIRECTION: [CallbackQueryHandler(pro_select_direction)],
            RISK_PERCENT: [CallbackQueryHandler(pro_select_risk)],
            DEPOSIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, pro_handle_deposit)],
            LEVERAGE: [CallbackQueryHandler(pro_select_leverage)],
            ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, pro_handle_entry)],
            STOP_LOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, pro_handle_stop_loss)],
            TAKE_PROFIT_SINGLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, pro_handle_take_profit)],
        },
        fallbacks=[CommandHandler('cancel', cancel), CommandHandler('start', start), CallbackQueryHandler(start, pattern='^main_menu$')]
    )

    # Регистрируем обработчики
    application.add_handler(CommandHandler('start', start))
    application.add_handler(pro_calc_conv)
    
    # Упрощенный обработчик состояний
    conv_handler = ConversationHandler(
        entry_points=[],
        states={
            MAIN_MENU: [CallbackQueryHandler(handle_main_menu)],
            PORTFOLIO_MENU: [CallbackQueryHandler(handle_main_menu)],
            ANALYTICS_MENU: [CallbackQueryHandler(handle_main_menu)],
        },
        fallbacks=[CommandHandler('start', start), CommandHandler('cancel', cancel)],
        allow_reentry=True
    )
    application.add_handler(conv_handler)
    
    # Обработчики команд
    application.add_handler(CommandHandler('portfolio', portfolio_command))
    application.add_handler(CommandHandler('analytics', analytics_command))
    application.add_handler(CommandHandler('info', pro_info_command))
    
    # Обработчик неизвестных команд
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    application.add_handler(MessageHandler(filters.TEXT, unknown_command))
    
    return application

def main():
    """Основная функция запуска"""
    application = create_application()
    if not application:
        return
    
    if WEBHOOK_URL:
        logger.info("🚀 Запуск в режиме Webhook...")
        asyncio.run(start_webhook_mode(application))
    else:
        logger.info("🚀 Запуск в режиме Polling...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
