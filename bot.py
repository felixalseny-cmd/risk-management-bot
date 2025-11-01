import os
import logging
import asyncio
import re
import time
import functools
import json
import io
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    ConversationHandler,
    CallbackQueryHandler
)

# Настройка логирования
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
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        if execution_time > 1.0:
            logger.warning(f"Медленная операция: {func.__name__} заняла {execution_time:.2f}с")
        return result
    return wrapper

# Состояния диалога
(
    MAIN_MENU, INSTRUMENT_TYPE, CUSTOM_INSTRUMENT, DIRECTION, 
    RISK_PERCENT, DEPOSIT, LEVERAGE, CURRENCY, ENTRY, 
    STOP_LOSS, TAKE_PROFIT_COUNT, TAKE_PROFIT_1, TAKE_PROFIT_2, TAKE_PROFIT_3,
    VOLUME_DISTRIBUTION,
    PORTFOLIO_MENU, ADD_TRADE_INSTRUMENT, ADD_TRADE_DIRECTION,
    ADD_TRADE_ENTRY, ADD_TRADE_EXIT, ADD_TRADE_VOLUME, ADD_TRADE_PROFIT,
    DEPOSIT_AMOUNT, WITHDRAW_AMOUNT, SETTINGS_MENU, SAVE_STRATEGY_NAME,
    PRO_DEPOSIT, PRO_LEVERAGE, PRO_RISK, PRO_ENTRY, PRO_STOPLOSS,
    PRO_TAKEPROFIT, PRO_VOLUME, STRATEGY_NAME, QUICK_INSTRUMENT,
    QUICK_DIRECTION, QUICK_DEPOSIT, QUICK_RISK, QUICK_ENTRY, QUICK_STOPLOSS,
    QUICK_TAKEPROFIT_COUNT, QUICK_TAKEPROFIT_1, QUICK_TAKEPROFIT_2, QUICK_TAKEPROFIT_3,
    ANALYTICS_MENU, EXPORT_CALCULATION
) = range(45)

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

PIP_VALUES = {
    # Forex - основные пары
    'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9, 'USDCHF': 10,
    'USDCAD': 10, 'AUDUSD': 10, 'NZDUSD': 10, 'EURGBP': 10,
    'EURJPY': 9, 'GBPJPY': 9, 'EURCHF': 10, 'AUDJPY': 9,
    # Криптовалюты
    'BTCUSD': 1, 'ETHUSD': 1, 'XRPUSD': 10, 'ADAUSD': 10,
    'DOTUSD': 1, 'LTCUSD': 1, 'BCHUSD': 1, 'LINKUSD': 1,
    # Индексы
    'US30': 1, 'NAS100': 1, 'SPX500': 1, 'DAX40': 1,
    'FTSE100': 1, 'NIKKEI225': 1, 'ASX200': 1,
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
CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']

# Файл для сохранения данных
DATA_FILE = "user_data.json"

# Менеджер данных с сохранением в файл
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
            logger.info("Данные успешно сохранены")
        except Exception as e:
            logger.error(f"Ошибка сохранения данных: {e}")

    @staticmethod
    def auto_save():
        """Автосохранение каждые 5 минут"""
        DataManager.save_data()
        asyncio.get_event_loop().call_later(300, DataManager.auto_save)

# Глобальное хранилище данных пользователей
user_data: Dict[int, Dict[str, Any]] = DataManager.load_data()

# Быстрый кэш
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
                    'max_drawdown': 0,
                    'sharpe_ratio': 0
                },
                'allocation': {},
                'history': [],
                'settings': {
                    'default_risk': 0.02,
                    'currency': 'USD',
                    'leverage': '1:100'
                },
                'saved_strategies': []
            }
        DataManager.save_data()

    @staticmethod
    def add_trade(user_id: int, trade_data: Dict):
        PortfolioManager.initialize_user_portfolio(user_id)
        trade_id = len(user_data[user_id]['portfolio']['trades']) + 1
        trade_data['id'] = trade_id
        trade_data['timestamp'] = datetime.now().isoformat()
        user_data[user_id]['portfolio']['trades'].append(trade_data)
        PortfolioManager.update_performance_metrics(user_id)
        instrument = trade_data.get('instrument', 'Unknown')
        if instrument not in user_data[user_id]['portfolio']['allocation']:
            user_data[user_id]['portfolio']['allocation'][instrument] = 0
        user_data[user_id]['portfolio']['allocation'][instrument] += 1
        user_data[user_id]['portfolio']['history'].append({
            'type': 'trade',
            'action': 'open' if trade_data.get('status') == 'open' else 'close',
            'instrument': instrument,
            'profit': trade_data.get('profit', 0),
            'timestamp': trade_data['timestamp']
        })
        DataManager.save_data()

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

            running_balance = portfolio['initial_balance']
            peak = running_balance
            max_drawdown = 0
            for trade in sorted(closed_trades, key=lambda x: x['timestamp']):
                running_balance += trade.get('profit', 0)
                if running_balance > peak:
                    peak = running_balance
                drawdown = (peak - running_balance) / peak * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            portfolio['performance']['max_drawdown'] = max_drawdown

        DataManager.save_data()

    @staticmethod
    def add_balance_operation(user_id: int, operation_type: str, amount: float, description: str = ""):
        PortfolioManager.initialize_user_portfolio(user_id)
        user_data[user_id]['portfolio']['history'].append({
            'type': 'balance',
            'action': operation_type,
            'amount': amount,
            'description': description,
            'timestamp': datetime.now().isoformat()
        })
        if operation_type == 'deposit':
            user_data[user_id]['portfolio']['current_balance'] += amount
            if user_data[user_id]['portfolio']['initial_balance'] == 0:
                user_data[user_id]['portfolio']['initial_balance'] = amount
        elif operation_type == 'withdrawal':
            user_data[user_id]['portfolio']['current_balance'] -= amount
        DataManager.save_data()

    @staticmethod
    def get_performance_recommendations(user_id: int) -> List[str]:
        portfolio = user_data[user_id]['portfolio']
        perf = portfolio['performance']
        recommendations = []

        if perf['win_rate'] < 40:
            recommendations.append("🎯 Увеличьте соотношение риск/прибыль до 1:3 для компенсации низкого Win Rate")
        elif perf['win_rate'] > 60:
            recommendations.append("✅ Отличный Win Rate! Рассмотрите увеличение размера позиций")
        else:
            recommendations.append("📊 Win Rate в норме. Сфокусируйтесь на управлении рисками")

        if perf['profit_factor'] < 1:
            recommendations.append("⚠️ Profit Factor ниже 1.0 - пересмотрите стратегию")
        elif perf['profit_factor'] > 2:
            recommendations.append("💰 Отличный Profit Factor! Стратегия очень эффективна")

        if perf['max_drawdown'] > 20:
            recommendations.append(f"📉 Максимальная просадка {perf['max_drawdown']:.1f}% слишком высока. Уменьшите риск на сделку")
        elif perf['max_drawdown'] < 5:
            recommendations.append("📈 Низкая просадка - можно рассмотреть увеличение агрессивности")

        if perf['average_profit'] > 0 and perf['average_loss'] > 0:
            reward_ratio = perf['average_profit'] / perf['average_loss']
            if reward_ratio < 1:
                recommendations.append("🔻 Соотношение прибыль/убыток меньше 1. Улучшайте тейк-профиты")
            elif reward_ratio > 2:
                recommendations.append("🔺 Отличное соотношение прибыль/убыток! Продолжайте в том же духе")

        allocation = portfolio.get('allocation', {})
        if len(allocation) < 3:
            recommendations.append("🌐 Диверсифицируйте портфель - торгуйте больше инструментов")
        elif len(allocation) > 10:
            recommendations.append("🎯 Слишком много инструментов - сфокусируйтесь на лучших")

        return recommendations

    @staticmethod
    def save_strategy(user_id: int, strategy_data: Dict):
        PortfolioManager.initialize_user_portfolio(user_id)
        strategy_id = len(user_data[user_id]['portfolio']['saved_strategies']) + 1
        strategy_data['id'] = strategy_id
        strategy_data['created_at'] = datetime.now().isoformat()
        user_data[user_id]['portfolio']['saved_strategies'].append(strategy_data)
        DataManager.save_data()
        return strategy_id

    @staticmethod
    def get_saved_strategies(user_id: int) -> List[Dict]:
        PortfolioManager.initialize_user_portfolio(user_id)
        return user_data[user_id]['portfolio']['saved_strategies']

# Ультра-быстрый калькулятор рисков
class FastRiskCalculator:
    """Оптимизированный калькулятор рисков с упрощенными расчетами"""

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
        direction: str,
        risk_percent: float = 0.02
    ) -> Dict[str, float]:
        """Ультра-быстрый расчет размера позиции"""
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
            logger.error(f"Ошибка в быстром расчете размера позиции: {e}")
            return {
                'position_size': 0.01,
                'risk_amount': 0,
                'stop_pips': 0,
                'required_margin': 0,
                'risk_percent': 0,
                'free_margin': deposit
            }

# Валидатор ввода данных
class InputValidator:
    """Класс для валидации вводимых данных"""

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

    @staticmethod
    def validate_percent(percent: str) -> Tuple[bool, float, str]:
        """Валидация процентного значения"""
        return InputValidator.validate_number(percent, 0.01, 100)

# Генератор TXT отчётов
class TXTReportGenerator:
    @staticmethod
    def generate_calculation_report(calc_data: Dict) -> str:
        """Генерация текстового отчёта по расчёту"""
        try:
            report = f"""
ОТЧЁТ ПО РАСЧЁТУ ПОЗИЦИИ
Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}

ПАРАМЕТРЫ СДЕЛКИ:
• Инструмент: {calc_data.get('instrument', 'N/A')}
• Направление: {calc_data.get('direction', 'N/A')}
• Депозит: ${calc_data.get('deposit', 0):,.2f}
• Плечо: {calc_data.get('leverage', 'N/A')}
• Уровень риска: {calc_data.get('risk_percent_input', 0)}%
• Цена входа: {calc_data.get('entry_price', 'N/A')}
• Стоп-лосс: {calc_data.get('stop_loss', 'N/A')}
• Дистанция SL: {calc_data.get('stop_pips', 0):.2f} пунктов

РЕЗУЛЬТАТЫ:
• Размер позиции: {calc_data.get('position_size', 0):.2f} лотов
• Сумма риска: ${calc_data.get('risk_amount', 0):.2f}
• Требуемая маржа: ${calc_data.get('required_margin', 0):.2f}
• Свободная маржа: ${calc_data.get('free_margin', 0):.2f}

ТЕЙК-ПРОФИТЫ:
"""
            take_profits = calc_data.get('take_profits', [])
            if take_profits:
                for i, tp in enumerate(take_profits, 1):
                    report += f"• TP{i}: {tp['price']} (прибыль: ${tp['profit']:.2f})\n"
            else:
                report += "• Не заданы\n"

            report += "\n— Сгенерировано PRO Risk Calculator v3.1 —"
            return report
        except Exception as e:
            logger.error(f"Ошибка генерации TXT отчёта: {e}")
            return "Ошибка при генерации отчёта"

# Обработчики портфеля
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
💼 *PRO ПОРТФЕЛЬ v3.1*
💰 *Баланс:* ${portfolio['current_balance']:,.2f}
📊 *Сделки:* {len(portfolio['trades'])}
🎯 *Win Rate:* {portfolio['performance']['win_rate']:.1f}%
*Выберите опцию:*
"""
        keyboard = [
            [InlineKeyboardButton("📈 Обзор сделок", callback_data="portfolio_trades")],
            [InlineKeyboardButton("💰 Баланс и распределение", callback_data="portfolio_balance")],
            [InlineKeyboardButton("📊 Анализ эффективности", callback_data="portfolio_performance")],
            [InlineKeyboardButton("📄 Сгенерировать отчет", callback_data="portfolio_report")],
            [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ]

        if update.message:
            await update.message.reply_text(
                portfolio_text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await update.callback_query.edit_message_text(
                portfolio_text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        return PORTFOLIO_MENU
    except Exception as e:
        logger.error(f"Ошибка в portfolio_command: {e}")

@log_performance
async def portfolio_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать обзор сделок"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        portfolio = user_data[user_id].get('portfolio', {})
        trades = portfolio.get('trades', [])
        if not trades:
            await query.edit_message_text(
                "📭 *У вас еще нет сделок*\n"
                "Используйте кнопку '➕ Добавить сделку' чтобы начать.",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]
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
                [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
                [InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в portfolio_trades: {e}")

@log_performance
async def portfolio_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать баланс и распределение"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        portfolio = user_data[user_id].get('portfolio', {})
        allocation = portfolio.get('allocation', {})
        performance = portfolio.get('performance', {})
        balance_text = "💰 *Баланс и распределение*\n"
        initial_balance = portfolio.get('initial_balance', 0)
        current_balance = portfolio.get('current_balance', 0)
        total_profit = performance.get('total_profit', 0)
        total_loss = performance.get('total_loss', 0)
        net_profit = total_profit - total_loss

        balance_text += f"💳 Начальный депозит: ${initial_balance:,.2f}\n"
        balance_text += f"💵 Текущий баланс: ${current_balance:,.2f}\n"
        balance_text += f"📈 Чистая прибыль: ${net_profit:.2f}\n"

        if allocation:
            balance_text += "🌐 *Распределение по инструментам:*\n"
            for instrument, count in list(allocation.items())[:5]:
                percentage = (count / len(portfolio['trades'])) * 100 if portfolio['trades'] else 0
                balance_text += f"• {instrument}: {count} сделок ({percentage:.1f}%)\n"
        else:
            balance_text += "🌐 *Распределение:* Нет данных\n"

        await query.edit_message_text(
            balance_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💸 Внести депозит", callback_data="portfolio_deposit")],
                [InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в portfolio_balance: {e}")

@log_performance
async def portfolio_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать анализ эффективности"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        portfolio = user_data[user_id].get('portfolio', {})
        performance = portfolio.get('performance', {})
        perf_text = "📊 *PRO АНАЛИЗ ЭФФЕКТИВНОСТИ*\n"
        total_trades = performance.get('total_trades', 0)
        win_rate = performance.get('win_rate', 0)
        avg_profit = performance.get('average_profit', 0)
        avg_loss = performance.get('average_loss', 0)
        profit_factor = performance.get('profit_factor', 0)
        max_drawdown = performance.get('max_drawdown', 0)

        perf_text += f"📈 Всего сделок: {total_trades}\n"
        perf_text += f"🎯 Процент прибыльных: {win_rate:.1f}%\n"
        perf_text += f"💰 Средняя прибыль: ${avg_profit:.2f}\n"
        perf_text += f"📉 Средний убыток: ${avg_loss:.2f}\n"
        perf_text += f"⚖️ Profit Factor: {profit_factor:.2f}\n"
        perf_text += f"📊 Макс. просадка: {max_drawdown:.1f}%\n"

        recommendations = PortfolioManager.get_performance_recommendations(user_id)
        if recommendations:
            perf_text += "💡 *PRO РЕКОМЕНДАЦИИ:*\n"
            for i, rec in enumerate(recommendations[:3], 1):
                perf_text += f"{i}. {rec}\n"

        await query.edit_message_text(
            perf_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📈 Обзор сделок", callback_data="portfolio_trades")],
                [InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в portfolio_performance: {e}")

@log_performance
async def portfolio_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Генерация отчёта по портфелю в TXT"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id

        portfolio = user_data[user_id]['portfolio']
        performance = portfolio['performance']
        report_lines = []
        report_lines.append("ОТЧЕТ ПО ПОРТФЕЛЮ v3.1")
        report_lines.append(f"Дата генерации: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        report_lines.append("")
        report_lines.append("БАЛАНС И СРЕДСТВА:")
        report_lines.append(f"• Начальный депозит: ${portfolio['initial_balance']:,.2f}")
        report_lines.append(f"• Текущий баланс: ${portfolio['current_balance']:,.2f}")
        report_lines.append(f"• Общая прибыль/убыток: ${portfolio['current_balance'] - portfolio['initial_balance']:,.2f}")
        report_lines.append("")
        report_lines.append("СТАТИСТИКА ТОРГОВЛИ:")
        report_lines.append(f"• Всего сделок: {performance['total_trades']}")
        report_lines.append(f"• Прибыльные сделки: {performance['winning_trades']}")
        report_lines.append(f"• Убыточные сделки: {performance['losing_trades']}")
        report_lines.append(f"• Win Rate: {performance['win_rate']:.1f}%")
        report_lines.append(f"• Profit Factor: {performance['profit_factor']:.2f}")
        report_lines.append(f"• Макс. просадка: {performance['max_drawdown']:.1f}%")
        report_lines.append("")
        report_lines.append("ДОХОДНОСТЬ:")
        report_lines.append(f"• Общая прибыль: ${performance['total_profit']:,.2f}")
        report_lines.append(f"• Общий убыток: ${performance['total_loss']:,.2f}")
        report_lines.append(f"• Средняя прибыль: ${performance['average_profit']:.2f}")
        report_lines.append(f"• Средний убыток: ${performance['average_loss']:.2f}")
        report_lines.append("")
        report_lines.append("РАСПРЕДЕЛЕНИЕ ПО ИНСТРУМЕНТАМ:")
        allocation = portfolio.get('allocation', {})
        for instrument, count in allocation.items():
            report_lines.append(f"• {instrument}: {count} сделок")
        report_lines.append("")
        recommendations = PortfolioManager.get_performance_recommendations(user_id)
        if recommendations:
            report_lines.append("PRO РЕКОМЕНДАЦИИ:")
            for i, rec in enumerate(recommendations[:3], 1):
                report_lines.append(f"{i}. {rec}")

        report_text = "\n".join(report_lines)
        bio = io.BytesIO()
        bio.name = f"portfolio_report_{user_id}.txt"
        bio.write(report_text.encode('utf-8'))
        bio.seek(0)

        await query.message.reply_document(
            document=InputFile(bio, filename=f"portfolio_report_{user_id}.txt"),
            caption="📄 Отчёт по портфелю успешно сгенерирован и отправлен!",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💼 Портфель", callback_data="portfolio")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в portfolio_report: {e}")
        await query.edit_message_text(
            "❌ *Ошибка генерации отчёта*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]
            ])
        )

@log_performance
async def portfolio_deposit_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Меню внесения депозита"""
    try:
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(
            "💸 *Внесение депозита*\n"
            "💰 *Введите сумму депозита:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="portfolio_balance")]
            ])
        )
        return DEPOSIT_AMOUNT
    except Exception as e:
        logger.error(f"Ошибка в portfolio_deposit_menu: {e}")

@log_performance
async def handle_deposit_amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода суммы депозита"""
    try:
        user_id = update.message.from_user.id
        text = update.message.text
        is_valid, amount, message = InputValidator.validate_number(text, 1, 1000000)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n💰 Введите сумму депозита:",
                parse_mode='Markdown'
            )
            return DEPOSIT_AMOUNT

        PortfolioManager.add_balance_operation(user_id, 'deposit', amount, "Депозит")
        await update.message.reply_text(
            f"✅ *Депозит на ${amount:,.2f} успешно внесен!*\n"
            f"💳 Текущий баланс: ${user_data[user_id]['portfolio']['current_balance']:,.2f}",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💰 Баланс", callback_data="portfolio_balance")],
                [InlineKeyboardButton("💼 Портфель", callback_data="portfolio")]
            ])
        )
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"Ошибка в handle_deposit_amount: {e}")
        await update.message.reply_text(
            "❌ *Произошла ошибка!*\n"
            "💰 Введите сумму депозита:",
            parse_mode='Markdown'
        )
        return DEPOSIT_AMOUNT

@log_performance
async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Настройки с полным функционалом"""
    try:
        if update.message:
            user_id = update.message.from_user.id
        else:
            user_id = update.callback_query.from_user.id
            await update.callback_query.answer()

        PortfolioManager.initialize_user_portfolio(user_id)
        settings = user_data[user_id]['portfolio']['settings']
        settings_text = f"""
⚙️ *Настройки PRO Трейдера*
*Текущие настройки:*
• 💰 Уровень риска: {settings['default_risk']*100}%
• 💵 Валюта депозита: {settings['currency']}
• ⚖️ Плечо по умолчанию: {settings['leverage']}
🔧 *Изменить настройки:*
"""
        keyboard = [
            [InlineKeyboardButton(f"💰 Уровень риска: {settings['default_risk']*100}%", callback_data="change_risk")],
            [InlineKeyboardButton(f"💵 Валюта: {settings['currency']}", callback_data="change_currency")],
            [InlineKeyboardButton(f"⚖️ Плечо: {settings['leverage']}", callback_data="change_leverage")],
            [InlineKeyboardButton("💾 Сохраненные стратегии", callback_data="saved_strategies")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ]

        if update.message:
            await update.message.reply_text(
                settings_text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await update.callback_query.edit_message_text(
                settings_text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        return SETTINGS_MENU
    except Exception as e:
        logger.error(f"Ошибка в settings_command: {e}")

@log_performance
async def change_risk_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(
            "🎯 *Выберите уровень риска по умолчанию:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🟢 1% (Консервативный)", callback_data="set_risk_0.01")],
                [InlineKeyboardButton("🟡 2% (Умеренный)", callback_data="set_risk_0.02")],
                [InlineKeyboardButton("🟠 3% (Сбалансированный)", callback_data="set_risk_0.03")],
                [InlineKeyboardButton("🔴 5% (Агрессивный)", callback_data="set_risk_0.05")],
                [InlineKeyboardButton("🔙 Назад", callback_data="settings")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в change_risk_setting: {e}")

@log_performance
async def change_currency_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        await query.answer()
        keyboard = []
        for currency in CURRENCIES:
            keyboard.append([InlineKeyboardButton(currency, callback_data=f"set_currency_{currency}")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="settings")])
        await query.edit_message_text(
            "💵 *Выберите валюту депозита:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    except Exception as e:
        logger.error(f"Ошибка в change_currency_setting: {e}")

@log_performance
async def change_leverage_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        await query.answer()
        keyboard = []
        for leverage in LEVERAGES:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"set_leverage_{leverage}")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="settings")])
        await query.edit_message_text(
            "⚖️ *Выберите плечо по умолчанию:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    except Exception as e:
        logger.error(f"Ошибка в change_leverage_setting: {e}")

@log_performance
async def save_risk_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        risk_level = float(query.data.replace("set_risk_", ""))
        user_data[user_id]['portfolio']['settings']['default_risk'] = risk_level
        DataManager.save_data()
        await query.edit_message_text(
            f"✅ *Уровень риска установлен: {risk_level*100}%*\n"
            "Настройки сохранены для будущих расчетов.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в save_risk_setting: {e}")

@log_performance
async def save_currency_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        currency = query.data.replace("set_currency_", "")
        user_data[user_id]['portfolio']['settings']['currency'] = currency
        DataManager.save_data()
        await query.edit_message_text(
            f"✅ *Валюта установлена: {currency}*\n"
            "Настройки сохранены для будущих расчетов.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в save_currency_setting: {e}")

@log_performance
async def save_leverage_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        leverage = query.data.replace("set_leverage_", "")
        user_data[user_id]['portfolio']['settings']['leverage'] = leverage
        DataManager.save_data()
        await query.edit_message_text(
            f"✅ *Плечо установлено: {leverage}*\n"
            "Настройки сохранены для будущих расчетов.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в save_leverage_setting: {e}")

# === НОВЫЕ ФУНКЦИИ: ТЕЙК-ПРОФИТ И ЭКСПОРТ ===

def calculate_take_profit_profit(position_size: float, entry: float, tp_price: float, direction: str, instrument_type: str, instrument: str) -> float:
    """Рассчитывает прибыль по TP"""
    if direction == "BUY":
        diff = tp_price - entry
    else:
        diff = entry - tp_price

    if instrument_type == 'forex':
        pips = diff * 10000
    elif instrument_type == 'crypto':
        pips = diff * 100
    elif instrument_type in ['indices', 'commodities', 'metals']:
        pips = diff * 10
    else:
        pips = diff * 10000

    pip_value = FastRiskCalculator.calculate_pip_value_fast(instrument_type, instrument, position_size)
    return pips * pip_value

# === ПРОФЕССИОНАЛЬНЫЙ РАСЧЁТ С TP ===

@log_performance
async def start_pro_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        query = update.callback_query
        await query.answer()
        keyboard = []
        for key, value in INSTRUMENT_TYPES.items():
            keyboard.append([InlineKeyboardButton(value, callback_data=f"pro_type_{key}")])
        keyboard.append([InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")])
        await query.edit_message_text(
            "📊 *ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ*\n"
            "🎯 Выберите тип инструмента:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return INSTRUMENT_TYPE
    except Exception as e:
        logger.error(f"Ошибка в start_pro_calculation: {e}")

@log_performance
async def pro_select_instrument_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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
            f"📊 *{INSTRUMENT_TYPES[instrument_type]}*\n"
            "Выберите инструмент из списка или введите свой:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return CUSTOM_INSTRUMENT
    except Exception as e:
        logger.error(f"Ошибка в pro_select_instrument_type: {e}")

@log_performance
async def pro_select_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        query = update.callback_query
        await query.answer()
        if query.data == "pro_custom":
            await query.edit_message_text(
                "✏️ *Введите название инструмента:*\n"
                "Пример: EURUSD, BTCUSD, XAUUSD",
                parse_mode='Markdown'
            )
            return CUSTOM_INSTRUMENT
        else:
            instrument = query.data.replace("pro_preset_", "")
            context.user_data['instrument'] = instrument
            keyboard = [
                [InlineKeyboardButton("📈 BUY", callback_data="BUY"),
                 InlineKeyboardButton("📉 SELL", callback_data="SELL")],
                [InlineKeyboardButton("🔙 Назад", callback_data=f"pro_type_{context.user_data['instrument_type']}")]
            ]
            await query.edit_message_text(
                f"🎯 *Инструмент:* {instrument}\n"
                "Выберите направление сделки:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return DIRECTION
    except Exception as e:
        logger.error(f"Ошибка в pro_select_instrument: {e}")

@log_performance
async def pro_handle_custom_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        user_id = update.message.from_user.id
        instrument = update.message.text.upper().strip()
        is_valid, validated_instrument, message = InputValidator.validate_instrument(instrument)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n✏️ Введите название инструмента:",
                parse_mode='Markdown'
            )
            return CUSTOM_INSTRUMENT

        context.user_data['instrument'] = validated_instrument
        keyboard = [
            [InlineKeyboardButton("📈 BUY", callback_data="BUY"),
             InlineKeyboardButton("📉 SELL", callback_data="SELL")],
            [InlineKeyboardButton("🔙 Назад", callback_data=f"pro_type_{context.user_data['instrument_type']}")]
        ]
        await update.message.reply_text(
            f"🎯 *Инструмент:* {validated_instrument}\n"
            "Выберите направление сделки:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return DIRECTION
    except Exception as e:
        logger.error(f"Ошибка в pro_handle_custom_instrument: {e}")

@log_performance
async def pro_select_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        query = update.callback_query
        await query.answer()
        direction = query.data
        context.user_data['direction'] = direction
        user_id = query.from_user.id
        PortfolioManager.initialize_user_portfolio(user_id)
        default_risk = user_data[user_id]['portfolio']['settings']['default_risk']
        keyboard = []
        for risk in RISK_LEVELS:
            keyboard.append([InlineKeyboardButton(risk, callback_data=f"pro_risk_{risk.replace('%', '')}")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="pro_custom" if 'custom' in context.user_data else f"pro_preset_{context.user_data['instrument']}")])
        await query.edit_message_text(
            f"🎯 *{context.user_data['instrument']}* | *{direction}*\n"
            "Выберите уровень риска (% от депозита):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return RISK_PERCENT
    except Exception as e:
        logger.error(f"Ошибка в pro_select_direction: {e}")

@log_performance
async def pro_select_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        query = update.callback_query
        await query.answer()
        risk_percent = float(query.data.replace("pro_risk_", "")) / 100
        context.user_data['risk_percent'] = risk_percent
        await query.edit_message_text(
            f"💰 *Уровень риска:* {risk_percent*100}%\n"
            "💵 Введите размер депозита:",
            parse_mode='Markdown'
        )
        return DEPOSIT
    except Exception as e:
        logger.error(f"Ошибка в pro_select_risk: {e}")

@log_performance
async def pro_handle_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        user_id = update.message.from_user.id
        text = update.message.text
        is_valid, deposit, message = InputValidator.validate_number(text, 1, 1000000)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n💵 Введите размер депозита:",
                parse_mode='Markdown'
            )
            return DEPOSIT

        context.user_data['deposit'] = deposit
        PortfolioManager.initialize_user_portfolio(user_id)
        default_leverage = user_data[user_id]['portfolio']['settings']['leverage']
        keyboard = []
        for leverage in LEVERAGES:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"pro_leverage_{leverage}")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data=f"pro_risk_{int(context.user_data['risk_percent']*100)}")])
        await update.message.reply_text(
            f"💰 *Депозит:* ${deposit:,.2f}\n"
            "⚖️ Выберите кредитное плечо:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return LEVERAGE
    except Exception as e:
        logger.error(f"Ошибка в pro_handle_deposit: {e}")

@log_performance
async def pro_select_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        query = update.callback_query
        await query.answer()
        leverage = query.data.replace("pro_leverage_", "")
        context.user_data['leverage'] = leverage
        await query.edit_message_text(
            f"⚖️ *Плечо:* {leverage}\n"
            "💎 Введите цену входа:",
            parse_mode='Markdown'
        )
        return ENTRY
    except Exception as e:
        logger.error(f"Ошибка в pro_select_leverage: {e}")

@log_performance
async def pro_handle_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, entry_price, message = InputValidator.validate_price(text)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n💎 Введите цену входа:",
                parse_mode='Markdown'
            )
            return ENTRY

        context.user_data['entry_price'] = entry_price
        direction = context.user_data.get('direction', 'BUY')
        direction_text = "выше" if direction == "BUY" else "ниже"
        await update.message.reply_text(
            f"💎 *Цена входа:* {entry_price}\n"
            f"🛑 Введите цену стоп-лосса ({direction_text} цены входа):",
            parse_mode='Markdown'
        )
        return STOP_LOSS
    except Exception as e:
        logger.error(f"Ошибка в pro_handle_entry: {e}")

@log_performance
async def pro_handle_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, stop_loss, message = InputValidator.validate_price(text)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n🛑 Введите цену стоп-лосса:",
                parse_mode='Markdown'
            )
            return STOP_LOSS

        context.user_data['stop_loss'] = stop_loss
        await update.message.reply_text(
            "🎯 *Сколько уровней тейк-профита вы хотите задать?*\n"
            "Выберите от 1 до 3:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("1 TP", callback_data="tp_count_1")],
                [InlineKeyboardButton("2 TP", callback_data="tp_count_2")],
                [InlineKeyboardButton("3 TP", callback_data="tp_count_3")],
                [InlineKeyboardButton("Пропустить", callback_data="tp_skip")]
            ])
        )
        return TAKE_PROFIT_COUNT
    except Exception as e:
        logger.error(f"Ошибка в pro_handle_stop_loss: {e}")

@log_performance
async def pro_handle_tp_count(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        query = update.callback_query
        await query.answer()
        choice = query.data
        if choice == "tp_skip":
            context.user_data['take_profits'] = []
            return await pro_calculate_and_show_results(update, context)

        tp_count = int(choice.replace("tp_count_", ""))
        context.user_data['tp_count'] = tp_count
        await query.edit_message_text(
            f"🎯 *Введите цену TP1:*",
            parse_mode='Markdown'
        )
        return TAKE_PROFIT_1
    except Exception as e:
        logger.error(f"Ошибка в pro_handle_tp_count: {e}")

@log_performance
async def pro_handle_tp1(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, tp1, message = InputValidator.validate_price(text)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n🎯 Введите цену TP1:",
                parse_mode='Markdown'
            )
            return TAKE_PROFIT_1

        context.user_data['tp1'] = tp1
        tp_count = context.user_data['tp_count']
        if tp_count == 1:
            return await pro_collect_tps_and_calculate(update, context)
        else:
            await update.message.reply_text(
                f"🎯 *Введите цену TP2:*",
                parse_mode='Markdown'
            )
            return TAKE_PROFIT_2
    except Exception as e:
        logger.error(f"Ошибка в pro_handle_tp1: {e}")

@log_performance
async def pro_handle_tp2(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, tp2, message = InputValidator.validate_price(text)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n🎯 Введите цену TP2:",
                parse_mode='Markdown'
            )
            return TAKE_PROFIT_2

        context.user_data['tp2'] = tp2
        tp_count = context.user_data['tp_count']
        if tp_count == 2:
            return await pro_collect_tps_and_calculate(update, context)
        else:
            await update.message.reply_text(
                f"🎯 *Введите цену TP3:*",
                parse_mode='Markdown'
            )
            return TAKE_PROFIT_3
    except Exception as e:
        logger.error(f"Ошибка в pro_handle_tp2: {e}")

@log_performance
async def pro_handle_tp3(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, tp3, message = InputValidator.validate_price(text)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n🎯 Введите цену TP3:",
                parse_mode='Markdown'
            )
            return TAKE_PROFIT_3

        context.user_data['tp3'] = tp3
        return await pro_collect_tps_and_calculate(update, context)
    except Exception as e:
        logger.error(f"Ошибка в pro_handle_tp3: {e}")

async def pro_collect_tps_and_calculate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        tps = []
        if 'tp1' in context.user_data:
            tps.append(context.user_data['tp1'])
        if 'tp2' in context.user_data:
            tps.append(context.user_data['tp2'])
        if 'tp3' in context.user_data:
            tps.append(context.user_data['tp3'])
        context.user_data['take_profits_raw'] = tps
        return await pro_calculate_and_show_results(update, context)
    except Exception as e:
        logger.error(f"Ошибка в pro_collect_tps_and_calculate: {e}")
        await update.message.reply_text("❌ Ошибка при сборе TP. Попробуйте снова.")
        return ConversationHandler.END

@log_performance
async def pro_calculate_and_show_results(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        user_data_context = context.user_data
        deposit = user_data_context['deposit']
        leverage = user_data_context['leverage']
        instrument_type = user_data_context['instrument_type']
        instrument = user_data_context['instrument']
        entry_price = user_data_context['entry_price']
        stop_loss = user_data_context['stop_loss']
        direction = user_data_context['direction']
        risk_percent = user_data_context['risk_percent']

        calculation = FastRiskCalculator.calculate_position_size_fast(
            deposit=deposit,
            leverage=leverage,
            instrument_type=instrument_type,
            currency_pair=instrument,
            entry_price=entry_price,
            stop_loss=stop_loss,
            direction=direction,
            risk_percent=risk_percent
        )

        take_profits = []
        tp_raw_list = user_data_context.get('take_profits_raw', [])
        position_size = calculation['position_size']
        for tp_price in tp_raw_list:
            profit = calculate_take_profit_profit(position_size, entry_price, tp_price, direction, instrument_type, instrument)
            take_profits.append({'price': tp_price, 'profit': profit})

        # Сохраняем полные данные для экспорта
        full_calc_data = {
            'instrument': instrument,
            'direction': direction,
            'deposit': deposit,
            'leverage': leverage,
            'risk_percent_input': risk_percent * 100,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'stop_pips': calculation['stop_pips'],
            'position_size': position_size,
            'risk_amount': calculation['risk_amount'],
            'required_margin': calculation['required_margin'],
            'free_margin': calculation['free_margin'],
            'take_profits': take_profits
        }

        context.user_data['last_calculation'] = full_calc_data

        result_text = f"""
🎯 *РЕЗУЛЬТАТЫ ПРОФЕССИОНАЛЬНОГО РАСЧЕТА*
📊 *Параметры сделки:*
• 💰 Инструмент: {instrument}
• 📈 Направление: {direction}
• 💵 Депозит: ${deposit:,.2f}
• ⚖️ Плечо: {leverage}
• 🎯 Риск: {risk_percent*100}%
💎 *Цены:*
• Вход: {entry_price}
• Стоп-лосс: {stop_loss}
• Дистанция: {calculation['stop_pips']:.2f} пунктов
📈 *Результаты расчета:*
• 📦 Размер позиции: {position_size:.2f} лотов
• 💸 Сумма риска: ${calculation['risk_amount']:.2f}
• 🏦 Требуемая маржа: ${calculation['required_margin']:.2f}
• 💰 Свободная маржа: ${calculation['free_margin']:.2f}
• 📊 Риск в %: {calculation['risk_percent']:.2f}%
"""
        if take_profits:
            result_text += "🎯 *Тейк-профиты:*\n"
            for i, tp in enumerate(take_profits, 1):
                result_text += f"• TP{i}: {tp['price']} → прибыль: ${tp['profit']:.2f}\n"
        else:
            result_text += "🎯 *Тейк-профиты:* не заданы\n"

        result_text += """
💡 *Рекомендации:*
• Всегда используйте стоп-лосс
• Следите за уровнем маржи
• Диверсифицируйте портфель
"""

        keyboard = [
            [InlineKeyboardButton("💾 Сохранить стратегию", callback_data="save_strategy")],
            [InlineKeyboardButton("📤 Выгрузить в TXT", callback_data="export_calculation")],
            [InlineKeyboardButton("📊 Новый расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ]

        if hasattr(update, 'message'):
            await update.message.reply_text(
                result_text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await update.callback_query.edit_message_text(
                result_text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

        return EXPORT_CALCULATION
    except Exception as e:
        logger.error(f"Ошибка в pro_calculate_and_show_results: {e}")
        await update.message.reply_text(
            "❌ Произошла ошибка при расчете. Попробуйте еще раз.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
        )
        return ConversationHandler.END

# === ЭКСПОРТ РАСЧЁТА В TXT ===

@log_performance
async def export_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id

        calc_data = context.user_data.get('last_calculation')
        if not calc_data:
            await query.message.reply_text("❌ Нет данных для экспорта.")
            return MAIN_MENU

        report_text = TXTReportGenerator.generate_calculation_report(calc_data)
        bio = io.BytesIO()
        bio.name = f"calculation_{user_id}_{int(time.time())}.txt"
        bio.write(report_text.encode('utf-8'))
        bio.seek(0)

        await query.message.reply_document(
            document=InputFile(bio, filename=bio.name),
            caption="📄 Расчёт успешно сохранён в TXT-файл!",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Новый расчёт", callback_data="pro_calculation")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return MAIN_MENU
    except Exception as e:
        logger.error(f"Ошибка в export_calculation: {e}")
        await query.message.reply_text("❌ Ошибка при экспорте.")
        return MAIN_MENU

# === БЫСТРЫЙ РАСЧЁТ С TP ===

@log_performance
async def start_quick_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(
            "⚡ *БЫСТРЫЙ РАСЧЕТ*\n"
            "✏️ Введите название инструмента:\n"
            "Пример: EURUSD, BTCUSD, XAUUSD",
            parse_mode='Markdown'
        )
        return QUICK_INSTRUMENT
    except Exception as e:
        logger.error(f"Ошибка в start_quick_calculation: {e}")

@log_performance
async def quick_handle_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        instrument = update.message.text.upper().strip()
        is_valid, validated_instrument, message = InputValidator.validate_instrument(instrument)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n✏️ Введите название инструмента:",
                parse_mode='Markdown'
            )
            return QUICK_INSTRUMENT

        context.user_data['instrument'] = validated_instrument
        keyboard = [
            [InlineKeyboardButton("📈 BUY", callback_data="quick_BUY"),
             InlineKeyboardButton("📉 SELL", callback_data="quick_SELL")]
        ]
        await update.message.reply_text(
            f"🎯 *Инструмент:* {validated_instrument}\n"
            "Выберите направление сделки:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return QUICK_DIRECTION
    except Exception as e:
        logger.error(f"Ошибка в quick_handle_instrument: {e}")

@log_performance
async def quick_select_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        query = update.callback_query
        await query.answer()
        direction = query.data.replace("quick_", "")
        context.user_data['direction'] = direction
        await query.edit_message_text(
            f"📊 *{context.user_data['instrument']}* | *{direction}*\n"
            "💵 Введите размер депозита:",
            parse_mode='Markdown'
        )
        return QUICK_DEPOSIT
    except Exception as e:
        logger.error(f"Ошибка в quick_select_direction: {e}")

@log_performance
async def quick_handle_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, deposit, message = InputValidator.validate_number(text, 1, 1000000)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n💵 Введите размер депозита:",
                parse_mode='Markdown'
            )
            return QUICK_DEPOSIT

        context.user_data['deposit'] = deposit
        await update.message.reply_text(
            f"💰 *Депозит:* ${deposit:,.2f}\n"
            "🎯 Введите уровень риска (% от депозита):\n"
            "Пример: 2 для 2% риска",
            parse_mode='Markdown'
        )
        return QUICK_RISK
    except Exception as e:
        logger.error(f"Ошибка в quick_handle_deposit: {e}")

@log_performance
async def quick_handle_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, risk_percent, message = InputValidator.validate_percent(text)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n🎯 Введите уровень риска (%):",
                parse_mode='Markdown'
            )
            return QUICK_RISK

        context.user_data['risk_percent'] = risk_percent / 100
        await update.message.reply_text(
            f"🎯 *Риск:* {risk_percent}%\n"
            "💎 Введите цену входа:",
            parse_mode='Markdown'
        )
        return QUICK_ENTRY
    except Exception as e:
        logger.error(f"Ошибка в quick_handle_risk: {e}")

@log_performance
async def quick_handle_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, entry_price, message = InputValidator.validate_price(text)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n💎 Введите цену входа:",
                parse_mode='Markdown'
            )
            return QUICK_ENTRY

        context.user_data['entry_price'] = entry_price
        direction = context.user_data.get('direction', 'BUY')
        direction_text = "выше" if direction == "BUY" else "ниже"
        await update.message.reply_text(
            f"💎 *Цена входа:* {entry_price}\n"
            f"🛑 Введите цену стоп-лосса ({direction_text} цены входа):",
            parse_mode='Markdown'
        )
        return QUICK_STOPLOSS
    except Exception as e:
        logger.error(f"Ошибка в quick_handle_entry: {e}")

@log_performance
async def quick_handle_stoploss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, stop_loss, message = InputValidator.validate_price(text)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n🛑 Введите цену стоп-лосса:",
                parse_mode='Markdown'
            )
            return QUICK_STOPLOSS

        context.user_data['stop_loss'] = stop_loss
        await update.message.reply_text(
            "🎯 *Сколько уровней тейк-профита вы хотите задать?*\n"
            "Выберите от 1 до 3:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("1 TP", callback_data="quick_tp_1")],
                [InlineKeyboardButton("2 TP", callback_data="quick_tp_2")],
                [InlineKeyboardButton("3 TP", callback_data="quick_tp_3")],
                [InlineKeyboardButton("Пропустить", callback_data="quick_tp_skip")]
            ])
        )
        return QUICK_TAKEPROFIT_COUNT
    except Exception as e:
        logger.error(f"Ошибка в quick_handle_stoploss: {e}")

@log_performance
async def quick_handle_tp_count(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        query = update.callback_query
        await query.answer()
        choice = query.data
        if choice == "quick_tp_skip":
            context.user_data['take_profits'] = []
            return await quick_calculate_and_show_results(update, context)

        tp_count = int(choice.replace("quick_tp_", ""))
        context.user_data['tp_count'] = tp_count
        await query.edit_message_text(
            f"🎯 *Введите цену TP1:*",
            parse_mode='Markdown'
        )
        return QUICK_TAKEPROFIT_1
    except Exception as e:
        logger.error(f"Ошибка в quick_handle_tp_count: {e}")

@log_performance
async def quick_handle_tp1(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, tp1, message = InputValidator.validate_price(text)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n🎯 Введите цену TP1:",
                parse_mode='Markdown'
            )
            return QUICK_TAKEPROFIT_1

        context.user_data['tp1'] = tp1
        tp_count = context.user_data['tp_count']
        if tp_count == 1:
            return await quick_collect_tps_and_calculate(update, context)
        else:
            await update.message.reply_text(
                f"🎯 *Введите цену TP2:*",
                parse_mode='Markdown'
            )
            return QUICK_TAKEPROFIT_2
    except Exception as e:
        logger.error(f"Ошибка в quick_handle_tp1: {e}")

@log_performance
async def quick_handle_tp2(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, tp2, message = InputValidator.validate_price(text)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n🎯 Введите цену TP2:",
                parse_mode='Markdown'
            )
            return QUICK_TAKEPROFIT_2

        context.user_data['tp2'] = tp2
        tp_count = context.user_data['tp_count']
        if tp_count == 2:
            return await quick_collect_tps_and_calculate(update, context)
        else:
            await update.message.reply_text(
                f"🎯 *Введите цену TP3:*",
                parse_mode='Markdown'
            )
            return QUICK_TAKEPROFIT_3
    except Exception as e:
        logger.error(f"Ошибка в quick_handle_tp2: {e}")

@log_performance
async def quick_handle_tp3(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, tp3, message = InputValidator.validate_price(text)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n🎯 Введите цену TP3:",
                parse_mode='Markdown'
            )
            return QUICK_TAKEPROFIT_3

        context.user_data['tp3'] = tp3
        return await quick_collect_tps_and_calculate(update, context)
    except Exception as e:
        logger.error(f"Ошибка в quick_handle_tp3: {e}")

async def quick_collect_tps_and_calculate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        tps = []
        if 'tp1' in context.user_data:
            tps.append(context.user_data['tp1'])
        if 'tp2' in context.user_data:
            tps.append(context.user_data['tp2'])
        if 'tp3' in context.user_data:
            tps.append(context.user_data['tp3'])
        context.user_data['take_profits_raw'] = tps
        return await quick_calculate_and_show_results(update, context)
    except Exception as e:
        logger.error(f"Ошибка в quick_collect_tps_and_calculate: {e}")
        await update.message.reply_text("❌ Ошибка при сборе TP. Попробуйте снова.")
        return ConversationHandler.END

@log_performance
async def quick_calculate_and_show_results(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        user_id = update.message.from_user.id
        PortfolioManager.initialize_user_portfolio(user_id)
        settings = user_data[user_id]['portfolio']['settings']

        instrument = context.user_data['instrument']
        instrument_type = 'forex'
        for key, presets in INSTRUMENT_PRESETS.items():
            if instrument in presets:
                instrument_type = key
                break

        calculation = FastRiskCalculator.calculate_position_size_fast(
            deposit=context.user_data['deposit'],
            leverage=settings['leverage'],
            instrument_type=instrument_type,
            currency_pair=instrument,
            entry_price=context.user_data['entry_price'],
            stop_loss=context.user_data['stop_loss'],
            direction=context.user_data['direction'],
            risk_percent=context.user_data['risk_percent']
        )

        take_profits = []
        tp_raw_list = context.user_data.get('take_profits_raw', [])
        position_size = calculation['position_size']
        for tp_price in tp_raw_list:
            profit = calculate_take_profit_profit(position_size, context.user_data['entry_price'], tp_price, context.user_data['direction'], instrument_type, instrument)
            take_profits.append({'price': tp_price, 'profit': profit})

        full_calc_data = {
            'instrument': instrument,
            'direction': context.user_data['direction'],
            'deposit': context.user_data['deposit'],
            'leverage': settings['leverage'],
            'risk_percent_input': context.user_data['risk_percent'] * 100,
            'entry_price': context.user_data['entry_price'],
            'stop_loss': context.user_data['stop_loss'],
            'stop_pips': calculation['stop_pips'],
            'position_size': position_size,
            'risk_amount': calculation['risk_amount'],
            'required_margin': calculation['required_margin'],
            'free_margin': calculation['free_margin'],
            'take_profits': take_profits
        }

        context.user_data['last_calculation'] = full_calc_data

        result_text = f"""
⚡ *РЕЗУЛЬТАТЫ БЫСТРОГО РАСЧЕТА*
📊 *Параметры:*
• 💰 Инструмент: {instrument}
• 📈 Направление: {context.user_data['direction']}
• 💵 Депозит: ${context.user_data['deposit']:,.2f}
• ⚖️ Плечо: {settings['leverage']}
• 🎯 Риск: {context.user_data['risk_percent']*100}%
💎 *Цены:*
• Вход: {context.user_data['entry_price']}
• Стоп-лосс: {context.user_data['stop_loss']}
• Дистанция: {calculation['stop_pips']:.2f} пунктов
📈 *Результаты:*
• 📦 Размер позиции: {position_size:.2f} лотов
• 💸 Сумма риска: ${calculation['risk_amount']:.2f}
• 🏦 Требуемая маржа: ${calculation['required_margin']:.2f}
• 💰 Свободная маржа: ${calculation['free_margin']:.2f}
"""
        if take_profits:
            result_text += "🎯 *Тейк-профиты:*\n"
            for i, tp in enumerate(take_profits, 1):
                result_text += f"• TP{i}: {tp['price']} → прибыль: ${tp['profit']:.2f}\n"
        else:
            result_text += "🎯 *Тейк-профиты:* не заданы\n"

        result_text += "💡 *Готово к торговле!*"

        keyboard = [
            [InlineKeyboardButton("📤 Выгрузить в TXT", callback_data="export_calculation")],
            [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("⚡ Новый быстрый расчет", callback_data="quick_calculation")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ]

        await update.message.reply_text(
            result_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return EXPORT_CALCULATION
    except Exception as e:
        logger.error(f"Ошибка в quick_calculate_and_show_results: {e}")
        await update.message.reply_text(
            "❌ Произошла ошибка при расчете. Попробуйте еще раз.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
        )
        return ConversationHandler.END

# === ДОБАВЛЕНИЕ СДЕЛКИ ===

@log_performance
async def portfolio_add_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(
            "➕ *ДОБАВЛЕНИЕ СДЕЛКИ*\n"
            "✏️ Введите название инструмента:\n"
            "Пример: EURUSD, BTCUSD, XAUUSD",
            parse_mode='Markdown'
        )
        return ADD_TRADE_INSTRUMENT
    except Exception as e:
        logger.error(f"Ошибка в portfolio_add_trade_start: {e}")

@log_performance
async def add_trade_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        instrument = update.message.text.upper().strip()
        is_valid, validated_instrument, message = InputValidator.validate_instrument(instrument)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n✏️ Введите название инструмента:",
                parse_mode='Markdown'
            )
            return ADD_TRADE_INSTRUMENT

        context.user_data['trade_instrument'] = validated_instrument
        keyboard = [
            [InlineKeyboardButton("📈 BUY", callback_data="trade_BUY"),
             InlineKeyboardButton("📉 SELL", callback_data="trade_SELL")]
        ]
        await update.message.reply_text(
            f"🎯 *Инструмент:* {validated_instrument}\n"
            "Выберите направление сделки:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return ADD_TRADE_DIRECTION
    except Exception as e:
        logger.error(f"Ошибка в add_trade_instrument: {e}")

@log_performance
async def add_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        query = update.callback_query
        await query.answer()
        direction = query.data.replace("trade_", "")
        context.user_data['trade_direction'] = direction
        await query.edit_message_text(
            f"📊 *{context.user_data['trade_instrument']}* | *{direction}*\n"
            "💎 Введите цену входа:",
            parse_mode='Markdown'
        )
        return ADD_TRADE_ENTRY
    except Exception as e:
        logger.error(f"Ошибка в add_trade_direction: {e}")

@log_performance
async def add_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, entry_price, message = InputValidator.validate_price(text)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n💎 Введите цену входа:",
                parse_mode='Markdown'
            )
            return ADD_TRADE_ENTRY

        context.user_data['trade_entry'] = entry_price
        await update.message.reply_text(
            f"💎 *Цена входа:* {entry_price}\n"
            "💰 Введите цену выхода:",
            parse_mode='Markdown'
        )
        return ADD_TRADE_EXIT
    except Exception as e:
        logger.error(f"Ошибка в add_trade_entry: {e}")

@log_performance
async def add_trade_exit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, exit_price, message = InputValidator.validate_price(text)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n💰 Введите цену выхода:",
                parse_mode='Markdown'
            )
            return ADD_TRADE_EXIT

        context.user_data['trade_exit'] = exit_price
        await update.message.reply_text(
            f"💰 *Цена выхода:* {exit_price}\n"
            "📦 Введите объем позиции (лоты):",
            parse_mode='Markdown'
        )
        return ADD_TRADE_VOLUME
    except Exception as e:
        logger.error(f"Ошибка в add_trade_exit: {e}")

@log_performance
async def add_trade_volume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, volume, message = InputValidator.validate_number(text, 0.01, 100)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n📦 Введите объем позиции (лоты):",
                parse_mode='Markdown'
            )
            return ADD_TRADE_VOLUME

        context.user_data['trade_volume'] = volume
        await update.message.reply_text(
            f"📦 *Объем:* {volume} лотов\n"
            "💵 Введите прибыль/убыток ($):\n"
            "Для убытка используйте отрицательное число (например: -50)",
            parse_mode='Markdown'
        )
        return ADD_TRADE_PROFIT
    except Exception as e:
        logger.error(f"Ошибка в add_trade_volume: {e}")

@log_performance
async def add_trade_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        text = update.message.text
        is_valid, profit, message = InputValidator.validate_number(text, -1000000, 1000000)
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n💵 Введите прибыль/убыток ($):",
                parse_mode='Markdown'
            )
            return ADD_TRADE_PROFIT

        user_id = update.message.from_user.id
        trade_data = {
            'instrument': context.user_data['trade_instrument'],
            'direction': context.user_data['trade_direction'],
            'entry_price': context.user_data['trade_entry'],
            'exit_price': context.user_data['trade_exit'],
            'volume': context.user_data['trade_volume'],
            'profit': profit,
            'status': 'closed'
        }

        PortfolioManager.add_trade(user_id, trade_data)
        profit_text = "прибылью" if profit > 0 else "убытком"
        await update.message.reply_text(
            f"✅ *Сделка добавлена!*\n"
            f"📊 *Детали сделки:*\n"
            f"• 💰 Инструмент: {trade_data['instrument']}\n"
            f"• 📈 Направление: {trade_data['direction']}\n"
            f"• 💎 Вход: {trade_data['entry_price']}\n"
            f"• 💰 Выход: {trade_data['exit_price']}\n"
            f"• 📦 Объем: {trade_data['volume']} лотов\n"
            f"• 💵 Результат: ${profit:.2f}\n"
            f"Сделка успешно добавлена в портфель с {profit_text}.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💼 Портфель", callback_data="portfolio")],
                [InlineKeyboardButton("➕ Новая сделка", callback_data="portfolio_add_trade")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"Ошибка в add_trade_profit: {e}")
        await update.message.reply_text(
            "❌ Произошла ошибка при добавлении сделки. Попробуйте еще раз.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
        )
        return ConversationHandler.END

# === СОХРАНЕННЫЕ СТРАТЕГИИ ===

@log_performance
async def show_saved_strategies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        strategies = PortfolioManager.get_saved_strategies(user_id)
        if not strategies:
            await query.edit_message_text(
                "💾 *СОХРАНЕННЫЕ СТРАТЕГИИ*\n"
                "📭 У вас пока нет сохраненных стратегий.\n"
                "Вы можете сохранить стратегию после профессионального расчета.",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
                    [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
                    [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
                ])
            )
            return

        strategies_text = "💾 *СОХРАНЕННЫЕ СТРАТЕГИИ*\n"
        for i, strategy in enumerate(strategies[-5:], 1):
            strategies_text += f"{i}. *{strategy.get('name', 'Без названия')}*\n"
            strategies_text += f"   📊 {strategy.get('instrument', 'N/A')} | "
            strategies_text += f"💵 ${strategy.get('deposit', 0):.0f} | "
            strategies_text += f"🎯 {strategy.get('risk_percent', 0)*100}%\n"
            strategies_text += f"   📅 {strategy.get('created_at', '')[:10]}\n"

        await query.edit_message_text(
            strategies_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
                [InlineKeyboardButton("🔙 Назад", callback_data="settings")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в show_saved_strategies: {e}")

# === АНАЛИТИКА ===

@log_performance
async def analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        if update.message:
            user_id = update.message.from_user.id
        else:
            user_id = update.callback_query.from_user.id
            await update.callback_query.answer()

        analytics_text = """
🔮 *АНАЛИТИКА И БУДУЩИЕ ВОЗМОЖНОСТИ*
🚀 *В РАЗРАБОТКЕ:*
🤖 *AI-АССИСТЕНТ*
• Прогнозирование движения цены на основе ML
• Интеллектуальные рекомендации по точкам входа/выхода
• Автоматическая оптимизация торговых стратегий
📈 *РЕАЛЬНЫЕ КОТИРОВКИ С БИРЖИ*
• Интеграция с Binance, Bybit, FTX API
• Автоматическое обновление котировок в реальном времени
• Price alerts и уведомления о достижении уровней
📊 *РАСШИРЕННАЯ АНАЛИТИКА ПОРТФЕЛЯ*
• Корреляция между активами
• Анализ волатильности и риска
• Оптимизация распределения капитала
🔄 *АВТОМАТИЧЕСКАЯ ТОРГОВЛЯ*
• Интеграция с торговыми API
• Исполнение сделок по сигналам
• Мониторинг и управление позициями в реальном времени
📱 *МОБИЛЬНОЕ ПРИЛОЖЕНИЕ*
• Push-уведомления на телефон
• Управление портфелем на ходу
• Полная функциональность в кармане
🔐 *ПОВЫШЕННАЯ БЕЗОПАСНОСТЬ*
• Двухфакторная аутентификация
• Шифрование данных
• Резервное копирование в облако
🌍 *МУЛЬТИВАЛЮТНАЯ ПОДДЕРЖКА*
• Поддержка всех основных валют
• Автоматическая конвертация
• Локализация для разных регионов
📚 *ОБУЧАЮЩИЕ МАТЕРИАЛЫ*
• Видео-уроки
• Торговые стратегии
• Анализ рынка и обзоры
*Следите за обновлениями! Новые функции появляются регулярно.*
"""
        keyboard = [
            [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("💼 Мой портфель", callback_data="portfolio")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ]

        if update.message:
            await update.message.reply_text(
                analytics_text,
                parse_mode='Markdown',
                disable_web_page_preview=True,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await update.callback_query.edit_message_text(
                analytics_text,
                parse_mode='Markdown',
                disable_web_page_preview=True,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        return ANALYTICS_MENU
    except Exception as e:
        logger.error(f"Ошибка в analytics_command: {e}")

# === ГЛАВНОЕ МЕНЮ ===

@log_performance
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        if update.message:
            user = update.message.from_user
        elif update.callback_query:
            user = update.callback_query.from_user
            await update.callback_query.answer()
        else:
            return ConversationHandler.END

        user_name = user.first_name or "Трейдер"
        welcome_text = f"""
👋 *Привет, {user_name}!*
🎯 *PRO Калькулятор Управления Рисками v3.1*
⚡ *АКТИВИРОВАННЫЕ ВОЗМОЖНОСТИ:*
• ✅ Профессиональный расчет с TP/SL
• ✅ Быстрый расчет с TP/SL
• ✅ Управление портфелем и сделками
• ✅ Сохранение стратегий
• ✅ Экспорт расчётов в TXT
• ✅ Расширенная аналитика
• ✅ Автосохранение данных
*Выберите опцию:*
"""
        user_id = user.id
        PortfolioManager.initialize_user_portfolio(user_id)
        keyboard = [
            [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("⚡ Быстрый расчет", callback_data="quick_calculation")],
            [InlineKeyboardButton("💼 Мой портфель", callback_data="portfolio")],
            [InlineKeyboardButton("🔮 Аналитика", callback_data="analytics")],
            [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")],
            [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")]
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
        return ConversationHandler.END

@log_performance
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        info_text = """
📚 *PRO ИНСТРУКЦИИ v3.1*
🎯 *ДЛЯ ПРОФЕССИОНАЛЬНЫХ ТРЕЙДЕРОВ:*
💡 *ИНТУИТИВНОЕ УПРАВЛЕНИЕ РИСКАМИ:*
• Рассчитывайте оптимальный размер позиции за секунды
• Автоматический учет типа инструмента (Форекс, крипто, индексы)
• Поддержка 1–3 уровней тейк-профита
• Мгновенный пересчет при изменении параметров
📊 *ПРОФЕССИОНАЛЬНАЯ АНАЛИТИКА:*
• Точный расчет стоимости пипса для любого инструмента
• Учет маржинальных требований и плеча
• Анализ риска в денежном и процентном выражении
• Рекомендации по оптимизации размера позиции
💼 *УПРАВЛЕНИЕ КАПИТАЛОМ:*
• Полный трекинг торгового портфеля
• Анализ эффективности стратегий
• Расчет ключевых метрик: Win Rate, Profit Factor, просадки
• Интеллектуальные рекомендации по улучшению
⚡ *БЫСТРЫЕ РАСЧЕТЫ:*
• Мгновенные вычисления с кэшированием
• Валидация вводимых данных
• Автоматическое сохранение прогресса
• История всех расчетов и сделок
📤 *ЭКСПОРТ В ФАЙЛ:*
• Сохраняйте расчёты в TXT для архива или анализа
• Отправляйте отчёты себе в личные сообщения
🔧 *КАК ИСПОЛЬЗОВАТЬ:*
1. *Профессиональный расчет* – полный цикл с TP/SL
2. *Быстрый расчет* – мгновенный расчёт с TP/SL
3. *Портфель* – управление сделками и аналитика
4. *Настройки* – персонализация параметров
💾 *СОХРАНЕНИЕ ДАННЫХ:*
• Все ваши расчеты и сделки сохраняются автоматически
• Доступ к истории после перезапуска бота
• Экспорт отчетов для дальнейшего анализа
🚀 *СОВЕТЫ ПРОФЕССИОНАЛА:*
• Всегда используйте стоп-лосс для ограничения рисков
• Диверсифицируйте портфель по разным инструментам
• Следите за соотношением риск/прибыль не менее 1:2
• Регулярно анализируйте статистику для оптимизации стратегии
👨‍💻 *Разработчик для профессионалов:* @fxfeelgood
*PRO v3.1 | Умно • Быстро • Надежно* 🚀
"""
        if update.message:
            await update.message.reply_text(
                info_text, 
                parse_mode='Markdown',
                disable_web_page_preview=True,
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
            )
        else:
            await update.callback_query.edit_message_text(
                info_text,
                parse_mode='Markdown',
                disable_web_page_preview=True,
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
            )
    except Exception as e:
        logger.error(f"Ошибка в pro_info_command: {e}")

@log_performance
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Операция отменена.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
    )
    return ConversationHandler.END

@log_performance
async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❌ Неизвестная команда. Используйте /start для начала работы.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
    )

# === ОБРАБОТЧИКИ МЕНЮ ===

@log_performance
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        query = update.callback_query
        if not query:
            return MAIN_MENU
        await query.answer()
        choice = query.data
        user_id = query.from_user.id
        if user_id in user_data:
            user_data[user_id]['last_activity'] = time.time()

        handlers_map = {
            "pro_calculation": start_pro_calculation,
            "quick_calculation": start_quick_calculation,
            "portfolio": portfolio_command,
            "analytics": analytics_command,
            "pro_info": pro_info_command,
            "settings": settings_command,
            "main_menu": start,
            "portfolio_deposit": portfolio_deposit_menu,
            "portfolio_trades": portfolio_trades,
            "portfolio_balance": portfolio_balance,
            "portfolio_performance": portfolio_performance,
            "portfolio_report": portfolio_report,
            "portfolio_add_trade": portfolio_add_trade_start,
            "change_risk": change_risk_setting,
            "change_currency": change_currency_setting,
            "change_leverage": change_leverage_setting,
            "saved_strategies": show_saved_strategies,
            "export_calculation": export_calculation,
        }

        if choice.startswith("set_risk_"):
            await save_risk_setting(update, context)
            return SETTINGS_MENU
        elif choice.startswith("set_currency_"):
            await save_currency_setting(update, context)
            return SETTINGS_MENU
        elif choice.startswith("set_leverage_"):
            await save_leverage_setting(update, context)
            return SETTINGS_MENU
        elif choice in handlers_map:
            handler = handlers_map[choice]
            if asyncio.iscoroutinefunction(handler):
                return await handler(update, context)
            else:
                return handler(update, context)

        # Обработка TP выбора
        if choice in ["tp_count_1", "tp_count_2", "tp_count_3", "tp_skip"]:
            return await pro_handle_tp_count(update, context)
        if choice in ["quick_tp_1", "quick_tp_2", "quick_tp_3", "quick_tp_skip"]:
            return await quick_handle_tp_count(update, context)

        return MAIN_MENU
    except Exception as e:
        logger.error(f"Ошибка в handle_main_menu: {e}")
        return await start(update, context)

# === ЗАПУСК ===

def main():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ Токен бота не найден!")
        return

    logger.info("🚀 Запуск ПРОФЕССИОНАЛЬНОГО калькулятора рисков v3.1...")

    try:
        DataManager.auto_save()
    except:
        pass

    application = Application.builder().token(token).build()

    # Профессиональный расчёт
    pro_calc_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(start_pro_calculation, pattern='^pro_calculation$')],
        states={
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
            TAKE_PROFIT_COUNT: [CallbackQueryHandler(pro_handle_tp_count)],
            TAKE_PROFIT_1: [MessageHandler(filters.TEXT & ~filters.COMMAND, pro_handle_tp1)],
            TAKE_PROFIT_2: [MessageHandler(filters.TEXT & ~filters.COMMAND, pro_handle_tp2)],
            TAKE_PROFIT_3: [MessageHandler(filters.TEXT & ~filters.COMMAND, pro_handle_tp3)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # Быстрый расчёт
    quick_calc_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(start_quick_calculation, pattern='^quick_calculation$')],
        states={
            QUICK_INSTRUMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, quick_handle_instrument)],
            QUICK_DIRECTION: [CallbackQueryHandler(quick_select_direction)],
            QUICK_DEPOSIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, quick_handle_deposit)],
            QUICK_RISK: [MessageHandler(filters.TEXT & ~filters.COMMAND, quick_handle_risk)],
            QUICK_ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, quick_handle_entry)],
            QUICK_STOPLOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, quick_handle_stoploss)],
            QUICK_TAKEPROFIT_COUNT: [CallbackQueryHandler(quick_handle_tp_count)],
            QUICK_TAKEPROFIT_1: [MessageHandler(filters.TEXT & ~filters.COMMAND, quick_handle_tp1)],
            QUICK_TAKEPROFIT_2: [MessageHandler(filters.TEXT & ~filters.COMMAND, quick_handle_tp2)],
            QUICK_TAKEPROFIT_3: [MessageHandler(filters.TEXT & ~filters.COMMAND, quick_handle_tp3)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # Добавление сделки
    add_trade_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(portfolio_add_trade_start, pattern='^portfolio_add_trade$')],
        states={
            ADD_TRADE_INSTRUMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_trade_instrument)],
            ADD_TRADE_DIRECTION: [CallbackQueryHandler(add_trade_direction)],
            ADD_TRADE_ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_trade_entry)],
            ADD_TRADE_EXIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_trade_exit)],
            ADD_TRADE_VOLUME: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_trade_volume)],
            ADD_TRADE_PROFIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_trade_profit)],
        },
        fallbacks=[C
