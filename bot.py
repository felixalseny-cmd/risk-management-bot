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
    STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION,
    PORTFOLIO_MENU, ADD_TRADE_INSTRUMENT, ADD_TRADE_DIRECTION,
    ADD_TRADE_ENTRY, ADD_TRADE_EXIT, ADD_TRADE_VOLUME, ADD_TRADE_PROFIT,
    DEPOSIT_AMOUNT, WITHDRAW_AMOUNT, SETTINGS_MENU, SAVE_STRATEGY_NAME,
    PRO_DEPOSIT, PRO_LEVERAGE, PRO_RISK, PRO_ENTRY, PRO_STOPLOSS,
    PRO_TAKEPROFIT, PRO_VOLUME, STRATEGY_NAME
) = range(35)

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
        # Планируем следующее автосохранение
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

# Генератор PDF отчетов
class PDFReportGenerator:
    @staticmethod
    def generate_portfolio_report(user_id: int) -> str:
        """Генерация текстового отчета"""
        try:
            portfolio = user_data[user_id]['portfolio']
            performance = portfolio['performance']
            
            report = f"""
ОТЧЕТ ПО ПОРТФЕЛЮ v3.0
Дата генерации: {datetime.now().strftime('%d.%m.%Y %H:%M')}

БАЛАНС И СРЕДСТВА:
• Начальный депозит: ${portfolio['initial_balance']:,.2f}
• Текущий баланс: ${portfolio['current_balance']:,.2f}
• Общая прибыль/убыток: ${portfolio['current_balance'] - portfolio['initial_balance']:,.2f}

СТАТИСТИКА ТОРГОВЛИ:
• Всего сделок: {performance['total_trades']}
• Прибыльные сделки: {performance['winning_trades']}
• Убыточные сделки: {performance['losing_trades']}
• Win Rate: {performance['win_rate']:.1f}%
• Profit Factor: {performance['profit_factor']:.2f}
• Макс. просадка: {performance['max_drawdown']:.1f}%

ДОХОДНОСТЬ:
• Общая прибыль: ${performance['total_profit']:,.2f}
• Общий убыток: ${performance['total_loss']:,.2f}
• Средняя прибыль: ${performance['average_profit']:.2f}
• Средний убыток: ${performance['average_loss']:.2f}

РАСПРЕДЕЛЕНИЕ ПО ИНСТРУМЕНТАМ:
"""
            
            allocation = portfolio.get('allocation', {})
            for instrument, count in allocation.items():
                report += f"• {instrument}: {count} сделок\n"
            
            recommendations = PortfolioManager.get_performance_recommendations(user_id)
            if recommendations:
                report += "\nPRO РЕКОМЕНДАЦИИ:\n"
                for i, rec in enumerate(recommendations[:3], 1):
                    report += f"{i}. {rec}\n"
            
            return report
        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")
            return "Ошибка при генерации отчета"

# Обработчики портфеля
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
                "📭 *У вас еще нет сделок*\n\n"
                "Используйте кнопку '➕ Добавить сделку' чтобы начать.",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]
                ])
            )
            return
        
        recent_trades = trades[-5:]
        trades_text = "📈 *Последние сделки:*\n\n"
        
        for trade in reversed(recent_trades):
            status_emoji = "🟢" if trade.get('profit', 0) > 0 else "🔴" if trade.get('profit', 0) < 0 else "⚪"
            trades_text += (
                f"{status_emoji} *{trade.get('instrument', 'N/A')}* | "
                f"{trade.get('direction', 'N/A')} | "
                f"Прибыль: ${trade.get('profit', 0):.2f}\n"
                f"📅 {trade.get('timestamp', '')[:16]}\n\n"
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
        
        balance_text = "💰 *Баланс и распределение*\n\n"
        
        initial_balance = portfolio.get('initial_balance', 0)
        current_balance = portfolio.get('current_balance', 0)
        total_profit = performance.get('total_profit', 0)
        total_loss = performance.get('total_loss', 0)
        net_profit = total_profit + total_loss
        
        balance_text += f"💳 Начальный депозит: ${initial_balance:,.2f}\n"
        balance_text += f"💵 Текущий баланс: ${current_balance:,.2f}\n"
        balance_text += f"📈 Чистая прибыль: ${net_profit:.2f}\n\n"
        
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
        
        perf_text = "📊 *PRO АНАЛИЗ ЭФФЕКТИВНОСТИ*\n\n"
        
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
        perf_text += f"📊 Макс. просадка: {max_drawdown:.1f}%\n\n"
        
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
    """Генерация отчета по портфелю"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        report_text = PDFReportGenerator.generate_portfolio_report(user_id)
        
        if len(report_text) > 4000:
            parts = [report_text[i:i+4000] for i in range(0, len(report_text), 4000)]
            for part in parts:
                await query.message.reply_text(
                    f"```\n{part}\n```",
                    parse_mode='Markdown'
                )
        else:
            await query.edit_message_text(
                f"```\n{report_text}\n```",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("💼 Портфель", callback_data="portfolio")],
                    [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
                ])
            )
        
        await query.message.reply_text(
            "📄 *Отчет сгенерирован!*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💼 Портфель", callback_data="portfolio")]
            ])
        )
            
    except Exception as e:
        logger.error(f"Ошибка в portfolio_report: {e}")
        await query.edit_message_text(
            "❌ *Ошибка генерации отчета*",
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
            "💸 *Внесение депозита*\n\n"
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
        
        # Валидация ввода
        is_valid, amount, message = InputValidator.validate_number(text, 1, 1000000)
        
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n\n💰 Введите сумму депозита:",
                parse_mode='Markdown'
            )
            return DEPOSIT_AMOUNT
        
        PortfolioManager.add_balance_operation(user_id, 'deposit', amount, "Депозит")
        
        await update.message.reply_text(
            f"✅ *Депозит на ${amount:,.2f} успешно внесен!*\n\n"
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
            "❌ *Произошла ошибка!*\n\n"
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
    """Изменение уровня риска"""
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
    """Изменение валюты"""
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
    """Изменение плеча"""
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
    """Сохранение уровня риска"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        risk_level = float(query.data.replace("set_risk_", ""))
        user_data[user_id]['portfolio']['settings']['default_risk'] = risk_level
        DataManager.save_data()
        
        await query.edit_message_text(
            f"✅ *Уровень риска установлен: {risk_level*100}%*\n\n"
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
    """Сохранение валюты"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        currency = query.data.replace("set_currency_", "")
        user_data[user_id]['portfolio']['settings']['currency'] = currency
        DataManager.save_data()
        
        await query.edit_message_text(
            f"✅ *Валюта установлена: {currency}*\n\n"
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
    """Сохранение плеча"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        leverage = query.data.replace("set_leverage_", "")
        user_data[user_id]['portfolio']['settings']['leverage'] = leverage
        DataManager.save_data()
        
        await query.edit_message_text(
            f"✅ *Плечо установлено: {leverage}*\n\n"
            "Настройки сохранены для будущих расчетов.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в save_leverage_setting: {e}")

# Профессиональный расчет - полный цикл
@log_performance
async def start_pro_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало профессионального расчета"""
    try:
        query = update.callback_query
        await query.answer()
        
        # Сбрасываем данные расчета
        context.user_data['pro_calc'] = {}
        
        await query.edit_message_text(
            "📊 *ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ РИСКОВ*\n\n"
            "💎 *Выберите тип инструмента:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 Форекс", callback_data="instrument_forex")],
                [InlineKeyboardButton("₿ Криптовалюты", callback_data="instrument_crypto")],
                [InlineKeyboardButton("📈 Индексы", callback_data="instrument_indices")],
                [InlineKeyboardButton("⚡ Сырьевые товары", callback_data="instrument_commodities")],
                [InlineKeyboardButton("🏅 Металлы", callback_data="instrument_metals")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return INSTRUMENT_TYPE
    except Exception as e:
        logger.error(f"Ошибка в start_pro_calculation: {e}")
        return MAIN_MENU

@log_performance
async def handle_instrument_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора типа инструмента"""
    try:
        query = update.callback_query
        await query.answer()
        
        instrument_type = query.data.replace("instrument_", "")
        context.user_data['pro_calc']['instrument_type'] = instrument_type
        
        # Показываем пресеты инструментов
        presets = INSTRUMENT_PRESETS.get(instrument_type, [])
        keyboard = []
        
        # Кнопки с пресетами (по 2 в ряд)
        for i in range(0, len(presets), 2):
            row = []
            if i < len(presets):
                row.append(InlineKeyboardButton(presets[i], callback_data=f"preset_{presets[i]}"))
            if i + 1 < len(presets):
                row.append(InlineKeyboardButton(presets[i+1], callback_data=f"preset_{presets[i+1]}"))
            keyboard.append(row)
        
        keyboard.append([InlineKeyboardButton("✏️ Свой инструмент", callback_data="custom_instrument")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="pro_calculation")])
        
        await query.edit_message_text(
            f"📊 *Выбран тип: {INSTRUMENT_TYPES[instrument_type]}*\n\n"
            "💎 *Выберите инструмент из списка или введите свой:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return CUSTOM_INSTRUMENT
    except Exception as e:
        logger.error(f"Ошибка в handle_instrument_type: {e}")

@log_performance
async def handle_instrument_preset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора пресета инструмента"""
    try:
        query = update.callback_query
        await query.answer()
        
        instrument = query.data.replace("preset_", "")
        context.user_data['pro_calc']['instrument'] = instrument
        
        await query.edit_message_text(
            f"💎 *Инструмент: {instrument}*\n\n"
            "📊 *Выберите направление сделки:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔼 BUY", callback_data="direction_BUY")],
                [InlineKeyboardButton("🔽 SELL", callback_data="direction_SELL")],
                [InlineKeyboardButton("🔙 Назад", callback_data="instrument_back")]
            ])
        )
        return DIRECTION
    except Exception as e:
        logger.error(f"Ошибка в handle_instrument_preset: {e}")

@log_performance
async def handle_custom_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода своего инструмента"""
    try:
        query = update.callback_query
        await query.answer()
        
        await query.edit_message_text(
            "✏️ *Введите свой инструмент:*\n\n"
            "Пример: EURUSD, BTCUSD, XAUUSD",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="instrument_back")]
            ])
        )
        return CUSTOM_INSTRUMENT
    except Exception as e:
        logger.error(f"Ошибка в handle_custom_instrument: {e}")

@log_performance
async def handle_custom_instrument_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода названия инструмента"""
    try:
        user_id = update.message.from_user.id
        instrument = update.message.text.upper().strip()
        
        # Валидация инструмента
        is_valid, validated_instrument = InputValidator.validate_instrument(instrument)
        
        if not is_valid:
            await update.message.reply_text(
                f"{validated_instrument}\n\n✏️ Введите свой инструмент:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="instrument_back")]
                ])
            )
            return CUSTOM_INSTRUMENT
        
        context.user_data['pro_calc']['instrument'] = validated_instrument
        
        await update.message.reply_text(
            f"💎 *Инструмент: {validated_instrument}*\n\n"
            "📊 *Выберите направление сделки:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔼 BUY", callback_data="direction_BUY")],
                [InlineKeyboardButton("🔽 SELL", callback_data="direction_SELL")],
                [InlineKeyboardButton("🔙 Назад", callback_data="instrument_back")]
            ])
        )
        return DIRECTION
    except Exception as e:
        logger.error(f"Ошибка в handle_custom_instrument_input: {e}")

@log_performance
async def handle_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора направления"""
    try:
        query = update.callback_query
        await query.answer()
        
        direction = query.data.replace("direction_", "")
        context.user_data['pro_calc']['direction'] = direction
        
        user_id = query.from_user.id
        settings = user_data[user_id]['portfolio']['settings']
        
        await query.edit_message_text(
            f"📊 *Направление: {direction}*\n\n"
            f"💰 *Текущий депозит: ${user_data[user_id]['portfolio']['current_balance']:,.2f}*\n\n"
            "💵 *Введите сумму депозита для расчета:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton(f"💳 Использовать текущий (${user_data[user_id]['portfolio']['current_balance']:,.2f})", 
                                    callback_data="use_current_deposit")],
                [InlineKeyboardButton("🔙 Назад", callback_data="instrument_back")]
            ])
        )
        return PRO_DEPOSIT
    except Exception as e:
        logger.error(f"Ошибка в handle_direction: {e}")

@log_performance
async def handle_pro_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода депозита для про расчета"""
    try:
        if update.callback_query:
            # Использовать текущий депозит
            query = update.callback_query
            await query.answer()
            user_id = query.from_user.id
            deposit = user_data[user_id]['portfolio']['current_balance']
        else:
            # Ввод депозита вручную
            user_id = update.message.from_user.id
            text = update.message.text
            
            is_valid, deposit, message = InputValidator.validate_number(text, 1, 1000000)
            if not is_valid:
                await update.message.reply_text(
                    f"{message}\n\n💵 Введите сумму депозита:",
                    parse_mode='Markdown'
                )
                return PRO_DEPOSIT
        
        context.user_data['pro_calc']['deposit'] = deposit
        
        user_id = update.message.from_user.id if update.message else query.from_user.id
        settings = user_data[user_id]['portfolio']['settings']
        
        await (update.message.reply_text if update.message else query.edit_message_text)(
            f"💰 *Депозит: ${deposit:,.2f}*\n\n"
            f"⚖️ *Текущее плечо: {settings['leverage']}*\n\n"
            "⚖️ *Выберите плечо:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("1:10", callback_data="leverage_1:10")],
                [InlineKeyboardButton("1:50", callback_data="leverage_1:50")],
                [InlineKeyboardButton("1:100", callback_data="leverage_1:100")],
                [InlineKeyboardButton("1:200", callback_data="leverage_1:200")],
                [InlineKeyboardButton("1:500", callback_data="leverage_1:500")],
                [InlineKeyboardButton(f"🔄 Использовать {settings['leverage']}", 
                                    callback_data=f"leverage_{settings['leverage']}")],
                [InlineKeyboardButton("🔙 Назад", callback_data="direction_back")]
            ])
        )
        return PRO_LEVERAGE
    except Exception as e:
        logger.error(f"Ошибка в handle_pro_deposit: {e}")

@log_performance
async def handle_pro_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора плеча"""
    try:
        query = update.callback_query
        await query.answer()
        
        leverage = query.data.replace("leverage_", "")
        context.user_data['pro_calc']['leverage'] = leverage
        
        user_id = query.from_user.id
        settings = user_data[user_id]['portfolio']['settings']
        
        await query.edit_message_text(
            f"⚖️ *Плечо: {leverage}*\n\n"
            f"🎯 *Текущий риск: {settings['default_risk']*100}%*\n\n"
            "🎯 *Выберите уровень риска:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("1%", callback_data="risk_0.01")],
                [InlineKeyboardButton("2%", callback_data="risk_0.02")],
                [InlineKeyboardButton("3%", callback_data="risk_0.03")],
                [InlineKeyboardButton("5%", callback_data="risk_0.05")],
                [InlineKeyboardButton(f"🔄 Использовать {settings['default_risk']*100}%", 
                                    callback_data=f"risk_{settings['default_risk']}")],
                [InlineKeyboardButton("🔙 Назад", callback_data="deposit_back")]
            ])
        )
        return PRO_RISK
    except Exception as e:
        logger.error(f"Ошибка в handle_pro_leverage: {e}")

@log_performance
async def handle_pro_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора уровня риска"""
    try:
        query = update.callback_query
        await query.answer()
        
        risk = float(query.data.replace("risk_", ""))
        context.user_data['pro_calc']['risk_percent'] = risk
        
        await query.edit_message_text(
            f"🎯 *Уровень риска: {risk*100}%*\n\n"
            "💰 *Введите цену входа:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="leverage_back")]
            ])
        )
        return PRO_ENTRY
    except Exception as e:
        logger.error(f"Ошибка в handle_pro_risk: {e}")

@log_performance
async def handle_pro_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода цены входа"""
    try:
        user_id = update.message.from_user.id
        text = update.message.text
        
        # Валидация цены
        is_valid, entry_price, message = InputValidator.validate_price(text)
        
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n\n💰 Введите цену входа:",
                parse_mode='Markdown'
            )
            return PRO_ENTRY
        
        context.user_data['pro_calc']['entry_price'] = entry_price
        
        await update.message.reply_text(
            f"💰 *Цена входа: {entry_price}*\n\n"
            "🛑 *Введите цену стоп-лосса:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="risk_back")]
            ])
        )
        return PRO_STOPLOSS
    except Exception as e:
        logger.error(f"Ошибка в handle_pro_entry: {e}")

@log_performance
async def handle_pro_stoploss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода стоп-лосса"""
    try:
        user_id = update.message.from_user.id
        text = update.message.text
        
        # Валидация стоп-лосса
        is_valid, stop_loss, message = InputValidator.validate_price(text)
        
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n\n🛑 Введите цену стоп-лосса:",
                parse_mode='Markdown'
            )
            return PRO_STOPLOSS
        
        context.user_data['pro_calc']['stop_loss'] = stop_loss
        
        await update.message.reply_text(
            f"🛑 *Стоп-лосс: {stop_loss}*\n\n"
            "🎯 *Введите первый тейк-профит:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="entry_back")]
            ])
        )
        return PRO_TAKEPROFIT
    except Exception as e:
        logger.error(f"Ошибка в handle_pro_stoploss: {e}")

@log_performance
async def handle_pro_takeprofit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода тейк-профита"""
    try:
        user_id = update.message.from_user.id
        text = update.message.text
        
        # Валидация тейк-профита
        is_valid, take_profit, message = InputValidator.validate_price(text)
        
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n\n🎯 Введите цену тейк-профита:",
                parse_mode='Markdown'
            )
            return PRO_TAKEPROFIT
        
        # Инициализируем список тейк-профитов
        if 'take_profits' not in context.user_data['pro_calc']:
            context.user_data['pro_calc']['take_profits'] = []
        
        context.user_data['pro_calc']['take_profits'].append(take_profit)
        
        # Предлагаем добавить еще тейк-профит или продолжить
        await update.message.reply_text(
            f"🎯 *Тейк-профит {len(context.user_data['pro_calc']['take_profits'])}: {take_profit}*\n\n"
            "Добавить еще один тейк-профит или продолжить?",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("➕ Добавить тейк-профит", callback_data="add_more_tp")],
                [InlineKeyboardButton("➡️ Продолжить", callback_data="continue_to_volume")],
                [InlineKeyboardButton("🔙 Назад", callback_data="stoploss_back")]
            ])
        )
        return PRO_TAKEPROFIT
    except Exception as e:
        logger.error(f"Ошибка в handle_pro_takeprofit: {e}")

@log_performance
async def handle_pro_volume_distribution(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка распределения объема"""
    try:
        query = update.callback_query
        await query.answer()
        
        take_profits = context.user_data['pro_calc']['take_profits']
        
        if query.data == "add_more_tp":
            await query.edit_message_text(
                f"🎯 *Текущие тейк-профиты: {len(take_profits)}*\n\n"
                "Введите следующий тейк-профит:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("➡️ Продолжить", callback_data="continue_to_volume")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="stoploss_back")]
                ])
            )
            return PRO_TAKEPROFIT
        
        # Переходим к распределению объема
        volume_text = "📊 *Распределение объема по тейк-профитам:*\n\n"
        for i, tp in enumerate(take_profits, 1):
            volume_text += f"Тейк-профит {i}: {tp}\n"
        
        volume_text += "\n💡 *Рекомендуемое распределение:*\n"
        
        # Автоматическое распределение объема
        num_tps = len(take_profits)
        if num_tps == 1:
            recommended = [100]
        elif num_tps == 2:
            recommended = [60, 40]
        elif num_tps == 3:
            recommended = [50, 30, 20]
        else:
            recommended = [40] + [60//(num_tps-1)]*(num_tps-1)
        
        for i, perc in enumerate(recommended, 1):
            volume_text += f"ТП{i}: {perc}%\n"
        
        context.user_data['pro_calc']['volume_distribution'] = recommended
        
        await query.edit_message_text(
            volume_text + "\n" +
            "✅ *Автоматическое распределение установлено*\n\n"
            "Хотите изменить распределение или продолжить?",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✏️ Изменить распределение", callback_data="edit_volume")],
                [InlineKeyboardButton("🧮 Рассчитать сделку", callback_data="calculate_trade")],
                [InlineKeyboardButton("🔙 Назад", callback_data="takeprofit_back")]
            ])
        )
        return PRO_VOLUME
    except Exception as e:
        logger.error(f"Ошибка в handle_pro_volume_distribution: {e}")

@log_performance
async def calculate_and_show_results(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Расчет и отображение результатов"""
    try:
        query = update.callback_query
        await query.answer()
        
        calc_data = context.user_data['pro_calc']
        
        # Выполняем расчеты
        position_data = FastRiskCalculator.calculate_position_size_fast(
            deposit=calc_data['deposit'],
            leverage=calc_data['leverage'],
            instrument_type=calc_data['instrument_type'],
            currency_pair=calc_data['instrument'],
            entry_price=calc_data['entry_price'],
            stop_loss=calc_data['stop_loss'],
            direction=calc_data['direction'],
            risk_percent=calc_data['risk_percent']
        )
        
        # Формируем результат
        result_text = f"""
🎯 *РЕЗУЛЬТАТЫ РАСЧЕТА*

💎 *Инструмент:* {calc_data['instrument']}
📊 *Направление:* {calc_data['direction']}
💰 *Депозит:* ${calc_data['deposit']:,.2f}
⚖️ *Плечо:* {calc_data['leverage']}
🎯 *Риск:* {calc_data['risk_percent']*100}%

📈 *ПАРАМЕТРЫ СДЕЛКИ:*
• Цена входа: {calc_data['entry_price']}
• Стоп-лосс: {calc_data['stop_loss']}
• Размер позиции: {position_data['position_size']:.2f} лотов
• Сумма риска: ${position_data['risk_amount']:.2f}
• Требуемая маржа: ${position_data['required_margin']:.2f}
• Свободная маржа: ${position_data['free_margin']:.2f}

🎯 *ТЕЙК-ПРОФИТЫ:*
"""
        
        for i, (tp, vol_perc) in enumerate(zip(calc_data['take_profits'], 
                                            calc_data['volume_distribution']), 1):
            result_text += f"• ТП{i}: {tp} ({vol_perc}% объема)\n"
        
        # Добавляем рекомендации
        result_text += "\n💡 *РЕКОМЕНДАЦИИ:*\n"
        if position_data['risk_percent'] > 5:
            result_text += "• ⚠️ Уровень риска высокий, рассмотрите уменьшение\n"
        else:
            result_text += "• ✅ Уровень риска в норме\n"
        
        if position_data['free_margin'] < position_data['required_margin'] * 0.5:
            result_text += "• ⚠️ Свободной маржи мало, увеличьте депозит\n"
        else:
            result_text += "• ✅ Маржинальный уровень достаточный\n"
        
        keyboard = [
            [InlineKeyboardButton("💾 Сохранить стратегию", callback_data="save_strategy")],
            [InlineKeyboardButton("🔄 Новый расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("💼 Добавить в портфель", callback_data="add_to_portfolio")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            result_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        # Сохраняем данные расчета для возможного сохранения
        context.user_data['last_calculation'] = {
            **calc_data,
            **position_data,
            'calculated_at': datetime.now().isoformat()
        }
        
        return MAIN_MENU
        
    except Exception as e:
        logger.error(f"Ошибка в calculate_and_show_results: {e}")
        await query.edit_message_text(
            "❌ *Ошибка расчета*\n\nПопробуйте еще раз.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Новый расчет", callback_data="pro_calculation")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return MAIN_MENU

@log_performance
async def save_strategy_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Запрос названия для сохранения стратегии"""
    try:
        query = update.callback_query
        await query.answer()
        
        await query.edit_message_text(
            "💾 *Сохранение стратегии*\n\n"
            "📝 Введите название для вашей стратегии:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="calculation_back")]
            ])
        )
        return STRATEGY_NAME
    except Exception as e:
        logger.error(f"Ошибка в save_strategy_prompt: {e}")

@log_performance
async def handle_strategy_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка названия стратегии"""
    try:
        user_id = update.message.from_user.id
        strategy_name = update.message.text.strip()
        
        if not strategy_name:
            await update.message.reply_text(
                "❌ *Название не может быть пустым!*\n\n"
                "📝 Введите название для вашей стратегии:",
                parse_mode='Markdown'
            )
            return STRATEGY_NAME
        
        # Сохраняем стратегию
        strategy_data = {
            'name': strategy_name,
            **context.user_data['last_calculation']
        }
        
        strategy_id = PortfolioManager.save_strategy(user_id, strategy_data)
        
        await update.message.reply_text(
            f"✅ *Стратегия '{strategy_name}' сохранена!*\n\n"
            f"🆔 ID: {strategy_id}\n"
            f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💾 Мои стратегии", callback_data="saved_strategies")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return ConversationHandler.END
        
    except Exception as e:
        logger.error(f"Ошибка в handle_strategy_name: {e}")

@log_performance
async def show_saved_strategies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать сохраненные стратегии"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        strategies = PortfolioManager.get_saved_strategies(user_id)
        
        if not strategies:
            await query.edit_message_text(
                "💾 *У вас нет сохраненных стратегий*\n\n"
                "Сохраните свою первую стратегию после расчета.",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="settings")]
                ])
            )
            return
        
        strategies_text = "💾 *Сохраненные стратегии:*\n\n"
        
        for strategy in strategies[-5:]:  # Показываем последние 5
            strategies_text += (
                f"🆔 *{strategy['id']}. {strategy['name']}*\n"
                f"💎 {strategy.get('instrument', 'N/A')} | "
                f"{strategy.get('direction', 'N/A')}\n"
                f"📅 {strategy.get('created_at', '')[:10]}\n\n"
            )
        
        await query.edit_message_text(
            strategies_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Новый расчет", callback_data="pro_calculation")],
                [InlineKeyboardButton("🔙 Назад", callback_data="settings")]
            ])
        )
        
    except Exception as e:
        logger.error(f"Ошибка в show_saved_strategies: {e}")

# Главное меню и основные команды
@log_performance
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Главное меню"""
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

🎯 *PRO Калькулятор Управления Рисками v3.0*

⚡ *Новые возможности в версии 3.0:*
• ✅ Полный цикл профессионального расчета
• 💾 Сохранение стратегий
• ⚙️ Расширенные настройки
• 💾 Автосохранение данных
• 🎯 Умные рекомендации

*Выберите опцию:*
"""
        
        user_id = user.id
        PortfolioManager.initialize_user_portfolio(user_id)
        
        keyboard = [
            [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("⚡ Быстрый расчет", callback_data="quick_calculation")],
            [InlineKeyboardButton("💼 Мой портфель", callback_data="portfolio")],
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
    """PRO Инструкции v3.0"""
    try:
        info_text = """
📚 *PRO ИНСТРУКЦИИ v3.0*

🎯 *ДЛЯ ПРОФЕССИОНАЛЬНЫХ ТРЕЙДЕРОВ:*

💡 *ИНТУИТИВНОЕ УПРАВЛЕНИЕ РИСКАМИ:*
• Рассчитывайте оптимальный размер позиции за секунды
• Автоматический учет типа инструмента (Форекс, крипто, индексы)
• Умное распределение объема по нескольким тейк-профитам
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

🔧 *КАК ИСПОЛЬЗОВАТЬ:*
1. *Профессиональный расчет* - полный цикл с настройкой всех параметров
2. *Быстрый расчет* - мгновенный расчет по основным параметрам  
3. *Портфель* - управление сделками и аналитика эффективности
4. *Настройки* - персонализация параметров по умолчанию

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

*PRO v3.0 | Умно • Быстро • Надежно* 🚀
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
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора в главном меню"""
    try:
        query = update.callback_query
        if not query:
            return MAIN_MENU
            
        await query.answer()
        choice = query.data
        
        user_id = query.from_user.id
        if user_id in user_data:
            user_data[user_id]['last_activity'] = time.time()
        
        # Основные опции меню
        if choice == "pro_calculation":
            return await start_pro_calculation(update, context)
        elif choice == "quick_calculation":
            return await start_quick_calculation(update, context)
        elif choice == "portfolio":
            return await portfolio_command(update, context)
        elif choice == "pro_info":
            await pro_info_command(update, context)
            return MAIN_MENU
        elif choice == "settings":
            return await settings_command(update, context)
        elif choice == "main_menu":
            return await start(update, context)
        
        # Настройки
        elif choice == "change_risk":
            await change_risk_setting(update, context)
            return SETTINGS_MENU
        elif choice == "change_currency":
            await change_currency_setting(update, context)
            return SETTINGS_MENU
        elif choice == "change_leverage":
            await change_leverage_setting(update, context)
            return SETTINGS_MENU
        elif choice.startswith("set_risk_"):
            await save_risk_setting(update, context)
            return SETTINGS_MENU
        elif choice.startswith("set_currency_"):
            await save_currency_setting(update, context)
            return SETTINGS_MENU
        elif choice.startswith("set_leverage_"):
            await save_leverage_setting(update, context)
            return SETTINGS_MENU
        elif choice == "saved_strategies":
            await show_saved_strategies(update, context)
            return SETTINGS_MENU
        
        # Портфель
        elif choice == "portfolio_deposit":
            return await portfolio_deposit_menu(update, context)
        elif choice == "portfolio_trades":
            await portfolio_trades(update, context)
            return PORTFOLIO_MENU
        elif choice == "portfolio_balance":
            await portfolio_balance(update, context)
            return PORTFOLIO_MENU
        elif choice == "portfolio_performance":
            await portfolio_performance(update, context)
            return PORTFOLIO_MENU
        elif choice == "portfolio_report":
            await portfolio_report(update, context)
            return PORTFOLIO_MENU
        elif choice == "portfolio_add_trade":
            return await portfolio_add_trade_start(update, context)
        
        # Профессиональный расчет
        elif choice.startswith("instrument_"):
            return await handle_instrument_type(update, context)
        elif choice.startswith("preset_"):
            return await handle_instrument_preset(update, context)
        elif choice == "custom_instrument":
            return await handle_custom_instrument(update, context)
        elif choice.startswith("direction_"):
            return await handle_direction(update, context)
        elif choice == "use_current_deposit":
            return await handle_pro_deposit(update, context)
        elif choice.startswith("leverage_"):
            return await handle_pro_leverage(update, context)
        elif choice.startswith("risk_"):
            return await handle_pro_risk(update, context)
        elif choice == "add_more_tp":
            return await handle_pro_volume_distribution(update, context)
        elif choice == "continue_to_volume":
            return await handle_pro_volume_distribution(update, context)
        elif choice == "calculate_trade":
            return await calculate_and_show_results(update, context)
        elif choice == "save_strategy":
            return await save_strategy_prompt(update, context)
        
        # Навигация назад
        elif choice == "portfolio":
            return await portfolio_command(update, context)
        elif choice == "settings":
            return await settings_command(update, context)
        elif choice == "instrument_back":
            return await start_pro_calculation(update, context)
        elif choice == "direction_back":
            return await handle_instrument_type(update, context)
        elif choice == "deposit_back":
            return await handle_direction(update, context)
        elif choice == "leverage_back":
            return await handle_pro_deposit(update, context)
        elif choice == "risk_back":
            return await handle_pro_leverage(update, context)
        elif choice == "entry_back":
            return await handle_pro_risk(update, context)
        elif choice == "stoploss_back":
            return await handle_pro_entry(update, context)
        elif choice == "takeprofit_back":
            return await handle_pro_stoploss(update, context)
        elif choice == "calculation_back":
            return await calculate_and_show_results(update, context)
        
        return MAIN_MENU
        
    except Exception as e:
        logger.error(f"Ошибка в handle_main_menu: {e}")
        return await start(update, context)

# Основная функция
def main():
    """Запуск бота v3.0"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ Токен бота не найден!")
        return

    logger.info("🚀 Запуск ПРОФЕССИОНАЛЬНОГО калькулятора рисков v3.0...")
    
    # Запускаем автосохранение
    DataManager.auto_save()
    
    application = Application.builder().token(token).build()

    # Расширенный обработчик диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MAIN_MENU: [CallbackQueryHandler(handle_main_menu)],
            SETTINGS_MENU: [CallbackQueryHandler(handle_main_menu)],
            PORTFOLIO_MENU: [CallbackQueryHandler(handle_main_menu)],
            
            # Портфель
            DEPOSIT_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_deposit_amount)],
            ADD_TRADE_INSTRUMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_trade_instrument)],
            ADD_TRADE_DIRECTION: [CallbackQueryHandler(handle_main_menu)],
            ADD_TRADE_ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_trade_entry)],
            ADD_TRADE_EXIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_trade_exit)],
            ADD_TRADE_VOLUME: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_trade_volume)],
            ADD_TRADE_PROFIT: [CallbackQueryHandler(handle_main_menu)],
            
            # Профессиональный расчет
            INSTRUMENT_TYPE: [CallbackQueryHandler(handle_main_menu)],
            CUSTOM_INSTRUMENT: [
                CallbackQueryHandler(handle_main_menu),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_instrument_input)
            ],
            DIRECTION: [CallbackQueryHandler(handle_main_menu)],
            PRO_DEPOSIT: [
                CallbackQueryHandler(handle_main_menu),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pro_deposit)
            ],
            PRO_LEVERAGE: [CallbackQueryHandler(handle_main_menu)],
            PRO_RISK: [CallbackQueryHandler(handle_main_menu)],
            PRO_ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pro_entry)],
            PRO_STOPLOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pro_stoploss)],
            PRO_TAKEPROFIT: [
                CallbackQueryHandler(handle_main_menu),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pro_takeprofit)
            ],
            PRO_VOLUME: [CallbackQueryHandler(handle_main_menu)],
            STRATEGY_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_strategy_name)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # Регистрируем обработчики
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('info', pro_info_command))
    application.add_handler(CommandHandler('help', pro_info_command))
    application.add_handler(CommandHandler('portfolio', portfolio_command))
    application.add_handler(CommandHandler('quick', start_quick_calculation))
    application.add_handler(CommandHandler('settings', settings_command))
    application.add_handler(CommandHandler('cancel', cancel))

    # Обработчик для неизвестных команд
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    
    # Запускаем бота
    port = int(os.environ.get('PORT', 10000))
    webhook_url = os.getenv('RENDER_EXTERNAL_URL', '')
    
    logger.info(f"🌐 PRO v3.0 запускается на порту {port}")
    
    try:
        if webhook_url and "render.com" in webhook_url:
            logger.info(f"🔗 PRO Webhook URL: {webhook_url}/webhook")
            application.run_webhook(
                listen="0.0.0.0",
                port=port,
                url_path="/webhook",
                webhook_url=webhook_url + "/webhook"
            )
        else:
            logger.info("🔄 PRO запускается в режиме polling...")
            application.run_polling()
    except Exception as e:
        logger.error(f"❌ Ошибка запуска PRO бота: {e}")

if __name__ == '__main__':
    main()
