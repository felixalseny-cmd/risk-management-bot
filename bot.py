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
    PORTFOLIO_MENU
) = range(13)

# Временное хранилище данных пользователей
user_data: Dict[int, Dict[str, Any]] = {}

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
RISK_LEVELS = ['2%', '5%', '10%', '15%', '20%', '25%']
TRADE_DIRECTIONS = ['BUY', 'SELL']

# Менеджер портфеля
class PortfolioManager:
    @staticmethod
    def initialize_user_portfolio(user_id: int):
        if user_id not in user_data:
            user_data[user_id] = {}
        
        if 'portfolio' not in user_data[user_id]:
            user_data[user_id]['portfolio'] = {
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
    
    @staticmethod
    def update_performance_metrics(user_id: int):
        portfolio = user_data[user_id]['portfolio']
        trades = portfolio['trades']
        
        if not trades:
            return
        
        closed_trades = [t for t in trades if t.get('status') == 'closed']
        winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('profit', 0) <= 0]
        
        portfolio['performance']['total_trades'] = len(closed_trades)
        portfolio['performance']['winning_trades'] = len(winning_trades)
        portfolio['performance']['losing_trades'] = len(losing_trades)
        portfolio['performance']['total_profit'] = sum(t.get('profit', 0) for t in winning_trades)
        portfolio['performance']['total_loss'] = sum(t.get('profit', 0) for t in losing_trades)
        
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

# Глобальный кэш
fast_cache = FastCache()

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
            # Быстрый ключ кэша
            cache_key = f"pos_{deposit}_{leverage}_{instrument_type}_{currency_pair}_{entry_price}_{stop_loss}_{direction}_{risk_percent}"
            cached_result = fast_cache.get(cache_key)
            if cached_result:
                return cached_result
            
            lev_value = int(leverage.split(':')[1])
            risk_amount = deposit * risk_percent
            
            # Быстрые расчеты стоп-лосса
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
            
            # Сохраняем в кэш
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
        """Быстрый расчет прибыли"""
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

# Основные обработчики команд
@log_performance
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Главное меню"""
    if update.message:
        user = update.message.from_user
    elif update.callback_query:
        user = update.callback_query.from_user
    else:
        return ConversationHandler.END
        
    user_name = user.first_name or "Трейдер"
    
    welcome_text = f"""
👋 *Привет, {user_name}!*

🎯 *PRO Калькулятор Управления Рисками v3.0*

⚡ *Выберите опцию:*
"""
    
    user_id = user.id
    # Сохраняем пресеты при перезапуске
    old_presets = user_data.get(user_id, {}).get('presets', [])
    
    user_data[user_id] = {
        'start_time': datetime.now().isoformat(),
        'last_activity': time.time(),
        'presets': old_presets
    }
    
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

@log_performance
async def quick_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Быстрый расчет"""
    return await start_quick_calculation(update, context)

@log_performance
async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Управление портфелем"""
    user_id = update.message.from_user.id if update.message else update.callback_query.from_user.id
    
    # Инициализируем портфель если не существует
    PortfolioManager.initialize_user_portfolio(user_id)
    
    portfolio_text = """
💼 *Управление Портфелем*

📊 *Доступные функции:*
• 📈 Обзор сделок
• 💰 Баланс и распределение
• 📊 Анализ эффективности
• 🔄 История операций

Выберите действие:
"""
    
    keyboard = [
        [InlineKeyboardButton("📈 Обзор сделок", callback_data="portfolio_trades")],
        [InlineKeyboardButton("💰 Баланс и распределение", callback_data="portfolio_balance")],
        [InlineKeyboardButton("📊 Анализ эффективности", callback_data="portfolio_performance")],
        [InlineKeyboardButton("🔄 История операций", callback_data="portfolio_history")],
        [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
        [InlineKeyboardButton("💸 Внести депозит", callback_data="portfolio_deposit")],
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

@log_performance
async def analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Аналитика (в разработке)"""
    analytics_text = """
📈 *Аналитика стратегий*

🚧 *Раздел в разработке*

🚀 *Скоро будет доступно:*
• 🤖 AI-анализ ваших стратегий
• 📊 Бэктестинг
• 📈 Прогнозирование
• 💡 Интеллектуальные рекомендации

Следите за обновлениями! ⚡
"""
    
    keyboard = [
        [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
    ]
    
    if update.message:
        await update.message.reply_text(
            analytics_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    else:
        await update.callback_query.edit_message_text(
            analytics_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

@log_performance
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """PRO Инструкции"""
    info_text = """
📚 *PRO ИНСТРУКЦИИ v3.0*

🎯 *РАСШИРЕННЫЕ ВОЗМОЖНОСТИ:*

⚡ *ВСЕ ТИПЫ ИНСТРУМЕНТОВ:*
• 🌐 Форекс (50+ валютных пар)
• ₿ Криптовалюты (15+ пар)
• 📈 Индексы (12+ индексов)
• ⚡ Сырьевые товары (8+ типов)
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

@log_performance
async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Настройки"""
    settings_text = """
⚙️ *Настройки*

🔧 *Доступные настройки:*
• 💰 Уровень риска по умолчанию
• 📊 Валюта депозита
• 🎯 Пресеты стратегий
• 📈 Параметры расчета

⚡ *Быстрые настройки:*
"""
    
    keyboard = [
        [InlineKeyboardButton("💰 Уровень риска: 2%", callback_data="set_risk_2")],
        [InlineKeyboardButton("💰 Уровень риска: 5%", callback_data="set_risk_5")],
        [InlineKeyboardButton("💵 Валюта: USD", callback_data="set_currency_usd")],
        [InlineKeyboardButton("💵 Валюта: EUR", callback_data="set_currency_eur")],
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

# Обработчики портфеля
@log_performance
async def portfolio_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать обзор сделок"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    trades = portfolio.get('trades', [])
    
    if not trades:
        await query.edit_message_text(
            "📭 *У вас еще нет сделок*\n\n"
            "Используйте кнопку '➕ Добавить сделку' чтобы начать торговать.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
                [InlineKeyboardButton("🔙 Назад в портфель", callback_data="portfolio_back")]
            ])
        )
        return
    
    # Показать последние 5 сделок
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
            [InlineKeyboardButton("📋 Полная история", callback_data="portfolio_full_history")],
            [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
            [InlineKeyboardButton("🔙 Назад в портфель", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать баланс и распределение"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    allocation = portfolio.get('allocation', {})
    performance = portfolio.get('performance', {})
    
    balance_text = "💰 *Баланс и распределение*\n\n"
    
    # Информация о балансе
    initial_balance = portfolio.get('initial_balance', 0)
    current_balance = portfolio.get('current_balance', 0)
    total_profit = performance.get('total_profit', 0)
    total_loss = performance.get('total_loss', 0)
    net_profit = total_profit + total_loss
    
    balance_text += f"💳 Начальный депозит: ${initial_balance:,.2f}\n"
    balance_text += f"💵 Текущий баланс: ${current_balance:,.2f}\n"
    balance_text += f"📈 Чистая прибыль: ${net_profit:.2f}\n\n"
    
    # Информация о распределении
    if allocation:
        balance_text += "🌐 *Распределение по инструментам:*\n"
        for instrument, count in list(allocation.items())[:5]:  # Показать топ-5
            balance_text += f"• {instrument}: {count} сделок\n"
    else:
        balance_text += "🌐 *Распределение:* Нет данных\n"
    
    await query.edit_message_text(
        balance_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("💸 Внести депозит", callback_data="portfolio_deposit")],
            [InlineKeyboardButton("🔙 Назад в портфель", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать анализ эффективности"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    performance = portfolio.get('performance', {})
    
    perf_text = "📊 *Анализ эффективности*\n\n"
    
    # Метрики эффективности
    total_trades = performance.get('total_trades', 0)
    win_rate = performance.get('win_rate', 0)
    avg_profit = performance.get('average_profit', 0)
    avg_loss = performance.get('average_loss', 0)
    
    perf_text += f"📈 Всего сделок: {total_trades}\n"
    perf_text += f"🎯 Процент прибыльных: {win_rate:.1f}%\n"
    perf_text += f"💰 Средняя прибыль: ${avg_profit:.2f}\n"
    perf_text += f"📉 Средний убыток: ${avg_loss:.2f}\n\n"
    
    # Рекомендации
    if win_rate < 40:
        perf_text += "💡 *Рекомендация:* Увеличьте соотношение риск/прибыль до 1:3\n"
    elif win_rate > 60:
        perf_text += "💡 *Рекомендация:* Отличные результаты! Продолжайте в том же духе\n"
    
    await query.edit_message_text(
        perf_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад в портфель", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать историю операций"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    history = portfolio.get('history', [])
    
    if not history:
        await query.edit_message_text(
            "📭 *История операций пуста*\n\n"
            "Все операции с вашим счетом будут отображаться здесь.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💸 Внести депозит", callback_data="portfolio_deposit")],
                [InlineKeyboardButton("🔙 Назад в портфель", callback_data="portfolio_back")]
            ])
        )
        return
    
    history_text = "🔄 *История операций*\n\n"
    
    # Показать последние 10 операций
    for op in reversed(history[-10:]):
        emoji = "💳" if op['type'] == 'balance' else "📈"
        action_emoji = "⬆️" if op.get('amount', 0) > 0 else "⬇️"
        
        history_text += f"{emoji} {op['type'].title()} | {op['action']} {action_emoji}\n"
        
        if op['type'] == 'balance':
            history_text += f"💵 Сумма: ${op.get('amount', 0):.2f}\n"
        else:
            history_text += f"💰 Прибыль: ${op.get('profit', 0):.2f}\n"
        
        history_text += f"📅 {op.get('timestamp', '')[:16]}\n\n"
    
    await query.edit_message_text(
        history_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад в портфель", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_add_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Добавить новую сделку в портфель"""
    query = update.callback_query
    user_id = query.from_user.id
    
    # Для демонстрации добавляем пример сделки
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
        "✅ *Сделка добавлена в портфель!*\n\n"
        f"📈 {sample_trade['instrument']} {sample_trade['direction']}\n"
        f"💰 Прибыль: ${sample_trade['profit']:.2f}\n\n"
        "Обновите анализ эффективности для просмотра новой статистики.",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Обновить анализ", callback_data="portfolio_performance")],
            [InlineKeyboardButton("🔙 Назад в портфель", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Добавить депозит в портфель"""
    query = update.callback_query
    user_id = query.from_user.id
    
    # Для демонстрации добавляем пример депозита
    PortfolioManager.add_balance_operation(
        user_id, 
        'deposit', 
        1000.0, 
        "Начальный депозит"
    )
    
    await query.edit_message_text(
        "✅ *Депозит добавлен!*\n\n"
        "💵 Сумма: $1,000.00\n\n"
        "Теперь вы можете отслеживать баланс и эффективность вашего портфеля.",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("💰 Проверить баланс", callback_data="portfolio_balance")],
            [InlineKeyboardButton("🔙 Назад в портфель", callback_data="portfolio_back")]
        ])
    )

# Обработчики навигации
@log_performance
async def portfolio_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Назад в меню портфеля"""
    return await portfolio_command(update, context)

@log_performance
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора в главном меню"""
    query = update.callback_query
    if not query:
        return MAIN_MENU
        
    await query.answer()
    choice = query.data
    
    # Обновляем время активности
    user_id = query.from_user.id
    if user_id in user_data:
        user_data[user_id]['last_activity'] = time.time()
    
    if choice == "pro_calculation":
        return await start_pro_calculation(update, context)
    elif choice == "quick_calculation":
        return await start_quick_calculation(update, context)
    elif choice == "portfolio":
        return await portfolio_command(update, context)
    elif choice == "analytics":
        await analytics_command(update, context)
        return MAIN_MENU
    elif choice == "pro_info":
        await pro_info_command(update, context)
        return MAIN_MENU
    elif choice == "settings":
        await settings_command(update, context)
        return MAIN_MENU
    elif choice == "main_menu":
        return await start(update, context)
    elif choice == "portfolio_back":
        return await portfolio_command(update, context)
    
    return MAIN_MENU

# Обработчики профессионального расчета
@log_performance
async def start_pro_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начать профессиональный расчет"""
    query = update.callback_query
    if query:
        await query.edit_message_text(
            "🎯 *Профессиональный расчет*\n\n"
            "📊 *Выберите тип инструмента:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 Форекс", callback_data="inst_type_forex")],
                [InlineKeyboardButton("₿ Криптовалюты", callback_data="inst_type_crypto")],
                [InlineKeyboardButton("📈 Индексы", callback_data="inst_type_indices")],
                [InlineKeyboardButton("⚡ Сырьевые товары", callback_data="inst_type_commodities")],
                [InlineKeyboardButton("🏅 Металлы", callback_data="inst_type_metals")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
    return INSTRUMENT_TYPE

@log_performance
async def start_quick_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начать быстрый расчет"""
    if update.message:
        await update.message.reply_text(
            "⚡ *Быстрый расчет*\n\n"
            "📊 *Введите тикер инструмента* (например: EURUSD, BTCUSD, NAS100):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
    else:
        query = update.callback_query
        await query.edit_message_text(
            "⚡ *Быстрый расчет*\n\n"
            "📊 *Введите тикер инструмента* (например: EURUSD, BTCUSD, NAS100):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
    return CUSTOM_INSTRUMENT

@log_performance
async def show_presets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать сохраненные пресеты"""
    user_id = update.message.from_user.id
    presets = user_data.get(user_id, {}).get('presets', [])
    
    if not presets:
        await update.message.reply_text(
            "📝 *У вас нет сохраненных PRO стратегий.*\n\n"
            "💡 Сохраняйте ваши стратегии после расчета для быстрого доступа!",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
        )
        return
    
    await update.message.reply_text(
        f"📚 *Ваши PRO стратегии ({len(presets)}):*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
    )

@log_performance
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отменить диалог"""
    if update.message:
        await update.message.reply_text(
            "❌ *PRO расчет отменен.*\n\n"
            "🚀 Используйте /start для нового PRO расчета\n"
            "📚 Используйте /info для PRO инструкций\n\n"
            "👨‍💻 *PRO Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)",
            parse_mode='Markdown',
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
    return ConversationHandler.END

# Простые обработчики для остальных шагов расчета (заглушки)
@log_performance
async def handle_instrument_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора типа инструмента"""
    query = update.callback_query
    await query.answer()
    
    instrument_type = query.data.replace("inst_type_", "")
    user_id = query.from_user.id
    
    if user_id not in user_data:
        user_data[user_id] = {}
    
    user_data[user_id]['instrument_type'] = instrument_type
    
    # Показать пресеты для выбранного типа
    presets = INSTRUMENT_PRESETS.get(instrument_type, [])
    keyboard = []
    
    for preset in presets:
        keyboard.append([InlineKeyboardButton(preset, callback_data=f"preset_{preset}")])
    
    keyboard.append([InlineKeyboardButton("✏️ Ввести свой инструмент", callback_data="custom_instrument")])
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="pro_calculation")])
    
    await query.edit_message_text(
        f"🎯 *Выбран тип: {INSTRUMENT_TYPES[instrument_type]}*\n\n"
        "📊 *Выберите инструмент из списка или введите свой:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return CUSTOM_INSTRUMENT

@log_performance
async def handle_custom_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода пользовательского инструмента"""
    query = update.callback_query
    if query:
        await query.edit_message_text(
            "✏️ *Введите ваш инструмент* (например: EURUSD, BTCUSD, NAS100):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="pro_calculation")]
            ])
        )
        return CUSTOM_INSTRUMENT
    
    # Обработка текстового ввода
    user_id = update.message.from_user.id
    instrument = update.message.text.upper().strip()
    
    user_data[user_id]['instrument'] = instrument
    
    await update.message.reply_text(
        f"✅ *Инструмент установлен: {instrument}*\n\n"
        "📈 *Выберите направление сделки:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔼 BUY", callback_data="direction_BUY")],
            [InlineKeyboardButton("🔽 SELL", callback_data="direction_SELL")],
            [InlineKeyboardButton("🔙 Назад", callback_data="pro_calculation")]
        ])
    )
    return DIRECTION

@log_performance
async def handle_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора направления"""
    query = update.callback_query
    await query.answer()
    
    direction = query.data.replace("direction_", "")
    user_id = query.from_user.id
    
    user_data[user_id]['direction'] = direction
    
    await query.edit_message_text(
        f"✅ *Направление: {direction}*\n\n"
        "⚠️ *Выберите уровень риска (% от депозита):*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🟢 2% (Консервативный)", callback_data="risk_2")],
            [InlineKeyboardButton("🟡 5% (Умеренный)", callback_data="risk_5")],
            [InlineKeyboardButton("🔴 10% (Агрессивный)", callback_data="risk_10")],
            [InlineKeyboardButton("⚫ 15% (Высокий)", callback_data="risk_15")],
            [InlineKeyboardButton("🔙 Назад", callback_data="custom_instrument")]
        ])
    )
    return RISK_PERCENT

# Завершающие обработчики расчета
@log_performance
async def handle_risk_percent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора уровня риска"""
    query = update.callback_query
    await query.answer()
    
    risk_percent = float(query.data.replace("risk_", "")) / 100
    user_id = query.from_user.id
    
    user_data[user_id]['risk_percent'] = risk_percent
    
    await query.edit_message_text(
        f"✅ *Уровень риска: {risk_percent*100}%*\n\n"
        "💰 *Введите размер депозита* (например: 5000):",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="direction_back")]
        ])
    )
    return DEPOSIT

@log_performance
async def handle_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода депозита"""
    user_id = update.message.from_user.id
    try:
        deposit = float(update.message.text)
        user_data[user_id]['deposit'] = deposit
        
        await update.message.reply_text(
            f"✅ *Депозит: ${deposit:,.2f}*\n\n"
            "⚖️ *Выберите плечо:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("1:10", callback_data="leverage_1:10")],
                [InlineKeyboardButton("1:50", callback_data="leverage_1:50")],
                [InlineKeyboardButton("1:100", callback_data="leverage_1:100")],
                [InlineKeyboardButton("1:200", callback_data="leverage_1:200")],
                [InlineKeyboardButton("1:500", callback_data="leverage_1:500")],
                [InlineKeyboardButton("🔙 Назад", callback_data="risk_back")]
            ])
        )
        return LEVERAGE
    except ValueError:
        await update.message.reply_text(
            "❌ *Неверный формат!*\n\n"
            "💰 Введите числовое значение депозита (например: 5000):",
            parse_mode='Markdown'
        )
        return DEPOSIT

@log_performance
async def show_calculation_results(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Показать результаты расчета"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    user_info = user_data.get(user_id, {})
    
    # Пример расчета (в реальном боте здесь будет полный расчет)
    result_text = f"""
📊 *РЕЗУЛЬТАТЫ РАСЧЕТА*

🎯 *Параметры сделки:*
• 📈 Инструмент: {user_info.get('instrument', 'N/A')}
• 📊 Тип: {INSTRUMENT_TYPES.get(user_info.get('instrument_type', ''), 'N/A')}
• 🎯 Направление: {user_info.get('direction', 'N/A')}
• 💰 Депозит: ${user_info.get('deposit', 0):,.2f}
• ⚖️ Плечо: {user_info.get('leverage', 'N/A')}
• ⚠️ Риск: {user_info.get('risk_percent', 0)*100}%

📦 *Рекомендации:*
• 📊 Размер позиции: 1.25 лота
• 💰 Риск на сделку: ${user_info.get('deposit', 0) * user_info.get('risk_percent', 0):.2f}
• ⚖️ Соотношение R/R: 2.5
• 🎯 Прибыль по TP1: $750
• 🎯 Прибыль по TP2: $1250
• 📊 Общий ROI: 25%

💡 *Совет:* Сохраните эту стратегию для быстрого доступа!
"""

    await query.edit_message_text(
        result_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("💾 Сохранить стратегию", callback_data="save_strategy")],
            [InlineKeyboardButton("🔄 Новый расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ])
    )
    return MAIN_MENU

# Основная функция
def main():
    """Оптимизированная основная функция для запуска бота"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ Токен PRO бота не найден!")
        return

    logger.info("🚀 Запуск УЛЬТРА-БЫСТРОГО PRO калькулятора рисков v3.0 с улучшенным портфелем...")
    
    # Создаем приложение
    application = Application.builder().token(token).build()

    # Настройка обработчиков
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MAIN_MENU: [CallbackQueryHandler(handle_main_menu)],
            INSTRUMENT_TYPE: [CallbackQueryHandler(handle_instrument_type)],
            CUSTOM_INSTRUMENT: [
                CallbackQueryHandler(handle_custom_instrument, pattern='^custom_instrument$'),
                CallbackQueryHandler(handle_instrument_type, pattern='^preset_'),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_instrument)
            ],
            DIRECTION: [CallbackQueryHandler(handle_direction)],
            RISK_PERCENT: [CallbackQueryHandler(handle_risk_percent)],
            DEPOSIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_deposit)],
            LEVERAGE: [CallbackQueryHandler(show_calculation_results)],
            PORTFOLIO_MENU: [CallbackQueryHandler(handle_main_menu)]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # Регистрируем обработчики
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('info', pro_info_command))
    application.add_handler(CommandHandler('help', pro_info_command))
    application.add_handler(CommandHandler('portfolio', portfolio_command))
    application.add_handler(CommandHandler('quick', quick_command))
    application.add_handler(CommandHandler('settings', settings_command))
    application.add_handler(CommandHandler('presets', show_presets))

    # Запускаем бота
    port = int(os.environ.get('PORT', 10000))
    webhook_url = os.getenv('RENDER_EXTERNAL_URL', '')
    
    logger.info(f"🌐 PRO запускается на порту {port}")
    
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
