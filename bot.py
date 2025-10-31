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
    DEPOSIT_AMOUNT, WITHDRAW_AMOUNT, SETTINGS_MENU, SAVE_STRATEGY_NAME
) = range(23)

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
    'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9, 'USDCHF': 10,
    'USDCAD': 10, 'AUDUSD': 10, 'NZDUSD': 10, 'EURGBP': 10,
    'EURJPY': 9, 'GBPJPY': 9, 'EURCHF': 10, 'AUDJPY': 9,
    'BTCUSD': 1, 'ETHUSD': 1, 'XRPUSD': 10, 'ADAUSD': 10,
    'DOTUSD': 1, 'LTCUSD': 1, 'BCHUSD': 1, 'LINKUSD': 1,
    'US30': 1, 'NAS100': 1, 'SPX500': 1, 'DAX40': 1,
    'FTSE100': 1, 'NIKKEI225': 1, 'ASX200': 1,
    'OIL': 10, 'NATGAS': 10, 'COPPER': 10,
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

# Генератор PDF отчетов
class PDFReportGenerator:
    @staticmethod
    def generate_portfolio_report(user_id: int) -> str:
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

# Основные обработчики команд
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

⚡ *Выберите опцию:*
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
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отменить диалог"""
    try:
        await update.message.reply_text(
            "❌ *Операция отменена.*\n\n"
            "🚀 Используйте /start для нового расчета",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"Ошибка в cancel: {e}")
        return ConversationHandler.END

@log_performance
async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Управление портфелем"""
    try:
        if update.message:
            user_id = update.message.from_user.id
        else:
            user_id = update.callback_query.from_user.id
            await update.callback_query.answer()
        
        PortfolioManager.initialize_user_portfolio(user_id)
        
        portfolio = user_data[user_id]['portfolio']
        current_balance = portfolio['current_balance']
        
        portfolio_text = f"""
💼 *Управление Портфелем*

💰 *Текущий баланс:* ${current_balance:,.2f}
📊 *Всего сделок:* {len(portfolio['trades'])}

🎯 *Доступные функции:*
"""
        
        keyboard = [
            [InlineKeyboardButton("📈 Обзор сделок", callback_data="portfolio_trades")],
            [InlineKeyboardButton("💰 Баланс", callback_data="portfolio_balance")],
            [InlineKeyboardButton("📊 Анализ эффективности", callback_data="portfolio_performance")],
            [InlineKeyboardButton("📄 Сгенерировать отчет", callback_data="portfolio_report")],
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
    except Exception as e:
        logger.error(f"Ошибка в portfolio_command: {e}")
        return await start(update, context)

@log_performance
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """PRO Инструкции"""
    try:
        info_text = """
📚 *PRO ИНСТРУКЦИИ v3.0*

🎯 *ЧТО УМЕЕТ БОТ:*
• 📊 Профессиональный расчет рисков
• 💼 Управление портфелем и аналитика
• ⚡ Быстрые расчеты позиций
• 📄 Генерация отчетов
• 💰 Трекинг сделок и баланса

🚀 *ИНСТРУКЦИЯ:*
1. Используйте кнопки для навигации
2. Для расчетов вводите числовые значения
3. Сохраняйте свои сделки для аналитики
4. Следите за рекомендациями системы

👨‍💻 *Разработчик:* @fxfeelgood
"""
        if update.message:
            await update.message.reply_text(
                info_text, 
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
            )
        else:
            await update.callback_query.edit_message_text(
                info_text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
            )
    except Exception as e:
        logger.error(f"Ошибка в pro_info_command: {e}")

@log_performance
async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Настройки"""
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
"""
        
        keyboard = [
            [InlineKeyboardButton(f"💰 Уровень риска: {settings['default_risk']*100}%", callback_data="change_risk")],
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
async def save_risk_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сохранение уровня риска"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        risk_level = float(query.data.replace("set_risk_", ""))
        user_data[user_id]['portfolio']['settings']['default_risk'] = risk_level
        
        await query.edit_message_text(
            f"✅ *Уровень риска установлен: {risk_level*100}%*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в save_risk_setting: {e}")

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
        try:
            amount = float(update.message.text)
            
            if amount <= 0:
                await update.message.reply_text(
                    "❌ *Сумма должна быть положительной!*\n\n"
                    "💰 Введите сумму депозита:",
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
            
        except ValueError:
            await update.message.reply_text(
                "❌ *Неверный формат!*\n\n"
                "💰 Введите числовое значение суммы депозита:",
                parse_mode='Markdown'
            )
            return DEPOSIT_AMOUNT
    except Exception as e:
        logger.error(f"Ошибка в handle_deposit_amount: {e}")

@log_performance
async def portfolio_add_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Начало добавления сделки"""
    try:
        query = update.callback_query
        await query.answer()
        
        await query.edit_message_text(
            "📈 *Добавление новой сделки*\n\n"
            "💎 *Введите тикер инструмента* (например: EURUSD, BTCUSD):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]
            ])
        )
        return ADD_TRADE_INSTRUMENT
    except Exception as e:
        logger.error(f"Ошибка в portfolio_add_trade_start: {e}")

@log_performance
async def handle_trade_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода инструмента для сделки"""
    try:
        user_id = update.message.from_user.id
        instrument = update.message.text.upper().strip()
        
        context.user_data['new_trade'] = {'instrument': instrument}
        
        await update.message.reply_text(
            f"✅ *Инструмент: {instrument}*\n\n"
            "📊 *Выберите направление сделки:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔼 BUY", callback_data="trade_direction_BUY")],
                [InlineKeyboardButton("🔽 SELL", callback_data="trade_direction_SELL")],
                [InlineKeyboardButton("🔙 Назад", callback_data="portfolio_add_trade")]
            ])
        )
        return ADD_TRADE_DIRECTION
    except Exception as e:
        logger.error(f"Ошибка в handle_trade_instrument: {e}")

@log_performance
async def handle_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора направления сделки"""
    try:
        query = update.callback_query
        await query.answer()
        
        direction = query.data.replace("trade_direction_", "")
        context.user_data['new_trade']['direction'] = direction
        
        await query.edit_message_text(
            f"✅ *Направление: {direction}*\n\n"
            "💰 *Введите цену входа:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="portfolio_add_trade")]
            ])
        )
        return ADD_TRADE_ENTRY
    except Exception as e:
        logger.error(f"Ошибка в handle_trade_direction: {e}")

@log_performance
async def handle_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода цены входа"""
    try:
        user_id = update.message.from_user.id
        try:
            entry_price = float(update.message.text)
            context.user_data['new_trade']['entry_price'] = entry_price
            
            await update.message.reply_text(
                f"✅ *Цена входа: {entry_price}*\n\n"
                "🛑 *Введите цену выхода:*",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="trade_direction_back")]
                ])
            )
            return ADD_TRADE_EXIT
        except ValueError:
            await update.message.reply_text(
                "❌ *Неверный формат цены!*\n\n"
                "💰 Введите числовое значение цены входа:",
                parse_mode='Markdown'
            )
            return ADD_TRADE_ENTRY
    except Exception as e:
        logger.error(f"Ошибка в handle_trade_entry: {e}")

@log_performance
async def handle_trade_exit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода цены выхода"""
    try:
        user_id = update.message.from_user.id
        try:
            exit_price = float(update.message.text)
            context.user_data['new_trade']['exit_price'] = exit_price
            
            await update.message.reply_text(
                f"✅ *Цена выхода: {exit_price}*\n\n"
                "📊 *Введите объем в лотах:*",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="trade_entry_back")]
                ])
            )
            return ADD_TRADE_VOLUME
        except ValueError:
            await update.message.reply_text(
                "❌ *Неверный формат цены!*\n\n"
                "🛑 Введите числовое значение цены выхода:",
                parse_mode='Markdown'
            )
            return ADD_TRADE_EXIT
    except Exception as e:
        logger.error(f"Ошибка в handle_trade_exit: {e}")

@log_performance
async def handle_trade_volume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода объема"""
    try:
        user_id = update.message.from_user.id
        try:
            volume = float(update.message.text)
            context.user_data['new_trade']['volume'] = volume
            
            entry = context.user_data['new_trade']['entry_price']
            exit_price = context.user_data['new_trade']['exit_price']
            direction = context.user_data['new_trade']['direction']
            
            if direction == 'BUY':
                profit = (exit_price - entry) * volume * 10000
            else:
                profit = (entry - exit_price) * volume * 10000
            
            context.user_data['new_trade']['profit'] = profit
            
            await update.message.reply_text(
                f"✅ *Объем: {volume} лотов*\n\n"
                f"💰 *Расчетная прибыль: ${profit:.2f}*\n\n"
                "Нажмите подтвердить для сохранения:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("✅ Подтвердить сделку", callback_data="confirm_trade")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="trade_exit_back")]
                ])
            )
            return ADD_TRADE_PROFIT
        except ValueError:
            await update.message.reply_text(
                "❌ *Неверный формат объема!*\n\n"
                "📊 Введите числовое значение объема:",
                parse_mode='Markdown'
            )
            return ADD_TRADE_VOLUME
    except Exception as e:
        logger.error(f"Ошибка в handle_trade_volume: {e}")

@log_performance
async def confirm_trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Подтверждение и сохранение сделки"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        trade_data = context.user_data['new_trade']
        trade_data['status'] = 'closed'
        
        PortfolioManager.add_trade(user_id, trade_data)
        
        context.user_data.pop('new_trade', None)
        
        await query.edit_message_text(
            f"✅ *Сделка успешно добавлена!*\n\n"
            f"📈 *Детали сделки:*\n"
            f"• 💎 Инструмент: {trade_data['instrument']}\n"
            f"• 📊 Направление: {trade_data['direction']}\n"
            f"• 💰 Прибыль: ${trade_data['profit']:.2f}\n"
            f"• 📅 Время: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Анализ эффективности", callback_data="portfolio_performance")],
                [InlineKeyboardButton("💼 Портфель", callback_data="portfolio")]
            ])
        )
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"Ошибка в confirm_trade: {e}")

@log_performance
async def start_pro_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало профессионального расчета"""
    try:
        query = update.callback_query
        await query.answer()
        
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
async def start_quick_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало быстрого расчета"""
    try:
        query = update.callback_query
        await query.answer()
        
        await query.edit_message_text(
            "⚡ *БЫСТРЫЙ РАСЧЕТ РИСКОВ*\n\n"
            "💰 *Введите сумму депозита:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return DEPOSIT
    except Exception as e:
        logger.error(f"Ошибка в start_quick_calculation: {e}")
        return MAIN_MENU

@log_performance
async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка неизвестных команд"""
    try:
        await update.message.reply_text(
            "❌ *Неизвестная команда*\n\n"
            "🎯 Используйте /start для начала работы",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🚀 Начать", callback_data="main_menu")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в unknown_command: {e}")

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
        elif choice == "change_risk":
            await change_risk_setting(update, context)
            return SETTINGS_MENU
        elif choice.startswith("set_risk_"):
            await save_risk_setting(update, context)
            return SETTINGS_MENU
        elif choice == "portfolio_deposit":
            return await portfolio_deposit_menu(update, context)
        elif choice == "portfolio_add_trade":
            return await portfolio_add_trade_start(update, context)
        elif choice.startswith("trade_direction_"):
            return await handle_trade_direction(update, context)
        elif choice == "confirm_trade":
            return await confirm_trade(update, context)
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
        elif choice == "portfolio":
            return await portfolio_command(update, context)
        
        return MAIN_MENU
    except Exception as e:
        logger.error(f"Ошибка в handle_main_menu: {e}")
        return await start(update, context)

# Основная функция
def main():
    """Запуск бота"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ Токен бота не найден!")
        return

    logger.info("🚀 Запуск ПРОФЕССИОНАЛЬНОГО калькулятора рисков v3.0...")
    
    application = Application.builder().token(token).build()

    # Обработчик диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MAIN_MENU: [CallbackQueryHandler(handle_main_menu)],
            SETTINGS_MENU: [CallbackQueryHandler(handle_main_menu)],
            PORTFOLIO_MENU: [CallbackQueryHandler(handle_main_menu)],
            DEPOSIT_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_deposit_amount)],
            ADD_TRADE_INSTRUMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_trade_instrument)],
            ADD_TRADE_DIRECTION: [CallbackQueryHandler(handle_main_menu)],
            ADD_TRADE_ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_trade_entry)],
            ADD_TRADE_EXIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_trade_exit)],
            ADD_TRADE_VOLUME: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_trade_volume)],
            ADD_TRADE_PROFIT: [CallbackQueryHandler(handle_main_menu)],
            INSTRUMENT_TYPE: [CallbackQueryHandler(handle_main_menu)],
            DEPOSIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_deposit_amount)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # Регистрируем обработчики
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('info', pro_info_command))
    application.add_handler(CommandHandler('help', pro_info_command))
    application.add_handler(CommandHandler('portfolio', portfolio_command))
    application.add_handler(CommandHandler('settings', settings_command))
    application.add_handler(CommandHandler('cancel', cancel))

    # Обработчик для неизвестных команд
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    
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
