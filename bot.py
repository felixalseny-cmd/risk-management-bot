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

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
(
    MAIN_MENU, INSTRUMENT_TYPE, CUSTOM_INSTRUMENT, DIRECTION, 
    RISK_PERCENT, DEPOSIT, LEVERAGE, CURRENCY, ENTRY, 
    STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION,
    PORTFOLIO_MENU, ANALYTICS_MENU, TRADE_HISTORY, PERFORMANCE_ANALYSIS
) = range(16)

# Temporary storage
user_data: Dict[int, Dict[str, Any]] = {}

# Portfolio Data Management
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
        
        # Update performance metrics
        PortfolioManager.update_performance_metrics(user_id)
        
        # Update allocation
        instrument = trade_data.get('instrument', 'Unknown')
        if instrument not in user_data[user_id]['portfolio']['allocation']:
            user_data[user_id]['portfolio']['allocation'][instrument] = 0
        user_data[user_id]['portfolio']['allocation'][instrument] += 1
        
        # Add to history
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
        
        # Simple risk score based on profit stability
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

# Ultra-fast cache manager (existing)
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

# Constants (existing)
INSTRUMENT_TYPES = {
    'forex': 'Forex',
    'crypto': 'Криптовалюты', 
    'indices': 'Индексы',
    'commodities': 'Товары',
    'metals': 'Металлы'
}

PIP_VALUES = {
    # ... (existing PIP_VALUES dictionary)
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

# Ultra-fast risk calculator (existing)
class FastRiskCalculator:
    # ... (existing FastRiskCalculator implementation)
    pass

# Performance logging decorator (existing)
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

# Enhanced Portfolio Management
@log_performance
async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Enhanced Portfolio Management"""
    user_id = update.message.from_user.id if update.message else update.callback_query.from_user.id
    
    # Initialize portfolio if not exists
    PortfolioManager.initialize_user_portfolio(user_id)
    
    portfolio_text = """
💼 *Управление Портфелем*

📊 *Доступные функции:*
• 📈 Обзор всех сделок
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
async def portfolio_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display trade overview"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    trades = portfolio.get('trades', [])
    
    if not trades:
        await query.edit_message_text(
            "📭 *У вас пока нет сделок*\n\n"
            "Используйте кнопку '➕ Добавить сделку' для начала торговли.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
                [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
            ])
        )
        return
    
    # Display last 5 trades
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
            [InlineKeyboardButton("📋 Вся история", callback_data="portfolio_full_history")],
            [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
            [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display balance and allocation"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    allocation = portfolio.get('allocation', {})
    performance = portfolio.get('performance', {})
    
    balance_text = "💰 *Баланс и распределение*\n\n"
    
    # Balance information
    initial_balance = portfolio.get('initial_balance', 0)
    current_balance = portfolio.get('current_balance', 0)
    total_profit = performance.get('total_profit', 0)
    total_loss = performance.get('total_loss', 0)
    net_profit = total_profit + total_loss
    
    balance_text += f"💳 Начальный депозит: ${initial_balance:,.2f}\n"
    balance_text += f"💵 Текущий баланс: ${current_balance:,.2f}\n"
    balance_text += f"📈 Чистая прибыль: ${net_profit:,.2f}\n\n"
    
    # Allocation information
    if allocation:
        balance_text += "🌐 *Распределение по инструментам:*\n"
        for instrument, count in list(allocation.items())[:5]:  # Show top 5
            balance_text += f"• {instrument}: {count} сделок\n"
    else:
        balance_text += "🌐 *Распределение:* Нет данных\n"
    
    await query.edit_message_text(
        balance_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("💸 Внести депозит", callback_data="portfolio_deposit")],
            [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display performance analysis"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    performance = portfolio.get('performance', {})
    
    perf_text = "📊 *Анализ эффективности*\n\n"
    
    # Performance metrics
    total_trades = performance.get('total_trades', 0)
    win_rate = performance.get('win_rate', 0)
    avg_profit = performance.get('average_profit', 0)
    avg_loss = performance.get('average_loss', 0)
    
    perf_text += f"📈 Всего сделок: {total_trades}\n"
    perf_text += f"🎯 Процент прибыльных: {win_rate:.1f}%\n"
    perf_text += f"💰 Средняя прибыль: ${avg_profit:.2f}\n"
    perf_text += f"📉 Средний убыток: ${avg_loss:.2f}\n\n"
    
    # Risk analysis
    risk_reward_data = AnalyticsEngine.calculate_risk_reward_analysis(
        portfolio.get('trades', [])
    )
    
    perf_text += f"⚡ Соотношение риск/вознаграждение: {risk_reward_data['average_risk_reward']:.2f}\n"
    perf_text += f"🏆 Лучшая сделка: ${risk_reward_data['best_trade']:.2f}\n"
    perf_text += f"🔻 Худшая сделка: ${risk_reward_data['worst_trade']:.2f}\n\n"
    
    # Recommendations
    recommendations = AnalyticsEngine.generate_strategy_recommendations(portfolio)
    if recommendations:
        perf_text += "💡 *Рекомендации:*\n"
        for rec in recommendations[:3]:  # Show top 3 recommendations
            perf_text += f"• {rec}\n"
    
    await query.edit_message_text(
        perf_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 Детальная аналитика", callback_data="analytics_detailed")],
            [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display operation history"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    history = portfolio.get('history', [])
    
    if not history:
        await query.edit_message_text(
            "📭 *История операций пуста*\n\n"
            "Здесь будут отображаться все ваши операции по счету.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💸 Внести депозит", callback_data="portfolio_deposit")],
                [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
            ])
        )
        return
    
    history_text = "🔄 *История операций*\n\n"
    
    # Show last 10 operations
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
            [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_add_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Add a new trade to portfolio"""
    query = update.callback_query
    user_id = query.from_user.id
    
    # For demo purposes, adding a sample trade
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
            [InlineKeyboardButton("📊 Обновить аналитику", callback_data="portfolio_performance")],
            [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
        ])
    )

@log_performance
async def portfolio_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Add deposit to portfolio"""
    query = update.callback_query
    user_id = query.from_user.id
    
    # For demo purposes, adding a sample deposit
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
            [InlineKeyboardButton("🔙 Назад к портфелю", callback_data="portfolio_back")]
        ])
    )

# Enhanced Analytics System
@log_performance
async def analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced Strategy Analytics"""
    user_id = update.message.from_user.id if update.message else update.callback_query.from_user.id
    
    analytics_text = """
📈 *Аналитика Стратегий*

📊 *Доступная аналитика:*
• 📈 Анализ риск/вознаграждение
• 💹 Эффективность стратегий
• 📊 Статистика сделок
• 🔄 Оптимизация параметров

🚀 *Скоро появится:*
• 🤖 AI-анализ стратегий
• 📊 Бэктестинг
• 📈 Прогнозирование
• 💡 Интеллектуальные рекомендации

Выберите тип анализа:
"""
    
    keyboard = [
        [InlineKeyboardButton("📈 Анализ риск/вознаграждение", callback_data="analytics_risk_reward")],
        [InlineKeyboardButton("💹 Эффективность стратегий", callback_data="analytics_strategy_perf")],
        [InlineKeyboardButton("📊 Статистика сделок", callback_data="analytics_trade_stats")],
        [InlineKeyboardButton("🔄 Оптимизация параметров", callback_data="analytics_optimization")],
        [InlineKeyboardButton("💡 Рекомендации", callback_data="analytics_recommendations")],
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
    return ANALYTICS_MENU

@log_performance
async def analytics_risk_reward(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Risk/Reward Analysis"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    trades = portfolio.get('trades', [])
    
    analysis = AnalyticsEngine.calculate_risk_reward_analysis(trades)
    
    risk_text = "📈 *Анализ Риск/Вознаграждение*\n\n"
    
    risk_text += f"⚡ Среднее соотношение R/R: {analysis['average_risk_reward']:.2f}\n"
    risk_text += f"🏆 Лучшая сделка: ${analysis['best_trade']:.2f}\n"
    risk_text += f"🔻 Худшая сделка: ${analysis['worst_trade']:.2f}\n"
    risk_text += f"🎯 Оценка стабильности: {analysis['consistency_score']:.1f}%\n"
    risk_text += f"⚠️ Уровень риска: {analysis['risk_score']:.1f}/100\n\n"
    
    # Recommendations based on risk analysis
    if analysis['average_risk_reward'] < 1:
        risk_text += "💡 *Рекомендация:* Увеличьте соотношение риск/вознаграждение до 1:3\n"
    elif analysis['average_risk_reward'] > 3:
        risk_text += "💡 *Рекомендация:* Отличное соотношение! Продолжайте в том же духе\n"
    
    if analysis['risk_score'] < 30:
        risk_text += "🔻 Снизьте риск на сделку до 1-2% от депозита\n"
    
    await query.edit_message_text(
        risk_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("💹 Эффективность стратегий", callback_data="analytics_strategy_perf")],
            [InlineKeyboardButton("🔙 Назад к аналитике", callback_data="analytics_back")]
        ])
    )

@log_performance
async def analytics_strategy_perf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Strategy Performance Analysis"""
    query = update.callback_query
    
    perf_text = "💹 *Эффективность Стратегий*\n\n"
    
    # Sample strategy performance data
    strategies = {
        'Breakout': {'win_rate': 65, 'avg_profit': 45, 'total_trades': 23},
        'Trend Following': {'win_rate': 58, 'avg_profit': 32, 'total_trades': 15},
        'Mean Reversion': {'win_rate': 72, 'avg_profit': 28, 'total_trades': 18}
    }
    
    for strategy, stats in strategies.items():
        perf_text += f"🎯 *{strategy}*\n"
        perf_text += f"   📊 Винрейт: {stats['win_rate']}%\n"
        perf_text += f"   💰 Средняя прибыль: ${stats['avg_profit']:.2f}\n"
        perf_text += f"   📈 Сделок: {stats['total_trades']}\n\n"
    
    perf_text += "💡 *Лучшая стратегия:* Breakout (65% успешных сделок)"
    
    await query.edit_message_text(
        perf_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Статистика сделок", callback_data="analytics_trade_stats")],
            [InlineKeyboardButton("🔙 Назад к аналитике", callback_data="analytics_back")]
        ])
    )

@log_performance
async def analytics_trade_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trade Statistics"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    performance = portfolio.get('performance', {})
    
    stats_text = "📊 *Статистика Сделок*\n\n"
    
    total_trades = performance.get('total_trades', 0)
    win_rate = performance.get('win_rate', 0)
    profit_factor = (
        abs(performance.get('total_profit', 0) / performance.get('total_loss', 1)) 
        if performance.get('total_loss', 0) != 0 else 0
    )
    
    stats_text += f"📈 Всего сделок: {total_trades}\n"
    stats_text += f"🎯 Процент прибыльных: {win_rate:.1f}%\n"
    stats_text += f"💰 Фактор прибыли: {profit_factor:.2f}\n"
    stats_text += f"⚡ Макс. серия прибылей: {performance.get('winning_trades', 0)}\n"
    stats_text += f"🔻 Макс. серия убытков: {performance.get('losing_trades', 0)}\n\n"
    
    # Performance rating
    if win_rate >= 60 and profit_factor >= 1.5:
        rating = "🏆 ОТЛИЧНО"
    elif win_rate >= 50 and profit_factor >= 1.2:
        rating = "✅ ХОРОШО"
    else:
        rating = "⚠️ ТРЕБУЕТСЯ ОПТИМИЗАЦИЯ"
    
    stats_text += f"📊 *Оценка эффективности:* {rating}"
    
    await query.edit_message_text(
        stats_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Оптимизация параметров", callback_data="analytics_optimization")],
            [InlineKeyboardButton("🔙 Назад к аналитике", callback_data="analytics_back")]
        ])
    )

@log_performance
async def analytics_optimization(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Parameter Optimization"""
    query = update.callback_query
    
    opt_text = "🔄 *Оптимизация Параметров*\n\n"
    
    opt_text += "🎯 *Рекомендуемые настройки:*\n"
    opt_text += "• 📉 Риск на сделку: 1-2% от депозита\n"
    opt_text += "• ⚡ Соотношение R/R: 1:3 или выше\n"
    opt_text += "• 📊 Размер позиции: Автоматический расчет\n"
    opt_text += "• 🛑 Стоп-лосс: Фиксированный процент\n\n"
    
    opt_text += "💡 *Советы по оптимизации:*\n"
    opt_text += "• Тестируйте стратегии на исторических данных\n"
    opt_text += "• Используйте разные таймфреймы\n"
    opt_text += "• Анализируйте результаты еженедельно\n"
    opt_text += "• Корректируйте параметры based on performance\n"
    
    await query.edit_message_text(
        opt_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("💡 Рекомендации", callback_data="analytics_recommendations")],
            [InlineKeyboardButton("🔙 Назад к аналитике", callback_data="analytics_back")]
        ])
    )

@log_performance
async def analytics_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Intelligent Recommendations"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    recommendations = AnalyticsEngine.generate_strategy_recommendations(portfolio)
    
    rec_text = "💡 *Интеллектуальные Рекомендации*\n\n"
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            rec_text += f"{i}. {rec}\n"
    else:
        rec_text += "✅ Ваша текущая стратегия показывает хорошие результаты!\n"
        rec_text += "Рекомендуется продолжать текущий подход.\n\n"
    
    rec_text += "\n🚀 *Скоро появится:*\n"
    rec_text += "• 🤖 AI-анализ ваших стратегий\n"
    rec_text += "• 📊 Автоматический бэктестинг\n"
    rec_text += "• 📈 Прогнозирование доходности\n"
    rec_text += "• 💡 Персональные торговые идеи"
    
    await query.edit_message_text(
        rec_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 Анализ риск/вознаграждение", callback_data="analytics_risk_reward")],
            [InlineKeyboardButton("🔙 Назад к аналитике", callback_data="analytics_back")]
        ])
    )

# Navigation handlers for portfolio and analytics
@log_performance
async def portfolio_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Back to portfolio menu"""
    return await portfolio_command(update, context)

@log_performance
async def analytics_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Back to analytics menu"""
    return await analytics_command(update, context)

# Update main menu handler to include new functionality
@log_performance
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle main menu selection"""
    query = update.callback_query
    if not query:
        return MAIN_MENU
        
    await query.answer()
    choice = query.data
    
    # Update activity time
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
        return await analytics_command(update, context)
    elif choice == "pro_info":
        await pro_info_command(update, context)
        return MAIN_MENU
    elif choice == "main_menu":
        return await start(update, context)
    elif choice == "portfolio_back":
        return await portfolio_command(update, context)
    elif choice == "analytics_back":
        return await analytics_command(update, context)
    
    return MAIN_MENU

# Update ConversationHandler states to include new functionality
def main():
    """Optimized main function to run bot"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ PRO Bot token not found!")
        return

    logger.info("🚀 Starting ULTRA-FAST PRO Risk Management Bot v3.0 with Enhanced Portfolio & Analytics...")
    
    # Create application
    application = Application.builder().token(token).build()

    # Configure ConversationHandler for main menu
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', start),
            CommandHandler('quick', quick_command),
            CommandHandler('portfolio', portfolio_command),
            CommandHandler('analytics', analytics_command),
            CommandHandler('info', pro_info_command),
            CommandHandler('presets', show_presets),
            CallbackQueryHandler(handle_main_menu, pattern='^(pro_calculation|quick_calculation|portfolio|analytics|pro_info|main_menu|portfolio_back|analytics_back)$')
        ],
        states={
            MAIN_MENU: [CallbackQueryHandler(handle_main_menu, pattern='^(pro_calculation|quick_calculation|portfolio|analytics|pro_info|main_menu|portfolio_back|analytics_back)$')],
            PORTFOLIO_MENU: [
                CallbackQueryHandler(portfolio_trades, pattern='^portfolio_trades$'),
                CallbackQueryHandler(portfolio_balance, pattern='^portfolio_balance$'),
                CallbackQueryHandler(portfolio_performance, pattern='^portfolio_performance$'),
                CallbackQueryHandler(portfolio_history, pattern='^portfolio_history$'),
                CallbackQueryHandler(portfolio_add_trade, pattern='^portfolio_add_trade$'),
                CallbackQueryHandler(portfolio_deposit, pattern='^portfolio_deposit$'),
                CallbackQueryHandler(handle_main_menu, pattern='^main_menu$'),
                CallbackQueryHandler(portfolio_back, pattern='^portfolio_back$')
            ],
            ANALYTICS_MENU: [
                CallbackQueryHandler(analytics_risk_reward, pattern='^analytics_risk_reward$'),
                CallbackQueryHandler(analytics_strategy_perf, pattern='^analytics_strategy_perf$'),
                CallbackQueryHandler(analytics_trade_stats, pattern='^analytics_trade_stats$'),
                CallbackQueryHandler(analytics_optimization, pattern='^analytics_optimization$'),
                CallbackQueryHandler(analytics_recommendations, pattern='^analytics_recommendations$'),
                CallbackQueryHandler(handle_main_menu, pattern='^main_menu$'),
                CallbackQueryHandler(analytics_back, pattern='^analytics_back$')
            ],
            # ... (rest of existing states)
        },
        fallbacks=[
            CommandHandler('cancel', cancel),
            CommandHandler('start', start),
            CommandHandler('presets', show_presets)
        ]
    )

    # Add handlers in correct order
    application.add_handler(conv_handler)
    
    # Add portfolio and analytics specific handlers
    application.add_handler(CallbackQueryHandler(portfolio_trades, pattern='^portfolio_trades$'))
    application.add_handler(CallbackQueryHandler(portfolio_balance, pattern='^portfolio_balance$'))
    application.add_handler(CallbackQueryHandler(portfolio_performance, pattern='^portfolio_performance$'))
    application.add_handler(CallbackQueryHandler(portfolio_history, pattern='^portfolio_history$'))
    application.add_handler(CallbackQueryHandler(portfolio_add_trade, pattern='^portfolio_add_trade$'))
    application.add_handler(CallbackQueryHandler(portfolio_deposit, pattern='^portfolio_deposit$'))
    
    application.add_handler(CallbackQueryHandler(analytics_risk_reward, pattern='^analytics_risk_reward$'))
    application.add_handler(CallbackQueryHandler(analytics_strategy_perf, pattern='^analytics_strategy_perf$'))
    application.add_handler(CallbackQueryHandler(analytics_trade_stats, pattern='^analytics_trade_stats$'))
    application.add_handler(CallbackQueryHandler(analytics_optimization, pattern='^analytics_optimization$'))
    application.add_handler(CallbackQueryHandler(analytics_recommendations, pattern='^analytics_recommendations$'))

    # ... (rest of existing handler additions)

    # Get webhook URL
    webhook_url = os.getenv('RENDER_EXTERNAL_URL', '')
    
    # Start webhook or polling
    port = int(os.environ.get('PORT', 10000))
    
    logger.info(f"🌐 PRO Starting on port {port}")
    
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
            logger.info("🔄 PRO Starting in polling mode...")
            application.run_polling()
    except Exception as e:
        logger.error(f"❌ Error starting PRO bot: {e}")

if __name__ == '__main__':
    main()
