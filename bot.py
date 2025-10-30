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

# Constants
INSTRUMENT_TYPES = {
    'forex': 'Forex',
    'crypto': 'Криптовалюты', 
    'indices': 'Индексы',
    'commodities': 'Товары',
    'metals': 'Металлы'
}

PIP_VALUES = {
    # Forex - major pairs
    'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9, 'USDCHF': 10,
    'USDCAD': 10, 'AUDUSD': 10, 'NZDUSD': 10, 'EURGBP': 10,
    'EURJPY': 9, 'GBPJPY': 9, 'EURCHF': 10, 'AUDJPY': 9,
    'NZDJPY': 9, 'CADJPY': 9, 'CHFJPY': 9, 'GBPCAD': 10,
    'GBPAUD': 10, 'GBPNZD': 10, 'EURAUD': 10, 'EURCAD': 10,
    'EURNZD': 10, 'AUDCAD': 10, 'AUDCHF': 10, 'AUDNZD': 10,
    'CADCHF': 10, 'NZDCAD': 10, 'NZDCHF': 10,
    # Forex - exotic pairs
    'USDSEK': 10, 'USDDKK': 10, 'USDNOK': 10, 'USDPLN': 10,
    'USDCZK': 10, 'USDHUF': 10, 'USDRON': 10, 'USDTRY': 10,
    'USDZAR': 10, 'USDMXN': 10, 'USDSGD': 10, 'USDHKD': 10,
    # Cryptocurrencies
    'BTCUSD': 1, 'ETHUSD': 1, 'XRPUSD': 10, 'ADAUSD': 10,
    'DOTUSD': 1, 'LTCUSD': 1, 'BCHUSD': 1, 'LINKUSD': 1,
    'BNBUSD': 1, 'SOLUSD': 1, 'DOGEUSD': 10, 'MATICUSD': 10,
    'AVAXUSD': 1, 'ATOMUSD': 1, 'UNIUSD': 1, 'XLMUSD': 10,
    # Indices
    'US30': 1, 'NAS100': 1, 'SPX500': 1, 'DAX40': 1,
    'FTSE100': 1, 'NIKKEI225': 1, 'ASX200': 1, 'CAC40': 1,
    'ESTX50': 1, 'HSI': 1, 'SENSEX': 1, 'IBOVESPA': 1,
    # Commodities
    'OIL': 10, 'NATGAS': 10, 'COPPER': 10, 'WHEAT': 10,
    'CORN': 10, 'SOYBEAN': 10, 'SUGAR': 10, 'COFFEE': 10,
    # Metals
    'XAUUSD': 10, 'XAGUSD': 50, 'XPTUSD': 10, 'XPDUSD': 10,
    'XAUAUD': 10, 'XAUEUR': 10, 'XAGGBP': 50
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

# Ultra-fast risk calculator
class FastRiskCalculator:
    """Optimized risk calculator with simplified calculations"""
    
    @staticmethod
    def calculate_pip_value_fast(instrument_type: str, currency_pair: str, lot_size: float) -> float:
        """Fast pip value calculation"""
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
        """Ultra-fast position size calculation"""
        try:
            # Fast cache key
            cache_key = f"pos_{deposit}_{leverage}_{instrument_type}_{currency_pair}_{entry_price}_{stop_loss}_{direction}_{risk_percent}"
            cached_result = fast_cache.get(cache_key)
            if cached_result:
                return cached_result
            
            lev_value = int(leverage.split(':')[1])
            risk_amount = deposit * risk_percent
            
            # Fast stop loss calculations
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
            
            # Save to cache
            fast_cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error in fast position size calculation: {e}")
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
        """Fast profit calculation"""
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

# Main command handlers
@log_performance
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Main menu"""
    if update.message:
        user = update.message.from_user
    elif update.callback_query:
        user = update.callback_query.from_user
    else:
        return ConversationHandler.END
        
    user_name = user.first_name or "Трейдер"
    
    welcome_text = f"""
👋 *Привет, {user_name}!*

🎯 *PRO Калькулятор Риск-Менеджмента v3.0*

⚡ *Выберите опцию:*
"""
    
    user_id = user.id
    # Preserve presets on restart
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
        [InlineKeyboardButton("📈 Аналитика", callback_data="analytics")],
        [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")]
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
    """Quick calculation"""
    return await start_quick_calculation(update, context)

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
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """PRO Instructions"""
    info_text = """
📚 *PRO ИНСТРУКЦИИ v3.0*

🎯 *РАСШИРЕННЫЕ ВОЗМОЖНОСТИ:*

⚡ *ВСЕ ТИПЫ ИНСТРУМЕНТОВ:*
• 🌐 Forex (50+ валютных пар)
• ₿ Криптовалюты (15+ пар)
• 📈 Индексы (12+ индексов)
• ⚡ Товары (8+ типов)
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

# Portfolio handlers
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
    balance_text += f"📈 Чистая прибыль: ${net_profit:.2f}\n\n"
    
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

# Analytics handlers
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
    opt_text += "• Корректируйте параметры на основе производительности\n"
    
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

# Navigation handlers
@log_performance
async def portfolio_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Back to portfolio menu"""
    return await portfolio_command(update, context)

@log_performance
async def analytics_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Back to analytics menu"""
    return await analytics_command(update, context)

# Main menu handlers
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

# Basic calculation handlers (simplified for demo)
@log_performance
async def start_pro_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start professional calculation"""
    query = update.callback_query
    if query:
        await query.edit_message_text(
            "🎯 *Профессиональный расчет*\n\n"
            "📊 *Выберите тип инструмента:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 Forex", callback_data="inst_type_forex")],
                [InlineKeyboardButton("₿ Криптовалюты", callback_data="inst_type_crypto")],
                [InlineKeyboardButton("📈 Индексы", callback_data="inst_type_indices")],
                [InlineKeyboardButton("⚡ Товары", callback_data="inst_type_commodities")],
                [InlineKeyboardButton("🏅 Металлы", callback_data="inst_type_metals")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
    return INSTRUMENT_TYPE

@log_performance
async def start_quick_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start quick calculation"""
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

# Additional required handlers
@log_performance
async def show_presets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show saved presets"""
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
    """Cancel conversation"""
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

# Simplified handlers for calculation flow
@log_performance
async def process_instrument_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process instrument type selection"""
    query = update.callback_query
    if not query:
        return INSTRUMENT_TYPE
        
    await query.answer()
    user_id = query.from_user.id
    instrument_type = query.data.replace('inst_type_', '')
    user_data[user_id]['instrument_type'] = instrument_type
    user_data[user_id]['last_activity'] = time.time()
    
    # For demo, show a simple message
    display_type = INSTRUMENT_TYPES.get(instrument_type, instrument_type)
    await query.edit_message_text(
        f"✅ *Тип инструмента:* {display_type}\n\n"
        "Эта функция находится в разработке. Используйте Быстрый расчет для демонстрации.",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("⚡ Быстрый расчет", callback_data="quick_calculation")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ])
    )
    return MAIN_MENU

@log_performance
async def process_currency_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process instrument ticker input for quick calculation"""
    if not update.message:
        return CUSTOM_INSTRUMENT
        
    user_id = update.message.from_user.id
    currency = update.message.text.upper().strip()
    
    # Basic ticker validation
    if not re.match(r'^[A-Z0-9]{2,10}$', currency):
        await update.message.reply_text(
            "❌ *Неверный формат тикера!*\n\n"
            "Пожалуйста, введите правильный тикер (только буквы и цифры, 2-10 символов):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return CUSTOM_INSTRUMENT
    
    # For demo, show calculation result
    user_data[user_id] = {
        'currency': currency,
        'instrument_type': 'forex',
        'direction': 'BUY',
        'risk_percent': 0.02,
        'deposit': 10000,
        'leverage': '1:100',
        'entry': 1.0850,
        'stop_loss': 1.0800,
        'take_profits': [1.0900, 1.0950],
        'volume_distribution': [70, 30]
    }
    
    # Perform calculation
    pos = FastRiskCalculator.calculate_position_size_fast(
        deposit=10000,
        leverage='1:100',
        instrument_type='forex',
        currency_pair=currency,
        entry_price=1.0850,
        stop_loss=1.0800,
        direction='BUY'
    )
    
    result_text = f"""
🎯 *РЕЗУЛЬТАТ БЫСТРОГО РАСЧЕТА*

📊 *Основные параметры:*
🌐 Инструмент: {currency}
💵 Депозит: $10,000
⚖️ Плечо: 1:100
📈 Вход: 1.0850
🛑 Стоп-лосс: 1.0800

⚠️ *Управление рисками:*
📦 Размер позиции: *{pos['position_size']:.2f} лотов*
💰 Риск на сделку: ${pos['risk_amount']:.2f}
📉 Стоп-лосс: {pos['stop_pips']:.0f} пунктов

💡 *Рекомендация:* Используйте профессиональный расчет для детального анализа
"""
    
    await update.message.reply_text(
        result_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("💼 Добавить в портфель", callback_data="portfolio_add_trade")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ])
    )
    return MAIN_MENU

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
            CallbackQueryHandler(handle_main_menu, pattern='^(pro_calculation|quick_calculation|portfolio|analytics|pro_info|main_menu)$')
        ],
        states={
            MAIN_MENU: [
                CallbackQueryHandler(handle_main_menu, pattern='^(pro_calculation|quick_calculation|portfolio|analytics|pro_info|main_menu|portfolio_back|analytics_back)$')
            ],
            INSTRUMENT_TYPE: [
                CallbackQueryHandler(process_instrument_type, pattern='^inst_type_')
            ],
            CUSTOM_INSTRUMENT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_currency_input)
            ],
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
        },
        fallbacks=[
            CommandHandler('cancel', cancel),
            CommandHandler('start', start),
            CommandHandler('presets', show_presets)
        ],
        allow_reentry=True
    )

    # Add handlers
    application.add_handler(conv_handler)
    
    # Add individual callback handlers
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

    # Start bot
    port = int(os.environ.get('PORT', 10000))
    webhook_url = os.getenv('RENDER_EXTERNAL_URL', '')
    
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
