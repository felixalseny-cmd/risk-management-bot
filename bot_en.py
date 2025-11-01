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

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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

# Conversation states - FIXED COUNT
(
    MAIN_MENU, INSTRUMENT_TYPE, CUSTOM_INSTRUMENT, DIRECTION, 
    RISK_PERCENT, DEPOSIT, LEVERAGE, CURRENCY, ENTRY, 
    STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION,
    PORTFOLIO_MENU, ADD_TRADE_INSTRUMENT, ADD_TRADE_DIRECTION,
    ADD_TRADE_ENTRY, ADD_TRADE_EXIT, ADD_TRADE_VOLUME, ADD_TRADE_PROFIT,
    DEPOSIT_AMOUNT, WITHDRAW_AMOUNT, SETTINGS_MENU, SAVE_STRATEGY_NAME,
    PRO_DEPOSIT, PRO_LEVERAGE, PRO_RISK, PRO_ENTRY, PRO_STOPLOSS,
    PRO_TAKEPROFIT, PRO_VOLUME, STRATEGY_NAME
) = range(31)  # FIXED: 31 states instead of 35

# Constants
INSTRUMENT_TYPES = {
    'forex': 'Forex',
    'crypto': 'Cryptocurrencies', 
    'indices': 'Indices',
    'commodities': 'Commodities',
    'metals': 'Metals'
}

INSTRUMENT_PRESETS = {
    'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'NZDUSD', 'EURGBP'],
    'crypto': ['BTCUSD', 'ETHUSD', 'XRPUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD'],
    'indices': ['US30', 'NAS100', 'SPX500', 'DAX40', 'FTSE100'],
    'commodities': ['OIL', 'NATGAS', 'COPPER', 'GOLD'],
    'metals': ['XAUUSD', 'XAGUSD', 'XPTUSD']
}

PIP_VALUES = {
    # Forex - major pairs
    'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9, 'USDCHF': 10,
    'USDCAD': 10, 'AUDUSD': 10, 'NZDUSD': 10, 'EURGBP': 10,
    'EURJPY': 9, 'GBPJPY': 9, 'EURCHF': 10, 'AUDJPY': 9,
    # Cryptocurrencies
    'BTCUSD': 1, 'ETHUSD': 1, 'XRPUSD': 10, 'ADAUSD': 10,
    'DOTUSD': 1, 'LTCUSD': 1, 'BCHUSD': 1, 'LINKUSD': 1,
    # Indices
    'US30': 1, 'NAS100': 1, 'SPX500': 1, 'DAX40': 1,
    'FTSE100': 1, 'NIKKEI225': 1, 'ASX200': 1,
    # Commodities
    'OIL': 10, 'NATGAS': 10, 'COPPER': 10,
    # Metals
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

# Data file
DATA_FILE = "user_data.json"

# Data manager with file saving
class DataManager:
    @staticmethod
    def load_data():
        """Load data from file"""
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {}

    @staticmethod
    def save_data():
        """Save data to file"""
        try:
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    @staticmethod
    def auto_save():
        """Autosave every 5 minutes"""
        DataManager.save_data()
        # Schedule next autosave
        asyncio.get_event_loop().call_later(300, DataManager.auto_save)

# Global user data storage
user_data: Dict[int, Dict[str, Any]] = DataManager.load_data()

# Fast cache
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

# Portfolio manager
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
            recommendations.append("🎯 Increase risk/reward ratio to 1:3 to compensate for low Win Rate")
        elif perf['win_rate'] > 60:
            recommendations.append("✅ Excellent Win Rate! Consider increasing position size")
        else:
            recommendations.append("📊 Win Rate is normal. Focus on risk management")
        
        if perf['profit_factor'] < 1:
            recommendations.append("⚠️ Profit Factor below 1.0 - review your strategy")
        elif perf['profit_factor'] > 2:
            recommendations.append("💰 Excellent Profit Factor! Strategy is very effective")
        
        if perf['max_drawdown'] > 20:
            recommendations.append(f"📉 Maximum drawdown {perf['max_drawdown']:.1f}% is too high. Reduce risk per trade")
        elif perf['max_drawdown'] < 5:
            recommendations.append("📈 Low drawdown - consider increasing aggressiveness")
        
        if perf['average_profit'] > 0 and perf['average_loss'] > 0:
            reward_ratio = perf['average_profit'] / perf['average_loss']
            if reward_ratio < 1:
                recommendations.append("🔻 Profit/loss ratio less than 1. Improve take profits")
            elif reward_ratio > 2:
                recommendations.append("🔺 Excellent profit/loss ratio! Keep it up")
        
        allocation = portfolio.get('allocation', {})
        if len(allocation) < 3:
            recommendations.append("🌐 Diversify your portfolio - trade more instruments")
        elif len(allocation) > 10:
            recommendations.append("🎯 Too many instruments - focus on the best ones")
        
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
            logger.error(f"Error in fast position size calculation: {e}")
            return {
                'position_size': 0.01,
                'risk_amount': 0,
                'stop_pips': 0,
                'required_margin': 0,
                'risk_percent': 0,
                'free_margin': deposit
            }

# Input validator
class InputValidator:
    """Class for input data validation"""
    
    @staticmethod
    def validate_number(text: str, min_val: float = 0, max_val: float = None) -> Tuple[bool, float, str]:
        """Validate numeric value"""
        try:
            value = float(text.replace(',', '.'))
            if value < min_val:
                return False, value, f"❌ Value cannot be less than {min_val}"
            if max_val and value > max_val:
                return False, value, f"❌ Value cannot be greater than {max_val}"
            return True, value, "✅ Valid value"
        except ValueError:
            return False, 0, "❌ Please enter a valid numeric value"
    
    @staticmethod
    def validate_instrument(instrument: str) -> Tuple[bool, str]:
        """Validate instrument name"""
        instrument = instrument.upper().strip()
        if not instrument:
            return False, "❌ Please enter instrument name"
        if len(instrument) > 20:
            return False, "❌ Instrument name is too long"
        return True, instrument
    
    @staticmethod
    def validate_price(price: str) -> Tuple[bool, float, str]:
        """Validate price"""
        return InputValidator.validate_number(price, 0.0001, 1000000)
    
    @staticmethod
    def validate_percent(percent: str) -> Tuple[bool, float, str]:
        """Validate percentage value"""
        return InputValidator.validate_number(percent, 0.01, 100)

# Portfolio handlers
@log_performance
async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Portfolio main menu"""
    try:
        if update.message:
            user_id = update.message.from_user.id
        else:
            user_id = update.callback_query.from_user.id
            await update.callback_query.answer()
        
        PortfolioManager.initialize_user_portfolio(user_id)
        portfolio = user_data[user_id]['portfolio']
        
        portfolio_text = f"""
💼 *PRO PORTFOLIO v3.0*

💰 *Balance:* ${portfolio['current_balance']:,.2f}
📊 *Trades:* {len(portfolio['trades'])}
🎯 *Win Rate:* {portfolio['performance']['win_rate']:.1f}%

*Select option:*
"""
        
        keyboard = [
            [InlineKeyboardButton("📈 Trades Overview", callback_data="portfolio_trades")],
            [InlineKeyboardButton("💰 Balance & Allocation", callback_data="portfolio_balance")],
            [InlineKeyboardButton("📊 Performance Analysis", callback_data="portfolio_performance")],
            [InlineKeyboardButton("📄 Generate Report", callback_data="portfolio_report")],
            [InlineKeyboardButton("➕ Add Trade", callback_data="portfolio_add_trade")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
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
        logger.error(f"Error in portfolio_command: {e}")

@log_performance
async def portfolio_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show trades overview"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        portfolio = user_data[user_id].get('portfolio', {})
        trades = portfolio.get('trades', [])
        
        if not trades:
            await query.edit_message_text(
                "📭 *You don't have any trades yet*\n\n"
                "Use the '➕ Add Trade' button to get started.",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("➕ Add Trade", callback_data="portfolio_add_trade")],
                    [InlineKeyboardButton("🔙 Back", callback_data="portfolio")]
                ])
            )
            return
        
        recent_trades = trades[-5:]
        trades_text = "📈 *Recent Trades:*\n\n"
        
        for trade in reversed(recent_trades):
            status_emoji = "🟢" if trade.get('profit', 0) > 0 else "🔴" if trade.get('profit', 0) < 0 else "⚪"
            trades_text += (
                f"{status_emoji} *{trade.get('instrument', 'N/A')}* | "
                f"{trade.get('direction', 'N/A')} | "
                f"Profit: ${trade.get('profit', 0):.2f}\n"
                f"📅 {trade.get('timestamp', '')[:16]}\n\n"
            )
        
        trades_text += f"📊 Total trades: {len(trades)}"
        
        await query.edit_message_text(
            trades_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("➕ Add Trade", callback_data="portfolio_add_trade")],
                [InlineKeyboardButton("🔙 Back", callback_data="portfolio")]
            ])
        )
    except Exception as e:
        logger.error(f"Error in portfolio_trades: {e}")

@log_performance
async def portfolio_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show balance and allocation"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        portfolio = user_data[user_id].get('portfolio', {})
        allocation = portfolio.get('allocation', {})
        performance = portfolio.get('performance', {})
        
        balance_text = "💰 *Balance & Allocation*\n\n"
        
        initial_balance = portfolio.get('initial_balance', 0)
        current_balance = portfolio.get('current_balance', 0)
        total_profit = performance.get('total_profit', 0)
        total_loss = performance.get('total_loss', 0)
        net_profit = total_profit + total_loss
        
        balance_text += f"💳 Initial deposit: ${initial_balance:,.2f}\n"
        balance_text += f"💵 Current balance: ${current_balance:,.2f}\n"
        balance_text += f"📈 Net profit: ${net_profit:.2f}\n\n"
        
        if allocation:
            balance_text += "🌐 *Instrument Allocation:*\n"
            for instrument, count in list(allocation.items())[:5]:
                percentage = (count / len(portfolio['trades'])) * 100 if portfolio['trades'] else 0
                balance_text += f"• {instrument}: {count} trades ({percentage:.1f}%)\n"
        else:
            balance_text += "🌐 *Allocation:* No data\n"
        
        await query.edit_message_text(
            balance_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💸 Make Deposit", callback_data="portfolio_deposit")],
                [InlineKeyboardButton("🔙 Back", callback_data="portfolio")]
            ])
        )
    except Exception as e:
        logger.error(f"Error in portfolio_balance: {e}")

@log_performance
async def portfolio_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show performance analysis"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        portfolio = user_data[user_id].get('portfolio', {})
        performance = portfolio.get('performance', {})
        
        perf_text = "📊 *PRO PERFORMANCE ANALYSIS*\n\n"
        
        total_trades = performance.get('total_trades', 0)
        win_rate = performance.get('win_rate', 0)
        avg_profit = performance.get('average_profit', 0)
        avg_loss = performance.get('average_loss', 0)
        profit_factor = performance.get('profit_factor', 0)
        max_drawdown = performance.get('max_drawdown', 0)
        
        perf_text += f"📈 Total trades: {total_trades}\n"
        perf_text += f"🎯 Win rate: {win_rate:.1f}%\n"
        perf_text += f"💰 Average profit: ${avg_profit:.2f}\n"
        perf_text += f"📉 Average loss: ${avg_loss:.2f}\n"
        perf_text += f"⚖️ Profit factor: {profit_factor:.2f}\n"
        perf_text += f"📊 Max drawdown: {max_drawdown:.1f}%\n\n"
        
        recommendations = PortfolioManager.get_performance_recommendations(user_id)
        
        if recommendations:
            perf_text += "💡 *PRO RECOMMENDATIONS:*\n"
            for i, rec in enumerate(recommendations[:3], 1):
                perf_text += f"{i}. {rec}\n"
        
        await query.edit_message_text(
            perf_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📈 Trades Overview", callback_data="portfolio_trades")],
                [InlineKeyboardButton("🔙 Back", callback_data="portfolio")]
            ])
        )
    except Exception as e:
        logger.error(f"Error in portfolio_performance: {e}")

# PDF report generator
class PDFReportGenerator:
    @staticmethod
    def generate_portfolio_report(user_id: int) -> str:
        """Generate text report"""
        try:
            portfolio = user_data[user_id]['portfolio']
            performance = portfolio['performance']
            
            report = f"""
PORTFOLIO REPORT v3.0
Generated: {datetime.now().strftime('%d.%m.%Y %H:%M')}

BALANCE & FUNDS:
• Initial deposit: ${portfolio['initial_balance']:,.2f}
• Current balance: ${portfolio['current_balance']:,.2f}
• Total P/L: ${portfolio['current_balance'] - portfolio['initial_balance']:,.2f}

TRADING STATISTICS:
• Total trades: {performance['total_trades']}
• Winning trades: {performance['winning_trades']}
• Losing trades: {performance['losing_trades']}
• Win Rate: {performance['win_rate']:.1f}%
• Profit Factor: {performance['profit_factor']:.2f}
• Max drawdown: {performance['max_drawdown']:.1f}%

PROFITABILITY:
• Total profit: ${performance['total_profit']:,.2f}
• Total loss: ${performance['total_loss']:,.2f}
• Average profit: ${performance['average_profit']:.2f}
• Average loss: ${performance['average_loss']:.2f}

INSTRUMENT ALLOCATION:
"""
            
            allocation = portfolio.get('allocation', {})
            for instrument, count in allocation.items():
                report += f"• {instrument}: {count} trades\n"
            
            recommendations = PortfolioManager.get_performance_recommendations(user_id)
            if recommendations:
                report += "\nPRO RECOMMENDATIONS:\n"
                for i, rec in enumerate(recommendations[:3], 1):
                    report += f"{i}. {rec}\n"
            
            return report
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return "Error generating report"

@log_performance
async def portfolio_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate portfolio report"""
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
                    [InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")],
                    [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
                ])
            )
        
        await query.message.reply_text(
            "📄 *Report generated!*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")]
            ])
        )
            
    except Exception as e:
        logger.error(f"Error in portfolio_report: {e}")
        await query.edit_message_text(
            "❌ *Report generation error*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data="portfolio")]
            ])
        )

@log_performance
async def portfolio_deposit_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Deposit menu"""
    try:
        query = update.callback_query
        await query.answer()
        
        await query.edit_message_text(
            "💸 *Make Deposit*\n\n"
            "💰 *Enter deposit amount:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data="portfolio_balance")]
            ])
        )
        return DEPOSIT_AMOUNT
    except Exception as e:
        logger.error(f"Error in portfolio_deposit_menu: {e}")

@log_performance
async def handle_deposit_amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle deposit amount input"""
    try:
        user_id = update.message.from_user.id
        text = update.message.text
        
        # Input validation
        is_valid, amount, message = InputValidator.validate_number(text, 1, 1000000)
        
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n\n💰 Enter deposit amount:",
                parse_mode='Markdown'
            )
            return DEPOSIT_AMOUNT
        
        PortfolioManager.add_balance_operation(user_id, 'deposit', amount, "Deposit")
        
        await update.message.reply_text(
            f"✅ *Deposit of ${amount:,.2f} successfully added!*\n\n"
            f"💳 Current balance: ${user_data[user_id]['portfolio']['current_balance']:,.2f}",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💰 Balance", callback_data="portfolio_balance")],
                [InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")]
            ])
        )
        return ConversationHandler.END
            
    except Exception as e:
        logger.error(f"Error in handle_deposit_amount: {e}")
        await update.message.reply_text(
            "❌ *An error occurred!*\n\n"
            "💰 Enter deposit amount:",
            parse_mode='Markdown'
        )
        return DEPOSIT_AMOUNT

@log_performance
async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Settings with full functionality"""
    try:
        if update.message:
            user_id = update.message.from_user.id
        else:
            user_id = update.callback_query.from_user.id
            await update.callback_query.answer()
        
        PortfolioManager.initialize_user_portfolio(user_id)
        settings = user_data[user_id]['portfolio']['settings']
        
        settings_text = f"""
⚙️ *PRO Trader Settings*

*Current settings:*
• 💰 Risk level: {settings['default_risk']*100}%
• 💵 Deposit currency: {settings['currency']}
• ⚖️ Default leverage: {settings['leverage']}

🔧 *Change settings:*
"""
        
        keyboard = [
            [InlineKeyboardButton(f"💰 Risk level: {settings['default_risk']*100}%", callback_data="change_risk")],
            [InlineKeyboardButton(f"💵 Currency: {settings['currency']}", callback_data="change_currency")],
            [InlineKeyboardButton(f"⚖️ Leverage: {settings['leverage']}", callback_data="change_leverage")],
            [InlineKeyboardButton("💾 Saved strategies", callback_data="saved_strategies")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
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
        logger.error(f"Error in settings_command: {e}")

@log_performance
async def change_risk_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Change risk level"""
    try:
        query = update.callback_query
        await query.answer()
        
        await query.edit_message_text(
            "🎯 *Select default risk level:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🟢 1% (Conservative)", callback_data="set_risk_0.01")],
                [InlineKeyboardButton("🟡 2% (Moderate)", callback_data="set_risk_0.02")],
                [InlineKeyboardButton("🟠 3% (Balanced)", callback_data="set_risk_0.03")],
                [InlineKeyboardButton("🔴 5% (Aggressive)", callback_data="set_risk_0.05")],
                [InlineKeyboardButton("🔙 Back", callback_data="settings")]
            ])
        )
    except Exception as e:
        logger.error(f"Error in change_risk_setting: {e}")

@log_performance
async def change_currency_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Change currency"""
    try:
        query = update.callback_query
        await query.answer()
        
        keyboard = []
        for currency in CURRENCIES:
            keyboard.append([InlineKeyboardButton(currency, callback_data=f"set_currency_{currency}")])
        keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="settings")])
        
        await query.edit_message_text(
            "💵 *Select deposit currency:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    except Exception as e:
        logger.error(f"Error in change_currency_setting: {e}")

@log_performance
async def change_leverage_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Change leverage"""
    try:
        query = update.callback_query
        await query.answer()
        
        keyboard = []
        for leverage in LEVERAGES:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"set_leverage_{leverage}")])
        keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="settings")])
        
        await query.edit_message_text(
            "⚖️ *Select default leverage:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    except Exception as e:
        logger.error(f"Error in change_leverage_setting: {e}")

@log_performance
async def save_risk_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Save risk level"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        risk_level = float(query.data.replace("set_risk_", ""))
        user_data[user_id]['portfolio']['settings']['default_risk'] = risk_level
        DataManager.save_data()
        
        await query.edit_message_text(
            f"✅ *Risk level set: {risk_level*100}%*\n\n"
            "Settings saved for future calculations.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
                [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
            ])
        )
    except Exception as e:
        logger.error(f"Error in save_risk_setting: {e}")

@log_performance
async def save_currency_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Save currency"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        currency = query.data.replace("set_currency_", "")
        user_data[user_id]['portfolio']['settings']['currency'] = currency
        DataManager.save_data()
        
        await query.edit_message_text(
            f"✅ *Currency set: {currency}*\n\n"
            "Settings saved for future calculations.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
                [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
            ])
        )
    except Exception as e:
        logger.error(f"Error in save_currency_setting: {e}")

@log_performance
async def save_leverage_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Save leverage"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        leverage = query.data.replace("set_leverage_", "")
        user_data[user_id]['portfolio']['settings']['leverage'] = leverage
        DataManager.save_data()
        
        await query.edit_message_text(
            f"✅ *Leverage set: {leverage}*\n\n"
            "Settings saved for future calculations.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
                [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
            ])
        )
    except Exception as e:
        logger.error(f"Error in save_leverage_setting: {e}")

# Main menu and basic commands
@log_performance
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Main menu"""
    try:
        if update.message:
            user = update.message.from_user
        elif update.callback_query:
            user = update.callback_query.from_user
            await update.callback_query.answer()
        else:
            return ConversationHandler.END
            
        user_name = user.first_name or "Trader"
        
        welcome_text = f"""
👋 *Hello, {user_name}!*

🎯 *PRO Risk Management Calculator v3.0*

⚡ *New features in version 3.0:*
• ✅ Full professional calculation cycle
• 💾 Strategy saving
• ⚙️ Advanced settings
• 💾 Auto data saving
• 🎯 Smart recommendations

*Select option:*
"""
        
        user_id = user.id
        PortfolioManager.initialize_user_portfolio(user_id)
        
        keyboard = [
            [InlineKeyboardButton("📊 Professional Calculation", callback_data="pro_calculation")],
            [InlineKeyboardButton("⚡ Quick Calculation", callback_data="quick_calculation")],
            [InlineKeyboardButton("💼 My Portfolio", callback_data="portfolio")],
            [InlineKeyboardButton("📚 PRO Instructions", callback_data="pro_info")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
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
        logger.error(f"Error in start: {e}")
        return ConversationHandler.END

@log_performance
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """PRO Instructions v3.0"""
    try:
        info_text = """
📚 *PRO INSTRUCTIONS v3.0*

🎯 *FOR PROFESSIONAL TRADERS:*

💡 *INTUITIVE RISK MANAGEMENT:*
• Calculate optimal position size in seconds
• Automatic instrument type consideration (Forex, crypto, indices)
• Smart volume distribution across multiple take profits
• Instant recalculation when parameters change

📊 *PROFESSIONAL ANALYTICS:*
• Accurate pip value calculation for any instrument
• Margin requirements and leverage consideration
• Risk analysis in monetary and percentage terms
• Position size optimization recommendations

💼 *CAPITAL MANAGEMENT:*
• Complete trading portfolio tracking
• Strategy performance analysis
• Key metrics calculation: Win Rate, Profit Factor, drawdowns
• Intelligent improvement recommendations

⚡ *FAST CALCULATIONS:*
• Instant computations with caching
• Input data validation
• Automatic progress saving
• History of all calculations and trades

🔧 *HOW TO USE:*
1. *Professional calculation* - full cycle with all parameter settings
2. *Quick calculation* - instant calculation based on main parameters  
3. *Portfolio* - trade management and performance analytics
4. *Settings* - personalization of default parameters

💾 *DATA SAVING:*
• All your calculations and trades are saved automatically
• Access to history after bot restart
• Report export for further analysis

🚀 *PROFESSIONAL TIPS:*
• Always use stop-loss to limit risks
• Diversify portfolio across different instruments
• Maintain risk/reward ratio of at least 1:2
• Regularly analyze statistics to optimize strategy

👨‍💻 *Developer for professionals:* @fxfeelgood

*PRO v3.0 | Smart • Fast • Reliable* 🚀
"""
        if update.message:
            await update.message.reply_text(
                info_text, 
                parse_mode='Markdown',
                disable_web_page_preview=True,
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
            )
        else:
            await update.callback_query.edit_message_text(
                info_text,
                parse_mode='Markdown',
                disable_web_page_preview=True,
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
            )
    except Exception as e:
        logger.error(f"Error in pro_info_command: {e}")

# Additional required functions
@log_performance
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel operation"""
    await update.message.reply_text(
        "Operation cancelled.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
    )
    return ConversationHandler.END

@log_performance
async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle unknown commands"""
    await update.message.reply_text(
        "❌ Unknown command. Use /start to begin.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
    )

# Stubs for missing functions
@log_performance
async def start_quick_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stub for quick calculation"""
    await update.message.reply_text(
        "⚡ *Quick calculation temporarily unavailable*\n\n"
        "Use professional calculation for full functionality.",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Professional Calculation", callback_data="pro_calculation")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
        ])
    )

@log_performance
async def portfolio_add_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stub for adding trade"""
    await update.callback_query.edit_message_text(
        "➕ *Adding trades temporarily unavailable*\n\n"
        "This feature will be added in the next update.",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
        ])
    )
    return ConversationHandler.END

@log_performance
async def show_saved_strategies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stub for saved strategies"""
    await update.callback_query.edit_message_text(
        "💾 *Saved strategies temporarily unavailable*\n\n"
        "This feature will be added in the next update.",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
        ])
    )

# Main function
def main():
    """Bot startup v3.0"""
    token = os.getenv('TELEGRAM_BOT_TOKEN_EN')
    if not token:
        logger.error("❌ Bot token not found!")
        return

    logger.info("🚀 Starting PROFESSIONAL risk calculator v3.0...")
    
    # Start autosave
    DataManager.auto_save()
    
    application = Application.builder().token(token).build()

    # Simplified conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MAIN_MENU: [CallbackQueryHandler(handle_main_menu)],
            SETTINGS_MENU: [CallbackQueryHandler(handle_main_menu)],
            PORTFOLIO_MENU: [CallbackQueryHandler(handle_main_menu)],
            DEPOSIT_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_deposit_amount)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # Register handlers
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('info', pro_info_command))
    application.add_handler(CommandHandler('help', pro_info_command))
    application.add_handler(CommandHandler('portfolio', portfolio_command))
    application.add_handler(CommandHandler('quick', start_quick_calculation))
    application.add_handler(CommandHandler('settings', settings_command))
    application.add_handler(CommandHandler('cancel', cancel))

    # Handler for unknown commands
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    
    # Main menu handler
    application.add_handler(CallbackQueryHandler(handle_main_menu, pattern="^(main_menu|portfolio|settings|pro_info|pro_calculation|quick_calculation|portfolio_trades|portfolio_balance|portfolio_performance|portfolio_report|portfolio_deposit|portfolio_add_trade|change_risk|change_currency|change_leverage|saved_strategies|set_risk_|set_currency_|set_leverage_)$"))
    
    # Start bot
    port = int(os.environ.get('PORT', 10000))
    webhook_url = os.getenv('RENDER_EXTERNAL_URL_EN', '')
    
    logger.info(f"🌐 PRO v3.0 starting on port {port}")
    
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
            logger.info("🔄 PRO starting in polling mode...")
            application.run_polling()
    except Exception as e:
        logger.error(f"❌ PRO bot startup error: {e}")

# Main menu handler
@log_performance
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle main menu selection"""
    try:
        query = update.callback_query
        if not query:
            return MAIN_MENU
            
        await query.answer()
        choice = query.data
        
        user_id = query.from_user.id
        if user_id in user_data:
            user_data[user_id]['last_activity'] = time.time()
        
        # Main menu options
        if choice == "pro_calculation":
            await query.edit_message_text(
                "📊 *Professional calculation temporarily unavailable*\n\n"
                "This feature will be added in the next update.",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
                ])
            )
            return MAIN_MENU
        elif choice == "quick_calculation":
            await start_quick_calculation(update, context)
            return MAIN_MENU
        elif choice == "portfolio":
            return await portfolio_command(update, context)
        elif choice == "pro_info":
            await pro_info_command(update, context)
            return MAIN_MENU
        elif choice == "settings":
            return await settings_command(update, context)
        elif choice == "main_menu":
            return await start(update, context)
        
        # Portfolio
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
        
        # Settings
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
        
        return MAIN_MENU
        
    except Exception as e:
        logger.error(f"Error in handle_main_menu: {e}")
        return await start(update, context)

if __name__ == '__main__':
    main()
