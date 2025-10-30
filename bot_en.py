import os
import logging
import asyncio
import re
import time
import functools
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
    STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION
) = range(12)

# Temporary storage
user_data: Dict[int, Dict[str, Any]] = {}

# Cache manager for faster calculations
class CacheManager:
    def __init__(self, max_size=1000, ttl=300):
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
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = (value, time.time())

# Global cache
calculation_cache = CacheManager()

# Extended constants
INSTRUMENT_TYPES = {
    'forex': 'Forex',
    'crypto': 'Cryptocurrencies', 
    'indices': 'Indices',
    'commodities': 'Commodities',
    'metals': 'Metals'
}

# Extended instruments list
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

# Optimized risk calculator with caching
class OptimizedRiskCalculator:
    """Optimized risk calculator with support for all instrument types"""
    
    @staticmethod
    @functools.lru_cache(maxsize=500)
    def calculate_pip_value_cached(instrument_type: str, currency_pair: str, lot_size: float) -> float:
        """Cached version of pip value calculation"""
        base_pip_value = PIP_VALUES.get(currency_pair, 10)
        
        if instrument_type == 'crypto':
            return base_pip_value * lot_size * 0.1
        elif instrument_type == 'indices':
            return base_pip_value * lot_size * 0.01
        else:
            return base_pip_value * lot_size

    @staticmethod
    def calculate_position_size(
        deposit: float,
        leverage: str,
        instrument_type: str,
        currency_pair: str,
        entry_price: float,
        stop_loss: float,
        direction: str,
        risk_percent: float = 0.02
    ) -> Dict[str, float]:
        """Optimized version of position size calculation"""
        try:
            # Create cache key
            cache_key = f"pos_{deposit}_{leverage}_{instrument_type}_{currency_pair}_{entry_price}_{stop_loss}_{direction}_{risk_percent}"
            cached_result = calculation_cache.get(cache_key)
            if cached_result:
                return cached_result
            
            lev_value = int(leverage.split(':')[1])
            risk_amount = deposit * risk_percent
            
            # Optimized stop loss calculations
            if instrument_type == 'forex':
                stop_pips = abs(entry_price - stop_loss) * 10000
            elif instrument_type == 'crypto':
                stop_pips = abs(entry_price - stop_loss) * 100
            elif instrument_type in ['indices', 'commodities', 'metals']:
                stop_pips = abs(entry_price - stop_loss) * 10
            else:
                stop_pips = abs(entry_price - stop_loss) * 10000

            pip_value_per_lot = OptimizedRiskCalculator.calculate_pip_value_cached(
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
            calculation_cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error in optimized position size calculation: {e}")
            return {
                'position_size': 0.01,
                'risk_amount': 0,
                'stop_pips': 0,
                'required_margin': 0,
                'risk_percent': 0,
                'free_margin': deposit
            }

    @staticmethod
    def calculate_profits(
        instrument_type: str,
        currency_pair: str,
        entry_price: float,
        take_profits: List[float],
        position_size: float,
        volume_distribution: List[float],
        direction: str
    ) -> List[Dict[str, Any]]:
        """Optimized profit calculation"""
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
            pip_value = OptimizedRiskCalculator.calculate_pip_value_cached(
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

    @staticmethod
    def calculate_risk_reward_ratio(
        entry_price: float,
        stop_loss: float,
        take_profits: List[float],
        volume_distribution: List[float],
        direction: str
    ) -> Dict[str, float]:
        """Optimized risk/reward ratio calculation"""
        try:
            risk = abs(entry_price - stop_loss)
            total_reward = 0
            
            for tp, vol in zip(take_profits, volume_distribution):
                reward = abs(entry_price - tp) * (vol / 100)
                total_reward += reward
            
            if risk > 0:
                risk_reward = total_reward / risk
            else:
                risk_reward = 0
                
            return {
                'risk_reward_ratio': risk_reward,
                'total_risk': risk,
                'total_reward': total_reward
            }
        except Exception as e:
            logger.error(f"Error in optimized risk/reward calculation: {e}")
            return {'risk_reward_ratio': 0, 'total_risk': 0, 'total_reward': 0}

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
        
    user_name = user.first_name or "Trader"
    
    welcome_text = f"""
👋 *Hello, {user_name}!*

🎯 *PRO Risk Management Calculator v3.0*

⚡ *Choose an option:*
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
        [InlineKeyboardButton("📊 Professional Calculation", callback_data="pro_calculation")],
        [InlineKeyboardButton("⚡ Quick Calculation", callback_data="quick_calculation")],
        [InlineKeyboardButton("💼 My Portfolio", callback_data="portfolio")],
        [InlineKeyboardButton("📈 Analytics", callback_data="analytics")],
        [InlineKeyboardButton("📚 PRO Instructions", callback_data="pro_info")]
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
async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Portfolio management"""
    portfolio_text = """
💼 *Portfolio Management*

*📊 Portfolio Features:*
• 📈 Overview of all trades
• 💰 Balance and allocation
• 📊 Performance analysis
• 🔄 Operation history

*🚀 Coming Soon:*
• 📊 Portfolio visualization
• 📈 Market comparison
• 💡 Diversification recommendations

*📚 Use professional calculation for risk management!*

👨‍💻 *PRO Developer:* [@fxfeelgood](https://t.me/fxfeelgood)
"""
    if update.message:
        await update.message.reply_text(
            portfolio_text,
            parse_mode='Markdown',
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
        )
    else:
        await update.callback_query.edit_message_text(
            portfolio_text,
            parse_mode='Markdown',
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
        )

@log_performance
async def analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Strategy analytics"""
    analytics_text = """
📈 *Strategy Analytics*

*📊 Available Analytics:*
• 📈 Risk/reward analysis
• 💹 Strategy performance
• 📊 Trade statistics
• 🔄 Parameter optimization

*🚀 Coming Soon:*
• 🤖 AI strategy analysis
• 📊 Backtesting
• 📈 Forecasting
• 💡 Intelligent recommendations

*📚 Use professional calculation for analysis!*

👨‍💻 *PRO Developer:* [@fxfeelgood](https://t.me/fxfeelgood)
"""
    if update.message:
        await update.message.reply_text(
            analytics_text,
            parse_mode='Markdown',
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
        )
    else:
        await update.callback_query.edit_message_text(
            analytics_text,
            parse_mode='Markdown',
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
        )

@log_performance
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """PRO Instructions"""
    info_text = """
📚 *PRO INSTRUCTIONS v3.0*

🎯 *ADVANCED CAPABILITIES:*

⚡ *ALL INSTRUMENT TYPES:*
• 🌐 Forex (50+ currency pairs)
• ₿ Cryptocurrencies (15+ pairs)
• 📈 Indices (12+ indices)
• ⚡ Commodities (8+ types)
• 🏅 Metals (6+ types)

🔄 *NEW FEATURES v3.0:*

1. *⚡ Quick Calculation*
   - Minimal parameter input
   - Automatic settings
   - Instant results

2. *🎯 Direction Selection*
   - BUY/SELL calculations
   - Direction consideration in risks
   - Optimized recommendations

3. *⚖️ Flexible Risk Management*
   - 6 risk levels (2%-25%)
   - Individual settings
   - Adaptive algorithms

4. *🔧 Custom Instruments*
   - Manual input of any ticker
   - Automatic parameter detection
   - Wide asset support

📋 *HOW TO USE:*

*Professional Calculation:*
1. Select instrument type
2. Choose specific instrument or enter custom
3. Specify trade direction (BUY/SELL)
4. Select risk level
5. Enter main parameters
6. Get detailed analysis

*Quick Calculation:*
1. Enter instrument
2. Specify basic parameters
3. Get instant result

🛠 *PRO COMMANDS v3.0:*
`/start` - Main menu
`/quick` - Quick calculation
`/portfolio` - Portfolio management
`/analytics` - Strategy analytics
`/info` - PRO instructions
`/presets` - My strategies

👨‍💻 *DEVELOPER:* [@fxfeelgood](https://t.me/fxfeelgood)

*PRO v3.0 | Fast • Smart • Accurate* 🚀
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
        await portfolio_command(update, context)
        return MAIN_MENU
    elif choice == "analytics":
        await analytics_command(update, context)
        return MAIN_MENU
    elif choice == "pro_info":
        await pro_info_command(update, context)
        return MAIN_MENU
    elif choice == "main_menu":
        return await start(update, context)
    
    return MAIN_MENU

@log_performance
async def start_pro_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start professional calculation"""
    query = update.callback_query
    if query:
        await query.edit_message_text(
            "🎯 *Professional Calculation*\n\n"
            "📊 *Select instrument type:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 Forex", callback_data="inst_type_forex")],
                [InlineKeyboardButton("₿ Cryptocurrencies", callback_data="inst_type_crypto")],
                [InlineKeyboardButton("📈 Indices", callback_data="inst_type_indices")],
                [InlineKeyboardButton("⚡ Commodities", callback_data="inst_type_commodities")],
                [InlineKeyboardButton("🏅 Metals", callback_data="inst_type_metals")],
                [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
            ])
        )
    return INSTRUMENT_TYPE

@log_performance
async def start_quick_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start quick calculation"""
    if update.message:
        await update.message.reply_text(
            "⚡ *Quick Calculation*\n\n"
            "📊 *Enter instrument ticker* (e.g.: EURUSD, BTCUSD, NAS100):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
        )
    else:
        query = update.callback_query
        await query.edit_message_text(
            "⚡ *Quick Calculation*\n\n"
            "📊 *Enter instrument ticker* (e.g.: EURUSD, BTCUSD, NAS100):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
        )
    return CUSTOM_INSTRUMENT

# Professional calculation handlers
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
    
    # Get instruments for selected type
    instruments = {
        'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD', 'Other pair'],
        'crypto': ['BTCUSD', 'ETHUSD', 'XRPUSD', 'ADAUSD', 'DOTUSD', 'Other crypto'],
        'indices': ['US30', 'NAS100', 'SPX500', 'DAX40', 'FTSE100', 'Other index'],
        'commodities': ['OIL', 'NATGAS', 'COPPER', 'WHEAT', 'Other commodity'],
        'metals': ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'Other metal']
    }.get(instrument_type, [])
    
    keyboard = []
    for i in range(0, len(instruments), 2):
        row = []
        for j in range(2):
            if i + j < len(instruments):
                inst = instruments[i + j]
                if inst.startswith('Other'):
                    row.append(InlineKeyboardButton("📝 " + inst, callback_data="custom_instrument"))
                else:
                    row.append(InlineKeyboardButton(inst, callback_data=f"currency_{inst}"))
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")])
    
    display_type = INSTRUMENT_TYPES.get(instrument_type, instrument_type)
    await query.edit_message_text(
        f"✅ *Instrument Type:* {display_type}\n\n"
        "🌐 *Select specific instrument:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return CURRENCY

@log_performance
async def process_custom_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process custom instrument input"""
    query = update.callback_query
    if query:
        await query.edit_message_text(
            "📝 *Enter instrument ticker manually*\n\n"
            "Examples:\n"
            "• EURGBP, USDSEK, GBPAUD\n"
            "• BNBUSD, SOLUSD, DOGEUSD\n"
            "• CAC40, ESTX50, HSI\n\n"
            "*Enter ticker:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_instruments")]])
        )
    return CUSTOM_INSTRUMENT

@log_performance
async def process_currency_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process instrument ticker input"""
    if not update.message:
        return CUSTOM_INSTRUMENT
        
    user_id = update.message.from_user.id
    currency = update.message.text.upper().strip()
    
    # Basic ticker validation
    if not re.match(r'^[A-Z0-9]{2,10}$', currency):
        await update.message.reply_text(
            "❌ *Invalid ticker format!*\n\n"
            "Please enter a valid ticker (letters and numbers only, 2-10 characters):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_instruments")]])
        )
        return CUSTOM_INSTRUMENT
    
    user_data[user_id]['currency'] = currency
    user_data[user_id]['last_activity'] = time.time()
    
    # Determine instrument type by default if not set
    if 'instrument_type' not in user_data[user_id]:
        if any(x in currency for x in ['BTC', 'ETH', 'XRP', 'ADA']):
            user_data[user_id]['instrument_type'] = 'crypto'
        elif any(x in currency for x in ['XAU', 'XAG', 'XPT', 'XPD']):
            user_data[user_id]['instrument_type'] = 'metals'
        elif currency.isalpha() and len(currency) == 6:
            user_data[user_id]['instrument_type'] = 'forex'
        else:
            user_data[user_id]['instrument_type'] = 'indices'
    
    await update.message.reply_text(
        f"✅ *Instrument:* {currency}\n\n"
        "🎯 *Select trade direction:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 BUY", callback_data="direction_BUY")],
            [InlineKeyboardButton("📉 SELL", callback_data="direction_SELL")],
            [InlineKeyboardButton("🔙 Back", callback_data="back_to_instruments")]
        ])
    )
    return DIRECTION

@log_performance
async def process_currency_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process instrument selection from list"""
    query = update.callback_query
    if not query:
        return CURRENCY
        
    await query.answer()
    user_id = query.from_user.id
    
    if query.data == "custom_instrument":
        return await process_custom_instrument(update, context)
    elif query.data == "back_to_instruments":
        user_data[user_id]['last_activity'] = time.time()
        return await start_pro_calculation(update, context)
    
    currency = query.data.replace('currency_', '')
    user_data[user_id]['currency'] = currency
    user_data[user_id]['last_activity'] = time.time()
    
    await query.edit_message_text(
        f"✅ *Instrument:* {currency}\n\n"
        "🎯 *Select trade direction:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 BUY", callback_data="direction_BUY")],
            [InlineKeyboardButton("📉 SELL", callback_data="direction_SELL")],
            [InlineKeyboardButton("🔙 Back", callback_data="back_to_instruments")]
        ])
    )
    return DIRECTION

@log_performance
async def process_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process direction selection"""
    query = update.callback_query
    if not query:
        return DIRECTION
        
    await query.answer()
    user_id = query.from_user.id
    direction = query.data.replace('direction_', '')
    user_data[user_id]['direction'] = direction
    user_data[user_id]['last_activity'] = time.time()
    
    await query.edit_message_text(
        f"✅ *Direction:* {direction}\n\n"
        "⚖️ *Select risk level per trade:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("2% (Conservative)", callback_data="risk_0.02")],
            [InlineKeyboardButton("5% (Moderate)", callback_data="risk_0.05")],
            [InlineKeyboardButton("10% (Aggressive)", callback_data="risk_0.10")],
            [InlineKeyboardButton("15% (High)", callback_data="risk_0.15")],
            [InlineKeyboardButton("20% (Very High)", callback_data="risk_0.20")],
            [InlineKeyboardButton("25% (Extreme)", callback_data="risk_0.25")],
            [InlineKeyboardButton("🔙 Back", callback_data="back_to_direction")]
        ])
    )
    return RISK_PERCENT

@log_performance
async def process_risk_percent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process risk level selection"""
    query = update.callback_query
    if not query:
        return RISK_PERCENT
        
    await query.answer()
    user_id = query.from_user.id
    user_data[user_id]['last_activity'] = time.time()
    
    if query.data == "back_to_direction":
        currency = user_data[user_id].get('currency', 'EURUSD')
        await query.edit_message_text(
            f"✅ *Instrument:* {currency}\n\n"
            "🎯 *Select trade direction:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📈 BUY", callback_data="direction_BUY")],
                [InlineKeyboardButton("📉 SELL", callback_data="direction_SELL")],
                [InlineKeyboardButton("🔙 Back", callback_data="back_to_instruments")]
            ])
        )
        return DIRECTION
    
    risk_percent = float(query.data.replace('risk_', ''))
    user_data[user_id]['risk_percent'] = risk_percent
    
    await query.edit_message_text(
        f"✅ *Risk level:* {risk_percent*100}%\n\n"
        "💵 *Enter deposit amount in USD:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_risk")]])
    )
    return DEPOSIT

@log_performance
async def process_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process deposit input"""
    if not update.message:
        return DEPOSIT
        
    user_id = update.message.from_user.id
    
    try:
        deposit = float(update.message.text.replace(',', '').replace(' ', ''))
        if deposit <= 0:
            await update.message.reply_text("❌ Deposit must be positive:")
            return DEPOSIT
        if deposit > 1000000:
            await update.message.reply_text("❌ Maximum deposit: $1,000,000:")
            return DEPOSIT
            
        user_data[user_id]['deposit'] = deposit
        user_data[user_id]['last_activity'] = time.time()
        
        # Create leverage keyboard
        keyboard = []
        for i in range(0, len(LEVERAGES), 3):
            row = []
            for j in range(3):
                if i + j < len(LEVERAGES):
                    lev = LEVERAGES[i + j]
                    row.append(InlineKeyboardButton(lev, callback_data=f"leverage_{lev}"))
            keyboard.append(row)
        
        keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="back_to_deposit")])
        
        await update.message.reply_text(
            f"✅ *Deposit:* ${deposit:,.2f}\n\n"
            "⚖️ *Choose your leverage:*",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
        return LEVERAGE
        
    except ValueError:
        await update.message.reply_text("❌ Please enter a valid deposit amount:")
        return DEPOSIT

@log_performance
async def process_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process leverage selection"""
    query = update.callback_query
    if not query:
        return LEVERAGE
        
    await query.answer()
    user_id = query.from_user.id
    user_data[user_id]['last_activity'] = time.time()
    
    if query.data == "back_to_deposit":
        await query.edit_message_text(
            "💵 *Enter deposit amount in USD:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_risk")]])
        )
        return DEPOSIT
    
    leverage = query.data.replace('leverage_', '')
    user_data[user_id]['leverage'] = leverage
    
    currency = user_data[user_id].get('currency', 'EURUSD')
    direction = user_data[user_id].get('direction', 'BUY')
    
    await query.edit_message_text(
        f"✅ *Leverage:* {leverage}\n"
        f"✅ *Direction:* {direction}\n\n"
        f"📈 *Enter entry price for {currency}:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_leverage")]])
    )
    return ENTRY

@log_performance
async def process_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process entry price input"""
    if not update.message:
        return ENTRY
        
    user_id = update.message.from_user.id
    
    try:
        entry = float(update.message.text)
        if entry <= 0:
            await update.message.reply_text("❌ Price must be positive:")
            return ENTRY
            
        user_data[user_id]['entry'] = entry
        user_data[user_id]['last_activity'] = time.time()
        
        currency = user_data[user_id].get('currency', 'EURUSD')
        direction = user_data[user_id].get('direction', 'BUY')
        
        await update.message.reply_text(
            f"✅ *Entry price:* {entry}\n"
            f"✅ *Direction:* {direction}\n\n"
            f"🛑 *Enter stop loss price for {currency}:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_entry")]])
        )
        return STOP_LOSS
        
    except ValueError:
        await update.message.reply_text("❌ Please enter a valid entry price:")
        return ENTRY

@log_performance
async def process_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process stop loss input"""
    if not update.message:
        return STOP_LOSS
        
    user_id = update.message.from_user.id
    
    try:
        sl = float(update.message.text)
        entry = user_data[user_id].get('entry', 0)
        
        if sl <= 0:
            await update.message.reply_text("❌ Price must be positive:")
            return STOP_LOSS
            
        user_data[user_id]['stop_loss'] = sl
        user_data[user_id]['last_activity'] = time.time()
        
        currency = user_data[user_id].get('currency', 'EURUSD')
        
        await update.message.reply_text(
            f"✅ *Stop loss:* {sl}\n\n"
            f"🎯 *Enter take profit prices for {currency} separated by commas* (e.g.: 1.0550, 1.0460):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_stop_loss")]])
        )
        return TAKE_PROFITS
        
    except ValueError:
        await update.message.reply_text("❌ Please enter a valid stop loss price:")
        return STOP_LOSS

@log_performance
async def process_take_profits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process take profits input"""
    if not update.message:
        return TAKE_PROFITS
        
    user_id = update.message.from_user.id
    
    try:
        tps = [float(x.strip()) for x in update.message.text.split(',')]
        
        if len(tps) > 5:
            await update.message.reply_text("❌ Maximum 5 take profits:")
            return TAKE_PROFITS
            
        user_data[user_id]['take_profits'] = tps
        user_data[user_id]['last_activity'] = time.time()
        
        await update.message.reply_text(
            f"✅ *Take profits:* {', '.join(map(str, tps))}\n\n"
            f"📊 *Enter volume distribution in % for each take profit separated by commas*\n"
            f"(total {len(tps)} values, sum must be 100%):\n"
            f"*Example:* 50, 30, 20",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_take_profits")]])
        )
        return VOLUME_DISTRIBUTION
        
    except ValueError:
        await update.message.reply_text("❌ Please enter valid take profit prices:")
        return TAKE_PROFITS

@log_performance
async def process_volume_distribution(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Optimized processing of volume distribution and results display"""
    if not update.message:
        return VOLUME_DISTRIBUTION
        
    user_id = update.message.from_user.id
    
    try:
        # Update activity time
        if user_id in user_data:
            user_data[user_id]['last_activity'] = time.time()
        
        dist = [float(x.strip()) for x in update.message.text.split(',')]
        
        # Fast validation
        if abs(sum(dist) - 100) > 1e-5:
            await update.message.reply_text(
                f"❌ *Distribution sum must be 100%. Your sum: {sum(dist)}%*\n"
                "Please enter distribution again:",
                parse_mode='Markdown'
            )
            return VOLUME_DISTRIBUTION
        
        user_tps = user_data[user_id].get('take_profits', [])
        if len(dist) != len(user_tps):
            await update.message.reply_text(
                f"❌ *Number of distribution values must match number of TPs ({len(user_tps)})*\n"
                "Please enter distribution again:",
                parse_mode='Markdown'
            )
            return VOLUME_DISTRIBUTION
        
        user_data[user_id]['volume_distribution'] = dist
        data = user_data[user_id]
        
        # Send quick message about starting calculations
        quick_response = await update.message.reply_text(
            "⚡ *Performing optimized calculations...*\n\n"
            "⏳ *Calculation will take a few seconds*",
            parse_mode='Markdown'
        )
        
        # Use optimized calculator
        pos = OptimizedRiskCalculator.calculate_position_size(
            deposit=data['deposit'],
            leverage=data['leverage'],
            instrument_type=data['instrument_type'],
            currency_pair=data['currency'],
            entry_price=data['entry'],
            stop_loss=data['stop_loss'],
            direction=data.get('direction', 'BUY'),
            risk_percent=data.get('risk_percent', 0.02)
        )
        
        profits = OptimizedRiskCalculator.calculate_profits(
            instrument_type=data['instrument_type'],
            currency_pair=data['currency'],
            entry_price=data['entry'],
            take_profits=data['take_profits'],
            position_size=pos['position_size'],
            volume_distribution=dist,
            direction=data.get('direction', 'BUY')
        )
        
        risk_reward = OptimizedRiskCalculator.calculate_risk_reward_ratio(
            entry_price=data['entry'],
            stop_loss=data['stop_loss'],
            take_profits=data['take_profits'],
            volume_distribution=dist,
            direction=data.get('direction', 'BUY')
        )
        
        # Professional results formatting
        instrument_display = INSTRUMENT_TYPES.get(data['instrument_type'], data['instrument_type'])
        direction_display = "📈 BUY" if data.get('direction', 'BUY') == 'BUY' else "📉 SELL"
        
        # Build response in parts for quick display
        resp_parts = []
        resp_parts.append("🎯 *PRO CALCULATION RESULTS*")
        resp_parts.append("\n*📊 Main Parameters:*")
        resp_parts.append(f"💼 Type: {instrument_display}")
        resp_parts.append(f"🌐 Instrument: {data['currency']}")
        resp_parts.append(f"🎯 Direction: {direction_display}")
        resp_parts.append(f"💵 Deposit: ${data['deposit']:,.2f}")
        resp_parts.append(f"⚖️ Leverage: {data['leverage']}")
        resp_parts.append(f"📈 Entry: {data['entry']}")
        resp_parts.append(f"🛑 Stop loss: {data['stop_loss']}")
        resp_parts.append(f"⚠️ Risk: {data.get('risk_percent', 0.02)*100}%")
        
        resp_parts.append("\n*⚠️ Risk Management:*")
        resp_parts.append(f"📦 Position size: *{pos['position_size']:.2f} lots*")
        resp_parts.append(f"💰 Risk per trade: ${pos['risk_amount']:.2f} ({pos['risk_percent']:.1f}% of deposit)")
        resp_parts.append(f"📉 Stop loss: {pos['stop_pips']:.0f} pips")
        resp_parts.append(f"💳 Required margin: ${pos['required_margin']:.2f}")
        resp_parts.append(f"🆓 Free margin: ${pos['free_margin']:.2f}")
        
        resp_parts.append("\n*📈 Analytics:*")
        resp_parts.append(f"⚖️ R/R ratio: {risk_reward['risk_reward_ratio']:.2f}")
        resp_parts.append(f"🎯 Total risk: {risk_reward['total_risk']:.4f}")
        resp_parts.append(f"🎯 Total reward: {risk_reward['total_reward']:.4f}")
        
        resp_parts.append("\n*🎯 Take profits and profit:*")
        
        total_roi = 0
        for p in profits:
            roi_display = f" | 📊 ROI: {p['roi_percent']:.1f}%" if p['roi_percent'] > 0 else ""
            resp_parts.append(f"\n🎯 TP{p['level']} ({p['volume_percent']}% volume):")
            resp_parts.append(f"   💰 Price: {p['price']}")
            resp_parts.append(f"   📦 Volume: {p['volume_lots']:.2f} lots")
            resp_parts.append(f"   📊 Pips: {p['pips']:.0f} pips")
            resp_parts.append(f"   💵 Profit: ${p['profit']:.2f}{roi_display}")
            resp_parts.append(f"   📈 Cumulative profit: ${p['cumulative_profit']:.2f}")
            total_roi += p['roi_percent']
        
        # Final metrics
        total_profit = profits[-1]['cumulative_profit'] if profits else 0
        overall_roi = (total_profit / data['deposit']) * 100 if data['deposit'] > 0 else 0
        
        resp_parts.append(f"\n*🏆 Final Metrics:*")
        resp_parts.append(f"💰 Total profit: ${total_profit:.2f}")
        resp_parts.append(f"📊 Overall ROI: {overall_roi:.2f}%")
        
        # Recommendations
        resp_parts.append(f"\n*💡 Recommendations:*")
        if risk_reward['risk_reward_ratio'] < 1:
            resp_parts.append(f"⚠️ Low R/R ratio. Consider revising TP/SL")
        elif risk_reward['risk_reward_ratio'] > 2:
            resp_parts.append(f"✅ Excellent R/R ratio!")
            
        if data.get('risk_percent', 0.02) > 0.03:
            resp_parts.append(f"⚠️ High risk! Recommended to reduce to 2-3%")
        else:
            resp_parts.append(f"✅ Risk level is normal")
        
        # Add developer info
        resp_parts.append(f"\n---")
        resp_parts.append(f"👨‍💻 *PRO Developer:* [@fxfeelgood](https://t.me/fxfeelgood)")
        resp_parts.append(f"⚡ *PRO Version 3.0 | Fast • Smart • Accurate*")
        
        # Combine all parts
        final_response = "\n".join(resp_parts)
        
        keyboard = [
            [InlineKeyboardButton("💾 Save Strategy", callback_data="save_preset")],
            [InlineKeyboardButton("🔄 New Calculation", callback_data="new_calculation")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
        ]
        
        # Delete temporary message and send final result
        await quick_response.delete()
        await update.message.reply_text(
            final_response, 
            parse_mode='Markdown', 
            reply_markup=InlineKeyboardMarkup(keyboard),
            disable_web_page_preview=True
        )
        return ConversationHandler.END
        
    except Exception as e:
        logger.error(f"Error in optimized volume distribution: {e}")
        await update.message.reply_text(
            "❌ Error occurred during calculation. Please start over with /start",
            parse_mode='Markdown'
        )
        return ConversationHandler.END

# Back button handlers
@log_performance
async def handle_back_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle all back buttons"""
    query = update.callback_query
    if not query:
        return MAIN_MENU
        
    await query.answer()
    user_id = query.from_user.id
    user_data[user_id]['last_activity'] = time.time()
    
    back_action = query.data
    
    if back_action == "back_to_instruments":
        return await start_pro_calculation(update, context)
    elif back_action == "back_to_direction":
        currency = user_data[user_id].get('currency', 'EURUSD')
        await query.edit_message_text(
            f"✅ *Instrument:* {currency}\n\n"
            "🎯 *Select trade direction:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📈 BUY", callback_data="direction_BUY")],
                [InlineKeyboardButton("📉 SELL", callback_data="direction_SELL")],
                [InlineKeyboardButton("🔙 Back", callback_data="back_to_instruments")]
            ])
        )
        return DIRECTION
    elif back_action == "back_to_risk":
        currency = user_data[user_id].get('currency', 'EURUSD')
        await query.edit_message_text(
            f"✅ *Instrument:* {currency}\n\n"
            "🎯 *Select trade direction:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📈 BUY", callback_data="direction_BUY")],
                [InlineKeyboardButton("📉 SELL", callback_data="direction_SELL")],
                [InlineKeyboardButton("🔙 Back", callback_data="back_to_instruments")]
            ])
        )
        return DIRECTION
    elif back_action == "back_to_deposit":
        await query.edit_message_text(
            "💵 *Enter deposit amount in USD:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_risk")]])
        )
        return DEPOSIT
    elif back_action == "back_to_leverage":
        currency = user_data[user_id].get('currency', 'EURUSD')
        direction = user_data[user_id].get('direction', 'BUY')
        
        await query.edit_message_text(
            f"✅ *Direction:* {direction}\n\n"
            f"📈 *Enter entry price for {currency}:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_entry")]])
        )
        return ENTRY
    elif back_action == "back_to_entry":
        currency = user_data[user_id].get('currency', 'EURUSD')
        direction = user_data[user_id].get('direction', 'BUY')
        
        await query.edit_message_text(
            f"✅ *Direction:* {direction}\n\n"
            f"🛑 *Enter stop loss price for {currency}:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_stop_loss")]])
        )
        return STOP_LOSS
    elif back_action == "back_to_stop_loss":
        currency = user_data[user_id].get('currency', 'EURUSD')
        
        await query.edit_message_text(
            f"✅ *Stop loss:* {user_data[user_id].get('stop_loss', 'N/A')}\n\n"
            f"🎯 *Enter take profit prices for {currency} separated by commas* (e.g.: 1.0550, 1.0460):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_take_profits")]])
        )
        return TAKE_PROFITS
    elif back_action == "back_to_take_profits":
        tps = user_data[user_id].get('take_profits', [])
        
        await query.edit_message_text(
            f"✅ *Take profits:* {', '.join(map(str, tps))}\n\n"
            f"📊 *Enter volume distribution in % for each take profit separated by commas*\n"
            f"(total {len(tps)} values, sum must be 100%):\n"
            f"*Example:* 50, 30, 20",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_volume_distribution")]])
        )
        return VOLUME_DISTRIBUTION
    
    # If action not recognized, return to main menu
    return await start(update, context)

# Quick calculation handlers
@log_performance
async def process_quick_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process quick calculation"""
    if not update.message:
        return CUSTOM_INSTRUMENT
        
    user_id = update.message.from_user.id
    currency = update.message.text.upper().strip()
    
    # Basic ticker validation
    if not re.match(r'^[A-Z0-9]{2,10}$', currency):
        await update.message.reply_text(
            "❌ *Invalid ticker format!*\n\n"
            "Please enter a valid ticker (letters and numbers only, 2-10 characters):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
        )
        return CUSTOM_INSTRUMENT
    
    user_data[user_id] = {
        'currency': currency,
        'direction': 'BUY',
        'risk_percent': 0.02,
        'leverage': '1:100',
        'take_profits': [],
        'volume_distribution': [100],
        'last_activity': time.time()
    }
    
    # Determine instrument type
    if any(x in currency for x in ['BTC', 'ETH', 'XRP', 'ADA']):
        user_data[user_id]['instrument_type'] = 'crypto'
    elif any(x in currency for x in ['XAU', 'XAG', 'XPT', 'XPD']):
        user_data[user_id]['instrument_type'] = 'metals'
    elif currency.isalpha() and len(currency) == 6:
        user_data[user_id]['instrument_type'] = 'forex'
    else:
        user_data[user_id]['instrument_type'] = 'indices'
    
    await update.message.reply_text(
        f"✅ *Instrument:* {currency}\n\n"
        "💵 *Enter deposit amount in USD:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
    )
    return DEPOSIT

# Additional handlers
@log_performance
async def save_preset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Save preset"""
    query = update.callback_query
    if not query:
        return
        
    await query.answer()
    uid = query.from_user.id
    
    if uid not in user_data:
        await query.edit_message_text("❌ Error: data not found. Start new calculation with /start")
        return
        
    if 'presets' not in user_data[uid]:
        user_data[uid]['presets'] = []
    
    # Limit number of saved presets
    if len(user_data[uid]['presets']) >= 20:
        user_data[uid]['presets'] = user_data[uid]['presets'][-19:]
    
    # Save only strategy keys
    strategy_data = {}
    keys_to_save = ['instrument_type', 'currency', 'direction', 'risk_percent', 
                   'deposit', 'leverage', 'entry', 'stop_loss', 'take_profits', 
                   'volume_distribution']
    
    for key in keys_to_save:
        if key in user_data[uid]:
            strategy_data[key] = user_data[uid][key]
    
    user_data[uid]['presets'].append({
        'timestamp': datetime.now().isoformat(),
        'data': strategy_data
    })
    
    await query.edit_message_text(
        "✅ *PRO Strategy successfully saved!*\n\n"
        "💾 Use /presets to view saved strategies\n"
        "🚀 Use /start for new PRO calculation\n\n"
        "👨‍💻 *PRO Developer:* [@fxfeelgood](https://t.me/fxfeelgood)",
        parse_mode='Markdown',
        disable_web_page_preview=True
    )

@log_performance
async def show_presets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show saved presets"""
    user_id = update.message.from_user.id
    presets = user_data.get(user_id, {}).get('presets', [])
    
    if not presets:
        await update.message.reply_text(
            "📝 *You have no saved PRO strategies.*\n\n"
            "💡 Save your strategies after calculation for quick access!",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
        )
        return
    
    await update.message.reply_text(
        f"📚 *Your PRO Strategies ({len(presets)}):*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]])
    )
    
    for i, p in enumerate(presets[-10:], 1):
        d = p['data']
        instrument_display = INSTRUMENT_TYPES.get(d.get('instrument_type', 'forex'), 'Forex')
        
        preset_text = f"""
📋 *PRO Strategy #{i}*
💼 Type: {instrument_display}
🌐 Instrument: {d.get('currency', 'N/A')}
💵 Deposit: ${d.get('deposit', 0):,.2f}
⚖️ Leverage: {d.get('leverage', 'N/A')}
📈 Entry: {d.get('entry', 'N/A')}
🛑 SL: {d.get('stop_loss', 'N/A')}
🎯 TP: {', '.join(map(str, d.get('take_profits', [])))}

👨‍💻 *PRO Developer:* [@fxfeelgood](https://t.me/fxfeelgood)
"""
        await update.message.reply_text(
            preset_text,
            parse_mode='Markdown',
            disable_web_page_preview=True
        )

@log_performance
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel conversation"""
    if update.message:
        await update.message.reply_text(
            "❌ *PRO Calculation cancelled.*\n\n"
            "🚀 Use /start for new PRO calculation\n"
            "📚 Use /info for PRO instructions\n\n"
            "👨‍💻 *PRO Developer:* [@fxfeelgood](https://t.me/fxfeelgood)",
            parse_mode='Markdown',
            disable_web_page_preview=True
        )
    return ConversationHandler.END

@log_performance
async def new_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """New calculation"""
    query = update.callback_query
    if query:
        await query.answer()
        await start(update, context)

def main():
    """Fixed main function to run bot"""
    token = os.getenv('TELEGRAM_BOT_TOKEN_EN')
    if not token:
        logger.error("❌ PRO Bot token not found!")
        return

    logger.info("🚀 Starting optimized PRO Risk Management Bot v3.0...")
    
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
            MAIN_MENU: [CallbackQueryHandler(handle_main_menu, pattern='^(pro_calculation|quick_calculation|portfolio|analytics|pro_info|main_menu)$')],
            INSTRUMENT_TYPE: [CallbackQueryHandler(process_instrument_type, pattern='^inst_type_')],
            CUSTOM_INSTRUMENT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_currency_input),
                CallbackQueryHandler(process_currency_selection, pattern='^(currency_|custom_instrument)'),
                CallbackQueryHandler(handle_back_buttons, pattern='^back_to_instruments$')
            ],
            CURRENCY: [
                CallbackQueryHandler(process_currency_selection, pattern='^(currency_|custom_instrument)'),
                CallbackQueryHandler(handle_back_buttons, pattern='^back_to_instruments$')
            ],
            DIRECTION: [
                CallbackQueryHandler(process_direction, pattern='^direction_'),
                CallbackQueryHandler(handle_back_buttons, pattern='^back_to_instruments$')
            ],
            RISK_PERCENT: [
                CallbackQueryHandler(process_risk_percent, pattern='^risk_'),
                CallbackQueryHandler(handle_back_buttons, pattern='^back_to_direction$')
            ],
            DEPOSIT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_deposit),
                CallbackQueryHandler(handle_back_buttons, pattern='^back_to_deposit$')
            ],
            LEVERAGE: [
                CallbackQueryHandler(process_leverage, pattern='^leverage_'),
                CallbackQueryHandler(handle_back_buttons, pattern='^back_to_deposit$')
            ],
            ENTRY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_entry),
                CallbackQueryHandler(handle_back_buttons, pattern='^back_to_entry$')
            ],
            STOP_LOSS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_stop_loss),
                CallbackQueryHandler(handle_back_buttons, pattern='^back_to_stop_loss$')
            ],
            TAKE_PROFITS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_take_profits),
                CallbackQueryHandler(handle_back_buttons, pattern='^back_to_take_profits$')
            ],
            VOLUME_DISTRIBUTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_volume_distribution),
                CallbackQueryHandler(handle_back_buttons, pattern='^back_to_volume_distribution$')
            ],
        },
        fallbacks=[
            CommandHandler('cancel', cancel),
            CommandHandler('start', start),
            CommandHandler('presets', show_presets)
        ]
    )

    # Add handlers in correct order
    application.add_handler(conv_handler)
    application.add_handler(CallbackQueryHandler(save_preset, pattern='^save_preset$'))
    application.add_handler(CallbackQueryHandler(new_calculation, pattern='^new_calculation$'))
    application.add_handler(CallbackQueryHandler(handle_back_buttons, pattern='^back_to_'))

    # Add global command handlers
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('presets', show_presets))
    application.add_handler(CommandHandler('portfolio', portfolio_command))
    application.add_handler(CommandHandler('analytics', analytics_command))
    application.add_handler(CommandHandler('info', pro_info_command))
    application.add_handler(CommandHandler('quick', quick_command))

    # Get webhook URL
    webhook_url = os.getenv('RENDER_EXTERNAL_URL_EN', '')
    
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
        import traceback
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()
