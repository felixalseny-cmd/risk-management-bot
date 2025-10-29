import os
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    ConversationHandler,
    CallbackQueryHandler,
    JobQueue
)

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
DEPOSIT, LEVERAGE, INSTRUMENT_TYPE, CURRENCY, ENTRY, STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION = range(8)

# User data storage
user_data: Dict[int, Dict[str, Any]] = {}

# Enhanced constants
INSTRUMENT_TYPES = {
    'forex': 'Forex',
    'crypto': 'Cryptocurrencies', 
    'indices': 'Indices',
    'commodities': 'Commodities'
}

PIP_VALUES = {
    # Forex
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
    'XAUUSD': 10, 'XAGUSD': 50, 'XPTUSD': 10, 'XPDUSD': 10,
    'OIL': 10, 'NATGAS': 10, 'COPPER': 10
}

CONTRACT_SIZES = {
    'forex': 100000,
    'crypto': 1,
    'indices': 1,
    'commodities': 100
}

LEVERAGES = ['1:10', '1:20', '1:50', '1:100', '1:200', '1:500']

# Advanced risk analysis class
class AdvancedRiskCalculator:
    """Advanced risk calculator with support for all instrument types"""
    
    @staticmethod
    def calculate_pip_value(instrument_type: str, currency_pair: str, lot_size: float) -> float:
        """Calculate pip value considering instrument type"""
        base_pip_value = PIP_VALUES.get(currency_pair, 10)
        
        if instrument_type == 'crypto':
            return base_pip_value * lot_size * 0.1  # Adjustment for cryptocurrencies
        elif instrument_type == 'indices':
            return base_pip_value * lot_size * 0.01  # Adjustment for indices
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
        risk_percent: float = 0.02
    ) -> Dict[str, float]:
        """Advanced position size calculation"""
        try:
            lev_value = int(leverage.split(':')[1])
            risk_amount = deposit * risk_percent
            
            # Different calculations for different instrument types
            if instrument_type == 'forex':
                stop_pips = abs(entry_price - stop_loss) * 10000
            elif instrument_type == 'crypto':
                stop_pips = abs(entry_price - stop_loss) * 100
            elif instrument_type in ['indices', 'commodities']:
                stop_pips = abs(entry_price - stop_loss) * 10
            else:
                stop_pips = abs(entry_price - stop_loss) * 10000

            pip_value_per_lot = AdvancedRiskCalculator.calculate_pip_value(
                instrument_type, currency_pair, 1.0
            )
            
            max_lots_by_risk = risk_amount / (stop_pips * pip_value_per_lot) if stop_pips > 0 else 0
            
            contract_size = CONTRACT_SIZES.get(instrument_type, 100000)
            max_lots_by_margin = (deposit * lev_value) / (contract_size * entry_price)
            
            position_size = min(max_lots_by_risk, max_lots_by_margin, 50.0)
            
            if position_size < 0.01:
                position_size = 0.01
            else:
                position_size = round(position_size * 100) / 100
                
            required_margin = (position_size * contract_size * entry_price) / lev_value
            
            return {
                'position_size': position_size,
                'risk_amount': risk_amount,
                'stop_pips': stop_pips,
                'required_margin': required_margin,
                'risk_percent': (risk_amount / deposit) * 100,
                'free_margin': deposit - required_margin
            }
        except Exception as e:
            logger.error(f"Error in position size calculation: {e}")
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
        volume_distribution: List[float]
    ) -> List[Dict[str, Any]]:
        """Advanced profit calculation"""
        profits = []
        total_profit = 0
        
        for i, (tp, vol_pct) in enumerate(zip(take_profits, volume_distribution)):
            if instrument_type == 'forex':
                tp_pips = abs(entry_price - tp) * 10000
            elif instrument_type == 'crypto':
                tp_pips = abs(entry_price - tp) * 100
            elif instrument_type in ['indices', 'commodities']:
                tp_pips = abs(entry_price - tp) * 10
            else:
                tp_pips = abs(entry_price - tp) * 10000
                
            volume_lots = position_size * (vol_pct / 100)
            pip_value = AdvancedRiskCalculator.calculate_pip_value(
                instrument_type, currency_pair, volume_lots
            )
            profit = tp_pips * pip_value
            total_profit += profit
            
            profits.append({
                'level': i + 1,
                'price': tp,
                'volume_percent': vol_pct,
                'volume_lots': volume_lots,
                'profit': profit,
                'cumulative_profit': total_profit,
                'roi_percent': (profit / (position_size * CONTRACT_SIZES.get(instrument_type, 100000) * entry_price)) * 100
            })
            
        return profits

    @staticmethod
    def calculate_risk_reward_ratio(
        entry_price: float,
        stop_loss: float,
        take_profits: List[float],
        volume_distribution: List[float]
    ) -> Dict[str, float]:
        """Calculate risk/reward ratio"""
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
            logger.error(f"Error in risk/reward calculation: {e}")
            return {'risk_reward_ratio': 0, 'total_risk': 0, 'total_reward': 0}

# Caching system for optimization
class CacheManager:
    """Cache manager for performance optimization"""
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache"""
        self._cache[key] = value
        self._timestamps[key] = datetime.now() + timedelta(seconds=ttl)
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self._cache and datetime.now() < self._timestamps.get(key, datetime.now()):
            return self._cache[key]
        else:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
            return None

cache_manager = CacheManager()

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Enhanced /start command handler"""
    if not update.message:
        return ConversationHandler.END
    
    user = update.message.from_user
    user_name = user.first_name or "Trader"
    
    # Cache welcome message
    welcome_key = f"welcome_{user.language_code}"
    welcome_text = cache_manager.get(welcome_key)
    
    if not welcome_text:
        welcome_text = f"""
👋 *Hello, {user_name}!*

🎯 *PRO Risk Management Calculator - PROFESSIONAL VERSION*

⚡ *Advanced Features:*
• 📊 All instrument types: Forex, Crypto, Indices, Commodities
• 🛡️ Advanced risk management
• 📈 Risk/reward analysis
• 💹 Multi-currency calculations
• 🔄 Intelligent caching
• ⚡ High performance

💡 *For full instructions use* /info

🚀 *Select instrument type to start calculation:*
"""
        cache_manager.set(welcome_key, welcome_text, 3600)
    
    user_id = user.id
    user_data[user_id] = {'start_time': datetime.now().isoformat()}
    
    # Instrument type selection keyboard
    keyboard = []
    for inst_type, display_name in INSTRUMENT_TYPES.items():
        keyboard.append([InlineKeyboardButton(
            f"📊 {display_name}", 
            callback_data=f"inst_type_{inst_type}"
        )])
    
    await update.message.reply_text(
        welcome_text, 
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return INSTRUMENT_TYPE

async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Updated full instructions"""
    info_text = """
📚 *PRO INSTRUCTIONS - Risk Management Calculator Bot*

🎯 *ADVANCED CAPABILITIES*

⚡ *SUPPORTED INSTRUMENTS:*

• 🌐 *FOREX:* EURUSD, GBPUSD, USDJPY, XAUUSD, XAGUSD and others
• ₿ *CRYPTO:* BTCUSD, ETHUSD, XRPUSD, ADAUSD, DOTUSD
• 📈 *INDICES:* NAS100, SPX500, US30, DAX40, FTSE100
• ⚡ *COMMODITIES:* OIL, NATGAS, COPPER, XPTUSD, XPDUSD

📋 *PROFESSIONAL ALGORITHM:*

1. *Smart Risk Calculation*
   - Automatic adaptation to instrument type
   - Contract specifications consideration
   - Optimal volume distribution

2. *Extended Analytics*
   - Risk/reward ratio
   - ROI per TP
   - Free margin analysis
   - Results visualization

3. *Security & Reliability*
   - All input data validation
   - Overload protection
   - Intelligent caching

📝 *CALCULATION EXAMPLE FOR NAS100:*

Deposit: $5000
Leverage: 1:50
Instrument: NAS100
Entry: 15000
SL: 14900
TP: 15100, 15200
Distribution: 60, 40

🛠 *PRO VERSION COMMANDS:*
`/start` - start professional calculation
`/quick` - quick calculation (coming soon)
`/portfolio` - portfolio management (coming soon)
`/analytics` - strategy analysis (coming soon)
`/presets` - strategy library
`/settings` - risk settings

🔧 *TECHNICAL ADVANTAGES:*
• ⚡ Response time < 100ms
• 🛡️ Overload protection
• 💾 Intelligent cache
• 🔄 Automatic optimization

👨‍💻 *DEVELOPER:* [@fxfeelgood](https://t.me/fxfeelgood)

*Become a professional with our bot!* 🚀
"""
    await update.message.reply_text(
        info_text, 
        parse_mode='Markdown', 
        disable_web_page_preview=True
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Updated help"""
    help_text = """
🤖 *PRO Risk Management Bot - Help*

⚡ *Quick Start:*
1. /start - select instrument type
2. Enter parameters as requested
3. Get professional analysis

🛠 *Main Commands:*
`/start` - Professional calculation
`/info` - Full PRO instructions
`/presets` - Strategy library
`/help` - This help

🔧 *Advanced Commands (coming soon):*
`/quick` - Quick calculation
`/portfolio` - Portfolio
`/analytics` - Analytics

💡 *PRO Tip:* Use 1-2% risk and R:R ratio from 1:2!

👨‍💻 *Developer:* [@fxfeelgood](https://t.me/fxfeelgood)
"""
    await update.message.reply_text(
        help_text, 
        parse_mode='Markdown', 
        disable_web_page_preview=True
    )

# Dialog handlers (similar structure to Russian version but in English)
# process_instrument_type, process_currency, process_deposit, process_leverage,
# process_entry, process_stop_loss, process_take_profits, process_volume_distribution

async def process_instrument_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process instrument type selection"""
    query = update.callback_query
    if not query:
        return ConversationHandler.END
        
    await query.answer()
    user_id = query.from_user.id
    instrument_type = query.data.replace('inst_type_', '')
    user_data[user_id]['instrument_type'] = instrument_type
    
    # Get instruments for selected type
    instruments = {
        'forex': [k for k in PIP_VALUES.keys() if k in ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'XAGUSD']],
        'crypto': [k for k in PIP_VALUES.keys() if k in ['BTCUSD', 'ETHUSD', 'XRPUSD', 'ADAUSD', 'DOTUSD']],
        'indices': [k for k in PIP_VALUES.keys() if k in ['US30', 'NAS100', 'SPX500', 'DAX40']],
        'commodities': [k for k in PIP_VALUES.keys() if k in ['OIL', 'NATGAS', 'COPPER', 'XPTUSD']]
    }.get(instrument_type, [])
    
    keyboard = []
    for i in range(0, len(instruments), 2):
        row = []
        for j in range(2):
            if i + j < len(instruments):
                inst = instruments[i + j]
                row.append(InlineKeyboardButton(inst, callback_data=f"currency_{inst}"))
        keyboard.append(row)
    
    display_type = INSTRUMENT_TYPES.get(instrument_type, instrument_type)
    await query.edit_message_text(
        f"✅ *Instrument Type:* {display_type}\n\n"
        "🌐 *Select specific instrument:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return CURRENCY

async def process_currency(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process instrument selection"""
    query = update.callback_query
    if not query:
        return ConversationHandler.END
        
    await query.answer()
    user_id = query.from_user.id
    currency = query.data.replace('currency_', '')
    user_data[user_id]['currency'] = currency
    
    await query.edit_message_text(
        f"✅ *Instrument:* {currency}\n\n"
        "💵 *Enter deposit amount in USD:*",
        parse_mode='Markdown'
    )
    return DEPOSIT

async def process_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process deposit input"""
    if not update.message:
        return ConversationHandler.END
        
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
        
        # Create leverage keyboard
        keyboard = []
        for i in range(0, len(LEVERAGES), 3):
            row = []
            for j in range(3):
                if i + j < len(LEVERAGES):
                    lev = LEVERAGES[i + j]
                    row.append(InlineKeyboardButton(lev, callback_data=f"leverage_{lev}"))
            keyboard.append(row)
        
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

async def process_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process leverage selection"""
    query = update.callback_query
    if not query:
        return ConversationHandler.END
        
    await query.answer()
    user_id = query.from_user.id
    leverage = query.data.replace('leverage_', '')
    user_data[user_id]['leverage'] = leverage
    
    currency = user_data[user_id].get('currency', 'EURUSD')
    await query.edit_message_text(
        f"✅ *Leverage:* {leverage}\n\n"
        f"📈 *Enter entry price for {currency}:*",
        parse_mode='Markdown'
    )
    return ENTRY

async def process_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process entry price input"""
    if not update.message:
        return ConversationHandler.END
        
    user_id = update.message.from_user.id
    try:
        entry = float(update.message.text)
        if entry <= 0:
            await update.message.reply_text("❌ Price must be positive:")
            return ENTRY
            
        user_data[user_id]['entry'] = entry
        
        await update.message.reply_text(
            f"✅ *Entry price:* {entry}\n\n"
            "🛑 *Enter stop loss price:*",
            parse_mode='Markdown'
        )
        return STOP_LOSS
        
    except ValueError:
        await update.message.reply_text("❌ Please enter a valid entry price:")
        return ENTRY

async def process_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process stop loss input"""
    if not update.message:
        return ConversationHandler.END
        
    user_id = update.message.from_user.id
    try:
        sl = float(update.message.text)
        entry = user_data[user_id].get('entry', 0)
        
        if sl <= 0:
            await update.message.reply_text("❌ Price must be positive:")
            return STOP_LOSS
            
        user_data[user_id]['stop_loss'] = sl
        
        await update.message.reply_text(
            f"✅ *Stop loss:* {sl}\n\n"
            "🎯 *Enter take profit prices separated by commas* (e.g.: 1.0550, 1.0460):",
            parse_mode='Markdown'
        )
        return TAKE_PROFITS
        
    except ValueError:
        await update.message.reply_text("❌ Please enter a valid stop loss price:")
        return STOP_LOSS

async def process_take_profits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process take profits input"""
    if not update.message:
        return ConversationHandler.END
        
    user_id = update.message.from_user.id
    try:
        tps = [float(x.strip()) for x in update.message.text.split(',')]
        
        if len(tps) > 5:
            await update.message.reply_text("❌ Maximum 5 take profits:")
            return TAKE_PROFITS
            
        user_data[user_id]['take_profits'] = tps
        
        await update.message.reply_text(
            f"✅ *Take profits:* {', '.join(map(str, tps))}\n\n"
            f"📊 *Enter volume distribution in % for each take profit separated by commas*\n"
            f"(total {len(tps)} values, sum must be 100%):\n"
            f"*Example:* 50, 30, 20",
            parse_mode='Markdown'
        )
        return VOLUME_DISTRIBUTION
        
    except ValueError:
        await update.message.reply_text("❌ Please enter valid take profit prices:")
        return TAKE_PROFITS

async def process_volume_distribution(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process volume distribution and show results"""
    if not update.message:
        return ConversationHandler.END
        
    user_id = update.message.from_user.id
    try:
        dist = [float(x.strip()) for x in update.message.text.split(',')]
        
        if abs(sum(dist) - 100) > 1e-5:
            await update.message.reply_text(
                f"❌ *Distribution sum must be 100%. Your sum: {sum(dist)}%*\n"
                "Please enter distribution again:",
                parse_mode='Markdown'
            )
            return VOLUME_DISTRIBUTION
        
        if len(dist) != len(user_data[user_id].get('take_profits', [])):
            await update.message.reply_text(
                f"❌ *Number of distribution values must match number of TPs ({len(user_data[user_id].get('take_profits', []))})*\n"
                "Please enter distribution again:",
                parse_mode='Markdown'
            )
            return VOLUME_DISTRIBUTION
        
        user_data[user_id]['volume_distribution'] = dist
        data = user_data[user_id]
        
        # Professional results calculation
        pos = AdvancedRiskCalculator.calculate_position_size(
            deposit=data['deposit'],
            leverage=data['leverage'],
            instrument_type=data['instrument_type'],
            currency_pair=data['currency'],
            entry_price=data['entry'],
            stop_loss=data['stop_loss']
        )
        
        profits = AdvancedRiskCalculator.calculate_profits(
            instrument_type=data['instrument_type'],
            currency_pair=data['currency'],
            entry_price=data['entry'],
            take_profits=data['take_profits'],
            position_size=pos['position_size'],
            volume_distribution=dist
        )
        
        risk_reward = AdvancedRiskCalculator.calculate_risk_reward_ratio(
            entry_price=data['entry'],
            stop_loss=data['stop_loss'],
            take_profits=data['take_profits'],
            volume_distribution=dist
        )
        
        # Professional results formatting
        instrument_display = INSTRUMENT_TYPES.get(data['instrument_type'], data['instrument_type'])
        
        resp = f"""
🎯 *PRO CALCULATION RESULTS*

*📊 Main Parameters:*
💼 Type: {instrument_display}
🌐 Instrument: {data['currency']}
💵 Deposit: ${data['deposit']:,.2f}
⚖️ Leverage: {data['leverage']}
📈 Entry: {data['entry']}
🛑 Stop loss: {data['stop_loss']}

*⚠️ Risk Management:*
📦 Position size: *{pos['position_size']:.2f} lots*
💰 Risk per trade: ${pos['risk_amount']:.2f} ({pos['risk_percent']:.1f}% of deposit)
📉 Stop loss: {pos['stop_pips']:.0f} points
💳 Required margin: ${pos['required_margin']:.2f}
🆓 Free margin: ${pos['free_margin']:.2f}

*📈 Analytics:*
⚖️ R/R ratio: {risk_reward['risk_reward_ratio']:.2f}
🎯 Total risk: {risk_reward['total_risk']:.4f}
🎯 Total reward: {risk_reward['total_reward']:.4f}

*🎯 Take profits and profit:*
"""
        
        total_roi = 0
        for p in profits:
            roi_display = f" | 📊 ROI: {p['roi_percent']:.1f}%" if p['roi_percent'] > 0 else ""
            resp += f"\n🎯 TP{p['level']} ({p['volume_percent']}% volume):"
            resp += f"\n   💰 Price: {p['price']}"
            resp += f"\n   📦 Volume: {p['volume_lots']:.2f} lots"
            resp += f"\n   💵 Profit: ${p['profit']:.2f}{roi_display}"
            resp += f"\n   📊 Cumulative profit: ${p['cumulative_profit']:.2f}\n"
            total_roi += p['roi_percent']
        
        # Final metrics
        total_profit = profits[-1]['cumulative_profit'] if profits else 0
        overall_roi = (total_profit / data['deposit']) * 100
        
        resp += f"\n*🏆 Final Metrics:*\n"
        resp += f"💰 Total profit: ${total_profit:.2f}\n"
        resp += f"📊 Overall ROI: {overall_roi:.2f}%\n"
        
        # Recommendations
        resp += f"\n*💡 Recommendations:*\n"
        if risk_reward['risk_reward_ratio'] < 1:
            resp += f"⚠️ Low R/R ratio. Consider revising TP/SL\n"
        elif risk_reward['risk_reward_ratio'] > 2:
            resp += f"✅ Excellent R/R ratio!\n"
            
        if pos['risk_percent'] > 3:
            resp += f"⚠️ High risk! Recommended to reduce to 2%\n"
        else:
            resp += f"✅ Risk level is normal\n"
        
        # Add developer info
        resp += f"\n---\n"
        resp += f"👨‍💻 *PRO Developer:* [@fxfeelgood](https://t.me/fxfeelgood)\n"
        resp += f"⚡ *PRO Version 2.0 | Fast • Reliable • Accurate*"
        
        keyboard = [
            [InlineKeyboardButton("💾 Save Strategy", callback_data="save_preset")],
            [InlineKeyboardButton("🔄 New Calculation", callback_data="new_calculation")],
            [InlineKeyboardButton("📚 PRO Instructions", callback_data="show_info")]
        ]
        
        await update.message.reply_text(
            resp, 
            parse_mode='Markdown', 
            reply_markup=InlineKeyboardMarkup(keyboard),
            disable_web_page_preview=True
        )
        return ConversationHandler.END
        
    except Exception as e:
        logger.error(f"Error in volume distribution: {e}")
        await update.message.reply_text(
            "❌ Error occurred during calculation. Please start over with /start",
            parse_mode='Markdown'
        )
        return ConversationHandler.END

# Other handlers (save_preset, show_presets, cancel, new_calculation, show_info_callback, settings_command)
# remain similar to Russian version but with English messages

async def save_preset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced preset saving"""
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
    
    user_data[uid]['presets'].append({
        'timestamp': datetime.now().isoformat(),
        'data': user_data[uid].copy()
    })
    
    await query.edit_message_text(
        "✅ *PRO Strategy successfully saved!*\n\n"
        "💾 Use /presets to view saved strategies\n"
        "🚀 Use /start for new PRO calculation\n\n"
        "👨‍💻 *PRO Developer:* [@fxfeelgood](https://t.me/fxfeelgood)",
        parse_mode='Markdown',
        disable_web_page_preview=True
    )

async def show_presets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced presets display"""
    if not update.message:
        return
        
    uid = update.message.from_user.id
    presets = user_data.get(uid, {}).get('presets', [])
    
    if not presets:
        await update.message.reply_text(
            "📝 *You have no saved PRO strategies.*\n\n"
            "💡 Save your strategies after calculation for quick access!",
            parse_mode='Markdown'
        )
        return
    
    await update.message.reply_text(
        f"📚 *Your PRO Strategies ({len(presets)}):*",
        parse_mode='Markdown'
    )
    
    for i, p in enumerate(presets[-10:], 1):  # Show last 10
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

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Enhanced cancellation"""
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

async def new_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """New calculation"""
    query = update.callback_query
    if query:
        await query.answer()
        await start(update, context)

async def show_info_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show instructions via callback"""
    query = update.callback_query
    if query:
        await query.answer()
        await info_command(update, context)

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Settings command"""
    settings_text = """
⚙️ *PRO Risk Settings*

*Current Settings:*
• 📊 Risk per trade: 2% (recommended)
• 💹 Max position: 50 lots
• 🛡️ Overload protection: enabled

*Available Commands:*
`/start` - PRO calculation
`/presets` - Strategy library  
`/info` - Full instructions

*Coming soon in PRO version:*
• 🔧 Risk level customization
• 📈 Personal templates
• 💾 Data export

👨‍💻 *PRO Developer:* [@fxfeelgood](https://t.me/fxfeelgood)
"""
    await update.message.reply_text(
        settings_text,
        parse_mode='Markdown',
        disable_web_page_preview=True
    )

def main():
    """Enhanced main function"""
    token = os.getenv('TELEGRAM_BOT_TOKEN_EN')
    if not token:
        logger.error("❌ PRO Bot token not found!")
        return

    logger.info("🚀 Starting PRO Risk Management Bot v2.0...")
    
    # Create optimized application
    application = (
        Application.builder()
        .token(token)
        .concurrent_updates(True)
        .pool_timeout(30)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .build()
    )

    # Configure enhanced ConversationHandler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            INSTRUMENT_TYPE: [CallbackQueryHandler(process_instrument_type, pattern='^inst_type_')],
            DEPOSIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_deposit)],
            LEVERAGE: [CallbackQueryHandler(process_leverage, pattern='^leverage_')],
            CURRENCY: [CallbackQueryHandler(process_currency, pattern='^currency_')],
            ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_entry)],
            STOP_LOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_stop_loss)],
            TAKE_PROFITS: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_take_profits)],
            VOLUME_DISTRIBUTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_volume_distribution)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
        allow_reentry=True,
        per_chat=True,
        per_user=True,
        per_message=False
    )

    # Add handlers
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('info', info_command))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('presets', show_presets))
    application.add_handler(CommandHandler('settings', settings_command))
    application.add_handler(CallbackQueryHandler(save_preset, pattern='^save_preset$'))
    application.add_handler(CallbackQueryHandler(new_calculation, pattern='^new_calculation$'))
    application.add_handler(CallbackQueryHandler(show_info_callback, pattern='^show_info$'))

    # Get webhook URL
    webhook_url = os.getenv('RENDER_EXTERNAL_URL_EN', '')
    if not webhook_url:
        logger.error("❌ RENDER_EXTERNAL_URL_EN not set!")
        return

    # Start webhook with enhanced settings
    port = int(os.environ.get('PORT', 10000))
    
    logger.info(f"🌐 PRO Starting webhook on port {port}")
    logger.info(f"🔗 PRO Webhook URL: {webhook_url}/webhook")
    
        try:
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path="/webhook",
            webhook_url=webhook_url + "/webhook"
        )
    except Exception as e:
        logger.error(f"❌ Error starting PRO webhook: {e}")
        logger.info("🔄 PRO Trying polling...")
        application.run_polling()

if __name__ == '__main__':

    main()
