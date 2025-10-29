import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any
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

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
DEPOSIT, LEVERAGE, CURRENCY, ENTRY, STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION = range(7)

# User data storage
user_data: Dict[int, Dict[str, Any]] = {}

# Constants
PIP_VALUES = {
    'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9, 'USDCHF': 10,
    'USDCAD': 10, 'AUDUSD': 10, 'NZDUSD': 10, 'EURGBP': 10,
    'EURJPY': 9, 'GBPJPY': 9, 'XAUUSD': 10, 'XAGUSD': 50,
    'BTCUSD': 1, 'ETHUSD': 1, 'XAUUSD': 10, 'XAGUSD': 50
}
LEVERAGES = ['1:100', '1:200', '1:500', '1:1000', '1:2000']


class RiskCalculator:
    @staticmethod
    def calculate_pip_value(currency_pair: str, lot_size: float) -> float:
        base_pip_value = PIP_VALUES.get(currency_pair, 10)
        return base_pip_value * lot_size

    @staticmethod
    def calculate_position_size(
        deposit: float,
        leverage: str,
        currency_pair: str,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = 0.02
    ) -> Dict[str, float]:
        lev_value = int(leverage.split(':')[1])
        risk_amount = deposit * risk_percent
        stop_pips = abs(entry_price - stop_loss) * 10000
        pip_value_per_lot = RiskCalculator.calculate_pip_value(currency_pair, 1.0)
        max_lots_by_risk = risk_amount / (stop_pips * pip_value_per_lot) if stop_pips > 0 else 0
        contract_size = 100000
        max_lots_by_margin = (deposit * lev_value) / contract_size
        position_size = min(max_lots_by_risk, max_lots_by_margin, 10.0)
        if position_size < 0.01:
            position_size = 0.01
        else:
            position_size = round(position_size * 100) / 100
        required_margin = (position_size * contract_size) / lev_value
        return {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'stop_pips': stop_pips,
            'required_margin': required_margin,
            'risk_percent': (risk_amount / deposit) * 100,
        }

    @staticmethod
    def calculate_profits(
        currency_pair: str,
        entry_price: float,
        take_profits: List[float],
        position_size: float,
        volume_distribution: List[float]
    ) -> List[Dict[str, Any]]:
        profits = []
        total_profit = 0
        for i, (tp, vol_pct) in enumerate(zip(take_profits, volume_distribution)):
            tp_pips = abs(entry_price - tp) * 10000
            volume_lots = position_size * (vol_pct / 100)
            pip_value = RiskCalculator.calculate_pip_value(currency_pair, volume_lots)
            profit = tp_pips * pip_value
            total_profit += profit
            profits.append({
                'level': i + 1,
                'price': tp,
                'volume_percent': vol_pct,
                'volume_lots': volume_lots,
                'profit': profit,
                'cumulative_profit': total_profit
            })
        return profits


# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message:
        return ConversationHandler.END
    
    user = update.message.from_user
    user_name = user.first_name or "Trader"
    
    welcome_text = f"""
👋 *Hello, {user_name}!*

🎯 *Risk Management Calculator Bot for FOREX*

I will help you calculate the optimal position size with professional risk management.

📊 *What I can do:*
• Calculate position size considering 2% risk of deposit
• Account for leverage and margin requirements
• Distribute volumes between take profits
• Calculate profit and risks in real-time

💡 *For full instructions use* /info

🚀 *Let's start! Enter your deposit amount in USD:*
"""
    
    user_id = user.id
    user_data[user_id] = {}
    await update.message.reply_text(welcome_text, parse_mode='Markdown')
    return DEPOSIT

async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Full usage instructions"""
    info_text = """
📚 *FULL INSTRUCTIONS - Risk Management Calculator Bot*

🎯 *DESCRIPTION*
Professional risk management calculator for FOREX market. Helps determine optimal position size considering deposit, leverage and stop loss.

📋 *HOW TO USE:*

1. *Getting Started*
   Command: `/start`
   - Bot will request main trade parameters

2. *Input Parameters*
   - 💰 Deposit (USD)
   - ⚖️ Leverage (1:100, 1:200, etc.)
   - 🌐 Currency pair (EURUSD, GBPUSD, XAUUSD, etc.)
   - 📈 Entry price
   - 🛑 Stop loss price
   - 🎯 Take profit prices (comma separated)
   - 📊 Volume distribution between TPs (in %)

3. *Calculation Results*
   - 📊 Optimal position size
   - ⚠️ Risk amount (2% of deposit)
   - 📉 Stop loss in pips
   - 💳 Required margin
   - 💰 Profit for each TP

📝 *CALCULATION EXAMPLE:*

Deposit: $1000
Leverage: 1:100
Pair: EURUSD
Entry: 1.0660
SL: 1.0640
TP: 1.0680, 1.0700
Distribution: 50, 50


🛠 *AVAILABLE COMMANDS:*
`/start` - start calculation
`/info` - full instructions  
`/help` - quick help
`/presets` - saved strategies

🔮 *COMING SOON:*
• Gold, Silver, WTI, NASDAQ, S&P500
• Bitcoin and other cryptocurrencies
• Advanced analysis tools

👨‍💻 *DEVELOPER:*
For questions and suggestions: [@fxfeelgood](https://t.me/fxfeelgood)

*Best regards,*
*Your reliable trading assistant!* 📈
"""
    await update.message.reply_text(info_text, parse_mode='Markdown', disable_web_page_preview=True)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick help"""
    help_text = """
🤖 *Risk Management Bot - Help*

📖 For full instructions use: /info

⚡ *Quick Start:*
1. /start - begin calculation
2. Enter parameters as requested by bot
3. Get ready risk calculations

🛠 *Main Commands:*
`/start` - New calculation
`/info` - Full instructions
`/presets` - My strategies
`/help` - This help

💡 *Tip:* Always use risk no more than 2% of deposit!

👨‍💻 *Developer:* [@fxfeelgood](https://t.me/fxfeelgood)
"""
    await update.message.reply_text(help_text, parse_mode='Markdown', disable_web_page_preview=True)

async def process_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message:
        return ConversationHandler.END
    user_id = update.message.from_user.id
    try:
        deposit = float(update.message.text.replace(',', '').replace(' ', ''))
        if deposit <= 0:
            raise ValueError
        user_data[user_id]['deposit'] = deposit
        keyboard = [[InlineKeyboardButton(l, callback_data=f"leverage_{l}")] for l in LEVERAGES]
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
    query = update.callback_query
    if not query:
        return ConversationHandler.END
    await query.answer()
    user_id = query.from_user.id
    leverage = query.data.replace('leverage_', '')
    user_data[user_id]['leverage'] = leverage
    
    pairs = list(PIP_VALUES.keys())
    keyboard = []
    for i in range(0, len(pairs), 3):
        row = []
        for j in range(3):
            if i + j < len(pairs):
                row.append(InlineKeyboardButton(pairs[i + j], callback_data=f"currency_{pairs[i + j]}"))
        keyboard.append(row)
    
    await query.edit_message_text(
        f"✅ *Leverage:* {leverage}\n\n"
        "🌐 *Select currency pair:*",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )
    return CURRENCY

async def process_currency(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if not query:
        return ConversationHandler.END
    await query.answer()
    user_id = query.from_user.id
    currency = query.data.replace('currency_', '')
    user_data[user_id]['currency'] = currency
    await query.edit_message_text(
        f"✅ *Currency pair:* {currency}\n\n"
        "📈 *Enter entry price* (e.g., 1.0660):",
        parse_mode='Markdown'
    )
    return ENTRY

async def process_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message:
        return ConversationHandler.END
    user_id = update.message.from_user.id
    try:
        entry = float(update.message.text)
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
    if not update.message:
        return ConversationHandler.END
    user_id = update.message.from_user.id
    try:
        sl = float(update.message.text)
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
    if not update.message:
        return ConversationHandler.END
    user_id = update.message.from_user.id
    try:
        tps = [float(x.strip()) for x in update.message.text.split(',')]
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
        
        user_data[user_id]['volume_distribution'] = dist
        data = user_data[user_id]
        
        # Calculate results
        pos = RiskCalculator.calculate_position_size(
            deposit=data['deposit'],
            leverage=data['leverage'],
            currency_pair=data['currency'],
            entry_price=data['entry'],
            stop_loss=data['stop_loss']
        )
        
        profits = RiskCalculator.calculate_profits(
            currency_pair=data['currency'],
            entry_price=data['entry'],
            take_profits=data['take_profits'],
            position_size=pos['position_size'],
            volume_distribution=dist
        )
        
        # Format results
        resp = f"""
📊 *CALCULATION RESULTS*

*🎯 Main parameters:*
💵 Deposit: ${data['deposit']:,.2f}
⚖️ Leverage: {data['leverage']}
🌐 Currency pair: {data['currency']}
📈 Entry price: {data['entry']}
🛑 Stop loss: {data['stop_loss']}

*⚠️ Risk management:*
📊 Position size: *{pos['position_size']:.2f} lots*
💰 Risk per trade: ${pos['risk_amount']:.2f} ({pos['risk_percent']:.1f}% of deposit)
📉 Stop loss: {pos['stop_pips']:.0f} pips
💳 Required margin: ${pos['required_margin']:.2f}

*🎯 Take profits and profit:*
"""
        
        for p in profits:
            resp += f"\n🎯 TP{p['level']} ({p['volume_percent']}% volume):"
            resp += f"\n   💰 Price: {p['price']}"
            resp += f"\n   📦 Volume: {p['volume_lots']:.2f} lots"
            resp += f"\n   💵 Profit: ${p['profit']:.2f}"
            resp += f"\n   📊 Cumulative profit: ${p['cumulative_profit']:.2f}\n"
        
        # Add developer info
        resp += f"\n---\n"
        resp += f"👨‍💻 *Developer:* [@fxfeelgood](https://t.me/fxfeelgood)\n"
        resp += f"💡 *Tip:* Always follow risk management rules!"
        
        keyboard = [
            [InlineKeyboardButton("💾 Save preset", callback_data="save_preset")],
            [InlineKeyboardButton("🔄 New calculation", callback_data="new_calculation")],
            [InlineKeyboardButton("📚 Instructions", callback_data="show_info")]
        ]
        
        await update.message.reply_text(
            resp, 
            parse_mode='Markdown', 
            reply_markup=InlineKeyboardMarkup(keyboard),
            disable_web_page_preview=True
        )
        return ConversationHandler.END
        
    except ValueError:
        await update.message.reply_text("❌ Please enter valid volume distribution:")
        return VOLUME_DISTRIBUTION

async def save_preset(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    
    user_data[uid]['presets'].append({
        'timestamp': datetime.now().isoformat(),
        'data': user_data[uid].copy()
    })
    
    await query.edit_message_text(
        "✅ *Preset successfully saved!*\n\n"
        "💾 Use /presets to view saved strategies\n"
        "🚀 Use /start for new calculation\n\n"
        "👨‍💻 *Developer:* [@fxfeelgood](https://t.me/fxfeelgood)",
        parse_mode='Markdown',
        disable_web_page_preview=True
    )

async def show_presets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    uid = update.message.from_user.id
    presets = user_data.get(uid, {}).get('presets', [])
    if not presets:
        await update.message.reply_text(
            "📝 *You have no saved presets.*\n\n"
            "💡 Save your strategies after calculation for quick access!",
            parse_mode='Markdown'
        )
        return
    
    for i, p in enumerate(presets[-5:], 1):
        d = p['data']
        await update.message.reply_text(
            f"📋 *Preset #{i}*\n"
            f"💵 Deposit: ${d['deposit']:,.2f}\n"
            f"⚖️ Leverage: {d['leverage']}\n"
            f"🌐 Pair: {d['currency']}\n"
            f"📈 Entry: {d['entry']}\n"
            f"🛑 SL: {d['stop_loss']}\n"
            f"🎯 TP: {', '.join(map(str, d['take_profits']))}\n\n"
            f"👨‍💻 *Developer:* [@fxfeelgood](https://t.me/fxfeelgood)",
            parse_mode='Markdown',
            disable_web_page_preview=True
        )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message:
        await update.message.reply_text(
            "❌ *Calculation cancelled.*\n\n"
            "🚀 Use /start for new calculation\n"
            "📚 Use /info for instructions\n\n"
            "👨‍💻 *Developer:* [@fxfeelgood](https://t.me/fxfeelgood)",
            parse_mode='Markdown',
            disable_web_page_preview=True
        )
    return ConversationHandler.END

async def new_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

def main():
    """Main function to run bot with webhook"""
    token = os.getenv('TELEGRAM_BOT_TOKEN_EN')
    if not token:
        logger.error("Bot token not found!")
        return

    # Create application
    application = Application.builder().token(token).build()

    # Configure ConversationHandler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            DEPOSIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_deposit)],
            LEVERAGE: [CallbackQueryHandler(process_leverage, pattern='^leverage_')],
            CURRENCY: [CallbackQueryHandler(process_currency, pattern='^currency_')],
            ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_entry)],
            STOP_LOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_stop_loss)],
            TAKE_PROFITS: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_take_profits)],
            VOLUME_DISTRIBUTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_volume_distribution)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # Add handlers
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('info', info_command))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('presets', show_presets))
    application.add_handler(CallbackQueryHandler(save_preset, pattern='^save_preset$'))
    application.add_handler(CallbackQueryHandler(new_calculation, pattern='^new_calculation$'))
    application.add_handler(CallbackQueryHandler(show_info_callback, pattern='^show_info$'))

    # Get webhook URL
    webhook_url = os.getenv('RENDER_EXTERNAL_URL_EN', '')
    if not webhook_url:
        logger.error("RENDER_EXTERNAL_URL_EN not set!")
        return

    # Start webhook
    port = int(os.environ.get('PORT', 10000))
    webhook_path = f"/webhook"
    
    logger.info(f"Starting webhook on port {port}, URL: {webhook_url}{webhook_path}")
    
    application.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=webhook_path,
        webhook_url=webhook_url + webhook_path
    )

if __name__ == '__main__':
    main()
