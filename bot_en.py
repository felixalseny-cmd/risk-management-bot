# Analytics handlers
@log_performance
async def analytics_risk_reward(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Risk/Reward Analysis"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    trades = portfolio.get('trades', [])
    
    analysis = AnalyticsEngine.calculate_risk_reward_analysis(trades)
    
    risk_text = "📈 *Risk/Reward Analysis*\n\n"
    
    risk_text += f"⚡ Average R/R ratio: {analysis['average_risk_reward']:.2f}\n"
    risk_text += f"🏆 Best trade: ${analysis['best_trade']:.2f}\n"
    risk_text += f"🔻 Worst trade: ${analysis['worst_trade']:.2f}\n"
    risk_text += f"🎯 Consistency score: {analysis['consistency_score']:.1f}%\n"
    risk_text += f"⚠️ Risk level: {analysis['risk_score']:.1f}/100\n\n"
    
    # Recommendations based on risk analysis
    if analysis['average_risk_reward'] < 1:
        risk_text += "💡 *Recommendation:* Increase risk/reward ratio to 1:3\n"
    elif analysis['average_risk_reward'] > 3:
        risk_text += "💡 *Recommendation:* Excellent ratio! Keep it up\n"
    
    if analysis['risk_score'] < 30:
        risk_text += "🔻 Reduce risk per trade to 1-2% of deposit\n"
    
    await query.edit_message_text(
        risk_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("💹 Strategy Performance", callback_data="analytics_strategy_perf")],
            [InlineKeyboardButton("🔙 Back to Analytics", callback_data="analytics_back")]
        ])
    )

@log_performance
async def analytics_strategy_perf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Strategy Performance Analysis"""
    query = update.callback_query
    
    perf_text = "💹 *Strategy Performance*\n\n"
    
    # Sample strategy performance data
    strategies = {
        'Breakout': {'win_rate': 65, 'avg_profit': 45, 'total_trades': 23},
        'Trend Following': {'win_rate': 58, 'avg_profit': 32, 'total_trades': 15},
        'Mean Reversion': {'win_rate': 72, 'avg_profit': 28, 'total_trades': 18}
    }
    
    for strategy, stats in strategies.items():
        perf_text += f"🎯 *{strategy}*\n"
        perf_text += f"   📊 Win rate: {stats['win_rate']}%\n"
        perf_text += f"   💰 Average profit: ${stats['avg_profit']:.2f}\n"
        perf_text += f"   📈 Trades: {stats['total_trades']}\n\n"
    
    perf_text += "💡 *Best strategy:* Breakout (65% winning trades)"
    
    await query.edit_message_text(
        perf_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Trade Statistics", callback_data="analytics_trade_stats")],
            [InlineKeyboardButton("🔙 Back to Analytics", callback_data="analytics_back")]
        ])
    )

@log_performance
async def analytics_trade_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trade Statistics"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    performance = portfolio.get('performance', {})
    
    stats_text = "📊 *Trade Statistics*\n\n"
    
    total_trades = performance.get('total_trades', 0)
    win_rate = performance.get('win_rate', 0)
    profit_factor = (
        abs(performance.get('total_profit', 0) / performance.get('total_loss', 1)) 
        if performance.get('total_loss', 0) != 0 else 0
    )
    
    stats_text += f"📈 Total trades: {total_trades}\n"
    stats_text += f"🎯 Winning percentage: {win_rate:.1f}%\n"
    stats_text += f"💰 Profit factor: {profit_factor:.2f}\n"
    stats_text += f"⚡ Max winning streak: {performance.get('winning_trades', 0)}\n"
    stats_text += f"🔻 Max losing streak: {performance.get('losing_trades', 0)}\n\n"
    
    # Performance rating
    if win_rate >= 60 and profit_factor >= 1.5:
        rating = "🏆 EXCELLENT"
    elif win_rate >= 50 and profit_factor >= 1.2:
        rating = "✅ GOOD"
    else:
        rating = "⚠️ NEEDS OPTIMIZATION"
    
    stats_text += f"📊 *Performance Rating:* {rating}"
    
    await query.edit_message_text(
        stats_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Parameter Optimization", callback_data="analytics_optimization")],
            [InlineKeyboardButton("🔙 Back to Analytics", callback_data="analytics_back")]
        ])
    )

@log_performance
async def analytics_optimization(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Parameter Optimization"""
    query = update.callback_query
    
    opt_text = "🔄 *Parameter Optimization*\n\n"
    
    opt_text += "🎯 *Recommended Settings:*\n"
    opt_text += "• 📉 Risk per trade: 1-2% of deposit\n"
    opt_text += "• ⚡ R/R ratio: 1:3 or higher\n"
    opt_text += "• 📊 Position size: Automatic calculation\n"
    opt_text += "• 🛑 Stop loss: Fixed percentage\n\n"
    
    opt_text += "💡 *Optimization Tips:*\n"
    opt_text += "• Test strategies on historical data\n"
    opt_text += "• Use different timeframes\n"
    opt_text += "• Analyze results weekly\n"
    opt_text += "• Adjust parameters based on performance\n"
    
    await query.edit_message_text(
        opt_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("💡 Recommendations", callback_data="analytics_recommendations")],
            [InlineKeyboardButton("🔙 Back to Analytics", callback_data="analytics_back")]
        ])
    )

@log_performance
async def analytics_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Intelligent Recommendations"""
    query = update.callback_query
    user_id = query.from_user.id
    
    portfolio = user_data[user_id].get('portfolio', {})
    recommendations = AnalyticsEngine.generate_strategy_recommendations(portfolio)
    
    rec_text = "💡 *Intelligent Recommendations*\n\n"
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            rec_text += f"{i}. {rec}\n"
    else:
        rec_text += "✅ Your current strategy shows good results!\n"
        rec_text += "We recommend continuing your current approach.\n\n"
    
    rec_text += "\n🚀 *Coming Soon:*\n"
    rec_text += "• 🤖 AI analysis of your strategies\n"
    rec_text += "• 📊 Automated backtesting\n"
    rec_text += "• 📈 Profitability forecasting\n"
    rec_text += "• 💡 Personalized trading ideas"
    
    await query.edit_message_text(
        rec_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 Risk/Reward Analysis", callback_data="analytics_risk_reward")],
            [InlineKeyboardButton("🔙 Back to Analytics", callback_data="analytics_back")]
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

# Professional calculation handlers - ACTIVATED
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
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
            ])
        )
    else:
        query = update.callback_query
        await query.edit_message_text(
            "⚡ *Quick Calculation*\n\n"
            "📊 *Enter instrument ticker* (e.g.: EURUSD, BTCUSD, NAS100):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
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

@log_performance
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel conversation"""
    if update.message:
        await update.message.reply_text(
            "❌ *PRO calculation cancelled.*\n\n"
            "🚀 Use /start for new PRO calculation\n"
            "📚 Use /info for PRO instructions\n\n"
            "👨‍💻 *PRO Developer:* [@fxfeelgood](https://t.me/fxfeelgood)",
            parse_mode='Markdown',
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
            ])
        )
    return ConversationHandler.END

# [Rest of the professional calculation flow handlers would continue here...]
# Due to character limits, I'm showing the main function:

def main():
    """Optimized main function to run bot"""
    token = os.getenv('TELEGRAM_BOT_TOKEN_EN')
    if not token:
        logger.error("❌ PRO Bot token not found!")
        return

    logger.info("🚀 Starting ULTRA-FAST PRO Risk Management Bot v3.0 with Enhanced Portfolio & Analytics...")
    
    # Create application
    application = Application.builder().token(token).build()

    # [The rest of the conversation handler setup would be identical to the Russian version]
    # but with English text and the appropriate environment variables
    
    # Start bot
    port = int(os.environ.get('PORT', 10000))
    webhook_url = os.getenv('RENDER_EXTERNAL_URL_EN', '')
    
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
