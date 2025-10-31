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
