import os
import logging
import asyncio
import re
import time
import functools
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

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Упрощенные состояния
MAIN_MENU, QUICK_PARAMS, PRO_PARAMS, CALCULATION, RESULTS = range(5)

# Глобальный кэш для ускорения
calculation_cache = {}

# Упрощенный калькулятор
class SimpleRiskCalculator:
    @staticmethod
    def calculate_fast(deposit: float, risk_percent: float, entry: float, stop_loss: float, instrument_type: str = "forex") -> Dict:
        """Сверхбыстрый расчет позиции"""
        cache_key = f"{deposit}_{risk_percent}_{entry}_{stop_loss}_{instrument_type}"
        
        if cache_key in calculation_cache:
            return calculation_cache[cache_key]
        
        risk_amount = deposit * risk_percent
        
        # Упрощенная формула для всех инструментов
        if instrument_type == "crypto":
            multiplier = 100
        elif instrument_type == "indices":
            multiplier = 1
        else:
            multiplier = 10000
            
        price_diff = abs(entry - stop_loss)
        if price_diff == 0:
            position_size = 0.01
        else:
            position_size = (risk_amount / (price_diff * multiplier)) * 100
            position_size = min(max(round(position_size, 2), 0.01), 50.0)
        
        result = {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent * 100,
            'stop_pips': price_diff * multiplier
        }
        
        calculation_cache[cache_key] = result
        return result

# Главное меню - полностью переработанное
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Упрощенное главное меню"""
    user = update.message.from_user if update.message else update.callback_query.from_user
    
    welcome_text = f"""
🎯 *Умный Калькулятор Рисков*

Привет, {user.first_name}! Я помогу рассчитать оптимальный размер позиции.

⚡ *Выберите режим:*
"""

    keyboard = [
        [InlineKeyboardButton("🚀 Мгновенный расчет", callback_data="instant_mode")],
        [InlineKeyboardButton("📊 Детальный расчет", callback_data="pro_mode")],
        [InlineKeyboardButton("❓ Как пользоваться", callback_data="help")]
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

# Мгновенный расчет - новый подход
async def instant_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Режим мгновенного расчета"""
    query = update.callback_query
    await query.answer()
    
    help_text = """
🚀 *Мгновенный Расчет*

Просто введите данные в ОДНОЙ строке:

`депозит риск% вход стоп-лосс`

*Пример:*
`1000 2 1.0850 1.0800`

Или нажмите кнопку для пошагового ввода:
"""

    await query.edit_message_text(
        help_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📝 Пошаговый ввод", callback_data="step_by_step")],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
        ])
    )
    return QUICK_PARAMS

# Обработка быстрого ввода одной строкой
async def process_instant_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка мгновенного ввода"""
    if not update.message:
        return QUICK_PARAMS
        
    text = update.message.text.strip()
    
    try:
        # Парсим ввод одной строкой
        parts = text.split()
        if len(parts) == 4:
            deposit, risk_percent, entry, stop_loss = map(float, parts)
            risk_percent /= 100  # Конвертируем в десятичную дробь
            
            # Быстрый расчет
            result = SimpleRiskCalculator.calculate_fast(deposit, risk_percent, entry, stop_loss)
            
            await show_simple_result(update, context, deposit, risk_percent, entry, stop_loss, result)
            return MAIN_MENU
            
        else:
            raise ValueError("Неверное количество параметров")
            
    except ValueError as e:
        await update.message.reply_text(
            "❌ *Неверный формат!*\n\n"
            "Пожалуйста, введите данные в формате:\n"
            "`депозит риск% вход стоп-лосс`\n\n"
            "*Пример:* `1000 2 1.0850 1.0800`",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="instant_mode")]])
        )
        return QUICK_PARAMS

# Пошаговый ввод - упрощенный
async def step_by_step_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Упрощенный пошаговый ввод"""
    query = update.callback_query
    await query.answer()
    
    context.user_data['step'] = 'deposit'
    
    await query.edit_message_text(
        "💵 *Шаг 1 из 4*\n\n"
        "Введите сумму депозита в USD:",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="instant_mode")]])
    )
    return QUICK_PARAMS

# Обработка пошагового ввода
async def process_step_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка пошагового ввода"""
    if not update.message:
        return QUICK_PARAMS
        
    user_data = context.user_data
    step = user_data.get('step', 'deposit')
    text = update.message.text
    
    try:
        if step == 'deposit':
            deposit = float(text)
            if deposit <= 0:
                await update.message.reply_text("❌ Депозит должен быть положительным:")
                return QUICK_PARAMS
                
            user_data['deposit'] = deposit
            user_data['step'] = 'risk'
            
            await update.message.reply_text(
                "⚖️ *Шаг 2 из 4*\n\n"
                "Введите уровень риска в %:\n"
                "(например: 2 для 2%)",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="step_by_step")]])
            )
            
        elif step == 'risk':
            risk_percent = float(text) / 100
            user_data['risk_percent'] = risk_percent
            user_data['step'] = 'entry'
            
            await update.message.reply_text(
                "📈 *Шаг 3 из 4*\n\n"
                "Введите цену входа:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="step_by_step")]])
            )
            
        elif step == 'entry':
            entry = float(text)
            user_data['entry'] = entry
            user_data['step'] = 'stop_loss'
            
            await update.message.reply_text(
                "🛑 *Шаг 4 из 4*\n\n"
                "Введите стоп-лосс:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="step_by_step")]])
            )
            
        elif step == 'stop_loss':
            stop_loss = float(text)
            deposit = user_data['deposit']
            risk_percent = user_data['risk_percent']
            entry = user_data['entry']
            
            # Быстрый расчет
            result = SimpleRiskCalculator.calculate_fast(deposit, risk_percent, entry, stop_loss)
            
            await show_simple_result(update, context, deposit, risk_percent, entry, stop_loss, result)
            return MAIN_MENU
            
    except ValueError:
        await update.message.reply_text(
            "❌ Пожалуйста, введите корректное число:",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="step_by_step")]])
        )
        return QUICK_PARAMS
    
    return QUICK_PARAMS

# Простой и понятный вывод результатов
async def show_simple_result(update: Update, context: ContextTypes.DEFAULT_TYPE, 
                           deposit: float, risk_percent: float, entry: float, stop_loss: float, result: Dict):
    """Простой вывод результатов"""
    
    message = f"""
🎯 *РЕЗУЛЬТАТ РАСЧЕТА*

*Основные параметры:*
💵 Депозит: ${deposit:,.2f}
⚖️ Риск: {risk_percent*100}% (${result['risk_amount']:.2f})
📈 Вход: {entry}
🛑 Стоп-лосс: {stop_loss}

*Рекомендации:*
📦 Размер позиции: *{result['position_size']:.2f} лота*
💰 Сумма риска: ${result['risk_amount']:.2f}
📊 Стоп-лосс: {result['stop_pips']:.0f} пунктов

💡 *Совет:* Всегда используйте стоп-лосс для управления рисками!
"""

    keyboard = [
        [InlineKeyboardButton("🔄 Новый расчет", callback_data="instant_mode")],
        [InlineKeyboardButton("📊 Детальный расчет", callback_data="pro_mode")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    
    if update.message:
        await update.message.reply_text(
            message,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    else:
        await update.callback_query.edit_message_text(
            message,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

# Профессиональный режим - упрощенный
async def pro_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Упрощенный профессиональный режим"""
    query = update.callback_query
    await query.answer()
    
    help_text = """
📊 *Профессиональный Режим*

В этом режиме вы можете указать:
• Конкретный инструмент (EURUSD, BTCUSD и т.д.)
• Кредитное плечо
• Несколько тейк-профитов

Выберите тип инструмента:
"""

    keyboard = [
        [InlineKeyboardButton("🌐 Форекс", callback_data="pro_forex")],
        [InlineKeyboardButton("₿ Крипто", callback_data="pro_crypto")],
        [InlineKeyboardButton("📈 Индексы", callback_data="pro_indices")],
        [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
    ]
    
    await query.edit_message_text(
        help_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return PRO_PARAMS

# Обработка кнопок
async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка всех callback кнопок"""
    query = update.callback_query
    await query.answer()
    
    choice = query.data
    
    if choice == "instant_mode":
        return await instant_mode(update, context)
    elif choice == "pro_mode":
        return await pro_mode(update, context)
    elif choice == "step_by_step":
        return await step_by_step_input(update, context)
    elif choice == "main_menu":
        return await start(update, context)
    elif choice == "help":
        await show_help(update, context)
        return MAIN_MENU
    
    return MAIN_MENU

# Простая помощь
async def show_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Упрощенная справка"""
    help_text = """
❓ *Как пользоваться ботом*

🚀 *Мгновенный расчет:*
Просто введите: `депозит риск% вход стоп-лосс`
Пример: `1000 2 1.0850 1.0800`

📊 *Профессиональный расчет:*
Позволяет указать инструмент, плечо и тейк-профиты

💡 *Советы:*
• Риск на сделку: 1-3% от депозита
• Всегда используйте стоп-лосс
• Рассчитывайте размер позиции заранее

Для начала нажмите /start
"""

    query = update.callback_query
    await query.edit_message_text(
        help_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]])
    )

# Основная функция
def main():
    """Оптимизированная основная функция"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ Токен бота не найден!")
        return

    logger.info("🚀 Запуск оптимизированного бота...")
    
    # Создаем приложение
    application = Application.builder().token(token).build()

    # Упрощенный обработчик диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MAIN_MENU: [CallbackQueryHandler(handle_buttons, pattern='^(instant_mode|pro_mode|help|main_menu)$')],
            QUICK_PARAMS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_instant_input),
                CallbackQueryHandler(handle_buttons, pattern='^(instant_mode|step_by_step|main_menu)$')
            ],
            PRO_PARAMS: [
                CallbackQueryHandler(handle_buttons, pattern='^(pro_forex|pro_crypto|pro_indices|main_menu)$')
            ],
        },
        fallbacks=[CommandHandler('start', start)]
    )

    application.add_handler(conv_handler)

    # Запускаем
    port = int(os.environ.get('PORT', 10000))
    webhook_url = os.getenv('RENDER_EXTERNAL_URL', '')
    
    try:
        if webhook_url and "render.com" in webhook_url:
            logger.info(f"🔗 Webhook URL: {webhook_url}/webhook")
            application.run_webhook(
                listen="0.0.0.0",
                port=port,
                url_path="/webhook",
                webhook_url=webhook_url + "/webhook"
            )
        else:
            logger.info("🔄 Запуск в режиме polling...")
            application.run_polling()
    except Exception as e:
        logger.error(f"❌ Ошибка при запуске бота: {e}")

if __name__ == '__main__':
    main()
