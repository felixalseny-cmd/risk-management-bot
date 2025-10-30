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
from fastapi import FastAPI, Request
import uvicorn

# Настройка FastAPI для webhook
app = FastAPI()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Быстрая инициализация бота
application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

# УЛЬТРА-ОПТИМИЗИРОВАННЫЙ КЭШ
class UltraCache:
    def __init__(self):
        self.data = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        if key in self.data:
            self.hits += 1
            return self.data[key]
        self.misses += 1
        return None
    
    def set(self, key, value):
        if len(self.data) > 1000:  # Ограничение размера
            self.data.clear()
        self.data[key] = value

ultra_cache = UltraCache()

# СУПЕР-УПРОЩЕННЫЙ КАЛЬКУЛЯТОР
class FastRiskCalculator:
    @staticmethod
    def calculate_position_size_fast(deposit: float, risk_percent: float, entry: float, stop_loss: float) -> Dict:
        """Молниеносный расчет позиции"""
        cache_key = f"pos_{deposit}_{risk_percent}_{entry}_{stop_loss}"
        cached = ultra_cache.get(cache_key)
        if cached:
            return cached
        
        risk_amount = deposit * risk_percent
        price_diff = abs(entry - stop_loss)
        
        if price_diff == 0:
            return {'position_size': 0.01, 'risk_amount': risk_amount}
        
        # Упрощенная формула для Forex
        position_size = min((risk_amount / price_diff) * 0.1, 50.0)
        position_size = max(round(position_size, 2), 0.01)
        
        result = {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent * 100
        }
        
        ultra_cache.set(cache_key, result)
        return result

# ОЧЕНЬ ПРОСТЫЕ СОСТОЯНИЯ
MAIN_MENU, QUICK_INPUT, CALCULATE = range(3)

# Webhook endpoint
@app.post("/webhook")
async def webhook(request: Request):
    try:
        json_data = await request.json()
        update = Update.de_json(json_data, application.bot)
        await application.process_update(update)
        return {"status": "ok"}
    except Exception as e:
        logging.error(f"Webhook error: {e}")
        return {"status": "error"}

# УПРОЩЕННОЕ ГЛАВНОЕ МЕНЮ
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user if update.message else update.callback_query.from_user
    
    welcome_text = f"""
🎯 *Risk Calculator PRO*

⚡ *Выберите действие:*
"""
    
    keyboard = [
        [InlineKeyboardButton("🚀 Быстрый расчет", callback_data="quick_calc")],
        [InlineKeyboardButton("📊 Детальный расчет", callback_data="pro_calc")],
        [InlineKeyboardButton("❓ Помощь", callback_data="help")]
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

# СУПЕР-ПРОСТОЙ БЫСТРЫЙ РАСЧЕТ
async def quick_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text(
        "🚀 *Быстрый расчет*\n\n"
        "💡 Просто введите данные в формате:\n"
        "`Депозит Риск% Вход Стоп-лосс`\n\n"
        "*Пример:*\n"
        "`1000 2 1.0850 1.0800`\n\n"
        "Или нажмите кнопку для пошагового ввода:",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📝 Пошаговый ввод", callback_data="step_by_step")],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
        ])
    )
    return QUICK_INPUT

# ОБРАБОТКА БЫСТРОГО ВВОДА
async def process_quick_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message:
        return QUICK_INPUT
        
    text = update.message.text.strip()
    
    # Обработка кнопки "Пошаговый ввод"
    if text == "Пошаговый ввод" or update.callback_query:
        await update.message.reply_text(
            "💵 *Введите сумму депозита в USD:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="quick_calc")]])
        )
        context.user_data['step'] = 'deposit'
        return QUICK_INPUT
    
    # Быстрый парсинг одной строки
    try:
        parts = text.split()
        if len(parts) == 4:
            deposit, risk_percent, entry, stop_loss = map(float, parts)
            risk_percent /= 100  # Конвертируем в десятичную дробь
            
            await calculate_and_show_result(update, context, deposit, risk_percent, entry, stop_loss)
            return MAIN_MENU
            
    except ValueError:
        pass
    
    # Если не распарсилось, переходим к пошаговому вводу
    await update.message.reply_text(
        "💵 *Введите сумму депозита в USD:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="quick_calc")]])
    )
    context.user_data['step'] = 'deposit'
    return QUICK_INPUT

# ПОШАГОВЫЙ ВВОД
async def process_step_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message:
        return QUICK_INPUT
        
    user_data = context.user_data
    step = user_data.get('step', 'deposit')
    text = update.message.text
    
    try:
        if step == 'deposit':
            deposit = float(text)
            if deposit <= 0:
                await update.message.reply_text("❌ Депозит должен быть положительным:")
                return QUICK_INPUT
                
            user_data['deposit'] = deposit
            user_data['step'] = 'risk'
            
            await update.message.reply_text(
                "⚖️ *Введите уровень риска в %:*\n(например: 2 для 2%)",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="quick_calc")]])
            )
            
        elif step == 'risk':
            risk_percent = float(text) / 100
            user_data['risk_percent'] = risk_percent
            user_data['step'] = 'entry'
            
            await update.message.reply_text(
                "📈 *Введите цену входа:*",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="quick_calc")]])
            )
            
        elif step == 'entry':
            entry = float(text)
            user_data['entry'] = entry
            user_data['step'] = 'stop_loss'
            
            await update.message.reply_text(
                "🛑 *Введите стоп-лосс:*",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="quick_calc")]])
            )
            
        elif step == 'stop_loss':
            stop_loss = float(text)
            deposit = user_data['deposit']
            risk_percent = user_data['risk_percent']
            entry = user_data['entry']
            
            await calculate_and_show_result(update, context, deposit, risk_percent, entry, stop_loss)
            return MAIN_MENU
            
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректное число:")
        return QUICK_INPUT
    
    return QUICK_INPUT

# БЫСТРЫЙ РАСЧЕТ И ПОКАЗ РЕЗУЛЬТАТА
async def calculate_and_show_result(update: Update, context: ContextTypes.DEFAULT_TYPE, 
                                  deposit: float, risk_percent: float, entry: float, stop_loss: float):
    
    # Молниеносный расчет
    result = FastRiskCalculator.calculate_position_size_fast(deposit, risk_percent, entry, stop_loss)
    
    # Простое и понятное сообщение
    message = f"""
🎯 *РЕЗУЛЬТАТ РАСЧЕТА*

💵 Депозит: ${deposit:,.2f}
⚖️ Риск: {risk_percent*100}% (${result['risk_amount']:.2f})
📈 Вход: {entry}
🛑 Стоп-лосс: {stop_loss}

📊 *РЕКОМЕНДАЦИЯ:*
📦 Размер позиции: *{result['position_size']:.2f} лота*

💡 *Объяснение:*
- Рискуете только ${result['risk_amount']:.2f} от депозита
- Это {result['risk_percent']:.1f}% от общего капитала
- Размер позиции рассчитан оптимально

⚡ *Совет:* Всегда используйте стоп-лосс!
"""
    
    if update.message:
        await update.message.reply_text(
            message,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Новый расчет", callback_data="quick_calc")],
                [InlineKeyboardButton("📊 Детальный расчет", callback_data="pro_calc")]
            ])
        )
    else:
        await update.callback_query.edit_message_text(
            message,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Новый расчет", callback_data="quick_calc")],
                [InlineKeyboardButton("📊 Детальный расчет", callback_data="pro_calc")]
            ])
        )

# ОБРАБОТКА КНОПОК
async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    choice = query.data
    
    if choice == "quick_calc":
        return await quick_calculation(update, context)
    elif choice == "main_menu":
        return await start(update, context)
    elif choice == "step_by_step":
        context.user_data.clear()
        context.user_data['step'] = 'deposit'
        await query.edit_message_text(
            "💵 *Введите сумму депозита в USD:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="quick_calc")]])
        )
        return QUICK_INPUT
    
    return MAIN_MENU

# НАСТРОЙКА HANDLERS
def setup_handlers():
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MAIN_MENU: [CallbackQueryHandler(handle_buttons, pattern='^(quick_calc|pro_calc|help|main_menu)$')],
            QUICK_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_quick_input),
                CallbackQueryHandler(handle_buttons, pattern='^(quick_calc|main_menu|step_by_step)$')
            ],
        },
        fallbacks=[CommandHandler('start', start)]
    )
    
    application.add_handler(conv_handler)

# ЗАПУСК СЕРВЕРА
if __name__ == '__main__':
    setup_handlers()
    
    port = int(os.environ.get('PORT', 10000))
    webhook_url = os.getenv('RENDER_EXTERNAL_URL')
    
    if webhook_url:
        # Webhook режим для Render
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path="/webhook",
            webhook_url=f"{webhook_url}/webhook"
        )
    else:
        # Polling режим для разработки
        application.run_polling()