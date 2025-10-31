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

# Оптимизированная настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot_errors.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Улучшенный декоратор для логирования производительности
def log_performance(max_time=1.0):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                if execution_time > max_time:
                    logger.warning(f"Медленная операция: {func.__name__} заняла {execution_time:.2f}с")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Ошибка в {func.__name__}: {e} (время: {execution_time:.2f}с)")
                raise
        return wrapper
    return decorator

# Состояния диалога (оптимизированы)
(
    MAIN_MENU, INSTRUMENT_TYPE, DIRECTION, 
    RISK_PERCENT, DEPOSIT, LEVERAGE, ENTRY, 
    STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION,
    PORTFOLIO_MENU, ADD_TRADE_INSTRUMENT, ADD_TRADE_DIRECTION,
    ADD_TRADE_ENTRY, ADD_TRADE_EXIT, ADD_TRADE_VOLUME,
    DEPOSIT_AMOUNT, WITHDRAW_AMOUNT, SETTINGS_MENU
) = range(19)

# Оптимизированное временное хранилище с TTL
class OptimizedUserData:
    def __init__(self, ttl=3600):  # 1 час TTL
        self.data = {}
        self.ttl = ttl
        self.access_times = {}
    
    def get(self, user_id: int):
        if user_id in self.data:
            if time.time() - self.access_times.get(user_id, 0) < self.ttl:
                self.access_times[user_id] = time.time()
                return self.data[user_id]
            else:
                # Удаляем устаревшие данные
                del self.data[user_id]
                del self.access_times[user_id]
        return None
    
    def set(self, user_id: int, value):
        self.data[user_id] = value
        self.access_times[user_id] = time.time()
        
        # Периодическая очистка старых данных
        if len(self.data) > 1000:  # Если больше 1000 пользователей
            self.cleanup()
    
    def cleanup(self):
        current_time = time.time()
        expired_users = [
            user_id for user_id, access_time in self.access_times.items()
            if current_time - access_time > self.ttl
        ]
        for user_id in expired_users:
            del self.data[user_id]
            del self.access_times[user_id]

user_data = OptimizedUserData()

# Оптимизированные константы
INSTRUMENT_TYPES = {
    'forex': 'Форекс',
    'crypto': 'Криптовалюты', 
    'indices': 'Индексы',
    'commodities': 'Сырьевые товары',
    'metals': 'Металлы'
}

# Оптимизированный менеджер портфеля
class OptimizedPortfolioManager:
    @staticmethod
    def initialize_user_portfolio(user_id: int):
        user_data_obj = user_data.get(user_id)
        if not user_data_obj:
            user_data_obj = {}
            user_data.set(user_id, user_data_obj)
        
        if 'portfolio' not in user_data_obj:
            user_data_obj['portfolio'] = {
                'initial_balance': 0.0,
                'current_balance': 0.0,
                'trades': [],
                'performance': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0.0,
                    'total_loss': 0.0,
                    'win_rate': 0.0,
                    'average_profit': 0.0,
                    'average_loss': 0.0,
                    'profit_factor': 0.0,
                    'max_drawdown': 0.0
                },
                'allocation': {},
                'settings': {
                    'default_risk': 0.02,
                    'currency': 'USD',
                    'leverage': '1:100'
                }
            }
    
    @staticmethod
    def add_trade(user_id: int, trade_data: Dict):
        OptimizedPortfolioManager.initialize_user_portfolio(user_id)
        user_data_obj = user_data.get(user_id)
        
        trade_id = len(user_data_obj['portfolio']['trades']) + 1
        trade_data['id'] = trade_id
        trade_data['timestamp'] = datetime.now().isoformat()
        
        user_data_obj['portfolio']['trades'].append(trade_data)
        OptimizedPortfolioManager.update_performance_metrics(user_id)
        
        instrument = trade_data.get('instrument', 'Unknown')
        user_data_obj['portfolio']['allocation'][instrument] = \
            user_data_obj['portfolio']['allocation'].get(instrument, 0) + 1

# Основные обработчики с оптимизацией
@log_performance(max_time=0.5)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Главное меню с оптимизацией"""
    try:
        user = update.effective_user
        if not user:
            return ConversationHandler.END
            
        user_name = user.first_name or "Трейдер"
        user_id = user.id
        
        # Быстрая инициализация
        OptimizedPortfolioManager.initialize_user_portfolio(user_id)
        
        welcome_text = f"""👋 *Привет, {user_name}!*

🎯 *PRO Калькулятор Управления Рисками v4.1*

⚡ *Выберите опцию:*"""
        
        keyboard = [
            [InlineKeyboardButton("📊 Проф. расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("⚡ Быстрый расчет", callback_data="quick_calculation")],
            [InlineKeyboardButton("💼 Мой портфель", callback_data="portfolio")],
            [InlineKeyboardButton("📚 Инструкции", callback_data="pro_info")],
            [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")]
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
        logger.error(f"Ошибка в start: {e}")
        return ConversationHandler.END

@log_performance(max_time=0.3)
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Быстрая отмена диалога"""
    try:
        await update.message.reply_text(
            "❌ *Операция отменена.*\n\n"
            "🚀 Используйте /start для нового расчета",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"Ошибка в cancel: {e}")
        return ConversationHandler.END

@log_performance(max_time=0.5)
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Оптимизированный обработчик главного меню"""
    try:
        query = update.callback_query
        if not query:
            return MAIN_MENU
            
        await query.answer()
        choice = query.data
        
        user_id = query.from_user.id
        
        # Быстрые обработчики для основных команд
        menu_handlers = {
            "pro_calculation": start_pro_calculation,
            "quick_calculation": start_quick_calculation,
            "portfolio": portfolio_command,
            "pro_info": pro_info_command,
            "settings": settings_command,
            "main_menu": start
        }
        
        handler = menu_handlers.get(choice)
        if handler:
            return await handler(update, context)
        
        return MAIN_MENU
        
    except Exception as e:
        logger.error(f"Ошибка в handle_main_menu: {e}")
        return await start(update, context)

# Оптимизированные команды
@log_performance(max_time=0.4)
async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Быстрое меню портфеля"""
    try:
        user_id = update.effective_user.id
        OptimizedPortfolioManager.initialize_user_portfolio(user_id)
        
        user_data_obj = user_data.get(user_id)
        current_balance = user_data_obj['portfolio']['current_balance']
        
        portfolio_text = f"""💼 *Управление Портфелем*

💰 *Баланс:* ${current_balance:,.2f}
📊 *Сделок:* {len(user_data_obj['portfolio']['trades'])}

🎯 *Выберите действие:*"""
        
        keyboard = [
            [InlineKeyboardButton("📈 Обзор сделок", callback_data="portfolio_trades")],
            [InlineKeyboardButton("💰 Баланс", callback_data="portfolio_balance")],
            [InlineKeyboardButton("📊 Аналитика", callback_data="portfolio_performance")],
            [InlineKeyboardButton("📄 Отчет", callback_data="portfolio_report")],
            [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
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
        
    except Exception as e:
        logger.error(f"Ошибка в portfolio_command: {e}")
        return await start(update, context)

@log_performance(max_time=0.3)
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Оптимизированные инструкции"""
    try:
        info_text = """📚 *PRO ИНСТРУКЦИИ v4.1*

🎯 *ОСНОВНЫЕ ВОЗМОЖНОСТИ:*

⚡ *МГНОВЕННЫЕ РАСЧЕТЫ:*
• Калькулятор рисков для всех активов
• Расчет позиций с учетом плеча
• Автоматические тейк-профиты

💼 *УПРАВЛЕНИЕ ПОРТФЕЛЕМ:*
• Трекинг сделок и баланса
• Аналитика эффективности
• Профессиональные рекомендации

🚀 *СКОРО В ОБНОВЛЕНИИ:*
• AI-анализ стратегий
• Реальные котировки онлайн
• Бэктестинг и аналитика

👨‍💻 *Разработчик:* @fxfeelgood
*PRO v4.1 | Быстро • Надежно • Профессионально* 🚀"""
        
        reply_markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ])
        
        if update.message:
            await update.message.reply_text(
                info_text, 
                parse_mode='Markdown', 
                disable_web_page_preview=True,
                reply_markup=reply_markup
            )
        else:
            await update.callback_query.edit_message_text(
                info_text,
                parse_mode='Markdown',
                disable_web_page_preview=True,
                reply_markup=reply_markup
            )
    except Exception as e:
        logger.error(f"Ошибка в pro_info_command: {e}")

# Оптимизированные функции-заглушки для отсутствующих обработчиков
@log_performance(max_time=0.3)
async def start_pro_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Быстрое начало профессионального расчета"""
    try:
        await update.callback_query.edit_message_text(
            "📊 *ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ*\n\n"
            "⚡ *Функция в разработке...*\n\n"
            "Используйте быстрый расчет для оперативных вычислений.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚡ Быстрый расчет", callback_data="quick_calculation")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return MAIN_MENU
    except Exception as e:
        logger.error(f"Ошибка в start_pro_calculation: {e}")
        return MAIN_MENU

@log_performance(max_time=0.3)
async def start_quick_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Быстрое начало быстрого расчета"""
    try:
        await update.callback_query.edit_message_text(
            "⚡ *БЫСТРЫЙ РАСЧЕТ*\n\n"
            "💰 *Введите сумму депозита:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return DEPOSIT
    except Exception as e:
        logger.error(f"Ошибка в start_quick_calculation: {e}")
        return MAIN_MENU

@log_performance(max_time=0.3)
async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Быстрые настройки"""
    try:
        user_id = update.effective_user.id
        OptimizedPortfolioManager.initialize_user_portfolio(user_id)
        
        user_data_obj = user_data.get(user_id)
        settings = user_data_obj['portfolio']['settings']
        
        settings_text = f"""⚙️ *Настройки*

*Текущие настройки:*
• Риск: {settings['default_risk']*100}%
• Валюта: {settings['currency']}
• Плечо: {settings['leverage']}

🔧 *Изменить настройки:*"""
        
        keyboard = [
            [InlineKeyboardButton(f"💰 Риск: {settings['default_risk']*100}%", callback_data="change_risk")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
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
        logger.error(f"Ошибка в settings_command: {e}")

# Оптимизированные обработчики портфеля
@log_performance(max_time=0.4)
async def portfolio_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Быстрый обзор сделок"""
    try:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        user_data_obj = user_data.get(user_id)
        trades = user_data_obj['portfolio']['trades']
        
        if not trades:
            await query.edit_message_text(
                "📭 *Сделок пока нет*\n\n"
                "Добавьте первую сделку для анализа.",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]
                ])
            )
            return
        
        # Показываем только последние 3 сделки для скорости
        recent_trades = trades[-3:]
        trades_text = "📈 *Последние сделки:*\n\n"
        
        for trade in reversed(recent_trades):
            profit = trade.get('profit', 0)
            status_emoji = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"
            trades_text += f"{status_emoji} {trade.get('instrument', 'N/A')} | ${profit:.2f}\n"
        
        await query.edit_message_text(
            trades_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
                [InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в portfolio_trades: {e}")

# Оптимизированная основная функция
def main():
    """Супер-оптимизированная основная функция"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ Токен бота не найден!")
        return

    logger.info("🚀 Запуск ОПТИМИЗИРОВАННОГО калькулятора рисков v4.1...")
    
    # Создаем application с оптимизированными настройками
    application = Application.builder().token(token).build()

    # Упрощенный обработчик диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MAIN_MENU: [CallbackQueryHandler(handle_main_menu)],
            PORTFOLIO_MENU: [CallbackQueryHandler(handle_main_menu)],
            SETTINGS_MENU: [CallbackQueryHandler(handle_main_menu)],
            DEPOSIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_deposit_amount)],
        },
        fallbacks=[
            CommandHandler('cancel', cancel),
            CommandHandler('start', start)
        ],
        allow_reentry=True
    )

    # Минимальный набор обработчиков
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('help', pro_info_command))
    application.add_handler(CommandHandler('info', pro_info_command))
    application.add_handler(CommandHandler('portfolio', portfolio_command))
    application.add_handler(CallbackQueryHandler(handle_main_menu))

    # Обработчик неизвестных команд
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    
    # Оптимизированный запуск
    port = int(os.environ.get('PORT', 10000))
    webhook_url = os.getenv('RENDER_EXTERNAL_URL', '')
    
    logger.info(f"🌐 Запуск на порту {port}")
    
    try:
        if webhook_url and "render.com" in webhook_url:
            logger.info("🔗 Webhook режим")
            application.run_webhook(
                listen="0.0.0.0",
                port=port,
                url_path="/webhook",
                webhook_url=f"{webhook_url}/webhook",
                drop_pending_updates=True  # Важно для избежания залипания
            )
        else:
            logger.info("🔄 Polling режим")
            application.run_polling(
                drop_pending_updates=True,  # Очищаем pending updates
                allowed_updates=['message', 'callback_query']  # Только нужные апдейты
            )
    except Exception as e:
        logger.error(f"❌ Ошибка запуска: {e}")

@log_performance(max_time=0.2)
async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Быстрая обработка неизвестных команд"""
    try:
        await update.message.reply_text(
            "❌ *Неизвестная команда*\n\n"
            "🎯 Используйте /start для начала",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🚀 Начать", callback_data="main_menu")]
            ])
        )
    except Exception as e:
        logger.error(f"Ошибка в unknown_command: {e}")

# Заглушки для отсутствующих функций
async def handle_deposit_amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Заглушка для обработки депозита"""
    try:
        await update.message.reply_text(
            "💰 *Функция в разработке*\n\n"
            "Скоро будет доступен полный функционал расчетов.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"Ошибка в handle_deposit_amount: {e}")
        return ConversationHandler.END

async def portfolio_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Заглушка для баланса"""
    await update.callback_query.edit_message_text(
        "💰 *Баланс*\n\nФункция в разработке...",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]
        ])
    )

async def portfolio_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Заглушка для аналитики"""
    await update.callback_query.edit_message_text(
        "📊 *Аналитика*\n\nФункция в разработке...",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]
        ])
    )

async def portfolio_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Заглушка для отчетов"""
    await update.callback_query.edit_message_text(
        "📄 *Отчет*\n\nФункция в разработке...",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]
        ])
    )

async def portfolio_add_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Заглушка для добавления сделки"""
    await update.callback_query.edit_message_text(
        "➕ *Добавление сделки*\n\nФункция в разработке...",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]
        ])
    )

async def change_risk_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Заглушка для изменения риска"""
    await update.callback_query.edit_message_text(
        "🎯 *Изменение риска*\n\nФункция в разработке...",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data="settings")]
        ])
    )

if __name__ == '__main__':
    main()
