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

# Логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Состояния
DEPOSIT, LEVERAGE, CURRENCY, ENTRY, STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION = range(7)

# Временное хранилище
user_data: Dict[int, Dict[str, Any]] = {}

# Константы
PIP_VALUES = {
    'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9, 'USDCHF': 10,
    'USDCAD': 10, 'AUDUSD': 10, 'NZDUSD': 10, 'EURGBP': 10,
    'EURJPY': 9, 'GBPJPY': 9, 'XAUUSD': 10, 'XAGUSD': 50,
    'BTCUSD': 1, 'ETHUSD': 1
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


# --- Обработчики команд ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message:
        return ConversationHandler.END
    
    user = update.message.from_user
    user_name = user.first_name or "Трейдер"
    
    welcome_text = f"""
👋 *Привет, {user_name}!*

🎯 *Risk Management Calculator Bot для FOREX*

Я помогу тебе рассчитать оптимальный объем позиции с профессиональным управлением рисками.

📊 *Что я умею:*
• Расчет размера позиции с учетом риска 2% от депозита
• Учет кредитного плеча и требований к марже
• Распределение объемов между тейк-профитами
• Расчет прибыли и рисков в реальном времени

💡 *Для полной инструкции используй* /info

🚀 *Давай начнем! Введи сумму депозита в USD:*
"""
    
    user_id = user.id
    user_data[user_id] = {}
    await update.message.reply_text(welcome_text, parse_mode='Markdown')
    return DEPOSIT

async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Полная инструкция по использованию бота"""
    info_text = """
📚 *ПОЛНАЯ ИНСТРУКЦИЯ - Risk Management Calculator Bot*

🎯 *ОПИСАНИЕ*
Профессиональный калькулятор управления рисками для рынка FOREX. Помогает определить оптимальный размер позиции с учетом депозита, плеча и стоп-лосса.

📋 *КАК ИСПОЛЬЗОВАТЬ:*

1. *Начало работы*
   Команда: `/start`
   - Бот запросит основные параметры сделки

2. *Ввод параметров*
   - 💰 Депозит (USD)
   - ⚖️ Кредитное плечо (1:100, 1:200 и т.д.)
   - 🌐 Валютная пара (EURUSD, GBPUSD, XAUUSD и др.)
   - 📈 Цена входа
   - 🛑 Цена стоп-лосса
   - 🎯 Цены тейк-профитов (через запятую)
   - 📊 Распределение объемов между TP (в %)

3. *Результаты расчета*
   - 📊 Оптимальный размер позиции
   - ⚠️ Сумма риска (2% от депозита)
   - 📉 Стоп-лосс в пипсах
   - 💳 Требуемая маржа
   - 💰 Прибыль по каждому TP

📝 *ПРИМЕР РАСЧЕТА:*
Депозит: $1000
Плечо: 1:100
Пара: EURUSD
Вход: 1.0660
SL: 1.0640
TP: 1.0680, 1.0700
Распределение: 50, 50


🛠 *ДОСТУПНЫЕ КОМАНДЫ:*
`/start` - начать расчет
`/info` - полная инструкция  
`/help` - краткая помощь
`/presets` - сохраненные стратегии

🔮 *СКОРО БУДУТ ДОБАВЛЕНЫ:*
• Gold, Silver, WTI, NASDAQ, S&P500
• Bitcoin и другие криптовалюты
• Расширенные инструменты анализа

👨‍💻 *РАЗРАБОТЧИК:*
По вопросам и предложениям: [@fxfeelgood](https://t.me/fxfeelgood)

*С наилучшими пожеланиями,*
*Ваш надежный помощник в трейдинге!* 📈
"""
    await update.message.reply_text(info_text, parse_mode='Markdown', disable_web_page_preview=True)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Краткая справка"""
    help_text = """
🤖 *Risk Management Bot - Помощь*

📖 Для полной инструкции используйте: /info

⚡ *Быстрый старт:*
1. /start - начать расчет
2. Вводите параметры по запросу бота
3. Получайте готовые расчеты рисков

🛠 *Основные команды:*
`/start` - Новый расчет
`/info` - Полная инструкция
`/presets` - Мои стратегии
`/help` - Эта справка

💡 *Совет:* Всегда используйте риск не более 2% от депозита!

👨‍💻 *Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)
"""
    await update.message.reply_text(help_text, parse_mode='Markdown', disable_web_page_preview=True)

# --- Существующие обработчики разговора (остаются без изменений) ---
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
            f"✅ *Депозит:* ${deposit:,.2f}\n\n"
            "⚖️ *Выберите кредитное плечо:*",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
        return LEVERAGE
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректную сумму депозита:")
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
        f"✅ *Плечо:* {leverage}\n\n"
        "🌐 *Выберите валютную пару:*",
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
        f"✅ *Валютная пара:* {currency}\n\n"
        "📈 *Введите цену входа* (например, 1.0660):",
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
            f"✅ *Цена входа:* {entry}\n\n"
            "🛑 *Введите цену стоп-лосса:*",
            parse_mode='Markdown'
        )
        return STOP_LOSS
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректную цену входа:")
        return ENTRY

async def process_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message:
        return ConversationHandler.END
    user_id = update.message.from_user.id
    try:
        sl = float(update.message.text)
        user_data[user_id]['stop_loss'] = sl
        await update.message.reply_text(
            f"✅ *Стоп-лосс:* {sl}\n\n"
            "🎯 *Введите цены тейк-профитов через запятую* (например: 1.0550, 1.0460):",
            parse_mode='Markdown'
        )
        return TAKE_PROFITS
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректную цену стоп-лосса:")
        return STOP_LOSS

async def process_take_profits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message:
        return ConversationHandler.END
    user_id = update.message.from_user.id
    try:
        tps = [float(x.strip()) for x in update.message.text.split(',')]
        user_data[user_id]['take_profits'] = tps
        await update.message.reply_text(
            f"✅ *Тейк-профиты:* {', '.join(map(str, tps))}\n\n"
            f"📊 *Введите распределение объемов в % для каждого тейк-профита через запятую*\n"
            f"(всего {len(tps)} значений, сумма должна быть 100%):\n"
            f"*Пример:* 50, 30, 20",
            parse_mode='Markdown'
        )
        return VOLUME_DISTRIBUTION
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректные цены тейк-профитов:")
        return TAKE_PROFITS

async def process_volume_distribution(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message:
        return ConversationHandler.END
    user_id = update.message.from_user.id
    try:
        dist = [float(x.strip()) for x in update.message.text.split(',')]
        if abs(sum(dist) - 100) > 1e-5:
            await update.message.reply_text(
                f"❌ *Сумма распределения должна быть 100%. Ваша сумма: {sum(dist)}%*\n"
                "Пожалуйста, введите распределение заново:",
                parse_mode='Markdown'
            )
            return VOLUME_DISTRIBUTION
        
        user_data[user_id]['volume_distribution'] = dist
        data = user_data[user_id]
        
        # Расчет результатов
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
        
        # Форматирование результатов
        resp = f"""
📊 *РЕЗУЛЬТАТЫ РАСЧЕТА*

*🎯 Основные параметры:*
💵 Депозит: ${data['deposit']:,.2f}
⚖️ Плечо: {data['leverage']}
🌐 Валютная пара: {data['currency']}
📈 Цена входа: {data['entry']}
🛑 Стоп-лосс: {data['stop_loss']}

*⚠️ Управление рисками:*
📊 Размер позиции: *{pos['position_size']:.2f} лота*
💰 Риск на сделку: ${pos['risk_amount']:.2f} ({pos['risk_percent']:.1f}% от депозита)
📉 Стоп-лосс: {pos['stop_pips']:.0f} пипсов
💳 Требуемая маржа: ${pos['required_margin']:.2f}

*🎯 Тейк-профиты и прибыль:*
"""
        
        for p in profits:
            resp += f"\n🎯 TP{p['level']} ({p['volume_percent']}% объема):"
            resp += f"\n   💰 Цена: {p['price']}"
            resp += f"\n   📦 Объем: {p['volume_lots']:.2f} лота"
            resp += f"\n   💵 Прибыль: ${p['profit']:.2f}"
            resp += f"\n   📊 Накопленная прибыль: ${p['cumulative_profit']:.2f}\n"
        
        # Добавляем информацию о разработчике
        resp += f"\n---\n"
        resp += f"👨‍💻 *Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)\n"
        resp += f"💡 *Совет:* Всегда соблюдайте правила управления рисками!"
        
        keyboard = [
            [InlineKeyboardButton("💾 Сохранить пресет", callback_data="save_preset")],
            [InlineKeyboardButton("🔄 Новый расчет", callback_data="new_calculation")],
            [InlineKeyboardButton("📚 Инструкция", callback_data="show_info")]
        ]
        
        await update.message.reply_text(
            resp, 
            parse_mode='Markdown', 
            reply_markup=InlineKeyboardMarkup(keyboard),
            disable_web_page_preview=True
        )
        return ConversationHandler.END
        
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректное распределение объемов:")
        return VOLUME_DISTRIBUTION

async def save_preset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    await query.answer()
    uid = query.from_user.id
    if uid not in user_data:
        await query.edit_message_text("❌ Ошибка: данные не найдены. Начните новый расчет с /start")
        return
    if 'presets' not in user_data[uid]:
        user_data[uid]['presets'] = []
    
    user_data[uid]['presets'].append({
        'timestamp': datetime.now().isoformat(),
        'data': user_data[uid].copy()
    })
    
    await query.edit_message_text(
        "✅ *Пресет успешно сохранен!*\n\n"
        "💾 Используйте /presets для просмотра сохраненных стратегий\n"
        "🚀 Используйте /start для нового расчета\n\n"
        "👨‍💻 *Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)",
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
            "📝 *У вас нет сохраненных пресетов.*\n\n"
            "💡 Сохраняйте свои стратегии после расчета для быстрого доступа!",
            parse_mode='Markdown'
        )
        return
    
    for i, p in enumerate(presets[-5:], 1):
        d = p['data']
        await update.message.reply_text(
            f"📋 *Пресет #{i}*\n"
            f"💵 Депозит: ${d['deposit']:,.2f}\n"
            f"⚖️ Плечо: {d['leverage']}\n"
            f"🌐 Пара: {d['currency']}\n"
            f"📈 Вход: {d['entry']}\n"
            f"🛑 SL: {d['stop_loss']}\n"
            f"🎯 TP: {', '.join(map(str, d['take_profits']))}\n\n"
            f"👨‍💻 *Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)",
            parse_mode='Markdown',
            disable_web_page_preview=True
        )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message:
        await update.message.reply_text(
            "❌ *Расчет отменен.*\n\n"
            "🚀 Используйте /start для нового расчета\n"
            "📚 Используйте /info для инструкции\n\n"
            "👨‍💻 *Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)",
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
    """Показать инструкцию через callback"""
    query = update.callback_query
    if query:
        await query.answer()
        await info_command(update, context)

def main():
    """Основная функция для запуска бота с вебхуком"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("Токен бота не найден!")
        return

    # Создаем приложение
    application = Application.builder().token(token).build()

    # Настраиваем ConversationHandler
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

    # Добавляем обработчики
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('info', info_command))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('presets', show_presets))
    application.add_handler(CallbackQueryHandler(save_preset, pattern='^save_preset$'))
    application.add_handler(CallbackQueryHandler(new_calculation, pattern='^new_calculation$'))
    application.add_handler(CallbackQueryHandler(show_info_callback, pattern='^show_info$'))

    # Получаем URL для вебхука
    webhook_url = os.getenv('RENDER_EXTERNAL_URL', '')
    if not webhook_url:
        logger.error("RENDER_EXTERNAL_URL не установлен!")
        return

    # Запускаем вебхук
    port = int(os.environ.get('PORT', 10000))
    webhook_path = f"/webhook"
    
    logger.info(f"Запуск вебхука на порту {port}, URL: {webhook_url}{webhook_path}")
    
    application.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=webhook_path,
        webhook_url=webhook_url + webhook_path
    )

if __name__ == '__main__':
    main()