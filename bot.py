import os
import logging
import json
from datetime import datetime
from typing import Dict, List
from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup
)
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

# Состояния разговора
DEPOSIT, LEVERAGE, CURRENCY, ENTRY, STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION = range(7)

# Данные пользователей
user_data = {}

# PIP значения для разных валютных пар
PIP_VALUES = {
    'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9, 'USDCHF': 10,
    'USDCAD': 10, 'AUDUSD': 10, 'NZDUSD': 10, 'EURGBP': 10,
    'EURJPY': 9, 'GBPJPY': 9, 'XAUUSD': 10, 'XAGUSD': 50
}

# Доступные кредитные плечи
LEVERAGES = ['1:100', '1:200', '1:500', '1:1000', '1:2000']

class RiskCalculator:
    @staticmethod
    def calculate_pip_value(currency_pair: str, lot_size: float) -> float:
        """Расчет стоимости пипса для валютной пары"""
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
    ) -> Dict:
        """Расчет размера позиции с учетом риска"""
        # Парсим плечо
        lev_value = int(leverage.split(':')[1])
        
        # Расчет риска в деньгах
        risk_amount = deposit * risk_percent
        
        # Расчет стоп-лосса в пипсах
        stop_pips = abs(entry_price - stop_loss) * 10000
        
        # Максимальный лот по риску
        pip_value_per_lot = RiskCalculator.calculate_pip_value(currency_pair, 1.0)
        max_lots_by_risk = risk_amount / (stop_pips * pip_value_per_lot)
        
        # Максимальный лот по марже
        contract_size = 100000  # Стандартный контракт
        max_lots_by_margin = (deposit * lev_value) / contract_size
        
        # Определяем финальный размер
        position_size = min(max_lots_by_risk, max_lots_by_margin, 10.0)  # Максимум 10 лотов
        
        # Округляем до допустимых значений
        if position_size < 0.01:
            position_size = 0.01
        else:
            position_size = round(position_size * 100) / 100
        
        # Расчет требуемой маржи
        required_margin = (position_size * contract_size) / lev_value
        
        return {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'stop_pips': stop_pips,
            'required_margin': required_margin,
            'risk_percent': (risk_amount / deposit) * 100,
            'max_risk_lots': max_lots_by_risk,
            'max_margin_lots': max_lots_by_margin
        }

    @staticmethod
    def calculate_profits(
        currency_pair: str,
        entry_price: float,
        take_profits: List[float],
        position_size: float,
        volume_distribution: List[float]
    ) -> List[Dict]:
        """Расчет прибыли для каждого тейк-профита"""
        profits = []
        total_profit = 0
        
        for i, (tp, volume_percent) in enumerate(zip(take_profits, volume_distribution)):
            tp_pips = abs(entry_price - tp) * 10000
            volume_lots = position_size * (volume_percent / 100)
            pip_value = RiskCalculator.calculate_pip_value(currency_pair, volume_lots)
            profit = tp_pips * pip_value
            total_profit += profit
            
            profits.append({
                'level': i + 1,
                'price': tp,
                'volume_percent': volume_percent,
                'volume_lots': volume_lots,
                'profit': profit,
                'cumulative_profit': total_profit
            })
        
        return profits

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало разговора"""
    user_id = update.message.from_user.id
    user_data[user_id] = {}
    
    await update.message.reply_text(
        "🎯 *Risk Management Calculator Bot*\n\n"
        "Я помогу рассчитать оптимальный объем позиции с учетом рисков.\n\n"
        "Давайте начнем! Введите сумму депозита в USD:",
        parse_mode='Markdown'
    )
    
    return DEPOSIT

async def process_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода депозита"""
    user_id = update.message.from_user.id
    
    try:
        deposit = float(update.message.text)
        if deposit <= 0:
            await update.message.reply_text("❌ Депозит должен быть положительным числом. Попробуйте еще раз:")
            return DEPOSIT
        
        user_data[user_id]['deposit'] = deposit
        
        # Создаем клавиатуру для выбора плеча
        keyboard = []
        for leverage in LEVERAGES:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"leverage_{leverage}")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"✅ Депозит: ${deposit:,.2f}\n\n"
            "Теперь выберите кредитное плечо:",
            reply_markup=reply_markup
        )
        
        return LEVERAGE
        
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректную сумму депозита:")
        return DEPOSIT

async def process_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора плеча"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    leverage = query.data.replace('leverage_', '')
    user_data[user_id]['leverage'] = leverage
    
    # Клавиатура для выбора валютной пары
    currency_pairs = list(PIP_VALUES.keys())
    keyboard = []
    row = []
    for i, pair in enumerate(currency_pairs):
        row.append(InlineKeyboardButton(pair, callback_data=f"currency_{pair}"))
        if (i + 1) % 3 == 0:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        f"✅ Плечо: {leverage}\n\n"
        "Выберите валютную пару:",
        reply_markup=reply_markup
    )
    
    return CURRENCY

async def process_currency(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора валютной пары"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    currency = query.data.replace('currency_', '')
    user_data[user_id]['currency'] = currency
    
    await query.edit_message_text(
        f"✅ Валютная пара: {currency}\n\n"
        "Введите цену входа (например, 1.0660):"
    )
    
    return ENTRY

async def process_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка цены входа"""
    user_id = update.message.from_user.id
    
    try:
        entry_price = float(update.message.text)
        user_data[user_id]['entry'] = entry_price
        
        await update.message.reply_text(
            f"✅ Цена входа: {entry_price}\n\n"
            "Введите цену стоп-лосса:"
        )
        
        return STOP_LOSS
        
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректную цену входа:")
        return ENTRY

async def process_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка стоп-лосса"""
    user_id = update.message.from_user.id
    
    try:
        stop_loss = float(update.message.text)
        user_data[user_id]['stop_loss'] = stop_loss
        
        await update.message.reply_text(
            f"✅ Стоп-лосс: {stop_loss}\n\n"
            "Введите цены тейк-профитов через запятую (например: 1.0550, 1.0460):"
        )
        
        return TAKE_PROFITS
        
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректную цену стоп-лосса:")
        return STOP_LOSS

async def process_take_profits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка тейк-профитов"""
    user_id = update.message.from_user.id
    
    try:
        take_profits_text = update.message.text
        take_profits = [float(tp.strip()) for tp in take_profits_text.split(',')]
        user_data[user_id]['take_profits'] = take_profits
        
        # Запрашиваем распределение объемов
        await update.message.reply_text(
            f"✅ Тейк-профиты: {', '.join(map(str, take_profits))}\n\n"
            f"Введите распределение объемов в % для каждого тейк-профита через запятую "
            f"(всего {len(take_profits)} значений, сумма должна быть 100%):\n"
            f"Пример: 50, 30, 20"
        )
        
        return VOLUME_DISTRIBUTION
        
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректные цены тейк-профитов:")
        return TAKE_PROFITS

async def process_volume_distribution(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка распределения объемов и расчет результатов"""
    user_id = update.message.from_user.id
    
    try:
        distribution_text = update.message.text
        distribution = [float(d.strip()) for d in distribution_text.split(',')]
        
        # Проверяем сумму распределения
        if sum(distribution) != 100:
            await update.message.reply_text(
                f"❌ Сумма распределения должна быть 100%. Ваша сумма: {sum(distribution)}%\n"
                "Пожалуйста, введите распределение заново:"
            )
            return VOLUME_DISTRIBUTION
        
        user_data[user_id]['volume_distribution'] = distribution
        
        # Получаем все данные
        data = user_data[user_id]
        
        # Расчет размера позиции
        position_result = RiskCalculator.calculate_position_size(
            deposit=data['deposit'],
            leverage=data['leverage'],
            currency_pair=data['currency'],
            entry_price=data['entry'],
            stop_loss=data['stop_loss']
        )
        
        # Расчет прибыли
        profits_result = RiskCalculator.calculate_profits(
            currency_pair=data['currency'],
            entry_price=data['entry'],
            take_profits=data['take_profits'],
            position_size=position_result['position_size'],
            volume_distribution=data['volume_distribution']
        )
        
        # Формируем ответ
        response = f"""
📊 *РЕЗУЛЬТАТЫ РАСЧЕТА*

*Основные параметры:*
💵 Депозит: ${data['deposit']:,.2f}
⚖️ Плечо: {data['leverage']}
🎯 Валютная пара: {data['currency']}
💰 Цена входа: {data['entry']}
🛑 Стоп-лосс: {data['stop_loss']}

*Управление рисками:*
📊 Размер позиции: *{position_result['position_size']:.2f} лота*
⚠️ Риск на сделку: ${position_result['risk_amount']:.2f} ({position_result['risk_percent']:.1f}% от депозита)
📉 Стоп-лосс: {position_result['stop_pips']:.0f} пипсов
💳 Требуемая маржа: ${position_result['required_margin']:.2f}

*Тейк-профиты и прибыль:*
"""
        
        for profit in profits_result:
            response += f"\n🎯 TP{profit['level']} ({profit['volume_percent']}% объема):"
            response += f"\n   Цена: {profit['price']}"
            response += f"\n   Объем: {profit['volume_lots']:.2f} лота"
            response += f"\n   Прибыль: ${profit['profit']:.2f}"
            response += f"\n   Накопленная прибыль: ${profit['cumulative_profit']:.2f}\n"
        
        # Предлагаем сохранить пресет
        keyboard = [
            [InlineKeyboardButton("💾 Сохранить пресет", callback_data="save_preset")],
            [InlineKeyboardButton("🔄 Новый расчет", callback_data="new_calculation")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(response, parse_mode='Markdown', reply_markup=reply_markup)
        
        # Сохраняем результаты для возможного сохранения пресета
        user_data[user_id]['last_calculation'] = {
            'position_result': position_result,
            'profits_result': profits_result
        }
        
        return ConversationHandler.END
        
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректное распределение объемов:")
        return VOLUME_DISTRIBUTION

async def save_preset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сохранение пресета стратегии"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    
    if 'presets' not in user_data[user_id]:
        user_data[user_id]['presets'] = []
    
    preset_data = {
        'timestamp': datetime.now().isoformat(),
        'data': user_data[user_id].copy()
    }
    
    user_data[user_id]['presets'].append(preset_data)
    
    await query.edit_message_text(
        "✅ Пресет успешно сохранен!\n\n"
        "Используйте /presets для просмотра сохраненных стратегий.\n"
        "Используйте /start для нового расчета."
    )

async def show_presets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать сохраненные пресеты"""
    user_id = update.message.from_user.id
    
    if 'presets' not in user_data[user_id] or not user_data[user_id]['presets']:
        await update.message.reply_text("📝 У вас нет сохраненных пресетов.")
        return
    
    presets = user_data[user_id]['presets']
    
    for i, preset in enumerate(presets[-5:], 1):  # Показываем последние 5 пресетов
        data = preset['data']
        await update.message.reply_text(
            f"📋 *Пресет #{i}*\n"
            f"Депозит: ${data['deposit']:,.2f}\n"
            f"Плечо: {data['leverage']}\n"
            f"Пара: {data['currency']}\n"
            f"Вход: {data['entry']}\n"
            f"SL: {data['stop_loss']}\n"
            f"TP: {', '.join(map(str, data['take_profits']))}",
            parse_mode='Markdown'
        )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена разговора"""
    await update.message.reply_text(
        'Расчет отменен. Используйте /start для нового расчета.'
    )
    return ConversationHandler.END

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Справка по боту"""
    help_text = """
🤖 *Risk Management Bot - Помощь*

*Доступные команды:*
/start - Начать новый расчет
/presets - Показать сохраненные пресеты
/help - Показать эту справку

*Процесс расчета:*
1. Ввод депозита
2. Выбор плеча
3. Выбор валютной пары
4. Ввод цены входа
5. Ввод стоп-лосса
6. Ввод тейк-профитов
7. Распределение объемов

*Особенности:*
- Автоматический расчет оптимального объема
- Учет кредитного плеча
- Распределение объемов между тейк-профитами
- Визуализация рисков в % от депозита
- Сохранение пресетов стратегий
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')

def main():
    """Запуск бота"""
    # Получаем токен из переменных окружения
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logging.error("Токен бота не найден! Убедитесь, что переменная TELEGRAM_BOT_TOKEN установлена.")
        return

    # Создаем Application
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
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('presets', show_presets))
    application.add_handler(CallbackQueryHandler(save_preset, pattern='^save_preset$'))
    application.add_handler(CallbackQueryHandler(start, pattern='^new_calculation$'))
    
    # Запускаем бота
    print("Бот запущен...")
    application.run_polling()

if __name__ == '__main__':
    main()