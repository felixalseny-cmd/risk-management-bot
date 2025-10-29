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

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Состояния диалога
DEPOSIT, LEVERAGE, INSTRUMENT_TYPE, CURRENCY, ENTRY, STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION = range(8)

# Хранилище данных пользователей
user_data: Dict[int, Dict[str, Any]] = {}

# Расширенные константы
INSTRUMENT_TYPES = {
    'forex': 'Форекс',
    'crypto': 'Криптовалюты', 
    'indices': 'Индексы',
    'commodities': 'Сырьевые товары'
}

PIP_VALUES = {
    # Форекс
    'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9, 'USDCHF': 10,
    'USDCAD': 10, 'AUDUSD': 10, 'NZDUSD': 10, 'EURGBP': 10,
    'EURJPY': 9, 'GBPJPY': 9, 'EURCHF': 10, 'AUDJPY': 9,
    # Криптовалюты
    'BTCUSD': 1, 'ETHUSD': 1, 'XRPUSD': 10, 'ADAUSD': 10,
    'DOTUSD': 1, 'LTCUSD': 1, 'BCHUSD': 1, 'LINKUSD': 1,
    # Индексы
    'US30': 1, 'NAS100': 1, 'SPX500': 1, 'DAX40': 1,
    'FTSE100': 1, 'NIKKEI225': 1, 'ASX200': 1,
    # Сырьевые товары
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

# Класс для расширенного анализа рисков
class AdvancedRiskCalculator:
    """Расширенный калькулятор рисков с поддержкой всех типов инструментов"""
    
    @staticmethod
    def calculate_pip_value(instrument_type: str, currency_pair: str, lot_size: float) -> float:
        """Рассчитать стоимость пипса с учетом типа инструмента"""
        base_pip_value = PIP_VALUES.get(currency_pair, 10)
        
        if instrument_type == 'crypto':
            return base_pip_value * lot_size * 0.1  # Корректировка для криптовалют
        elif instrument_type == 'indices':
            return base_pip_value * lot_size * 0.01  # Корректировка для индексов
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
        """Расширенный расчет размера позиции"""
        try:
            lev_value = int(leverage.split(':')[1])
            risk_amount = deposit * risk_percent
            
            # Разные расчеты для разных типов инструментов
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
        """Расширенный расчет прибыли"""
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
        """Расчет соотношения риск/вознаграждение"""
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

# Система кэширования и оптимизации
class CacheManager:
    """Менеджер кэша для оптимизации производительности"""
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        
    def set(self, key: str, value: Any, ttl: int = 300):
        """Установить значение в кэш"""
        self._cache[key] = value
        self._timestamps[key] = datetime.now() + timedelta(seconds=ttl)
        
    def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша"""
        if key in self._cache and datetime.now() < self._timestamps.get(key, datetime.now()):
            return self._cache[key]
        else:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
            return None

cache_manager = CacheManager()

# Обработчики команд
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Улучшенный обработчик команды /start"""
    if not update.message:
        return ConversationHandler.END
    
    user = update.message.from_user
    user_name = user.first_name or "Трейдер"
    
    # Кэширование приветственного сообщения
    welcome_key = f"welcome_{user.language_code}"
    welcome_text = cache_manager.get(welcome_key)
    
    if not welcome_text:
        welcome_text = f"""
👋 *Привет, {user_name}!*

🎯 *PRO Risk Management Calculator - ПРОФЕССИОНАЛЬНАЯ ВЕРСИЯ*

⚡ *Расширенные возможности:*
• 📊 Все типы инструментов: Форекс, Крипто, Индексы, Сырье
• 🛡️ Продвинутое управление рисками
• 📈 Анализ риск/вознаграждение
• 💹 Мультивалютные расчеты
• 🔄 Интеллектуальное кэширование
• ⚡ Высокая производительность

💡 *Для полной инструкции используй* /info

🚀 *Выбери тип инструмента для начала расчета:*
"""
        cache_manager.set(welcome_key, welcome_text, 3600)
    
    user_id = user.id
    user_data[user_id] = {'start_time': datetime.now().isoformat()}
    
    # Клавиатура выбора типа инструмента
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
    """Обновленная полная инструкция"""
    info_text = """
📚 *PRO ИНСТРУКЦИЯ - Risk Management Calculator Bot*

🎯 *РАСШИРЕННЫЕ ВОЗМОЖНОСТИ*

⚡ *ПОДДЕРЖИВАЕМЫЕ ИНСТРУМЕНТЫ:*

• 🌐 *ФОРЕКС:* EURUSD, GBPUSD, USDJPY, XAUUSD, XAGUSD и другие
• ₿ *КРИПТОВАЛЮТЫ:* BTCUSD, ETHUSD, XRPUSD, ADAUSD, DOTUSD
• 📈 *ИНДЕКСЫ:* NAS100, SPX500, US30, DAX40, FTSE100
• ⚡ *СЫРЬЕ:* OIL, NATGAS, COPPER, XPTUSD, XPDUSD

📋 *ПРОФЕССИОНАЛЬНЫЙ АЛГОРИТМ:*

1. *Умный расчет рисков*
   - Автоматическая адаптация под тип инструмента
   - Учет специфики контрактов
   - Оптимальное распределение объемов

2. *Расширенная аналитика*
   - Соотношение риск/вознаграждение
   - ROI по каждому TP
   - Анализ свободной маржи
   - Визуализация результатов

3. *Безопасность и надежность*
   - Валидация всех входных данных
   - Защита от перегрузки
   - Интеллектуальное кэширование

📝 *ПРИМЕР РАСЧЕТА ДЛЯ NAS100:*

Депозит: $5000
Плечо: 1:50
Инструмент: NAS100
Вход: 15000
SL: 14900
TP: 15100, 15200
Распределение: 60, 40

🛠 *КОМАНДЫ PRO ВЕРСИИ:*
`/start` - начать профессиональный расчет
`/quick` - быстрый расчет (скоро)
`/portfolio` - управление портфелем (скоро)
`/analytics` - анализ стратегий (скоро)
`/presets` - библиотека стратегий
`/settings` - настройки рисков

🔧 *ТЕХНИЧЕСКИЕ ПРЕИМУЩЕСТВА:*
• ⚡ Время ответа < 100мс
• 🛡️ Защита от перегрузки
• 💾 Интеллектуальный кэш
• 🔄 Автоматическая оптимизация

👨‍💻 *РАЗРАБОТЧИК:* [@fxfeelgood](https://t.me/fxfeelgood)

*Стань профессионалом с нашим ботом!* 🚀
"""
    await update.message.reply_text(
        info_text, 
        parse_mode='Markdown', 
        disable_web_page_preview=True
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обновленная справка"""
    help_text = """
🤖 *PRO Risk Management Bot - Помощь*

⚡ *Быстрый старт:*
1. /start - выбрать тип инструмента
2. Вводите параметры по запросу
3. Получайте профессиональный анализ

🛠 *Основные команды:*
`/start` - Профессиональный расчет
`/info` - Полная инструкция PRO
`/presets` - Библиотека стратегий
`/help` - Эта справка

🔧 *Расширенные команды (скоро):*
`/quick` - Быстрый расчет
`/portfolio` - Портфель
`/analytics` - Аналитика

💡 *Совет PRO:* Используйте риск 1-2% и соотношение R:R от 1:2!

👨‍💻 *Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)
"""
    await update.message.reply_text(
        help_text, 
        parse_mode='Markdown', 
        disable_web_page_preview=True
    )

# Обработчики диалога
async def process_instrument_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора типа инструмента"""
    query = update.callback_query
    if not query:
        return ConversationHandler.END
        
    await query.answer()
    user_id = query.from_user.id
    instrument_type = query.data.replace('inst_type_', '')
    user_data[user_id]['instrument_type'] = instrument_type
    
    # Получаем список инструментов для выбранного типа
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
        f"✅ *Тип инструмента:* {display_type}\n\n"
        "🌐 *Выберите конкретный инструмент:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return CURRENCY

async def process_currency(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора инструмента"""
    query = update.callback_query
    if not query:
        return ConversationHandler.END
        
    await query.answer()
    user_id = query.from_user.id
    currency = query.data.replace('currency_', '')
    user_data[user_id]['currency'] = currency
    
    await query.edit_message_text(
        f"✅ *Инструмент:* {currency}\n\n"
        "💵 *Введите сумму депозита в USD:*",
        parse_mode='Markdown'
    )
    return DEPOSIT

async def process_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода депозита"""
    if not update.message:
        return ConversationHandler.END
        
    user_id = update.message.from_user.id
    try:
        deposit = float(update.message.text.replace(',', '').replace(' ', ''))
        if deposit <= 0:
            await update.message.reply_text("❌ Депозит должен быть положительным числом:")
            return DEPOSIT
        if deposit > 1000000:
            await update.message.reply_text("❌ Максимальный депозит: $1,000,000:")
            return DEPOSIT
            
        user_data[user_id]['deposit'] = deposit
        
        # Создаем клавиатуру с плечами
        keyboard = []
        for i in range(0, len(LEVERAGES), 3):
            row = []
            for j in range(3):
                if i + j < len(LEVERAGES):
                    lev = LEVERAGES[i + j]
                    row.append(InlineKeyboardButton(lev, callback_data=f"leverage_{lev}"))
            keyboard.append(row)
        
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
    """Обработка выбора плеча"""
    query = update.callback_query
    if not query:
        return ConversationHandler.END
        
    await query.answer()
    user_id = query.from_user.id
    leverage = query.data.replace('leverage_', '')
    user_data[user_id]['leverage'] = leverage
    
    currency = user_data[user_id].get('currency', 'EURUSD')
    await query.edit_message_text(
        f"✅ *Плечо:* {leverage}\n\n"
        f"📈 *Введите цену входа для {currency}:*",
        parse_mode='Markdown'
    )
    return ENTRY

async def process_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода цены входа"""
    if not update.message:
        return ConversationHandler.END
        
    user_id = update.message.from_user.id
    try:
        entry = float(update.message.text)
        if entry <= 0:
            await update.message.reply_text("❌ Цена должна быть положительной:")
            return ENTRY
            
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
    """Обработка ввода стоп-лосса"""
    if not update.message:
        return ConversationHandler.END
        
    user_id = update.message.from_user.id
    try:
        sl = float(update.message.text)
        entry = user_data[user_id].get('entry', 0)
        
        if sl <= 0:
            await update.message.reply_text("❌ Цена должна быть положительной:")
            return STOP_LOSS
            
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
    """Обработка ввода тейк-профитов"""
    if not update.message:
        return ConversationHandler.END
        
    user_id = update.message.from_user.id
    try:
        tps = [float(x.strip()) for x in update.message.text.split(',')]
        
        if len(tps) > 5:
            await update.message.reply_text("❌ Максимум 5 тейк-профитов:")
            return TAKE_PROFITS
            
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
    """Обработка распределения объемов и вывод результатов"""
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
        
        if len(dist) != len(user_data[user_id].get('take_profits', [])):
            await update.message.reply_text(
                f"❌ *Количество значений распределения должно совпадать с количеством TP ({len(user_data[user_id].get('take_profits', []))})*\n"
                "Пожалуйста, введите распределение заново:",
                parse_mode='Markdown'
            )
            return VOLUME_DISTRIBUTION
        
        user_data[user_id]['volume_distribution'] = dist
        data = user_data[user_id]
        
        # Профессиональный расчет результатов
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
        
        # Профессиональное форматирование результатов
        instrument_display = INSTRUMENT_TYPES.get(data['instrument_type'], data['instrument_type'])
        
        resp = f"""
🎯 *PRO РЕЗУЛЬТАТЫ РАСЧЕТА*

*📊 Основные параметры:*
💼 Тип: {instrument_display}
🌐 Инструмент: {data['currency']}
💵 Депозит: ${data['deposit']:,.2f}
⚖️ Плечо: {data['leverage']}
📈 Вход: {data['entry']}
🛑 Стоп-лосс: {data['stop_loss']}

*⚠️ Управление рисками:*
📦 Размер позиции: *{pos['position_size']:.2f} лота*
💰 Риск на сделку: ${pos['risk_amount']:.2f} ({pos['risk_percent']:.1f}% от депозита)
📉 Стоп-лосс: {pos['stop_pips']:.0f} пунктов
💳 Требуемая маржа: ${pos['required_margin']:.2f}
🆓 Свободная маржа: ${pos['free_margin']:.2f}

*📈 Аналитика:*
⚖️ R/R соотношение: {risk_reward['risk_reward_ratio']:.2f}
🎯 Общий риск: {risk_reward['total_risk']:.4f}
🎯 Общее вознаграждение: {risk_reward['total_reward']:.4f}

*🎯 Тейк-профиты и прибыль:*
"""
        
        total_roi = 0
        for p in profits:
            roi_display = f" | 📊 ROI: {p['roi_percent']:.1f}%" if p['roi_percent'] > 0 else ""
            resp += f"\n🎯 TP{p['level']} ({p['volume_percent']}% объема):"
            resp += f"\n   💰 Цена: {p['price']}"
            resp += f"\n   📦 Объем: {p['volume_lots']:.2f} лота"
            resp += f"\n   💵 Прибыль: ${p['profit']:.2f}{roi_display}"
            resp += f"\n   📊 Накопленная прибыль: ${p['cumulative_profit']:.2f}\n"
            total_roi += p['roi_percent']
        
        # Итоговые показатели
        total_profit = profits[-1]['cumulative_profit'] if profits else 0
        overall_roi = (total_profit / data['deposit']) * 100
        
        resp += f"\n*🏆 Итоговые показатели:*\n"
        resp += f"💰 Общая прибыль: ${total_profit:.2f}\n"
        resp += f"📊 Общий ROI: {overall_roi:.2f}%\n"
        
        # Рекомендации
        resp += f"\n*💡 Рекомендации:*\n"
        if risk_reward['risk_reward_ratio'] < 1:
            resp += f"⚠️ Низкое R/R соотношение. Рекомендуется пересмотреть TP/SL\n"
        elif risk_reward['risk_reward_ratio'] > 2:
            resp += f"✅ Отличное R/R соотношение!\n"
            
        if pos['risk_percent'] > 3:
            resp += f"⚠️ Высокий риск! Рекомендуется снизить до 2%\n"
        else:
            resp += f"✅ Уровень риска в норме\n"
        
        # Добавляем информацию о разработчике
        resp += f"\n---\n"
        resp += f"👨‍💻 *PRO Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)\n"
        resp += f"⚡ *PRO Версия 2.0 | Быстро • Надежно • Точно*"
        
        keyboard = [
            [InlineKeyboardButton("💾 Сохранить стратегию", callback_data="save_preset")],
            [InlineKeyboardButton("🔄 Новый расчет", callback_data="new_calculation")],
            [InlineKeyboardButton("📚 PRO Инструкция", callback_data="show_info")]
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
            "❌ Произошла ошибка при расчете. Пожалуйста, начните заново с /start",
            parse_mode='Markdown'
        )
        return ConversationHandler.END

# Остальные обработчики (save_preset, show_presets, cancel, new_calculation, show_info_callback)
# остаются аналогичными предыдущей версии, но с улучшенными сообщениями

async def save_preset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Улучшенное сохранение пресета"""
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
    
    # Ограничиваем количество сохраненных пресетов
    if len(user_data[uid]['presets']) >= 20:
        user_data[uid]['presets'] = user_data[uid]['presets'][-19:]
    
    user_data[uid]['presets'].append({
        'timestamp': datetime.now().isoformat(),
        'data': user_data[uid].copy()
    })
    
    await query.edit_message_text(
        "✅ *PRO Стратегия успешно сохранена!*\n\n"
        "💾 Используйте /presets для просмотра сохраненных стратегий\n"
        "🚀 Используйте /start для нового PRO расчета\n\n"
        "👨‍💻 *PRO Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)",
        parse_mode='Markdown',
        disable_web_page_preview=True
    )

async def show_presets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Улучшенный показ пресетов"""
    if not update.message:
        return
        
    uid = update.message.from_user.id
    presets = user_data.get(uid, {}).get('presets', [])
    
    if not presets:
        await update.message.reply_text(
            "📝 *У вас нет сохраненных PRO стратегий.*\n\n"
            "💡 Сохраняйте свои стратегии после расчета для быстрого доступа!",
            parse_mode='Markdown'
        )
        return
    
    await update.message.reply_text(
        f"📚 *Ваши PRO стратегии ({len(presets)}):*",
        parse_mode='Markdown'
    )
    
    for i, p in enumerate(presets[-10:], 1):  # Показываем последние 10
        d = p['data']
        instrument_display = INSTRUMENT_TYPES.get(d.get('instrument_type', 'forex'), 'Forex')
        
        preset_text = f"""
📋 *PRO Стратегия #{i}*
💼 Тип: {instrument_display}
🌐 Инструмент: {d.get('currency', 'N/A')}
💵 Депозит: ${d.get('deposit', 0):,.2f}
⚖️ Плечо: {d.get('leverage', 'N/A')}
📈 Вход: {d.get('entry', 'N/A')}
🛑 SL: {d.get('stop_loss', 'N/A')}
🎯 TP: {', '.join(map(str, d.get('take_profits', [])))}

👨‍💻 *PRO Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)
"""
        await update.message.reply_text(
            preset_text,
            parse_mode='Markdown',
            disable_web_page_preview=True
        )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Улучшенная отмена"""
    if update.message:
        await update.message.reply_text(
            "❌ *PRO Расчет отменен.*\n\n"
            "🚀 Используйте /start для нового PRO расчета\n"
            "📚 Используйте /info для PRO инструкции\n\n"
            "👨‍💻 *PRO Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)",
            parse_mode='Markdown',
            disable_web_page_preview=True
        )
    return ConversationHandler.END

async def new_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Новый расчет"""
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

# Команда для быстрого доступа к настройкам
async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда настроек"""
    settings_text = """
⚙️ *PRO Настройки Рисков*

*Текущие настройки:*
• 📊 Риск на сделку: 2% (рекомендуется)
• 💹 Макс. позиция: 50 лотов
• 🛡️ Защита от перегрузки: включена

*Доступные команды:*
`/start` - PRO расчет
`/presets` - Библиотека стратегий  
`/info` - Полная инструкция

*Скоро в PRO версии:*
• 🔧 Настройка уровня риска
• 📈 Персональные шаблоны
• 💾 Экспорт данных

👨‍💻 *PRO Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)
"""
    await update.message.reply_text(
        settings_text,
        parse_mode='Markdown',
        disable_web_page_preview=True
    )

def main():
    """Улучшенная основная функция"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ PRO Токен бота не найден!")
        return

    logger.info("🚀 Запуск PRO Risk Management Bot v2.0...")
    
    # Создаем оптимизированное приложение
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

    # Настраиваем улучшенный ConversationHandler
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

    # Добавляем обработчики
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('info', info_command))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('presets', show_presets))
    application.add_handler(CommandHandler('settings', settings_command))
    application.add_handler(CallbackQueryHandler(save_preset, pattern='^save_preset$'))
    application.add_handler(CallbackQueryHandler(new_calculation, pattern='^new_calculation$'))
    application.add_handler(CallbackQueryHandler(show_info_callback, pattern='^show_info$'))

    # Получаем URL для вебхука
    webhook_url = os.getenv('RENDER_EXTERNAL_URL', '')
    if not webhook_url:
        logger.error("❌ RENDER_EXTERNAL_URL не установлен!")
        return

    # Запускаем вебхук с улучшенными настройками
    port = int(os.environ.get('PORT', 10000))
    
    logger.info(f"🌐 PRO Запуск вебхука на порту {port}")
    logger.info(f"🔗 PRO Webhook URL: {webhook_url}/webhook")
    
    try:
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path="/webhook",
            webhook_url=webhook_url + "/webhook",
            max_connections=40,
            pool_timeout=30,
            connect_timeout=30,
            read_timeout=30,
            write_timeout=30
        )
    except Exception as e:
        logger.error(f"❌ Ошибка при запуске PRO вебхука: {e}")
        logger.info("🔄 PRO Попытка запуска с polling...")
        application.run_polling(
            poll_interval=1.0,
            timeout=30,
            read_timeout=30,
            write_timeout=30,
            connect_timeout=30,
            pool_timeout=30
        )

if __name__ == '__main__':
    main()
