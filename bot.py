import os
import logging
import asyncio
import re
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

# Логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Состояния диалога
(
    MAIN_MENU, INSTRUMENT_TYPE, CUSTOM_INSTRUMENT, DIRECTION, 
    RISK_PERCENT, DEPOSIT, LEVERAGE, CURRENCY, ENTRY, 
    STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION
) = range(12)

# Временное хранилище
user_data: Dict[int, Dict[str, Any]] = {}

# Расширенные константы
INSTRUMENT_TYPES = {
    'forex': 'Форекс',
    'crypto': 'Криптовалюты', 
    'indices': 'Индексы',
    'commodities': 'Сырьевые товары',
    'metals': 'Металлы'
}

# Расширенный список инструментов
PIP_VALUES = {
    # Форекс - основные пары
    'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9, 'USDCHF': 10,
    'USDCAD': 10, 'AUDUSD': 10, 'NZDUSD': 10, 'EURGBP': 10,
    'EURJPY': 9, 'GBPJPY': 9, 'EURCHF': 10, 'AUDJPY': 9,
    'NZDJPY': 9, 'CADJPY': 9, 'CHFJPY': 9, 'GBPCAD': 10,
    'GBPAUD': 10, 'GBPNZD': 10, 'EURAUD': 10, 'EURCAD': 10,
    'EURNZD': 10, 'AUDCAD': 10, 'AUDCHF': 10, 'AUDNZD': 10,
    'CADCHF': 10, 'NZDCAD': 10, 'NZDCHF': 10,
    # Форекс - экзотические пары
    'USDSEK': 10, 'USDDKK': 10, 'USDNOK': 10, 'USDPLN': 10,
    'USDCZK': 10, 'USDHUF': 10, 'USDRON': 10, 'USDTRY': 10,
    'USDZAR': 10, 'USDMXN': 10, 'USDSGD': 10, 'USDHKD': 10,
    # Криптовалюты
    'BTCUSD': 1, 'ETHUSD': 1, 'XRPUSD': 10, 'ADAUSD': 10,
    'DOTUSD': 1, 'LTCUSD': 1, 'BCHUSD': 1, 'LINKUSD': 1,
    'BNBUSD': 1, 'SOLUSD': 1, 'DOGEUSD': 10, 'MATICUSD': 10,
    'AVAXUSD': 1, 'ATOMUSD': 1, 'UNIUSD': 1, 'XLMUSD': 10,
    # Индексы
    'US30': 1, 'NAS100': 1, 'SPX500': 1, 'DAX40': 1,
    'FTSE100': 1, 'NIKKEI225': 1, 'ASX200': 1, 'CAC40': 1,
    'ESTX50': 1, 'HSI': 1, 'SENSEX': 1, 'IBOVESPA': 1,
    # Сырьевые товары
    'OIL': 10, 'NATGAS': 10, 'COPPER': 10, 'WHEAT': 10,
    'CORN': 10, 'SOYBEAN': 10, 'SUGAR': 10, 'COFFEE': 10,
    # Металлы
    'XAUUSD': 10, 'XAGUSD': 50, 'XPTUSD': 10, 'XPDUSD': 10,
    'XAUAUD': 10, 'XAUEUR': 10, 'XAGGBP': 50
}

CONTRACT_SIZES = {
    'forex': 100000,
    'crypto': 1,
    'indices': 1,
    'commodities': 100,
    'metals': 100
}

LEVERAGES = ['1:10', '1:20', '1:50', '1:100', '1:200', '1:500', '1:1000']
RISK_LEVELS = ['2%', '5%', '10%', '15%', '20%', '25%']
TRADE_DIRECTIONS = ['BUY', 'SELL']

# Класс для расширенного анализа рисков
class AdvancedRiskCalculator:
    """Расширенный калькулятор рисков с поддержкой всех типов инструментов"""
    
    @staticmethod
    def calculate_pip_value(instrument_type: str, currency_pair: str, lot_size: float) -> float:
        """Рассчитать стоимость пипса с учетом типа инструмента"""
        base_pip_value = PIP_VALUES.get(currency_pair, 10)
        
        if instrument_type == 'crypto':
            return base_pip_value * lot_size * 0.1
        elif instrument_type == 'indices':
            return base_pip_value * lot_size * 0.01
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
        direction: str,
        risk_percent: float = 0.02
    ) -> Dict[str, float]:
        """Расширенный расчет размера позиции с учетом направления"""
        try:
            lev_value = int(leverage.split(':')[1])
            risk_amount = deposit * risk_percent
            
            # Разные расчеты для разных типов инструментов
            if instrument_type == 'forex':
                stop_pips = abs(entry_price - stop_loss) * 10000
            elif instrument_type == 'crypto':
                stop_pips = abs(entry_price - stop_loss) * 100
            elif instrument_type in ['indices', 'commodities', 'metals']:
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
        volume_distribution: List[float],
        direction: str
    ) -> List[Dict[str, Any]]:
        """Расширенный расчет прибыли с учетом направления"""
        profits = []
        total_profit = 0
        
        for i, (tp, vol_pct) in enumerate(zip(take_profits, volume_distribution)):
            if instrument_type == 'forex':
                tp_pips = abs(entry_price - tp) * 10000
            elif instrument_type == 'crypto':
                tp_pips = abs(entry_price - tp) * 100
            elif instrument_type in ['indices', 'commodities', 'metals']:
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
                'pips': tp_pips,
                'roi_percent': (profit / (position_size * CONTRACT_SIZES.get(instrument_type, 100000) * entry_price)) * 100
            })
            
        return profits

    @staticmethod
    def calculate_risk_reward_ratio(
        entry_price: float,
        stop_loss: float,
        take_profits: List[float],
        volume_distribution: List[float],
        direction: str
    ) -> Dict[str, float]:
        """Расчет соотношения риск/вознаграждение с учетом направления"""
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

# Основные обработчики команд
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Главное меню"""
    if not update.message:
        return ConversationHandler.END
    
    user = update.message.from_user
    user_name = user.first_name or "Трейдер"
    
    welcome_text = f"""
👋 *Привет, {user_name}!*

🎯 *PRO Risk Management Calculator v3.0*

⚡ *Выберите опцию:*
"""
    
    user_id = user.id
    user_data[user_id] = {'start_time': datetime.now().isoformat()}
    
    keyboard = [
        [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("⚡ Быстрый расчет", callback_data="quick_calculation")],
        [InlineKeyboardButton("💼 Мой портфель", callback_data="portfolio")],
        [InlineKeyboardButton("📈 Аналитика", callback_data="analytics")],
        [InlineKeyboardButton("📚 PRO Инструкция", callback_data="pro_info")]
    ]
    
    await update.message.reply_text(
        welcome_text, 
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return MAIN_MENU

async def quick_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Быстрый расчет"""
    return await start_quick_calculation(update, context)

async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Управление портфелем"""
    portfolio_text = """
💼 *Управление Портфелем*

*📊 Функции портфеля:*
• 📈 Обзор всех сделок
• 💰 Баланс и распределение
• 📊 Анализ эффективности
• 🔄 История операций

*🚀 Скоро будет доступно:*
• 📊 Визуализация портфеля
• 📈 Сравнение с рынком
• 💡 Рекомендации по диверсификации

*📚 Используйте профессиональный расчет для управления рисками!*

👨‍💻 *PRO Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)
"""
    await update.message.reply_text(
        portfolio_text,
        parse_mode='Markdown',
        disable_web_page_preview=True,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
    )

async def analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Аналитика стратегий"""
    analytics_text = """
📈 *Аналитика Стратегий*

*📊 Доступная аналитика:*
• 📈 Анализ риск/вознаграждение
• 💹 Эффективность стратегий
• 📊 Статистика сделок
• 🔄 Оптимизация параметров

*🚀 Скоро будет доступно:*
• 🤖 AI-анализ стратегий
• 📊 Бэктестинг
• 📈 Прогнозирование
• 💡 Интеллектуальные рекомендации

*📚 Используйте профессиональный расчет для анализа!*

👨‍💻 *PRO Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)
"""
    await update.message.reply_text(
        analytics_text,
        parse_mode='Markdown',
        disable_web_page_preview=True,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
    )

async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """PRO Инструкция"""
    info_text = """
📚 *PRO ИНСТРУКЦИЯ v3.0*

🎯 *РАСШИРЕННЫЕ ВОЗМОЖНОСТИ:*

⚡ *ВСЕ ТИПЫ ИНСТРУМЕНТОВ:*
• 🌐 Форекс (50+ валютных пар)
• ₿ Криптовалюты (15+ пар)
• 📈 Индексы (12+ индексов)
• ⚡ Сырьевые товары (8+ видов)
• 🏅 Металлы (6+ видов)

🔄 *НОВЫЕ ФУНКЦИИ v3.0:*

1. *⚡ Быстрый расчет*
   - Минимальный ввод параметров
   - Автоматические настройки
   - Мгновенные результаты

2. *🎯 Выбор направления*
   - BUY/SELL расчеты
   - Учет направления в рисках
   - Оптимизированные рекомендации

3. *⚖️ Гибкое управление риском*
   - 6 уровней риска (2%-25%)
   - Индивидуальные настройки
   - Адаптивные алгоритмы

4. *🔧 Произвольные инструменты*
   - Ввод любого тикера вручную
   - Автоматическое определение параметров
   - Широкая поддержка активов

📋 *КАК ИСПОЛЬЗОВАТЬ:*

*Профессиональный расчет:*
1. Выберите тип инструмента
2. Выберите конкретный инструмент или введите свой
3. Укажите направление сделки (BUY/SELL)
4. Выберите уровень риска
5. Введите основные параметры
6. Получите детальный анализ

*Быстрый расчет:*
1. Введите инструмент
2. Укажите базовые параметры
3. Получите моментальный результат

🛠 *КОМАНДЫ PRO v3.0:*
`/start` - Главное меню
`/quick` - Быстрый расчет
`/portfolio` - Управление портфелем
`/analytics` - Аналитика стратегий
`/info` - PRO инструкция

👨‍💻 *РАЗРАБОТЧИК:* [@fxfeelgood](https://t.me/fxfeelgood)

*PRO v3.0 | Быстро • Умно • Точно* 🚀
"""
    await update.message.reply_text(
        info_text, 
        parse_mode='Markdown', 
        disable_web_page_preview=True,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
    )

# Обработчики главного меню
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора в главном меню"""
    query = update.callback_query
    if not query:
        return MAIN_MENU
        
    await query.answer()
    choice = query.data
    
    if choice == "pro_calculation":
        return await start_pro_calculation(update, context)
    elif choice == "quick_calculation":
        return await start_quick_calculation(update, context)
    elif choice == "portfolio":
        await portfolio_command(update, context)
        return MAIN_MENU
    elif choice == "analytics":
        await analytics_command(update, context)
        return MAIN_MENU
    elif choice == "pro_info":
        await pro_info_command(update, context)
        return MAIN_MENU
    elif choice == "main_menu":
        return await start(update, context)
    
    return MAIN_MENU

async def start_pro_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало профессионального расчета"""
    query = update.callback_query
    if query:
        await query.edit_message_text(
            "🎯 *Профессиональный расчет*\n\n"
            "📊 *Выберите тип инструмента:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 Форекс", callback_data="inst_type_forex")],
                [InlineKeyboardButton("₿ Криптовалюты", callback_data="inst_type_crypto")],
                [InlineKeyboardButton("📈 Индексы", callback_data="inst_type_indices")],
                [InlineKeyboardButton("⚡ Сырьевые товары", callback_data="inst_type_commodities")],
                [InlineKeyboardButton("🏅 Металлы", callback_data="inst_type_metals")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ])
        )
    return INSTRUMENT_TYPE

async def start_quick_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало быстрого расчета"""
    if update.message:
        await update.message.reply_text(
            "⚡ *Быстрый расчет*\n\n"
            "📊 *Введите тикер инструмента* (например: EURUSD, BTCUSD, NAS100):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
        )
    else:
        query = update.callback_query
        await query.edit_message_text(
            "⚡ *Быстрый расчет*\n\n"
            "📊 *Введите тикер инструмента* (например: EURUSD, BTCUSD, NAS100):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
        )
    return CUSTOM_INSTRUMENT

# Обработчики профессионального расчета
async def process_instrument_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора типа инструмента"""
    query = update.callback_query
    if not query:
        return INSTRUMENT_TYPE
        
    await query.answer()
    user_id = query.from_user.id
    instrument_type = query.data.replace('inst_type_', '')
    user_data[user_id]['instrument_type'] = instrument_type
    
    # Получаем список инструментов для выбранного типа
    instruments = {
        'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD', 'Другая пара'],
        'crypto': ['BTCUSD', 'ETHUSD', 'XRPUSD', 'ADAUSD', 'DOTUSD', 'Другая крипта'],
        'indices': ['US30', 'NAS100', 'SPX500', 'DAX40', 'FTSE100', 'Другой индекс'],
        'commodities': ['OIL', 'NATGAS', 'COPPER', 'WHEAT', 'Другое сырье'],
        'metals': ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'Другой металл']
    }.get(instrument_type, [])
    
    keyboard = []
    for i in range(0, len(instruments), 2):
        row = []
        for j in range(2):
            if i + j < len(instruments):
                inst = instruments[i + j]
                if inst.startswith('Друг'):
                    row.append(InlineKeyboardButton("📝 " + inst, callback_data="custom_instrument"))
                else:
                    row.append(InlineKeyboardButton(inst, callback_data=f"currency_{inst}"))
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")])
    
    display_type = INSTRUMENT_TYPES.get(instrument_type, instrument_type)
    await query.edit_message_text(
        f"✅ *Тип инструмента:* {display_type}\n\n"
        "🌐 *Выберите конкретный инструмент:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return CURRENCY

async def process_custom_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода произвольного инструмента"""
    query = update.callback_query
    if query:
        await query.edit_message_text(
            "📝 *Введите тикер инструмента вручную*\n\n"
            "Примеры:\n"
            "• EURGBP, USDSEK, GBPAUD\n"
            "• BNBUSD, SOLUSD, DOGEUSD\n"
            "• CAC40, ESTX50, HSI\n\n"
            "*Введите тикер:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="back_to_instruments")]])
        )
    return CUSTOM_INSTRUMENT

async def process_currency_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода тикера инструмента"""
    if not update.message:
        return CUSTOM_INSTRUMENT
        
    user_id = update.message.from_user.id
    currency = update.message.text.upper().strip()
    
    # Базовая валидация тикера
    if not re.match(r'^[A-Z0-9]{2,10}$', currency):
        await update.message.reply_text(
            "❌ *Неверный формат тикера!*\n\n"
            "Пожалуйста, введите корректный тикер (только буквы и цифры, 2-10 символов):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="back_to_instruments")]])
        )
        return CUSTOM_INSTRUMENT
    
    user_data[user_id]['currency'] = currency
    
    # Определяем тип инструмента по умолчанию, если не установлен
    if 'instrument_type' not in user_data[user_id]:
        if any(x in currency for x in ['BTC', 'ETH', 'XRP', 'ADA']):
            user_data[user_id]['instrument_type'] = 'crypto'
        elif any(x in currency for x in ['XAU', 'XAG', 'XPT', 'XPD']):
            user_data[user_id]['instrument_type'] = 'metals'
        elif currency.isalpha() and len(currency) == 6:
            user_data[user_id]['instrument_type'] = 'forex'
        else:
            user_data[user_id]['instrument_type'] = 'indices'
    
    await update.message.reply_text(
        f"✅ *Инструмент:* {currency}\n\n"
        "🎯 *Выберите направление сделки:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 BUY (Покупка)", callback_data="direction_BUY")],
            [InlineKeyboardButton("📉 SELL (Продажа)", callback_data="direction_SELL")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_instruments")]
        ])
    )
    return DIRECTION

async def process_currency_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора инструмента из списка"""
    query = update.callback_query
    if not query:
        return CURRENCY
        
    await query.answer()
    user_id = query.from_user.id
    
    if query.data == "custom_instrument":
        return await process_custom_instrument(update, context)
    elif query.data == "back_to_instruments":
        return await start_pro_calculation(update, context)
    
    currency = query.data.replace('currency_', '')
    user_data[user_id]['currency'] = currency
    
    await query.edit_message_text(
        f"✅ *Инструмент:* {currency}\n\n"
        "🎯 *Выберите направление сделки:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 BUY (Покупка)", callback_data="direction_BUY")],
            [InlineKeyboardButton("📉 SELL (Продажа)", callback_data="direction_SELL")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_instruments")]
        ])
    )
    return DIRECTION

async def process_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора направления"""
    query = update.callback_query
    if not query:
        return DIRECTION
        
    await query.answer()
    user_id = query.from_user.id
    direction = query.data.replace('direction_', '')
    user_data[user_id]['direction'] = direction
    
    await query.edit_message_text(
        f"✅ *Направление:* {direction}\n\n"
        "⚖️ *Выберите уровень риска на сделку:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("2% (Консервативный)", callback_data="risk_0.02")],
            [InlineKeyboardButton("5% (Умеренный)", callback_data="risk_0.05")],
            [InlineKeyboardButton("10% (Агрессивный)", callback_data="risk_0.10")],
            [InlineKeyboardButton("15% (Высокий)", callback_data="risk_0.15")],
            [InlineKeyboardButton("20% (Очень высокий)", callback_data="risk_0.20")],
            [InlineKeyboardButton("25% (Экстремальный)", callback_data="risk_0.25")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_direction")]
        ])
    )
    return RISK_PERCENT

async def process_risk_percent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора уровня риска"""
    query = update.callback_query
    if not query:
        return RISK_PERCENT
        
    await query.answer()
    user_id = query.from_user.id
    
    if query.data == "back_to_direction":
        currency = user_data[user_id].get('currency', 'EURUSD')
        await query.edit_message_text(
            f"✅ *Инструмент:* {currency}\n\n"
            "🎯 *Выберите направление сделки:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📈 BUY (Покупка)", callback_data="direction_BUY")],
                [InlineKeyboardButton("📉 SELL (Продажа)", callback_data="direction_SELL")],
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_instruments")]
            ])
        )
        return DIRECTION
    
    risk_percent = float(query.data.replace('risk_', ''))
    user_data[user_id]['risk_percent'] = risk_percent
    
    await query.edit_message_text(
        f"✅ *Уровень риска:* {risk_percent*100}%\n\n"
        "💵 *Введите сумму депозита в USD:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="back_to_risk")]])
    )
    return DEPOSIT

async def process_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода депозита"""
    if not update.message:
        return DEPOSIT
        
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
        
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_deposit")])
        
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
        return LEVERAGE
        
    await query.answer()
    user_id = query.from_user.id
    
    if query.data == "back_to_deposit":
        await query.edit_message_text(
            "💵 *Введите сумму депозита в USD:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="back_to_risk")]])
        )
        return DEPOSIT
    
    leverage = query.data.replace('leverage_', '')
    user_data[user_id]['leverage'] = leverage
    
    currency = user_data[user_id].get('currency', 'EURUSD')
    direction = user_data[user_id].get('direction', 'BUY')
    
    await query.edit_message_text(
        f"✅ *Плечо:* {leverage}\n"
        f"✅ *Направление:* {direction}\n\n"
        f"📈 *Введите цену входа для {currency}:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="back_to_leverage")]])
    )
    return ENTRY

async def process_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода цены входа"""
    if not update.message:
        return ENTRY
        
    user_id = update.message.from_user.id
    
    try:
        entry = float(update.message.text)
        if entry <= 0:
            await update.message.reply_text("❌ Цена должна быть положительной:")
            return ENTRY
            
        user_data[user_id]['entry'] = entry
        
        currency = user_data[user_id].get('currency', 'EURUSD')
        direction = user_data[user_id].get('direction', 'BUY')
        
        await update.message.reply_text(
            f"✅ *Цена входа:* {entry}\n"
            f"✅ *Направление:* {direction}\n\n"
            f"🛑 *Введите цену стоп-лосса для {currency}:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="back_to_entry")]])
        )
        return STOP_LOSS
        
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректную цену входа:")
        return ENTRY

async def process_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода стоп-лосса"""
    if not update.message:
        return STOP_LOSS
        
    user_id = update.message.from_user.id
    
    try:
        sl = float(update.message.text)
        entry = user_data[user_id].get('entry', 0)
        
        if sl <= 0:
            await update.message.reply_text("❌ Цена должна быть положительной:")
            return STOP_LOSS
            
        user_data[user_id]['stop_loss'] = sl
        
        currency = user_data[user_id].get('currency', 'EURUSD')
        
        await update.message.reply_text(
            f"✅ *Стоп-лосс:* {sl}\n\n"
            f"🎯 *Введите цены тейк-профитов для {currency} через запятую* (например: 1.0550, 1.0460):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="back_to_stop_loss")]])
        )
        return TAKE_PROFITS
        
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректную цену стоп-лосса:")
        return STOP_LOSS

async def process_take_profits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода тейк-профитов"""
    if not update.message:
        return TAKE_PROFITS
        
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
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="back_to_take_profits")]])
        )
        return VOLUME_DISTRIBUTION
        
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите корректные цены тейк-профитов:")
        return TAKE_PROFITS

async def process_volume_distribution(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка распределения объемов и вывод результатов"""
    if not update.message:
        return VOLUME_DISTRIBUTION
        
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
            stop_loss=data['stop_loss'],
            direction=data.get('direction', 'BUY'),
            risk_percent=data.get('risk_percent', 0.02)
        )
        
        profits = AdvancedRiskCalculator.calculate_profits(
            instrument_type=data['instrument_type'],
            currency_pair=data['currency'],
            entry_price=data['entry'],
            take_profits=data['take_profits'],
            position_size=pos['position_size'],
            volume_distribution=dist,
            direction=data.get('direction', 'BUY')
        )
        
        risk_reward = AdvancedRiskCalculator.calculate_risk_reward_ratio(
            entry_price=data['entry'],
            stop_loss=data['stop_loss'],
            take_profits=data['take_profits'],
            volume_distribution=dist,
            direction=data.get('direction', 'BUY')
        )
        
        # Профессиональное форматирование результатов
        instrument_display = INSTRUMENT_TYPES.get(data['instrument_type'], data['instrument_type'])
        direction_display = "📈 BUY" if data.get('direction', 'BUY') == 'BUY' else "📉 SELL"
        
        resp = f"""
🎯 *PRO РЕЗУЛЬТАТЫ РАСЧЕТА*

*📊 Основные параметры:*
💼 Тип: {instrument_display}
🌐 Инструмент: {data['currency']}
🎯 Направление: {direction_display}
💵 Депозит: ${data['deposit']:,.2f}
⚖️ Плечо: {data['leverage']}
📈 Вход: {data['entry']}
🛑 Стоп-лосс: {data['stop_loss']}
⚠️ Риск: {data.get('risk_percent', 0.02)*100}%

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
            resp += f"\n   📊 Пункты: {p['pips']:.0f} pips"
            resp += f"\n   💵 Прибыль: ${p['profit']:.2f}{roi_display}"
            resp += f"\n   📈 Накопленная прибыль: ${p['cumulative_profit']:.2f}\n"
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
            
        if data.get('risk_percent', 0.02) > 0.03:
            resp += f"⚠️ Высокий риск! Рекомендуется снизить до 2-3%\n"
        else:
            resp += f"✅ Уровень риска в норме\n"
        
        # Добавляем информацию о разработчике
        resp += f"\n---\n"
        resp += f"👨‍💻 *PRO Разработчик:* [@fxfeelgood](https://t.me/fxfeelgood)\n"
        resp += f"⚡ *PRO Версия 3.0 | Быстро • Умно • Точно*"
        
        keyboard = [
            [InlineKeyboardButton("💾 Сохранить стратегию", callback_data="save_preset")],
            [InlineKeyboardButton("🔄 Новый расчет", callback_data="new_calculation")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
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

# Обработчики кнопок "Назад"
async def process_risk_percent_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Назад к выбору риска"""
    query = update.callback_query if hasattr(update, 'callback_query') else None
    user_id = update.message.from_user.id if update.message else query.from_user.id
    
    risk_percent = user_data[user_id].get('risk_percent', 0.02)
    
    if query:
        await query.edit_message_text(
            f"✅ *Уровень риска:* {risk_percent*100}%\n\n"
            "⚖️ *Выберите уровень риска на сделку:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("2% (Консервативный)", callback_data="risk_0.02")],
                [InlineKeyboardButton("5% (Умеренный)", callback_data="risk_0.05")],
                [InlineKeyboardButton("10% (Агрессивный)", callback_data="risk_0.10")],
                [InlineKeyboardButton("15% (Высокий)", callback_data="risk_0.15")],
                [InlineKeyboardButton("20% (Очень высокий)", callback_data="risk_0.20")],
                [InlineKeyboardButton("25% (Экстремальный)", callback_data="risk_0.25")],
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_direction")]
            ])
        )
    else:
        await update.message.reply_text(
            f"✅ *Уровень риска:* {risk_percent*100}%\n\n"
            "⚖️ *Выберите уровень риска на сделку:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("2% (Консервативный)", callback_data="risk_0.02")],
                [InlineKeyboardButton("5% (Умеренный)", callback_data="risk_0.05")],
                [InlineKeyboardButton("10% (Агрессивный)", callback_data="risk_0.10")],
                [InlineKeyboardButton("15% (Высокий)", callback_data="risk_0.15")],
                [InlineKeyboardButton("20% (Очень высокий)", callback_data="risk_0.20")],
                [InlineKeyboardButton("25% (Экстремальный)", callback_data="risk_0.25")],
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_direction")]
            ])
        )
    return RISK_PERCENT

# Обработчики для быстрого расчета
async def process_quick_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка быстрого расчета"""
    if not update.message:
        return CUSTOM_INSTRUMENT
        
    user_id = update.message.from_user.id
    currency = update.message.text.upper().strip()
    
    # Базовая валидация тикера
    if not re.match(r'^[A-Z0-9]{2,10}$', currency):
        await update.message.reply_text(
            "❌ *Неверный формат тикера!*\n\n"
            "Пожалуйста, введите корректный тикер (только буквы и цифры, 2-10 символов):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
        )
        return CUSTOM_INSTRUMENT
    
    user_data[user_id] = {
        'currency': currency,
        'direction': 'BUY',
        'risk_percent': 0.02,
        'leverage': '1:100',
        'take_profits': [],
        'volume_distribution': [100]
    }
    
    # Определяем тип инструмента
    if any(x in currency for x in ['BTC', 'ETH', 'XRP', 'ADA']):
        user_data[user_id]['instrument_type'] = 'crypto'
    elif any(x in currency for x in ['XAU', 'XAG', 'XPT', 'XPD']):
        user_data[user_id]['instrument_type'] = 'metals'
    elif currency.isalpha() and len(currency) == 6:
        user_data[user_id]['instrument_type'] = 'forex'
    else:
        user_data[user_id]['instrument_type'] = 'indices'
    
    await update.message.reply_text(
        f"✅ *Инструмент:* {currency}\n\n"
        "💵 *Введите сумму депозита в USD:*",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
    )
    return DEPOSIT

# Дополнительные обработчики
async def save_preset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сохранение пресета"""
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
    """Показать сохраненные пресеты"""
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
    
    for i, p in enumerate(presets[-10:], 1):
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
    """Отмена диалога"""
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

def main():
    """Основная функция для запуска бота"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ PRO Токен бота не найден!")
        return

    logger.info("🚀 Запуск PRO Risk Management Bot v3.0...")
    
    # Создаем приложение
    application = Application.builder().token(token).build()

    # Настраиваем ConversationHandler для главного меню
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', start),
            CommandHandler('quick', quick_command),
            CommandHandler('portfolio', portfolio_command),
            CommandHandler('analytics', analytics_command),
            CommandHandler('info', pro_info_command)
        ],
        states={
            MAIN_MENU: [CallbackQueryHandler(handle_main_menu, pattern='^(pro_calculation|quick_calculation|portfolio|analytics|pro_info|main_menu)$')],
            INSTRUMENT_TYPE: [CallbackQueryHandler(process_instrument_type, pattern='^inst_type_')],
            CUSTOM_INSTRUMENT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_currency_input),
                CallbackQueryHandler(process_currency_selection, pattern='^(back_to_instruments|custom_instrument)')
            ],
            CURRENCY: [CallbackQueryHandler(process_currency_selection, pattern='^(currency_|back_to_instruments|custom_instrument)')],
            DIRECTION: [CallbackQueryHandler(process_direction, pattern='^(direction_|back_to_instruments)')],
            RISK_PERCENT: [CallbackQueryHandler(process_risk_percent, pattern='^(risk_|back_to_direction|back_to_risk)')],
            DEPOSIT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_deposit),
                CallbackQueryHandler(process_risk_percent_back, pattern='^back_to_deposit$')
            ],
            LEVERAGE: [CallbackQueryHandler(process_leverage, pattern='^(leverage_|back_to_deposit)')],
            ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_entry)],
            STOP_LOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_stop_loss)],
            TAKE_PROFITS: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_take_profits)],
            VOLUME_DISTRIBUTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_volume_distribution)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # Добавляем обработчики
    application.add_handler(conv_handler)
    application.add_handler(CallbackQueryHandler(save_preset, pattern='^save_preset$'))
    application.add_handler(CallbackQueryHandler(new_calculation, pattern='^new_calculation$'))

    # Получаем URL для вебхука
    webhook_url = os.getenv('RENDER_EXTERNAL_URL', '')
    if not webhook_url:
        logger.error("❌ RENDER_EXTERNAL_URL не установлен!")
        return

    # Запускаем вебхук
    port = int(os.environ.get('PORT', 10000))
    
    logger.info(f"🌐 PRO Запуск вебхука на порту {port}")
    logger.info(f"🔗 PRO Webhook URL: {webhook_url}/webhook")
    
    try:
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path="/webhook",
            webhook_url=webhook_url + "/webhook"
        )
    except Exception as e:
        logger.error(f"❌ Ошибка при запуске PRO вебхука: {e}")
        logger.info("🔄 PRO Попытка запуска с polling...")
        application.run_polling()

if __name__ == '__main__':
    main()
