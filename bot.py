# bot.py — PRO Risk Calculator v3.0 | Render + .env + orjson
import os
import logging
import asyncio
import time
import functools
import json
import io
from datetime import datetime
from typing import Dict, List, Any, Tuple
from aiohttp import web
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    CallbackQueryHandler
)

# --- Загрузка .env ---
from dotenv import load_dotenv
load_dotenv()

# --- Настройки ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found! Set it in .env or environment.")

PORT = int(os.getenv("PORT", 10000))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").rstrip("/")
WEBHOOK_PATH = f"/webhook/{TOKEN}"

# --- Логи ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("pro_risk_bot")

# ---------------------------
# Константы и справочники
# ---------------------------
INSTRUMENT_TYPES = {
    'forex': 'Форекс',
    'crypto': 'Криптовалюты',
    'indices': 'Индексы',
    'commodities': 'Сырьевые товары',
    'metals': 'Металлы'
}

INSTRUMENT_PRESETS = {
    'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'NZDUSD', 'EURGBP'],
    'crypto': ['BTCUSD', 'ETHUSD', 'XRPUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD'],
    'indices': ['US30', 'NAS100', 'SPX500', 'DAX40', 'FTSE100'],
    'commodities': ['OIL', 'NATGAS', 'COPPER', 'GOLD'],
    'metals': ['XAUUSD', 'XAGUSD', 'XPTUSD']
}

CORRELATION_MATRIX = {
    'EURUSD': {'GBPUSD': 0.8, 'USDJPY': -0.7, 'USDCAD': -0.8, 'AUDUSD': 0.6, 'XAUUSD': 0.3},
    'GBPUSD': {'EURUSD': 0.8, 'USDJPY': -0.6, 'USDCAD': -0.7, 'AUDUSD': 0.5, 'XAUUSD': 0.2},
    'USDJPY': {'EURUSD': -0.7, 'GBPUSD': -0.6, 'USDCAD': 0.9, 'AUDUSD': -0.5, 'XAUUSD': -0.4},
    'USDCAD': {'EURUSD': -0.8, 'GBPUSD': -0.7, 'USDJPY': 0.9, 'AUDUSD': -0.6, 'XAUUSD': -0.3},
    'AUDUSD': {'EURUSD': 0.6, 'GBPUSD': 0.5, 'USDJPY': -0.5, 'USDCAD': -0.6, 'XAUUSD': 0.4},
    'XAUUSD': {'EURUSD': 0.3, 'GBPUSD': 0.2, 'USDJPY': -0.4, 'USDCAD': -0.3, 'AUDUSD': 0.4}
}

VOLATILITY_DATA = {
    'EURUSD': 8.5, 'GBPUSD': 9.2, 'USDJPY': 7.8, 'USDCAD': 7.5,
    'AUDUSD': 10.1, 'NZDUSD': 9.8, 'EURGBP': 6.5,
    'BTCUSD': 65.2, 'ETHUSD': 70.5, 'XRPUSD': 85.3,
    'US30': 15.2, 'NAS100': 18.5, 'SPX500': 16.1,
    'XAUUSD': 14.5, 'XAGUSD': 25.3, 'OIL': 35.2
}

PIP_VALUES = {
    'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9, 'USDCHF': 10,
    'USDCAD': 10, 'AUDUSD': 10, 'NZDUSD': 10, 'EURGBP': 10,
    'EURJPY': 9, 'GBPJPY': 9, 'EURCHF': 10, 'AUDJPY': 9,
    'BTCUSD': 1, 'ETHUSD': 1, 'XRPUSD': 10, 'ADAUSD': 10,
    'DOTUSD': 1, 'LTCUSD': 1, 'BCHUSD': 1, 'LINKUSD': 1,
    'US30': 1, 'NAS100': 1, 'SPX500': 1, 'DAX40': 1,
    'FTSE100': 1, 'NIKKEI225': 1, 'ASX200': 1,
    'OIL': 10, 'NATGAS': 10, 'COPPER': 10,
    'XAUUSD': 10, 'XAGUSD': 50, 'XPTUSD': 10
}

CONTRACT_SIZES = {
    'forex': 100000,
    'crypto': 1,
    'indices': 1,
    'commodities': 100,
    'metals': 100
}

LEVERAGES = ['1:10', '1:20', '1:50', '1:100', '1:200', '1:500', '1:1000']
RISK_LEVELS = ['1%', '2%', '3%', '5%', '7%', '10%', '15%']
DATA_FILE = "user_data.json"

# ---------------------------
# DataManager
# ---------------------------
class DataManager:
    @staticmethod
    def load_data() -> Dict[int, Dict[str, Any]]:
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                return {int(k): v for k, v in raw.items()}
        except Exception as e:
            logger.error("Ошибка загрузки: %s", e)
        return {}

    @staticmethod
    def save_data(data: Dict[int, Dict[str, Any]]):
        try:
            serializable = {str(k): v for k, v in data.items()}
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error("Ошибка сохранения: %s", e)

user_data: Dict[int, Dict[str, Any]] = DataManager.load_data()

# ---------------------------
# FastCache
# ---------------------------
class FastCache:
    def __init__(self, max_size=200, ttl=300):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key):
        entry = self.cache.get(key)
        if not entry:
            return None
        value, ts = entry
        if time.time() - ts > self.ttl:
            del self.cache[key]
            return None
        return value

    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.items(), key=lambda kv: kv[1][1])[0]
            del self.cache[oldest]
        self.cache[key] = (value, time.time())

fast_cache = FastCache()

# ---------------------------
# InputValidator
# ---------------------------
class InputValidator:
    @staticmethod
    def validate_number(text: str, min_val: float = 0, max_val: float = None) -> Tuple[bool, float, str]:
        try:
            value = float(text.replace(',', '.'))
            if value < min_val:
                return False, value, f"Минимум: {min_val}"
            if max_val is not None and value > max_val:
                return False, value, f"Максимум: {max_val}"
            return True, value, "OK"
        except Exception:
            return False, 0.0, "Введите число"

    @staticmethod
    def validate_instrument(instr: str) -> Tuple[bool, str, str]:
        s = instr.upper().strip()
        if not s or len(s) > 20:
            return False, s, "Некорректный инструмент"
        return True, s, "OK"

    @staticmethod
    def validate_price(price: str) -> Tuple[bool, float, str]:
        return InputValidator.validate_number(price, 0.0000001, 1_000_000_000)

    @staticmethod
    def validate_percent(percent: str) -> Tuple[bool, float, str]:
        return InputValidator.validate_number(percent, 0.1, 100)

# ---------------------------
# PortfolioAnalyzer
# ---------------------------
class PortfolioAnalyzer:
    @staticmethod
    def analyze_correlations(trades: List[Dict]) -> List[str]:
        if len(trades) < 2:
            return ["Минимум 2 позиции для анализа"]
        res = []
        for i in range(len(trades)):
            for j in range(i + 1, len(trades)):
                a, b = trades[i], trades[j]
                inst1, dir1 = a['instrument'], a['direction']
                inst2, dir2 = b['instrument'], b['direction']
                corr = None
                if inst1 in CORRELATION_MATRIX and inst2 in CORRELATION_MATRIX[inst1]:
                    corr = CORRELATION_MATRIX[inst1][inst2]
                elif inst2 in CORRELATION_MATRIX and inst1 in CORRELATION_MATRIX[inst2]:
                    corr = CORRELATION_MATRIX[inst2][inst1]
                if corr is None or abs(corr) <= 0.7:
                    continue
                if dir1 == dir2:
                    res.append(f"Высокая корреляция ({corr:+.2f}) {inst1}/{inst2}")
                else:
                    res.append(f"Противоположные позиции с корр. ({corr:+.2f})")
        return res if res else ["Корреляции в норме"]

    @staticmethod
    def analyze_volatility(trades: List[Dict]) -> List[str]:
        out, high = [], 0
        for t in trades:
            vol = VOLATILITY_DATA.get(t['instrument'])
            if not vol: continue
            if vol > 20:
                out.append(f"ВЫСОКАЯ волатильность {t['instrument']}: {vol}%")
                high += 1
            elif vol > 10:
                out.append(f"Средняя волатильность {t['instrument']}: {vol}%")
        if high >= 3:
            out.append("ВНИМАНИЕ: Много высоковолатильных активов")
        return out or ["Волатильность под контролем"]

    @staticmethod
    def calculate_metrics(trades: List[Dict]) -> Dict:
        if not trades:
            return {}
        total_risk = sum(t.get('risk_percent', 0) for t in trades)
        avg_vol = sum(VOLATILITY_DATA.get(t['instrument'], 15) for t in trades) / len(trades)
        buys = sum(1 for t in trades if t.get('direction') == 'BUY')
        return {
            'total_risk': total_risk,
            'avg_volatility': avg_vol,
            'direction_balance': abs(buys - (len(trades) - buys)) / len(trades),
            'diversity': len(set(t['instrument'] for t in trades)) / 5.0
        }

# ---------------------------
# PortfolioManager
# ---------------------------
class PortfolioManager:
    @staticmethod
    def ensure_user(user_id: int):
        if user_id not in user_data:
            user_data[user_id] = {
                'portfolio': {
                    'initial_balance': 0.0,
                    'current_balance': 0.0,
                    'trades': [],
                    'performance': {k: 0.0 for k in ['total_trades', 'winning_trades', 'losing_trades', 'total_profit', 'total_loss', 'win_rate', 'average_profit', 'average_loss', 'profit_factor', 'max_drawdown']},
                    'allocation': {},
                    'history': [],
                    'settings': {'default_risk': 0.02, 'currency': 'USD', 'leverage': '1:100'},
                    'saved_strategies': [],
                    'multi_trade_mode': False
                }
            }
            DataManager.save_data(user_data)

    @staticmethod
    def add_trade(user_id: int, trade: Dict) -> int:
        PortfolioManager.ensure_user(user_id)
        trades = user_data[user_id]['portfolio']['trades']
        if len(trades) >= 50:
            raise ValueError("Лимит: 50 сделок")
        trade_id = len(trades) + 1
        trade.update({'id': trade_id, 'timestamp': datetime.now().isoformat(), 'status': 'open', 'profit': 0.0})
        trades.append(trade)
        inst = trade['instrument']
        alloc = user_data[user_id]['portfolio']['allocation']
        alloc[inst] = alloc.get(inst, 0) + 1
        DataManager.save_data(user_data)
        return trade_id

    @staticmethod
    def add_multi_trades(user_id: int, trades: List[Dict], deposit: float, leverage: str):
        for trade in trades:
            calc = FastRiskCalculator.calculate_position_size_fast(
                deposit=deposit, leverage=leverage, instrument_type='forex',
                currency_pair=trade['instrument'], entry_price=trade['entry_price'],
                stop_loss=trade['stop_loss'], take_profit=trade['take_profit'],
                direction=trade['direction'], risk_percent=trade['risk_percent']
            )
            trade['position_size'] = calc['position_size']
            PortfolioManager.add_trade(user_id, trade.copy())

# ---------------------------
# FastRiskCalculator
# ---------------------------
class FastRiskCalculator:
    @staticmethod
    def calculate_pip_value_fast(instrument_type: str, pair: str, lot_size: float) -> float:
        base = PIP_VALUES.get(pair, 10)
        return base * lot_size * (0.1 if instrument_type in ['crypto', 'commodities'] else 1.0)

    @staticmethod
    def calculate_position_size_fast(
        deposit: float, leverage: str, instrument_type: str, currency_pair: str,
        entry_price: float, stop_loss: float, take_profit: float,
        direction: str, risk_percent: float = 0.02
    ) -> Dict:
        try:
            lev_value = int(leverage.split(':')[1]) if ':' in leverage else 100
            risk_amount = deposit * risk_percent
            
            # Расчет стоп-лосса в пунктах
            if instrument_type == 'forex':
                stop_pips = abs(entry_price - stop_loss) * 10000
                take_profit_pips = abs(entry_price - take_profit) * 10000
            elif instrument_type == 'crypto':
                stop_pips = abs(entry_price - stop_loss) * 100
                take_profit_pips = abs(entry_price - take_profit) * 100
            else:
                stop_pips = abs(entry_price - stop_loss) * 10
                take_profit_pips = abs(entry_price - take_profit) * 10

            pip_value = FastRiskCalculator.calculate_pip_value_fast(instrument_type, currency_pair, 1.0)
            
            if stop_pips > 0 and pip_value > 0:
                max_lots_risk = risk_amount / (stop_pips * pip_value)
            else:
                max_lots_risk = 0
                
            max_lots_margin = (deposit * lev_value) / (CONTRACT_SIZES.get(instrument_type, 100000) * entry_price) if entry_price > 0 else 0
            
            position_size = max(0.01, min(max_lots_risk, max_lots_margin, 50.0))
            position_size = round(position_size, 2)
            
            required_margin = (position_size * CONTRACT_SIZES.get(instrument_type, 100000) * entry_price) / lev_value
            potential_profit = take_profit_pips * pip_value * position_size
            potential_loss = stop_pips * pip_value * position_size
            
            return {
                'position_size': position_size,
                'risk_amount': risk_amount,
                'stop_pips': stop_pips,
                'take_profit_pips': take_profit_pips,
                'potential_profit': potential_profit,
                'potential_loss': potential_loss,
                'reward_risk_ratio': potential_profit / risk_amount if risk_amount > 0 else 0,
                'required_margin': required_margin,
                'free_margin': deposit - required_margin,
                'risk_percent': risk_percent * 100
            }
        except Exception as e:
            logger.exception("Calc error: %s", e)
            return {'position_size': 0.01, 'risk_amount': 0, 'potential_profit': 0, 'reward_risk_ratio': 0}

# ---------------------------
# ReportGenerator
# ---------------------------
class ReportGenerator:
    @staticmethod
    def generate_single_trade_report(calculation_data: Dict, trade_data: Dict) -> str:
        report = f"""
🎯 *ОТЧЕТ ПО СДЕЛКЕ*
Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}

*ПАРАМЕТРЫ СДЕЛКИ:*
• Инструмент: {trade_data.get('instrument', 'N/A')}
• Направление: {trade_data.get('direction', 'N/A')}
• Депозит: ${trade_data.get('deposit', 0):,.2f}
• Плечо: {trade_data.get('leverage', 'N/A')}
• Уровень риска: {trade_data.get('risk_percent', 0)*100}%

*ЦЕНОВЫЕ УРОВНИ:*
• Цена входа: {trade_data.get('entry_price', 0)}
• Стоп-лосс: {trade_data.get('stop_loss', 0)}
• Тейк-профит: {trade_data.get('take_profit', 0)}
• Дистанция SL: {calculation_data.get('stop_pips', 0):.2f} пунктов
• Дистанция TP: {calculation_data.get('take_profit_pips', 0):.2f} пунктов

*РЕЗУЛЬТАТЫ РАСЧЕТА:*
• Размер позиции: {calculation_data.get('position_size', 0):.2f} лотов
• Сумма риска: ${calculation_data.get('risk_amount', 0):.2f}
• Потенциальная прибыль: ${calculation_data.get('potential_profit', 0):.2f}
• Соотношение прибыль/риск: {calculation_data.get('reward_risk_ratio', 0):.2f}
• Требуемая маржа: ${calculation_data.get('required_margin', 0):.2f}
• Свободная маржа: ${calculation_data.get('free_margin', 0):.2f}

*РЕКОМЕНДАЦИИ:*
{ReportGenerator.get_single_trade_recommendations(calculation_data)}
"""
        return report

    @staticmethod
    def get_single_trade_recommendations(calculation_data: Dict) -> str:
        recommendations = []
        rr_ratio = calculation_data.get('reward_risk_ratio', 0)
        
        if rr_ratio < 1:
            recommendations.append("• ❌ Соотношение риск/прибыль меньше 1 - reconsider your strategy")
        elif rr_ratio > 2:
            recommendations.append("• ✅ Отличное соотношение риск/прибыль!")
        else:
            recommendations.append("• ⚠️ Хорошее соотношение риск/прибыль")
        
        risk_percent = calculation_data.get('risk_percent', 0)
        if risk_percent > 5:
            recommendations.append("• ❌ Риск на сделку слишком высок (>5%)")
        elif risk_percent < 1:
            recommendations.append("• ℹ️ Риск на сделку очень низкий (<1%)")
        else:
            recommendations.append("• ✅ Уровень риска оптимальный")
        
        return "\n".join(recommendations)

    @staticmethod
    def generate_multi_report(trades: List[Dict], deposit: float, leverage: str) -> str:
        total_risk = sum(t.get('risk_percent', 0) for t in trades) * 100
        corr = PortfolioAnalyzer.analyze_correlations(trades)
        vol = PortfolioAnalyzer.analyze_volatility(trades)
        metrics = PortfolioAnalyzer.calculate_metrics(trades)
        
        lines = [
            f"*МУЛЬТИПОЗИЦИОННЫЙ ОТЧЕТ*\n",
            f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n",
            f"*ОБЩАЯ ИНФОРМАЦИЯ:*",
            f"• Депозит: ${deposit:,.2f}",
            f"• Плечо: {leverage}",
            f"• Сделок: {len(trades)}",
            f"• Общий риск: {total_risk:.2f}%\n\n",
            f"*КОРРЕЛЯЦИИ:*",
            *corr[:3],
            f"\n*ВОЛАТИЛЬНОСТЬ:*",
            *vol[:3],
            f"\n*МЕТРИКИ ПОРТФЕЛЯ:*",
            f"• Баланс направлений: {metrics.get('direction_balance', 0):.2f}",
            f"• Диверсификация: {metrics.get('diversity', 0):.1f}/5.0",
            f"• Средняя волатильность: {metrics.get('avg_volatility', 0):.1f}%\n\n",
            f"*PRO РЕКОМЕНДАЦИИ:*",
            f"• Поддерживайте общий риск < 10%",
            f"• Стремитесь к RR > 1.5",
            f"• Диверсифицируйте по инструментам",
            f"• Следите за корреляциями"
        ]
        return "\n".join(lines)

# ---------------------------
# UI / Handlers
# ---------------------------
def performance_logger(func):
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        start = time.time()
        try:
            return await func(update, context)
        finally:
            if time.time() - start > 1.0:
                logger.warning("Slow: %s", func.__name__)
    return wrapper

@performance_logger
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    PortfolioManager.ensure_user(user_id)
    
    text = (
        f"👋 *Привет, {user.first_name}!*\n\n"
        "🎯 *PRO Калькулятор Управления Рисками v3.0*\n\n"
        "⚡ *АКТИВИРОВАННЫЕ ВОЗМОЖНОСТИ:*\n"
        "• ✅ Профессиональный расчет позиций\n"
        "• ✅ Многопозиционный анализ\n"
        "• ✅ Анализ корреляций и волатильности\n"
        "• ✅ Умные рекомендации\n"
        "• ✅ Управление портфелем\n\n"
        "*Выберите опцию:*"
    )
    
    keyboard = [
        [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("💼 Мой портфель", callback_data="portfolio")],
        [InlineKeyboardButton("🔮 Будущие разработки", callback_data="coming_soon")],
        [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")]
    ]
    
    await (update.message or update.callback_query.message).reply_text(
        text, 
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

@performance_logger
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query: 
        return
        
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    if data == "pro_calculation":
        keyboard = [
            [InlineKeyboardButton("🎯 Одна сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade")],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
        ]
        await query.edit_message_text(
            "*Выберите тип расчета:*", 
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
    elif data == "single_trade":
        context.user_data['single_trade'] = {}
        context.user_data['awaiting'] = 'single_deposit'
        await query.edit_message_text(
            "*🎯 ОДНА СДЕЛКА*\n\nВведите размер депозита (USD):*", 
            parse_mode='Markdown'
        )
        
    elif data == "multi_trade":
        context.user_data['multi_trades'] = []
        context.user_data['awaiting'] = 'multi_deposit'
        await query.edit_message_text(
            "*📊 МУЛЬТИПОЗИЦИЯ*\n\nВведите депозит (USD):*", 
            parse_mode='Markdown'
        )
        
    elif data == "portfolio":
        p = user_data[user_id]['portfolio']
        trades = p['trades']
        
        if not trades:
            text = "*💼 ВАШ ПОРТФЕЛЬ*\n\n📭 Портфель пуст.\n\nИспользуйте расчеты для добавления сделок."
        else:
            total_trades = len(trades)
            open_trades = len([t for t in trades if t.get('status') == 'open'])
            total_profit = sum(t.get('profit', 0) for t in trades)
            
            text = (
                f"*💼 ВАШ ПОРТФЕЛЬ*\n\n"
                f"• Всего сделок: {total_trades}\n"
                f"• Открытых: {open_trades}\n"
                f"• Общая прибыль: ${total_profit:.2f}\n\n"
                f"Используйте расширенную аналитику для детального анализа."
            )
            
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]]
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        
    elif data == "coming_soon":
        text = """
🔮 *БУДУЩИЕ РАЗРАБОТКИ - COMING SOON*

🚀 *В РАЗРАБОТКЕ:*

🤖 *AI-АССИСТЕНТ*
• Прогнозирование движения цены на основе ML
• Интеллектуальные рекомендации по точкам входа/выхода
• Автоматическая оптимизация торговых стратегий

📈 *РЕАЛЬНЫЕ КОТИРОВКИ С БИРЖИ*
• Интеграция с Binance, Bybit, FTX API
• Автоматическое обновление котировок в реальном времени
• Price alerts и уведомления о достижении уровней

📊 *РАСШИРЕННАЯ АНАЛИТИКА ПОРТФЕЛЯ*
• Корреляция между активами
• Анализ волатильности и риска
• Оптимизация распределения капитала

🔄 *АВТОМАТИЧЕСКАЯ ТОРГОВЛЯ*
• Интеграция с торговыми API
• Исполнение сделок по сигналам
• Мониторинг и управление позициями в реальном времени

📱 *МОБИЛЬНОЕ ПРИЛОЖЕНИЕ*
• Push-уведомления на телефон
• Управление портфелем на ходу
• Полная функциональность в кармане

🔐 *ПОВЫШЕННАЯ БЕЗОПАСНОСТЬ*
• Двухфакторная аутентификация
• Шифрование данных
• Резервное копирование в облако

🌍 *МУЛЬТИВАЛЮТНАЯ ПОДДЕРЖКА*
• Поддержка всех основных валют
• Автоматическая конвертация
• Локализация для разных регионов

📚 *ОБУЧАЮЩИЕ МАТЕРИАЛЫ*
• Видео-уроки
• Торговые стратегии
• Анализ рынка и обзоры

*Следите за обновлениями! Новые функции появляются регулярно.*
"""
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]]
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        
    elif data == "pro_info":
        text = """
📚 *PRO ИНСТРУКЦИИ v3.0*

🎯 *ДЛЯ ПРОФЕССИОНАЛЬНЫХ ТРЕЙДЕРОВ:*

💡 *ИНТУИТИВНОЕ УПРАВЛЕНИЕ РИСКАМИ:*
• Рассчитывайте оптимальный размер позиции за секунды
• Автоматический учет типа инструмента (Форекс, крипто, индексы)
• Умное распределение объема по нескольким тейк-профитам
• Мгновенный пересчет при изменении параметров

📊 *ПРОФЕССИОНАЛЬНАЯ АНАЛИТИКА:*
• Точный расчет стоимости пипса для любого инструмента
• Учет маржинальных требований и плеча
• Анализ риска в денежном и процентном выражении
• Рекомендации по оптимизации размера позиции

💼 *УПРАВЛЕНИЕ КАПИТАЛОМ:*
• Полный трекинг торгового портфеля
• Анализ эффективности стратегий
• Расчет ключевых метрик: Win Rate, Profit Factor, просадки
• Интеллектуальные рекомендации по улучшению

⚡ *БЫСТРЫЕ РАСЧЕТЫ:*
• Мгновенные вычисления с кэшированием
• Валидация вводимых данных
• Автоматическое сохранение прогресса
• История всех расчетов и сделок

🔧 *КАК ИСПОЛЬЗОВАТЬ:*
1. *Профессиональный расчет* - полный цикл с настройкой всех параметров
2. *Быстрый расчет* - мгновенный расчет по основным параметрам  
3. *Портфель* - управление сделками и аналитика эффективности
4. *Настройки* - персонализация параметров по умолчанию

💾 *СОХРАНЕНИЕ ДАННЫХ:*
• Все ваши расчеты и сделки сохраняются автоматически
• Доступ к истории после перезапуска бота
• Экспорт отчетов для дальнейшего анализа

🚀 *СОВЕТЫ ПРОФЕССИОНАЛА:*
• Всегда используйте стоп-лосс для ограничения рисков
• Диверсифицируйте портфель по разным инструментам
• Следите за соотношением риск/прибыль не менее 1:2
• Регулярно анализируйте статистику для оптимизации стратегии

👨‍💻 *Разработчик для профессионалов:* @fxfeelgood

*PRO v3.0 | Умно • Быстро • Надежно* 🚀
"""
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]]
        await query.edit_message_text(
            text, 
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
    elif data == "main_menu":
        await start_command(update, context)
        
    elif data.startswith("single_lev_"):
        leverage = data.replace("single_lev_", "")
        context.user_data['single_trade']['leverage'] = leverage
        context.user_data['awaiting'] = 'single_instrument'
        await query.edit_message_text(
            f"*⚖️ Плечо: {leverage}*\n\nВведите инструмент (например EURUSD):*", 
            parse_mode='Markdown'
        )
        
    elif data.startswith("single_risk_"):
        risk_percent = float(data.replace("single_risk_", "")) / 100
        context.user_data['single_trade']['risk_percent'] = risk_percent
        context.user_data['awaiting'] = 'single_direction'
        await query.edit_message_text(
            f"*🎯 Уровень риска: {risk_percent*100}%*\n\nВыберите направление:*", 
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📈 BUY", callback_data="single_direction_BUY"),
                 InlineKeyboardButton("📉 SELL", callback_data="single_direction_SELL")]
            ])
        )
        
    elif data.startswith("single_direction_"):
        direction = data.replace("single_direction_", "")
        context.user_data['single_trade']['direction'] = direction
        context.user_data['awaiting'] = 'single_entry'
        await query.edit_message_text(
            f"*📊 Направление: {direction}*\n\nВведите цену входа:*", 
            parse_mode='Markdown'
        )
        
    elif data.startswith("multi_lev_"):
        leverage = data.replace("multi_lev_", "")
        context.user_data['multi_leverage'] = leverage
        context.user_data['awaiting'] = 'multi_instrument'
        await query.edit_message_text(
            f"*⚖️ Плечо: {leverage}*\n\nВведите инструмент (например EURUSD):*", 
            parse_mode='Markdown'
        )
        
    elif data == "multi_add_another":
        context.user_data['awaiting'] = 'multi_instrument'
        await query.edit_message_text("*➕ ДОБАВЛЕНИЕ СДЕЛКИ*\n\nВведите инструмент:*", parse_mode='Markdown')
        
    elif data == "multi_calculate":
        trades = context.user_data.get('multi_trades', [])
        deposit = context.user_data.get('multi_deposit', 0)
        leverage = context.user_data.get('multi_leverage', '1:100')
        
        if not trades:
            await query.edit_message_text("*❌ Нет сделок для расчета*", parse_mode='Markdown')
            return
            
        PortfolioManager.add_multi_trades(user_id, trades, deposit, leverage)
        report = ReportGenerator.generate_multi_report(trades, deposit, leverage)
        
        bio = io.BytesIO(report.encode('utf-8'))
        bio.name = "multi_report.txt"
        
        await query.message.reply_document(
            document=InputFile(bio), 
            caption="*📊 Мультипозиционный отчет готов!*",
            parse_mode='Markdown'
        )
        context.user_data.clear()
        
    elif data == "single_calculate":
        trade_data = context.user_data.get('single_trade', {})
        
        if not trade_data:
            await query.edit_message_text("*❌ Нет данных для расчета*", parse_mode='Markdown')
            return
            
        # Определяем тип инструмента автоматически
        instrument = trade_data.get('instrument', '').upper()
        instrument_type = 'forex'
        for inst_type, presets in INSTRUMENT_PRESETS.items():
            if instrument in presets:
                instrument_type = inst_type
                break
                
        calculation = FastRiskCalculator.calculate_position_size_fast(
            deposit=trade_data['deposit'],
            leverage=trade_data['leverage'],
            instrument_type=instrument_type,
            currency_pair=trade_data['instrument'],
            entry_price=trade_data['entry_price'],
            stop_loss=trade_data['stop_loss'],
            take_profit=trade_data['take_profit'],
            direction=trade_data['direction'],
            risk_percent=trade_data['risk_percent']
        )
        
        report = ReportGenerator.generate_single_trade_report(calculation, trade_data)
        
        bio = io.BytesIO(report.encode('utf-8'))
        bio.name = "single_trade_report.txt"
        
        await query.message.reply_document(
            document=InputFile(bio),
            caption="*🎯 Отчет по сделке готов!*",
            parse_mode='Markdown'
        )
        
        # Добавляем сделку в портфель
        trade_data.update({
            'position_size': calculation['position_size'],
            'potential_profit': calculation['potential_profit'],
            'potential_loss': calculation['potential_loss']
        })
        PortfolioManager.add_trade(user_id, trade_data)
        
        context.user_data.clear()
        
    else:
        await query.edit_message_text("*⚙️ Функция в разработке*", parse_mode='Markdown')

@performance_logger
async def generic_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: 
        return
        
    text = update.message.text.strip()
    awaiting = context.user_data.get('awaiting')
    user_id = update.message.from_user.id

    # ОДНА СДЕЛКА
    if awaiting == 'single_deposit':
        ok, val, msg = InputValidator.validate_number(text, 100)
        if not ok: 
            await update.message.reply_text(f"*❌ {msg}*", parse_mode='Markdown')
            return
            
        context.user_data['single_trade']['deposit'] = val
        context.user_data['awaiting'] = None
        
        await update.message.reply_text(
            f"*💰 Депозит: ${val:,.2f}*\n\n*Выберите уровень риска:*", 
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🟢 1%", callback_data="single_risk_1"),
                 InlineKeyboardButton("🟡 2%", callback_data="single_risk_2")],
                [InlineKeyboardButton("🟠 3%", callback_data="single_risk_3"),
                 InlineKeyboardButton("🔴 5%", callback_data="single_risk_5")],
                [InlineKeyboardButton("⚫ 10%", callback_data="single_risk_10")]
            ])
        )
    
    elif awaiting == 'single_instrument':
        ok, inst, msg = InputValidator.validate_instrument(text)
        if not ok: 
            await update.message.reply_text(f"*❌ {msg}*", parse_mode='Markdown')
            return
            
        context.user_data['single_trade']['instrument'] = inst
        context.user_data['awaiting'] = None
        
        await update.message.reply_text(
            f"*💰 Депозит: ${context.user_data['single_trade']['deposit']:,.2f}*\n"
            f"*🎯 Риск: {context.user_data['single_trade']['risk_percent']*100}%*\n"
            f"*📊 Инструмент: {inst}*\n\n"
            f"*Выберите плечо:*", 
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton(l, callback_data=f"single_lev_{l}") for l in LEVERAGES[:3]],
                [InlineKeyboardButton(l, callback_data=f"single_lev_{l}") for l in LEVERAGES[3:6]],
                [InlineKeyboardButton(LEVERAGES[6], callback_data=f"single_lev_{LEVERAGES[6]}")]
            ])
        )
    
    elif awaiting == 'single_entry':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok: 
            await update.message.reply_text(f"*❌ {msg}*", parse_mode='Markdown')
            return
            
        context.user_data['single_trade']['entry_price'] = val
        context.user_data['awaiting'] = 'single_stop_loss'
        
        direction = context.user_data['single_trade']['direction']
        direction_text = "выше" if direction == "BUY" else "ниже"
        
        await update.message.reply_text(
            f"*💎 Цена входа: {val}*\n\n"
            f"*🛑 Введите цену стоп-лосса ({direction_text} цены входа):*", 
            parse_mode='Markdown'
        )
    
    elif awaiting == 'single_stop_loss':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok: 
            await update.message.reply_text(f"*❌ {msg}*", parse_mode='Markdown')
            return
            
        context.user_data['single_trade']['stop_loss'] = val
        context.user_data['awaiting'] = 'single_take_profit'
        
        await update.message.reply_text(
            f"*🛑 Стоп-лосс: {val}*\n\n*🎯 Введите цену тейк-профита:*", 
            parse_mode='Markdown'
        )
    
    elif awaiting == 'single_take_profit':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok: 
            await update.message.reply_text(f"*❌ {msg}*", parse_mode='Markdown')
            return
            
        context.user_data['single_trade']['take_profit'] = val
        context.user_data['awaiting'] = None
        
        trade_data = context.user_data['single_trade']
        
        summary = (
            f"*📋 СВОДКА СДЕЛКИ:*\n\n"
            f"• 📊 Инструмент: {trade_data['instrument']}\n"
            f"• 📈 Направление: {trade_data['direction']}\n"
            f"• 💰 Депозит: ${trade_data['deposit']:,.2f}\n"
            f"• ⚖️ Плечо: {trade_data['leverage']}\n"
            f"• 🎯 Риск: {trade_data['risk_percent']*100}%\n"
            f"• 💎 Вход: {trade_data['entry_price']}\n"
            f"• 🛑 SL: {trade_data['stop_loss']}\n"
            f"• 🎯 TP: {trade_data['take_profit']}\n\n"
            f"*Готовы рассчитать?*"
        )
        
        keyboard = [
            [InlineKeyboardButton("✅ Рассчитать", callback_data="single_calculate")],
            [InlineKeyboardButton("🔄 Начать заново", callback_data="single_trade")]
        ]
        
        await update.message.reply_text(
            summary, 
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # МУЛЬТИПОЗИЦИЯ
    elif awaiting == 'multi_deposit':
        ok, val, msg = InputValidator.validate_number(text, 100)
        if not ok: 
            await update.message.reply_text(f"*❌ {msg}*", parse_mode='Markdown')
            return
            
        context.user_data['multi_deposit'] = val
        context.user_data['awaiting'] = None
        
        await update.message.reply_text(
            f"*💰 Депозит: ${val:,.2f}*\n\n*Выберите плечо:*", 
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton(l, callback_data=f"multi_lev_{l}") for l in LEVERAGES[:3]],
                [InlineKeyboardButton(l, callback_data=f"multi_lev_{l}") for l in LEVERAGES[3:6]],
                [InlineKeyboardButton(LEVERAGES[6], callback_data=f"multi_lev_{LEVERAGES[6]}")]
            ])
        )
    
    elif awaiting == 'multi_instrument':
        ok, inst, msg = InputValidator.validate_instrument(text)
        if not ok: 
            await update.message.reply_text(f"*❌ {msg}*", parse_mode='Markdown')
            return
            
        context.user_data['multi_current'] = {'instrument': inst}
        context.user_data['awaiting'] = 'multi_direction'
        
        await update.message.reply_text(
            "*Выберите направление:*", 
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📈 BUY", callback_data="multi_direction_BUY"),
                 InlineKeyboardButton("📉 SELL", callback_data="multi_direction_SELL")]
            ])
        )
    
    elif awaiting == 'multi_entry':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok: 
            await update.message.reply_text(f"*❌ {msg}*", parse_mode='Markdown')
            return
            
        context.user_data['multi_current']['entry_price'] = val
        context.user_data['awaiting'] = 'multi_stop_loss'
        
        direction = context.user_data['multi_current']['direction']
        direction_text = "выше" if direction == "BUY" else "ниже"
        
        await update.message.reply_text(
            f"*💎 Цена входа: {val}*\n\n"
            f"*🛑 Введите цену стоп-лосса ({direction_text} цены входа):*", 
            parse_mode='Markdown'
        )
    
    elif awaiting == 'multi_stop_loss':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok: 
            await update.message.reply_text(f"*❌ {msg}*", parse_mode='Markdown')
            return
            
        context.user_data['multi_current']['stop_loss'] = val
        context.user_data['awaiting'] = 'multi_take_profit'
        
        await update.message.reply_text(
            f"*🛑 Стоп-лосс: {val}*\n\n*🎯 Введите цену тейк-профита:*", 
            parse_mode='Markdown'
        )
    
    elif awaiting == 'multi_take_profit':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok: 
            await update.message.reply_text(f"*❌ {msg}*", parse_mode='Markdown')
            return
            
        context.user_data['multi_current']['take_profit'] = val
        context.user_data['awaiting'] = 'multi_risk'
        
        await update.message.reply_text(
            f"*🎯 Тейк-профит: {val}*\n\n*📊 Введите риск в %:*", 
            parse_mode='Markdown'
        )
    
    elif awaiting == 'multi_risk':
        ok, val, msg = InputValidator.validate_percent(text)
        if not ok: 
            await update.message.reply_text(f"*❌ {msg}*", parse_mode='Markdown')
            return
            
        trade = context.user_data['multi_current']
        trade['risk_percent'] = val / 100.0
        
        if 'multi_trades' not in context.user_data:
            context.user_data['multi_trades'] = []
            
        context.user_data['multi_trades'].append(trade.copy())
        
        keyboard = [
            [InlineKeyboardButton("➕ Добавить еще", callback_data="multi_add_another")],
            [InlineKeyboardButton("🧮 Рассчитать", callback_data="multi_calculate")],
            [InlineKeyboardButton("❌ Отмена", callback_data="main_menu")]
        ]
        
        await update.message.reply_text(
            f"*✅ Сделка добавлена: {trade['instrument']} {trade['direction']}*\n"
            f"*📊 Всего сделок: {len(context.user_data['multi_trades'])}*", 
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        context.user_data.pop('awaiting', None)
        context.user_data.pop('multi_current', None)

# Обработка выбора направления для мультипозиции
@performance_logger
async def handle_multi_direction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    direction = query.data.replace("multi_direction_", "")
    context.user_data['multi_current']['direction'] = direction
    context.user_data['awaiting'] = 'multi_entry'
    
    await query.edit_message_text(
        f"*📊 Направление: {direction}*\n\n*💎 Введите цену входа:*", 
        parse_mode='Markdown'
    )

# ---------------------------
# Webhook & Main
# ---------------------------
async def set_webhook(application):
    """Установка вебхука"""
    try:
        webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
        await application.bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook установлен: {webhook_url}")
        return True
    except Exception as e:
        logger.error(f"Ошибка установки вебхука: {e}")
        return False

async def start_http_server(application):
    """Запуск HTTP сервера"""
    app = web.Application()
    
    async def handle_webhook(request):
        """Обработчик вебхука"""
        try:
            data = await request.json()
            update = Update.de_json(data, application.bot)
            await application.process_update(update)
            return web.Response(status=200)
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return web.Response(status=400)
    
    app.router.add_post(WEBHOOK_PATH, handle_webhook)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    
    logger.info(f"HTTP сервер запущен на порту {PORT}")
    return runner

async def main():
    """Основная функция"""
    application = Application.builder().token(TOKEN).build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(callback_router))
    application.add_handler(CallbackQueryHandler(handle_multi_direction, pattern="^multi_direction_"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generic_text_handler))

    # Режим запуска
    if WEBHOOK_URL and WEBHOOK_URL.strip():
        logger.info("Запуск в режиме WEBHOOK")
        await application.initialize()
        
        if await set_webhook(application):
            await start_http_server(application)
            logger.info("Бот запущен в режиме WEBHOOK")
            await asyncio.Event().wait()  # Бесконечное ожидание
        else:
            logger.error("Не удалось установить вебхук, запуск в режиме polling")
            await application.run_polling()
    else:
        logger.info("Запуск в режиме POLLING")
        await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
