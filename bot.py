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
load_dotenv()  # <-- автоматически читает .env

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
            stop_pips = abs(entry_price - stop_loss) * (10000 if instrument_type == 'forex' else 100 if instrument_type == 'crypto' else 10)
            pip_value = FastRiskCalculator.calculate_pip_value_fast(instrument_type, currency_pair, 1.0)
            max_lots_risk = risk_amount / (stop_pips * pip_value) if stop_pips > 0 else 0
            max_lots_margin = (deposit * lev_value) / (CONTRACT_SIZES.get(instrument_type, 100000) * entry_price) if entry_price > 0 else 0
            position_size = max(0.01, min(max_lots_risk, max_lots_margin, 50.0))
            position_size = round(position_size, 2)
            required_margin = (position_size * CONTRACT_SIZES.get(instrument_type, 100000) * entry_price) / lev_value
            profit_pips = abs(take_profit - entry_price) * (10000 if instrument_type == 'forex' else 100 if instrument_type == 'crypto' else 10)
            potential_profit = profit_pips * pip_value * position_size
            potential_loss = stop_pips * pip_value * position_size
            return {
                'position_size': position_size,
                'risk_amount': risk_amount,
                'stop_pips': stop_pips,
                'take_profit_pips': profit_pips,
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
    def generate_multi_report(trades: List[Dict], deposit: float, leverage: str) -> str:
        total_risk = sum(t.get('risk_percent', 0) for t in trades) * 100
        corr = PortfolioAnalyzer.analyze_correlations(trades)
        vol = PortfolioAnalyzer.analyze_volatility(trades)
        metrics = PortfolioAnalyzer.calculate_metrics(trades)
        lines = [
            f"МУЛЬТИПОЗИЦИОННЫЙ ОТЧЕТ\n",
            f"Депозит: ${deposit:,.2f} | Плечо: {leverage}\n",
            f"Сделок: {len(trades)} | Общий риск: {total_risk:.2f}%\n\n",
            "КОРРЕЛЯЦИИ:\n" + "\n".join(corr[:3]) + ("\n..." if len(corr) > 3 else "") + "\n\n",
            "ВОЛАТИЛЬНОСТЬ:\n" + "\n".join(vol[:3]) + ("\n..." if len(vol) > 3 else "") + "\n\n",
            f"Баланс направлений: {metrics.get('direction_balance', 0):.2f} | Диверсификация: {metrics.get('diversity', 0):.1f}\n",
            "Рекомендация: Риск < 10%, RR > 1.5"
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
        f"Привет, {user.first_name}!\n\n"
        "PRO Калькулятор Управления Рисками v3.0\n\n"
        "МОИ ВОЗМОЖНОСТИ:\n"
        "• Многопозиционный расчет\n"
        "• Анализ корреляций и волатильности\n"
        "• Портфельные стратегии\n"
        "• Умные рекомендации\n\n"
        "Выберите:"
    )
    keyboard = [
        [InlineKeyboardButton("Профессиональный расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("Мой портфель", callback_data="portfolio")],
        [InlineKeyboardButton("Расширенная аналитика", callback_data="analytics")],
        [InlineKeyboardButton("PRO Инструкции", callback_data="pro_info")]
    ]
    await (update.message or update.callback_query.message).reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

@performance_logger
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query: return
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    if data == "pro_calculation":
        keyboard = [
            [InlineKeyboardButton("Одна сделка", callback_data="single_trade")],
            [InlineKeyboardButton("Мультипозиция", callback_data="multi_trade")],
            [InlineKeyboardButton("Назад", callback_data="main_menu")]
        ]
        await query.edit_message_text("Выберите тип расчета:", reply_markup=InlineKeyboardMarkup(keyboard))
    elif data == "multi_trade":
        context.user_data['multi_trades'] = []
        context.user_data['awaiting'] = 'multi_deposit'
        await query.edit_message_text("МУЛЬТИПОЗИЦИЯ\nВведите депозит (USD):")
    elif data == "analytics":
        p = user_data[user_id]['portfolio']
        trades = p['trades']
        if not trades:
            await query.edit_message_text("Портфель пуст.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="main_menu")]]))
            return
        corr = PortfolioAnalyzer.analyze_correlations(trades)
        vol = PortfolioAnalyzer.analyze_volatility(trades)
        metrics = PortfolioAnalyzer.calculate_metrics(trades)
        text = (
            f"РАСШИРЕННАЯ АНАЛИТИКА\n\n"
            f"КОРРЕЛЯЦИИ:\n" + "\n".join(corr) + "\n\n"
            f"ВОЛАТИЛЬНОСТЬ:\n" + "\n".join(vol) + "\n\n"
            f"Общий риск: {metrics.get('total_risk', 0)*100:.1f}%\n"
            f"Диверсификация: {metrics.get('diversity', 0):.1f}\n"
            f"Баланс направлений: {metrics.get('direction_balance', 0):.2f}\n\n"
            "Coming soon: AI-прогнозы, реал-тайм котировки"
        )
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="main_menu")]]))
    elif data == "pro_info":
        text = (
            "PRO ИНСТРУКЦИИ v3.0\n\n"
            "ДЛЯ ПРОФЕССИОНАЛЬНЫХ ТРЕЙДЕРОВ:\n"
            "• Рассчитывайте оптимальный размер позиции за секунды\n"
            "• Учитывайте корреляции и волатильность\n"
            "• Мгновенный пересчет при изменении параметров\n\n"
            "ПРОФЕССИОНАЛЬНАЯ АНАЛИТИКА:\n"
            "• Точный расчет для любого инструмента\n"
            "• Анализ риска в денежном и процентном выражении\n"
            "• Рекомендации по оптимизации размера позиции\n\n"
            "УПРАВЛЕНИЕ КАПИТАЛОМ:\n"
            "• Полный трекинг портфеля\n"
            "• Расчет ключевых метрик: Win Rate, Profit Factor\n\n"
            "СОВЕТЫ ПРОФЕССИОНАЛА:\n"
            "• Всегда используйте стоп-лосс\n"
            "• Диверсифицируйте портфель\n"
            "• Следите за соотношением риск/прибыль\n\n"
            "Разработчик: @fxfeelgood"
        )
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="main_menu")]]))
    elif data == "main_menu":
        await start_command(update, context)
    elif data.startswith("multi_lev_"):
        leverage = data.replace("multi_lev_", "")
        context.user_data['multi_leverage'] = leverage
        context.user_data['awaiting'] = 'multi_instrument'
        await query.edit_message_text(f"Плечо: {leverage}\nВведите инструмент (например EURUSD):")
    elif data == "multi_add_another":
        context.user_data['awaiting'] = 'multi_instrument'
        await query.edit_message_text("Добавьте еще инструмент:")
    elif data == "multi_calculate":
        trades = context.user_data.get('multi_trades', [])
        deposit = context.user_data.get('multi_deposit', 0)
        leverage = context.user_data.get('multi_leverage', '1:100')
        
        if not trades:
            await query.edit_message_text("Нет сделок для расчета")
            return
            
        PortfolioManager.add_multi_trades(user_id, trades, deposit, leverage)
        report = ReportGenerator.generate_multi_report(trades, deposit, leverage)
        bio = io.BytesIO(report.encode('utf-8'))
        bio.name = "multi_report.txt"
        await query.message.reply_document(document=InputFile(bio), caption="Мультиотчет")
        context.user_data.clear()
    else:
        await query.edit_message_text("Функция в разработке")

@performance_logger
async def generic_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text.strip()
    awaiting = context.user_data.get('awaiting')
    user_id = update.message.from_user.id

    # МУЛЬТИПОЗИЦИЯ
    if awaiting == 'multi_deposit':
        ok, val, msg = InputValidator.validate_number(text, 100)
        if not ok: 
            await update.message.reply_text(msg)
            return
        context.user_data['multi_deposit'] = val
        context.user_data['awaiting'] = None
        await update.message.reply_text(f"Депозит: ${val:,.2f}\nВыберите плечо:", 
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(l, callback_data=f"multi_lev_{l}")] for l in LEVERAGES]))
    
    elif awaiting == 'multi_instrument':
        ok, inst, msg = InputValidator.validate_instrument(text)
        if not ok: 
            await update.message.reply_text(msg)
            return
        context.user_data['multi_current'] = {'instrument': inst}
        context.user_data['awaiting'] = 'multi_direction'
        await update.message.reply_text("Направление: BUY / SELL")
    
    elif awaiting == 'multi_direction':
        d = text.upper()
        if d not in ("BUY", "SELL"): 
            await update.message.reply_text("BUY или SELL")
            return
        context.user_data['multi_current']['direction'] = d
        context.user_data['awaiting'] = 'multi_entry'
        await update.message.reply_text("Цена входа:")
    
    elif awaiting == 'multi_entry':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok: 
            await update.message.reply_text(msg)
            return
        context.user_data['multi_current']['entry_price'] = val
        context.user_data['awaiting'] = 'multi_sl'
        await update.message.reply_text("Стоп-лосс:")
    
    elif awaiting == 'multi_sl':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok: 
            await update.message.reply_text(msg)
            return
        context.user_data['multi_current']['stop_loss'] = val
        context.user_data['awaiting'] = 'multi_tp'
        await update.message.reply_text("Тейк-профит:")
    
    elif awaiting == 'multi_tp':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok: 
            await update.message.reply_text(msg)
            return
        context.user_data['multi_current']['take_profit'] = val
        context.user_data['awaiting'] = 'multi_risk'
        await update.message.reply_text("Риск в %:")
    
    elif awaiting == 'multi_risk':
        ok, val, msg = InputValidator.validate_number(text, 0.1, 20)
        if not ok: 
            await update.message.reply_text(msg)
            return
        
        trade = context.user_data['multi_current']
        trade['risk_percent'] = val / 100.0
        
        if 'multi_trades' not in context.user_data:
            context.user_data['multi_trades'] = []
            
        context.user_data['multi_trades'].append(trade.copy())
        
        keyboard = [
            [InlineKeyboardButton("Добавить еще", callback_data="multi_add_another")],
            [InlineKeyboardButton("Рассчитать", callback_data="multi_calculate")],
            [InlineKeyboardButton("Отмена", callback_data="main_menu")]
        ]
        await update.message.reply_text(
            f"Сделка добавлена: {trade['instrument']} {trade['direction']}\nВсего: {len(context.user_data['multi_trades'])}", 
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        context.user_data.pop('awaiting', None)
        context.user_data.pop('multi_current', None)

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
