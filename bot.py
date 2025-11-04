import os
import logging
import asyncio
import time
import functools
import json
import io
import math
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
---------------------------
Настройки и логирование
---------------------------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
raise ValueError("TELEGRAM_BOT_TOKEN не найден! Добавь в переменные окружения.")
PORT = int(os.getenv("PORT", 5000))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_PATH = f"/webhook/{TOKEN}"
logging.basicConfig(
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
level=logging.INFO
)
logger = logging.getLogger("pro_risk_bot")
---------------------------
Константы и справочники
---------------------------
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
TRADE_DIRECTIONS = ['BUY', 'SELL']
CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']
DATA_FILE = "user_data.json"
---------------------------
Сохранение/загрузка данных
---------------------------
class DataManager:
@staticmethod
def load_data() -> Dict[int, Dict[str, Any]]:
try:
if os.path.exists(DATA_FILE):
with open(DATA_FILE, 'r', encoding='utf-8') as f:
raw = json.load(f)
# Ensure keys are ints
return {int(k): v for k, v in raw.items()}
return {}
except Exception as e:
logger.error("Ошибка загрузки данных: %s", e)
return {}
@staticmethod
def save_data(data: Dict[int, Dict[str, Any]]):
    try:
        # convert keys to str for JSON
        serializable = {str(k): v for k, v in data.items()}
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
        logger.debug("Данные сохранены")
    except Exception as e:
        logger.error("Ошибка сохранения данных: %s", e)

user_data: Dict[int, Dict[str, Any]] = DataManager.load_data()
---------------------------
Быстрый кеш
---------------------------
class FastCache:
def init(self, max_size=200, ttl=300):
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
        # remove oldest
        oldest = sorted(self.cache.items(), key=lambda kv: kv[1][1])[0][0]
        del self.cache[oldest]
    self.cache[key] = (value, time.time())

fast_cache = FastCache()
---------------------------
Проверки ввода
---------------------------
class InputValidator:
@staticmethod
def validate_number(text: str, min_val: float = 0, max_val: float = None) -> Tuple[bool, float, str]:
try:
value = float(text.replace(',', '.'))
if value < min_val:
return False, value, f"❌ Значение не может быть меньше {min_val}"
if max_val is not None and value > max_val:
return False, value, f"❌ Значение не может быть больше {max_val}"
return True, value, "✅ Корректное значение"
except Exception:
return False, 0.0, "❌ Введите корректное числовое значение"
@staticmethod
def validate_instrument(instr: str) -> Tuple[bool, str, str]:
    s = instr.upper().strip()
    if not s:
        return False, s, "❌ Введите название инструмента"
    if len(s) > 20:
        return False, s, "❌ Название инструмента слишком длинное"
    return True, s, "✅ OK"

@staticmethod
def validate_price(price: str) -> Tuple[bool, float, str]:
    return InputValidator.validate_number(price, 0.0000001, 1_000_000_000)

@staticmethod
def validate_percent(percent: str) -> Tuple[bool, float, str]:
    return InputValidator.validate_number(percent, 0.01, 100.0)

---------------------------
Аналитика портфеля
---------------------------
class PortfolioAnalyzer:
@staticmethod
def analyze_correlations(trades: List[Dict]) -> List[str]:
if len(trades) < 2:
return ["ℹ️ Для анализа корреляций нужно минимум 2 позиции"]
res = []
for i in range(len(trades)):
for j in range(i + 1, len(trades)):
a = trades[i]
b = trades[j]
inst1, dir1 = a['instrument'], a['direction']
inst2, dir2 = b['instrument'], b['direction']
corr = None
if inst1 in CORRELATION_MATRIX and inst2 in CORRELATION_MATRIX[inst1]:
corr = CORRELATION_MATRIX[inst1][inst2]
elif inst2 in CORRELATION_MATRIX and inst1 in CORRELATION_MATRIX[inst2]:
corr = CORRELATION_MATRIX[inst2][inst1]
if corr is None:
continue
if abs(corr) > 0.7:
if dir1 == dir2:
if corr > 0:
res.append(f"⚠️ Высокая позитивная корреляция ({corr:.2f}) между {inst1} и {inst2} — позиции идут в одну сторону")
else:
res.append(f"🔄 Высокая негативная корреляция ({corr:.2f}) между {inst1} и {inst2} — возможное хеджирование")
else:
if corr > 0:
res.append(f"⚡ Кореллированные противоположные позиции ({corr:.2f}) — следите за риском")
else:
res.append(f"🎯 Противонаправленные позиции с негативной корреляцией ({corr:.2f})")
return res if res else ["✅ Корреляционный риск под контролем"]
@staticmethod
def analyze_volatility(trades: List[Dict]) -> List[str]:
    out = []
    high = 0
    for t in trades:
        inst = t['instrument']
        vol = VOLATILITY_DATA.get(inst)
        if vol is None:
            continue
        if vol > 20:
            out.append(f"⚡ Высокая волатильность {inst}: {vol}%")
            high += 1
        elif vol > 10:
            out.append(f"📊 Средняя волатильность {inst}: {vol}%")
        else:
            out.append(f"✅ Низкая волатильность {inst}: {vol}%")
    if high >= 3:
        out.append("🚨 ВНИМАНИЕ: много высоковолатильных инструментов")
    return out

@staticmethod
def calculate_portfolio_metrics(trades: List[Dict]) -> Dict[str, float]:
    if not trades:
        return {}
    total_risk = sum(t.get('risk_percent', 0) for t in trades)
    avg_vol = sum(VOLATILITY_DATA.get(t['instrument'], 15) for t in trades) / len(trades)
    buys = sum(1 for t in trades if t.get('direction') == 'BUY')
    sells = len(trades) - buys
    return {
        'total_risk': total_risk,
        'avg_volatility': avg_vol,
        'diversity_score': min(len(trades) / 5.0, 1.0),
        'direction_balance': abs(buys - sells) / len(trades)
    }

---------------------------
Менеджер портфеля
---------------------------
class PortfolioManager:
@staticmethod
def ensure_user(user_id: int):
if user_id not in user_data:
user_data[user_id] = {}
if 'portfolio' not in user_data[user_id]:
user_data[user_id]['portfolio'] = {
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
'max_drawdown': 0.0,
'sharpe_ratio': 0.0
},
'allocation': {},
'history': [],
'settings': {
'default_risk': 0.02,
'currency': 'USD',
'leverage': '1:100'
},
'saved_strategies': [],
'multi_trade_mode': False
}
DataManager.save_data(user_data)
@staticmethod
def add_trade(user_id: int, trade: Dict[str, Any]) -> int:
    PortfolioManager.ensure_user(user_id)
    trades = user_data[user_id]['portfolio']['trades']
    if len(trades) >= 50:
        raise ValueError("Лимит сделок достигнут (50). Удалите старые.")
    trade_id = len(trades) + 1
    trade['id'] = trade_id
    trade['timestamp'] = datetime.now().isoformat()
    trades.append(trade)
    # update allocation
    alloc = user_data[user_id]['portfolio']['allocation']
    inst = trade.get('instrument', 'Unknown')
    alloc[inst] = alloc.get(inst, 0) + 1
    # update balance/history if closed
    if trade.get('status') == 'closed':
        profit = float(trade.get('profit', 0.0))
        user_data[user_id]['portfolio']['current_balance'] += profit
        user_data[user_id]['portfolio']['history'].append({
            'type': 'trade',
            'action': 'close',
            'instrument': inst,
            'profit': profit,
            'timestamp': trade['timestamp']
        })
    DataManager.save_data(user_data)
    PortfolioManager.recalculate_performance(user_id)
    return trade_id

@staticmethod
def recalculate_performance(user_id: int):
    PortfolioManager.ensure_user(user_id)
    p = user_data[user_id]['portfolio']
    closed = [t for t in p['trades'] if t.get('status') == 'closed']
    if not closed:
        DataManager.save_data(user_data)
        return
    winners = [t for t in closed if t.get('profit', 0) > 0]
    losers = [t for t in closed if t.get('profit', 0) < 0]
    p['performance']['total_trades'] = len(closed)
    p['performance']['winning_trades'] = len(winners)
    p['performance']['losing_trades'] = len(losers)
    p['performance']['total_profit'] = sum(t.get('profit', 0) for t in winners)
    p['performance']['total_loss'] = abs(sum(t.get('profit', 0) for t in losers))
    if len(closed) > 0:
        p['performance']['win_rate'] = (len(winners) / len(closed)) * 100.0
    p['performance']['average_profit'] = (p['performance']['total_profit'] / len(winners)) if winners else 0.0
    p['performance']['average_loss'] = (p['performance']['total_loss'] / len(losers)) if losers else 0.0
    if p['performance']['total_loss'] > 0:
        p['performance']['profit_factor'] = p['performance']['total_profit'] / p['performance']['total_loss']
    else:
        p['performance']['profit_factor'] = float('inf') if p['performance']['total_profit'] > 0 else 0.0
    # max drawdown calculation from history
    balance_history = []
    running = p['initial_balance']
    for ev in sorted(p['history'], key=lambda x: x['timestamp']):
        if ev['type'] == 'balance':
            if ev['action'] == 'deposit':
                running += ev.get('amount', 0)
            elif ev['action'] == 'withdrawal':
                running -= ev.get('amount', 0)
        elif ev['type'] == 'trade' and ev['action'] == 'close':
            running += ev.get('profit', 0)
        balance_history.append(running)
    max_drawdown = 0.0
    if balance_history:
        peak = balance_history[0]
        for b in balance_history:
            if b > peak:
                peak = b
            drawdown = (peak - b) / peak * 100 if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    p['performance']['max_drawdown'] = max_drawdown
    DataManager.save_data(user_data)

@staticmethod
def add_balance_operation(user_id: int, op_type: str, amount: float, description: str = ""):
    PortfolioManager.ensure_user(user_id)
    p = user_data[user_id]['portfolio']
    p['history'].append({
        'type': 'balance',
        'action': op_type,
        'amount': amount,
        'description': description,
        'timestamp': datetime.now().isoformat()
    })
    if op_type == 'deposit':
        p['current_balance'] += amount
        if p['initial_balance'] == 0:
            p['initial_balance'] = amount
    elif op_type == 'withdrawal':
        if p['current_balance'] >= amount:
            p['current_balance'] -= amount
        else:
            raise ValueError("Недостаточно средств")
    DataManager.save_data(user_data)
    PortfolioManager.recalculate_performance(user_id)

@staticmethod
def save_strategy(user_id: int, strategy: Dict[str, Any]) -> int:
    PortfolioManager.ensure_user(user_id)
    lst = user_data[user_id]['portfolio']['saved_strategies']
    sid = len(lst) + 1
    strategy['id'] = sid
    strategy['created_at'] = datetime.now().isoformat()
    lst.append(strategy)
    DataManager.save_data(user_data)
    return sid

---------------------------
Быстрый калькулятор рисков
---------------------------
class FastRiskCalculator:
@staticmethod
def calculate_pip_value_fast(instrument_type: str, pair: str, lot_size: float) -> float:
base = PIP_VALUES.get(pair, 10)
if instrument_type == 'crypto':
return base * lot_size * 0.1
elif instrument_type == 'indices':
return base * lot_size * 0.01
elif instrument_type == 'commodities':
return base * lot_size * 0.1
else:
return base * lot_size
@staticmethod
def calculate_position_size_fast(
    deposit: float,
    leverage: str,
    instrument_type: str,
    currency_pair: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
    risk_percent: float = 0.02
) -> Dict[str, Any]:
    try:
        cache_key = f"pos_{deposit}_{leverage}_{instrument_type}_{currency_pair}_{entry_price}_{stop_loss}_{take_profit}_{direction}_{risk_percent}"
        cached = fast_cache.get(cache_key)
        if cached:
            return cached

        lev_value = int(leverage.split(':')[1]) if ':' in leverage else int(leverage)
        risk_amount = deposit * risk_percent

        # pips calc
        if instrument_type == 'forex':
            stop_pips = abs(entry_price - stop_loss) * 10000
            take_pips = abs(entry_price - take_profit) * 10000
        elif instrument_type == 'crypto':
            stop_pips = abs(entry_price - stop_loss) * 100
            take_pips = abs(entry_price - take_profit) * 100
        else:
            stop_pips = abs(entry_price - stop_loss) * 10
            take_pips = abs(entry_price - take_profit) * 10

        pip_value = FastRiskCalculator.calculate_pip_value_fast(instrument_type, currency_pair, 1.0)
        max_lots_by_risk = (risk_amount / (stop_pips * pip_value)) if stop_pips > 0 and pip_value > 0 else 0.0

        contract_size = CONTRACT_SIZES.get(instrument_type, 100000)
        max_lots_by_margin = (deposit * lev_value) / (contract_size * entry_price) if entry_price > 0 else 0.0

        position_size = min(max_lots_by_risk, max_lots_by_margin, 50.0)
        position_size = max(0.01, round(position_size, 2))

        required_margin = (position_size * contract_size * entry_price) / lev_value if lev_value > 0 else 0.0

        if direction == 'BUY':
            potential_profit = (take_profit - entry_price) * pip_value * position_size
            potential_loss = (entry_price - stop_loss) * pip_value * position_size
        else:
            potential_profit = (entry_price - take_profit) * pip_value * position_size
            potential_loss = (stop_loss - entry_price) * pip_value * position_size

        potential_loss = abs(potential_loss)
        potential_profit = max(0.0, potential_profit)
        reward_risk_ratio = (potential_profit / risk_amount) if risk_amount > 0 else 0.0

        res = {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'stop_pips': stop_pips,
            'take_profit_pips': take_pips,
            'potential_profit': potential_profit,
            'potential_loss': potential_loss,
            'reward_risk_ratio': reward_risk_ratio,
            'required_margin': required_margin,
            'risk_percent': (risk_amount / deposit) * 100 if deposit > 0 else 0.0,
            'free_margin': deposit - required_margin,
            'is_profitable': potential_profit > 0.0
        }
        fast_cache.set(cache_key, res)
        return res
    except Exception as e:
        logger.exception("Ошибка расчета позиции: %s", e)
        return {
            'position_size': 0.01,
            'risk_amount': 0.0,
            'stop_pips': 0.0,
            'take_profit_pips': 0.0,
            'potential_profit': 0.0,
            'potential_loss': 0.0,
            'reward_risk_ratio': 0.0,
            'required_margin': 0.0,
            'risk_percent': 0.0,
            'free_margin': deposit,
            'is_profitable': False
        }

---------------------------
Генератор отчетов
---------------------------
class ReportGenerator:
@staticmethod
def generate_calculation_report(calc: Dict[str, Any], context: Dict[str, Any]) -> str:
try:
instrument = context.get('instrument', 'N/A')
direction = context.get('direction', 'N/A')
deposit = context.get('deposit', 0.0)
risk_level = context.get('risk_percent', 0.0)
entry = context.get('entry_price', 0.0)
sl = context.get('stop_loss', 0.0)
tp = context.get('take_profit', 0.0)
return (
f"ОТЧЕТ О РАСЧЕТЕ ПОЗИЦИИ\nДата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n"
f"ПАРАМЕТРЫ СДЕЛКИ:\n• Инструмент: {instrument}\n• Направление: {direction}\n• Депозит: ${deposit:,.2f}\n• Плечо: {context.get('leverage', 'N/A')}\n• Уровень риска: {risk_level*100:.2f}%\n\n"
f"ЦЕНОВЫЕ УРОВНИ:\n• Цена входа: {entry}\n• Стоп-лосс: {sl}\n• Тейк-профит: {tp}\n• Дистанция SL: {calc.get('stop_pips', 0):.2f} пунктов\n• Дистанция TP: {calc.get('take_profit_pips', 0):.2f} пунктов\n\n"
f"РЕЗУЛЬТАТЫ РАСЧЕТА:\n• Размер позиции: {calc.get('position_size', 0):.4f} лотов\n• Сумма риска: ${calc.get('risk_amount', 0):.2f}\n• Потенциальная прибыль: ${calc.get('potential_profit', 0):.2f}\n• Потенциальный убыток: ${calc.get('potential_loss', 0):.2f}\n• Соотношение прибыль/риск: {calc.get('reward_risk_ratio', 0):.2f}\n• Требуемая маржа: ${calc.get('required_margin', 0):.2f}\n• Свободная маржа: ${calc.get('free_margin', 0):.2f}\n\n"
f"РЕКОМЕНДАЦИИ:\n{ReportGenerator.get_professional_recommendations(calc, context)}\n"
)
except Exception as e:
logger.exception("Ошибка генерации отчета: %s", e)
return "Ошибка при генерации отчета"
@staticmethod
def get_professional_recommendations(calc: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    recs = []
    rr = calc.get('reward_risk_ratio', 0.0)
    risk_percent = calc.get('risk_percent', 0.0)
    deposit = ctx.get('deposit', 0.0)
    is_profitable = calc.get('is_profitable', True)
    required_margin = calc.get('required_margin', 0.0)
    margin_usage = (required_margin / deposit * 100.0) if deposit > 0 else 0.0

    if not is_profitable:
        recs.append("🔴 Убыточная сделка (TP/SL неверно настроены).")
    if rr < 1:
        recs.append("🔴 Соотношение прибыль/риск < 1 — пересмотрите TP/SL.")
    elif rr < 2:
        recs.append("🟡 Соотношение 1-2 — можно улучшить TP или уменьшить SL.")
    else:
        recs.append("🟢 Хорошее соотношение риск/прибыль (>2).")

    if risk_percent > 0.05:
        recs.append("🔴 Риск >5% — высокий риск, рекомендуется 1-3%.")
    elif risk_percent < 0.01:
        recs.append("🟡 Риск <1% — консервативно, можно увеличить до 1-3% при уверенности.")
    else:
        recs.append("🟢 Риск в пределах норм (1-5%).")

    if margin_usage > 50:
        recs.append("🔴 Загруженность маржи >50% — уменьшите размер позиции.")
    elif margin_usage > 30:
        recs.append("🟡 Маржа 30-50% — следите за свободной маржей.")
    else:
        recs.append("🟢 Низкая загрузка маржи — есть запас для других позиций.")

    if is_profitable and rr >= 1.5 and risk_percent <= 0.03 and margin_usage <= 40:
        recs.append("🚀 ИДЕАЛЬНЫЕ ПАРАМЕТРЫ: можно рассмотреть масштабирование позиции.")
    if not is_profitable or rr < 1 or risk_percent > 0.05:
        recs.append("⚡ ВНИМАНИЕ: пересмотрите параметры сделки.")

    return "\n".join(recs)

@staticmethod
def generate_portfolio_report(user_id: int) -> str:
    try:
        PortfolioManager.ensure_user(user_id)
        p = user_data[user_id]['portfolio']
        perf = p['performance']
        trades = p['trades']
        total_return = ((p['current_balance'] - p['initial_balance']) / p['initial_balance'] * 100) if p['initial_balance'] > 0 else 0.0
        report = [
            f"ОТЧЕТ ПО ПОРТФЕЛЮ",
            f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            "",
            f"Баланс:",
            f"• Начальный депозит: ${p['initial_balance']:,.2f}",
            f"• Текущий баланс: ${p['current_balance']:,.2f}",
            f"• Общая прибыль/убыток: ${p['current_balance'] - p['initial_balance']:,.2f}",
            f"• Доходность: {total_return:.2f}%",
            "",
            "Статистика:",
            f"• Всего сделок: {perf.get('total_trades', 0)}",
            f"• Прибыльные: {perf.get('winning_trades', 0)}",
            f"• Убыточные: {perf.get('losing_trades', 0)}",
            f"• Win rate: {perf.get('win_rate', 0.0):.1f}%",
            f"• Profit factor: {perf.get('profit_factor', 0.0):.2f}",
            f"• Макс. просадка: {perf.get('max_drawdown', 0.0):.1f}%",
            f"• Средняя прибыль: ${perf.get('average_profit', 0.0):.2f}",
            f"• Средний убыток: ${perf.get('average_loss', 0.0):.2f}",
            "",
            "Распределение по инструментам:"
        ]
        alloc = p.get('allocation', {})
        total_trades = len(trades)
        for inst, cnt in alloc.items():
            pct = (cnt / total_trades * 100) if total_trades else 0.0
            report.append(f"• {inst}: {cnt} сделок ({pct:.1f}%)")
        report.append("")
        if trades:
            report.append("Основные выводы:")
            corr = PortfolioAnalyzer.analyze_correlations(trades)
            vol = PortfolioAnalyzer.analyze_volatility(trades)
            metrics = PortfolioAnalyzer.calculate_portfolio_metrics(trades)
            report.extend(["• " + s for s in corr[:5]])
            report.extend(["• " + s for s in vol[:5]])
            if metrics:
                report.append(f"• Общий риск: {metrics['total_risk']:.1f}%")
                report.append(f"• Диверсификация: {metrics['diversity_score']*100:.0f}%")
        return "\n".join(report)
    except Exception as e:
        logger.exception("Ошибка генерации отчета портфеля: %s", e)
        return "Ошибка при генерации отчета портфеля"

---------------------------
UI / Handlers
---------------------------
def performance_logger(func):
@functools.wraps(func)
async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
start = time.time()
try:
return await func(update, context)
finally:
dt = time.time() - start
if dt > 1.0:
logger.warning("Slow handler %s: %.2fs", func.name, dt)
return wrapper
@performance_logger
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
try:
user = update.effective_user
user_name = user.first_name if user else "Трейдер"
PortfolioManager.ensure_user(user.id if user else 0)
welcome_text = (
f"👋 Привет, {user_name}!\n\n"
"🎯 PRO Калькулятор Управления Рисками v3.0\n\n"
"⚡ МОИ ВОЗМОЖНОСТИ:\n"
"• ✅ Многопозиционный расчет\n"
"• ✅ Анализ корреляций между активами\n"
"• ✅ Учет волатильности инструментов\n"
"• ✅ Портфельные стратегии и рекомендации\n"
"• ✅ Статистика за 5 лет по основным парам\n"
"• ✅ Умные рекомендации для PRO трейдеров\n\n"
"Выберите опцию:"
)
keyboard = [
[InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
[InlineKeyboardButton("💼 Мой портфель", callback_data="portfolio")],
[InlineKeyboardButton("🔮 Расширенная аналитика", callback_data="analytics")],
[InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")]
]
await update.message.reply_text(welcome_text, reply_markup=InlineKeyboardMarkup(keyboard))
except Exception as e:
logger.exception("Ошибка в start_command: %s", e)
await update.message.reply_text("Произошла ошибка. Попробуйте позже.")
@performance_logger
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
try:
query = update.callback_query
if not query:
return
await query.answer()
data = query.data
    # Main menu entry points
    if data == "pro_calculation":
        await start_pro_calculation(query, context)
        return
    if data == "portfolio":
        await show_portfolio_menu(query, context)
        return
    if data == "analytics":
        await query.edit_message_text("🔮 Раздел аналитики: генерируем краткий отчет...", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))
        return
    if data == "pro_info":
        await query.edit_message_text("📚 PRO инструкции:\n\n1) Используйте профессиональные расчеты для точного управления риском.\n2) Сохраняйте стратегии.\n3) Следите за корреляциями.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))
        return
    if data == "main_menu":
        # emulate /start
        await start_command(update, context)
        return

    # PRO flow internal callbacks are delegated to the specific function
    # Many callbacks are handled inside handle_main_menu equivalent
    await handle_main_menu_callbacks(query, context)
except Exception as e:
    logger.exception("Ошибка в callback_router: %s", e)
    await handle_error_generic(update, context, e)

@performance_logger
async def start_pro_calculation(query, context):
keyboard = [
[InlineKeyboardButton("📈 Одна сделка", callback_data="single_trade")],
[InlineKeyboardButton("🗂️ Мультипозиционный расчет (группа)", callback_data="multi_trade")],
[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
]
await query.edit_message_text("📊 ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ\nВыберите тип расчета:", reply_markup=InlineKeyboardMarkup(keyboard))
@performance_logger
async def show_portfolio_menu(query, context):
user_id = query.from_user.id
PortfolioManager.ensure_user(user_id)
p = user_data[user_id]['portfolio']
text = (
f"💼 PRO ПОРТФЕЛЬ v3.0\n\n"
f"💰 Баланс: ${p['current_balance']:,.2f}\n"
f"📊 Сделки: {len(p['trades'])}\n"
f"🎯 Win Rate: {p['performance'].get('win_rate', 0.0):.1f}%\n\n"
"Выберите опцию:"
)
keyboard = [
[InlineKeyboardButton("📈 Обзор сделок", callback_data="portfolio_trades")],
[InlineKeyboardButton("💰 Баланс и распределение", callback_data="portfolio_balance")],
[InlineKeyboardButton("📊 Анализ эффективности", callback_data="portfolio_performance")],
[InlineKeyboardButton("📄 Сгенерировать отчет", callback_data="portfolio_report")],
[InlineKeyboardButton("💾 Выгрузить отчет", callback_data="export_portfolio")],
[InlineKeyboardButton("➕ Добавить сделку", callback_data="portfolio_add_trade")],
[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
]
await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
Helper for many callback actions
@performance_logger
async def handle_main_menu_callbacks(query, context):
data = query.data
user_id = query.from_user.id
# single trade flow
if data == "single_trade":
    keyboard = []
    for key, label in INSTRUMENT_TYPES.items():
        keyboard.append([InlineKeyboardButton(label, callback_data=f"pro_type_{key}")])
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="pro_calculation")])
    await query.edit_message_text("📊 Одна сделка — выберите тип инструмента:", reply_markup=InlineKeyboardMarkup(keyboard))
    return

# choose instrument type -> list presets
if data.startswith("pro_type_"):
    instrument_type = data.replace("pro_type_", "")
    context.user_data['instrument_type'] = instrument_type
    presets = INSTRUMENT_PRESETS.get(instrument_type, [])
    keyboard = [[InlineKeyboardButton(p, callback_data=f"pro_preset_{p}")] for p in presets]
    keyboard.append([InlineKeyboardButton("✏️ Ввести свой инструмент", callback_data="pro_custom")])
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="single_trade")])
    await query.edit_message_text(f"📊 {INSTRUMENT_TYPES.get(instrument_type, instrument_type)} — выберите инструмент:", reply_markup=InlineKeyboardMarkup(keyboard))
    return

# preset selected or custom
if data.startswith("pro_preset_") or data == "pro_custom":
    if data == "pro_custom":
        # ask user to type instrument
        context.user_data['awaiting'] = 'custom_instrument'
        await query.edit_message_text("✏️ Введите название инструмента (пример: EURUSD, BTCUSD):")
        return
    instrument = data.replace("pro_preset_", "")
    context.user_data['instrument'] = instrument
    keyboard = [
        [InlineKeyboardButton("📈 BUY", callback_data="BUY"), InlineKeyboardButton("📉 SELL", callback_data="SELL")],
        [InlineKeyboardButton("🔙 Назад", callback_data=f"pro_type_{context.user_data.get('instrument_type', 'forex')}")]
    ]
    await query.edit_message_text(f"🎯 Инструмент: {instrument}\nВыберите направление:", reply_markup=InlineKeyboardMarkup(keyboard))
    return

# direction chosen
if data in ("BUY", "SELL"):
    context.user_data['direction'] = data
    keyboard = [[InlineKeyboardButton(r, callback_data=f"pro_risk_{r.replace('%','')}")] for r in RISK_LEVELS]
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data=f"pro_preset_{context.user_data.get('instrument','')}")])
    await query.edit_message_text(f"🎯 {context.user_data.get('instrument','')} | {data}\nВыберите уровень риска (% от депозита):", reply_markup=InlineKeyboardMarkup(keyboard))
    return

# risk chosen
if data.startswith("pro_risk_"):
    val = float(data.replace("pro_risk_", "")) / 100.0
    context.user_data['risk_percent'] = val
    context.user_data['awaiting'] = 'deposit'
    await query.edit_message_text(f"💰 Уровень риска установлен: {val*100:.2f}%\n\n💵 Введите размер депозита (в USD):")
    return

# leverage
if data.startswith("pro_leverage_"):
    lev = data.replace("pro_leverage_", "")
    context.user_data['leverage'] = lev
    context.user_data['awaiting'] = 'entry_price'
    await query.edit_message_text(f"⚖️ Плечо: {lev}\n\n💎 Введите цену входа:")
    return

# options in result menu
if data == "export_calculation":
    # export last calculation of user
    cd = context.user_data.get('last_calculation')
    meta = context.user_data.get('calculation_meta')
    if not cd or not meta:
        await query.edit_message_text("❌ Нет данных для экспорта.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))
        return
    txt = ReportGenerator.generate_calculation_report(cd, meta)
    bio = io.BytesIO(txt.encode('utf-8'))
    bio.name = "calculation_report.txt"
    await query.edit_message_document(document=InputFile(bio), caption="Отчет расчета")
    return

if data == "save_trade_from_pro":
    # save last calculation as trade in portfolio
    calc = context.user_data.get('last_calculation')
    meta = context.user_data.get('calculation_meta')
    if not calc or not meta:
        await query.edit_message_text("❌ Нет данных для сохранения.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))
        return
    # create trade object
    trade = {
        'instrument': meta.get('instrument'),
        'direction': meta.get('direction'),
        'entry_price': meta.get('entry_price'),
        'stop_loss': meta.get('stop_loss'),
        'take_profit': meta.get('take_profit'),
        'position_size': calc.get('position_size'),
        'risk_percent': meta.get('risk_percent'),
        'status': 'open',
        'profit': 0.0
    }
    trade_id = PortfolioManager.add_trade(query.from_user.id, trade)
    await query.edit_message_text(f"✅ Сделка сохранена в портфеле (ID: {trade_id})", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))
    return

# portfolio actions
if data == "portfolio_trades":
    PortfolioManager.ensure_user(user_id)
    trades = user_data[user_id]['portfolio'].get('trades', [])
    if not trades:
        await query.edit_message_text("📈 Сделок ещё нет.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]]))
        return
    text_lines = ["📈 Обзор сделок:"]
    for t in trades[-10:]:
        text_lines.append(f"ID {t.get('id')} | {t.get('instrument')} {t.get('direction')} | Size: {t.get('position_size')} | Status: {t.get('status')}")
    await query.edit_message_text("\n".join(text_lines), reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]]))
    return

if data == "portfolio_balance":
    PortfolioManager.ensure_user(user_id)
    p = user_data[user_id]['portfolio']
    await query.edit_message_text(f"💰 Баланс: ${p['current_balance']:,.2f}\nНачальный депозит: ${p['initial_balance']:,.2f}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]]))
    return

if data == "portfolio_performance":
    PortfolioManager.ensure_user(user_id)
    text = ReportGenerator.generate_portfolio_report(user_id)
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("💾 Выгрузить", callback_data="export_portfolio")],[InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]]))
    return

if data == "portfolio_report":
    text = ReportGenerator.generate_portfolio_report(user_id)
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("💾 Выгрузить", callback_data="export_portfolio")],[InlineKeyboardButton("🔙 Назад", callback_data="portfolio")]]))
    return

if data == "export_portfolio":
    txt = ReportGenerator.generate_portfolio_report(user_id)
    bio = io.BytesIO(txt.encode('utf-8'))
    bio.name = "portfolio_report.txt"
    await query.edit_message_document(document=InputFile(bio), caption="Отчет по портфелю")
    return

if data == "portfolio_add_trade":
    context.user_data['awaiting'] = 'portfolio_new_instrument'
    await query.edit_message_text("➕ Добавление сделки: введите инструмент (например EURUSD):")
    return

# fallback
await query.edit_message_text("❌ Неизвестное действие. Вернитесь в главное меню.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))

Generic message handler for text inputs — routes based on context.user_data['awaiting']
@performance_logger
async def generic_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
try:
text = update.message.text.strip()
awaiting = context.user_data.get('awaiting')
if awaiting == 'custom_instrument':
ok, inst, msg = InputValidator.validate_instrument(text)
if not ok:
await update.message.reply_text(msg + "\nВведите инструмент заново:")
return
context.user_data['instrument'] = inst
context.user_data.pop('awaiting', None)
keyboard = [
[InlineKeyboardButton("📈 BUY", callback_data="BUY"), InlineKeyboardButton("📉 SELL", callback_data="SELL")],
[InlineKeyboardButton("🔙 Назад", callback_data=f"pro_type_{context.user_data.get('instrument_type','forex')}")]
]
await update.message.reply_text(f"🎯 Инструмент: {inst}\nВыберите направление:", reply_markup=InlineKeyboardMarkup(keyboard))
return
    if awaiting == 'deposit':
        ok, val, msg = InputValidator.validate_number(text, 1, 1_000_000_000)
        if not ok:
            await update.message.reply_text(msg + "\nВведите депозит:")
            return
        context.user_data['deposit'] = val
        context.user_data.pop('awaiting', None)
        keyboard = [[InlineKeyboardButton(l, callback_data=f"pro_leverage_{l}")] for l in LEVERAGES]
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data=f"pro_risk_{int(context.user_data.get('risk_percent',0)*100)}")])
        await update.message.reply_text(f"💰 Депозит: ${val:,.2f}\nВыберите кредитное плечо:", reply_markup=InlineKeyboardMarkup(keyboard))
        return

    if awaiting == 'entry_price':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok:
            await update.message.reply_text(msg + "\nВведите цену входа:")
            return
        context.user_data['entry_price'] = val
        context.user_data.pop('awaiting', None)
        # Ask for stop loss
        dir_text = "ниже" if context.user_data.get('direction') == 'BUY' else "выше"
        context.user_data['awaiting'] = 'stop_loss'
        await update.message.reply_text(f"💎 Цена входа: {val}\n🛑 Введите цену стоп-лосса ({dir_text} цены входа):")
        return

    if awaiting == 'stop_loss':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok:
            await update.message.reply_text(msg + "\nВведите стоп-лосс:")
            return
        context.user_data['stop_loss'] = val
        context.user_data.pop('awaiting', None)
        context.user_data['awaiting'] = 'take_profit'
        await update.message.reply_text(f"🛑 Стоп-лосс: {val}\n🎯 Введите цену тейк-профита:")
        return

    if awaiting == 'take_profit':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok:
            await update.message.reply_text(msg + "\nВведите тейк-профит:")
            return
        context.user_data['take_profit'] = val
        context.user_data.pop('awaiting', None)
        # perform calculation
        calc = FastRiskCalculator.calculate_position_size_fast(
            deposit=context.user_data.get('deposit', 0.0),
            leverage=context.user_data.get('leverage', '1:100'),
            instrument_type=context.user_data.get('instrument_type', 'forex'),
            currency_pair=context.user_data.get('instrument', 'EURUSD'),
            entry_price=context.user_data.get('entry_price', 0.0),
            stop_loss=context.user_data.get('stop_loss', 0.0),
            take_profit=context.user_data.get('take_profit', 0.0),
            direction=context.user_data.get('direction', 'BUY'),
            risk_percent=context.user_data.get('risk_percent', 0.02)
        )
        context.user_data['last_calculation'] = calc
        context.user_data['calculation_meta'] = {
            'instrument': context.user_data.get('instrument'),
            'direction': context.user_data.get('direction'),
            'deposit': context.user_data.get('deposit'),
            'leverage': context.user_data.get('leverage'),
            'risk_percent': context.user_data.get('risk_percent'),
            'entry_price': context.user_data.get('entry_price'),
            'stop_loss': context.user_data.get('stop_loss'),
            'take_profit': context.user_data.get('take_profit')
        }
        # render result
        is_prof = calc.get('is_profitable', True)
        status_emoji = "🟢" if is_prof else "🔴"
        status_text = "ПРИБЫЛЬНАЯ" if is_prof else "УБЫТОЧНАЯ"
        result_text = (
            f"🎯 РЕЗУЛЬТАТЫ ПРОФЕССИОНАЛЬНОГО РАСЧЕТА\n{status_emoji} СТАТУС: {status_text}\n\n"
            f"Параметры: {context.user_data.get('instrument')} | {context.user_data.get('direction')}\n"
            f"Депозит: ${context.user_data.get('deposit'):,.2f} | Плечо: {context.user_data.get('leverage')}\n"
            f"Риск: {context.user_data.get('risk_percent')*100:.2f}%\n\n"
            f"Вход: {context.user_data.get('entry_price')}\n"
            f"SL: {context.user_data.get('stop_loss')} ({calc.get('stop_pips'):.2f} пунктов)\n"
            f"TP: {context.user_data.get('take_profit')} ({calc.get('take_profit_pips'):.2f} пунктов)\n\n"
            f"Размер позиции: {calc.get('position_size'):.4f} лотов\n"
            f"Сумма риска: ${calc.get('risk_amount'):.2f}\n"
            f"Потенц. прибыль: ${calc.get('potential_profit'):.2f}\n"
            f"Потенц. убыток: ${calc.get('potential_loss'):.2f}\n"
            f"Соотношение P/R: {calc.get('reward_risk_ratio'):.2f}\n"
            f"Требуемая маржа: ${calc.get('required_margin'):.2f}\n"
            f"Свободная маржа: ${calc.get('free_margin'):.2f}\n\n"
            f"Рекомендации:\n{ReportGenerator.get_professional_recommendations(calc, context.user_data)}"
        )
        keyboard = [
            [InlineKeyboardButton("💾 Выгрузить расчет", callback_data="export_calculation")],
            [InlineKeyboardButton("💼 Сохранить сделку", callback_data="save_trade_from_pro")],
            [InlineKeyboardButton("📊 Новый расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ]
        await update.message.reply_text(result_text, reply_markup=InlineKeyboardMarkup(keyboard))
        return

    # adding trade manually to portfolio
    if awaiting == 'portfolio_new_instrument':
        ok, inst, msg = InputValidator.validate_instrument(text)
        if not ok:
            await update.message.reply_text(msg)
            return
        context.user_data['new_trade'] = {'instrument': inst}
        context.user_data['awaiting'] = 'portfolio_new_direction'
        await update.message.reply_text("Введите направление сделки: BUY или SELL")
        return

    if awaiting == 'portfolio_new_direction':
        d = text.upper()
        if d not in ("BUY", "SELL"):
            await update.message.reply_text("Введите BUY или SELL")
            return
        context.user_data['new_trade']['direction'] = d
        context.user_data['awaiting'] = 'portfolio_new_entry'
        await update.message.reply_text("Введите цену входа:")
        return

    if awaiting == 'portfolio_new_entry':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok:
            await update.message.reply_text(msg)
            return
        context.user_data['new_trade']['entry_price'] = val
        context.user_data['awaiting'] = 'portfolio_new_sl'
        await update.message.reply_text("Введите цену стоп-лосса:")
        return

    if awaiting == 'portfolio_new_sl':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok:
            await update.message.reply_text(msg)
            return
        context.user_data['new_trade']['stop_loss'] = val
        context.user_data['awaiting'] = 'portfolio_new_tp'
        await update.message.reply_text("Введите тейк-профит:")
        return

    if awaiting == 'portfolio_new_tp':
        ok, val, msg = InputValidator.validate_price(text)
        if not ok:
            await update.message.reply_text(msg)
            return
        context.user_data['new_trade']['take_profit'] = val
        context.user_data['awaiting'] = 'portfolio_new_risk'
        await update.message.reply_text("Введите риск в процентах (например 1.5):")
        return

    if awaiting == 'portfolio_new_risk':
        ok, val, msg = InputValidator.validate_number(text, 0.01, 100.0)
        if not ok:
            await update.message.reply_text(msg)
            return
        ctx_trade = context.user_data['new_trade']
        ctx_trade['risk_percent'] = val / 100.0
        ctx_trade['position_size'] = None
        ctx_trade['status'] = 'open'
        # calculate estimated position size using default leverage and deposit managed in portfolio
        p = user_data.get(update.message.from_user.id, {}).get('portfolio', {})
        deposit = p.get('current_balance') or p.get('initial_balance') or 1000.0
        calc = FastRiskCalculator.calculate_position_size_fast(
            deposit=deposit,
            leverage=p.get('settings', {}).get('leverage', '1:100'),
            instrument_type='forex',
            currency_pair=ctx_trade['instrument'],
            entry_price=ctx_trade['entry_price'],
            stop_loss=ctx_trade['stop_loss'],
            take_profit=ctx_trade['take_profit'],
            direction=ctx_trade['direction'],
            risk_percent=ctx_trade['risk_percent']
        )
        ctx_trade['position_size'] = calc.get('position_size')
        try:
            tid = PortfolioManager.add_trade(update.message.from_user.id, ctx_trade)
            await update.message.reply_text(f"✅ Сделка добавлена в портфель (ID: {tid})", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка при добавлении сделки: {e}")
        finally:
            context.user_data.pop('new_trade', None)
            context.user_data.pop('awaiting', None)
        return

    # default fallback
    await update.message.reply_text("ℹ️ Не распознано. Используйте меню /start или клавиатуру.")
except Exception as e:
    logger.exception("Ошибка в generic_text_handler: %s", e)
    await update.message.reply_text("Произошла ошибка при обработке сообщения.")

@performance_logger
async def handle_error_generic(update: Update, context: ContextTypes.DEFAULT_TYPE, exc: Exception):
logger.exception("Unhandled error: %s", exc)
try:
if update and update.callback_query:
await update.callback_query.edit_message_text("❌ Произошла непредвиденная ошибка.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]]))
elif update and update.message:
await update.message.reply_text("❌ Произошла непредвиденная ошибка.")
except Exception:
logger.exception("Ошибка при отправке сообщения об ошибке")
---------------------------
HTTP Server (Webhook support) for Render
---------------------------
async def health_check(request):
return web.Response(text="OK")
async def handle_webhook(request, application: Application):
try:
data = await request.json()
update = Update.de_json(data, application.bot)
await application.process_update(update)
return web.Response(status=200)
except Exception as e:
logger.exception("Ошибка обработки вебхука: %s", e)
return web.Response(status=500)
async def start_http_server(application: Application):
app = web.Application()
app.router.add_get('/', health_check)
app.router.add_get('/health', health_check)
app.router.add_post(WEBHOOK_PATH, lambda request: handle_webhook(request, application))
runner = web.AppRunner(app)
await runner.setup()
site = web.TCPSite(runner, '0.0.0.0', PORT)
await site.start()
logger.info("HTTP сервер запущен на порту %s", PORT)
return runner
async def set_webhook(application: Application) -> bool:
if not WEBHOOK_URL:
logger.warning("WEBHOOK_URL не установлен, перейдем в polling режим")
return False
try:
webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
await application.bot.set_webhook(url=webhook_url, drop_pending_updates=True)
logger.info("Webhook установлен: %s", webhook_url)
return True
except Exception as e:
logger.exception("Ошибка установки webhook: %s", e)
return False
---------------------------
Запуск бота
---------------------------
async def main():
application = Application.builder().token(TOKEN).build()
# Handlers
application.add_handler(CommandHandler("start", start_command))
application.add_handler(CallbackQueryHandler(callback_router))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generic_text_handler))

# start webhook / polling depending on environment
if WEBHOOK_URL:
    ok = await set_webhook(application)
    if ok:
        runner = await start_http_server(application)
        logger.info("Bot running in webhook mode.")
        # keep running until cancelled
        await asyncio.Event().wait()
        await runner.cleanup()
        return
    else:
        logger.warning("Не удалось установить webhook. Переходим в polling.")
# polling fallback
logger.info("Запуск бота в режиме polling")
await application.initialize()
await application.start()
await application.updater.start_polling()
try:
    await asyncio.Event().wait()
finally:
    await application.updater.stop()
    await application.stop()
    await application.shutdown()

if name == "main":
try:
asyncio.run(main())
except (KeyboardInterrupt, SystemExit):
logger.info("Завершение работы бота...")
