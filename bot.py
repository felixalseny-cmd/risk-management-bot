# bot.py
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

# ---------------------------
# Настройки и логирование
# ---------------------------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found! Set it in environment variables.")
PORT = int(os.getenv("PORT", 5000))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").rstrip("/")
WEBHOOK_PATH = f"/webhook/{TOKEN}"

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
TRADE_DIRECTIONS = ['BUY', 'SELL']
CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']
DATA_FILE = "user_data.json"

# ---------------------------
# DataManager: сохранение/загрузка
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
            logger.error("Ошибка загрузки данных: %s", e)
        return {}

    @staticmethod
    def save_data(data: Dict[int, Dict[str, Any]]):
        try:
            serializable = {str(k): v for k, v in data.items()}
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error("Ошибка сохранения данных: %s", e)

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
                return False, value, f"Значение не может быть меньше {min_val}"
            if max_val is not None and value > max_val:
                return False, value, f"Значение не может быть больше {max_val}"
            return True, value, "OK"
        except Exception:
            return False, 0.0, "Введите корректное числовое значение"

    @staticmethod
    def validate_instrument(instr: str) -> Tuple[bool, str, str]:
        s = instr.upper().strip()
        if not s:
            return False, s, "Введите название инструмента"
        if len(s) > 20:
            return False, s, "Название инструмента слишком длинное"
        return True, s, "OK"

    @staticmethod
    def validate_price(price: str) -> Tuple[bool, float, str]:
        return InputValidator.validate_number(price, 0.0000001, 1_000_000_000)

    @staticmethod
    def validate_percent(percent: str) -> Tuple[bool, float, str]:
        return InputValidator.validate_number(percent, 0.01, 100.0)

# ---------------------------
# PortfolioAnalyzer
# ---------------------------
class PortfolioAnalyzer:
    @staticmethod
    def analyze_correlations(trades: List[Dict]) -> List[str]:
        if len(trades) < 2:
            return ["Для анализа корреляций нужно минимум 2 позиции"]
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
                if corr is None:
                    continue
                if abs(corr) > 0.7:
                    if dir1 == dir2:
                        res.append(f"Высокая {'позитивная' if corr > 0 else 'негативная'} корреляция ({corr:.2f}) между {inst1} и {inst2}")
                    else:
                        res.append(f"Противоположные позиции с высокой корреляцией ({corr:.2f})")
        return res if res else ["Корреляционный риск под контролем"]

    @staticmethod
    def analyze_volatility(trades: List[Dict]) -> List[str]:
        out, high = [], 0
        for t in trades:
            inst = t['instrument']
            vol = VOLATILITY_DATA.get(inst)
            if not vol:
                continue
            if vol > 20:
                out.append(f"Высокая волатильность {inst}: {vol}%")
                high += 1
            elif vol > 10:
                out.append(f"Средняя волатильность {inst}: {vol}%")
            else:
                out.append(f"Низкая волатильность {inst}: {vol}%")
        if high >= 3:
            out.append("ВНИМАНИЕ: много высоковолатильных инструментов")
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
                    'performance': {
                        'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                        'total_profit': 0.0, 'total_loss': 0.0, 'win_rate': 0.0,
                        'average_profit': 0.0, 'average_loss': 0.0,
                        'profit_factor': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0
                    },
                    'allocation': {},
                    'history': [],
                    'settings': {'default_risk': 0.02, 'currency': 'USD', 'leverage': '1:100'},
                    'saved_strategies': [],
                    'multi_trade_mode': False
                }
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
        alloc = user_data[user_id]['portfolio']['allocation']
        inst = trade.get('instrument', 'Unknown')
        alloc[inst] = alloc.get(inst, 0) + 1
        if trade.get('status') == 'closed':
            profit = float(trade.get('profit', 0.0))
            user_data[user_id]['portfolio']['current_balance'] += profit
            user_data[user_id]['portfolio']['history'].append({
                'type': 'trade', 'action': 'close', 'instrument': inst,
                'profit': profit, 'timestamp': trade['timestamp']
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

        balance_history = []
        running = p['initial_balance']
        for ev in sorted(p['history'], key=lambda x: x['timestamp']):
            if ev['type'] == 'balance':
                running += ev.get('amount', 0) if ev['action'] == 'deposit' else -ev.get('amount', 0)
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
                max_drawdown = max(max_drawdown, drawdown)
        p['performance']['max_drawdown'] = max_drawdown
        DataManager.save_data(user_data)

    @staticmethod
    def add_balance_operation(user_id: int, op_type: str, amount: float, description: str = ""):
        PortfolioManager.ensure_user(user_id)
        p = user_data[user_id]['portfolio']
        p['history'].append({
            'type': 'balance', 'action': op_type, 'amount': amount,
            'description': description, 'timestamp': datetime.now().isoformat()
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

# ---------------------------
# FastRiskCalculator
# ---------------------------
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
        deposit: float, leverage: str, instrument_type: str, currency_pair: str,
        entry_price: float, stop_loss: float, take_profit: float,
        direction: str, risk_percent: float = 0.02
    ) -> Dict[str, Any]:
        try:
            cache_key = f"pos_{deposit}_{leverage}_{instrument_type}_{currency_pair}_{entry_price}_{stop_loss}_{take_profit}_{direction}_{risk_percent}"
            cached = fast_cache.get(cache_key)
            if cached:
                return cached

            lev_value = int(leverage.split(':')[1]) if ':' in leverage else int(leverage)
            risk_amount = deposit * risk_percent

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
                'position_size': 0.01, 'risk_amount': 0.0, 'stop_pips': 0.0,
                'take_profit_pips': 0.0, 'potential_profit': 0.0, 'potential_loss': 0.0,
                'reward_risk_ratio': 0.0, 'required_margin': 0.0, 'risk_percent': 0.0,
                'free_margin': deposit, 'is_profitable': False
            }

# ---------------------------
# ReportGenerator
# ---------------------------
class ReportGenerator:
    @staticmethod
    def generate_calculation_report(calc: Dict[str, Any], context: Dict[str, Any]) -> str:
        try:
            return (
                f"ОТЧЕТ О РАСЧЕТЕ ПОЗИЦИИ\n"
                f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n"
                f"ПАРАМЕТРЫ СДЕЛКИ:\n"
                f"• Инструмент: {context.get('instrument', 'N/A')}\n"
                f"• Направление: {context.get('direction', 'N/A')}\n"
                f"• Депозит: ${context.get('deposit', 0.0):,.2f}\n"
                f"• Плечо: {context.get('leverage', 'N/A')}\n"
                f"• Уровень риска: {context.get('risk_percent', 0.0)*100:.2f}%\n\n"
                f"ЦЕНОВЫЕ УРОВНИ:\n"
                f"• Цена входа: {context.get('entry_price', 0.0)}\n"
                f"• Стоп-лосс: {context.get('stop_loss', 0.0)}\n"
                f"• Тейк-профит: {context.get('take_profit', 0.0)}\n"
                f"• Дистанция SL: {calc.get('stop_pips', 0):.2f} пунктов\n"
                f"• Дистанция TP: {calc.get('take_profit_pips', 0):.2f} пунктов\n\n"
                f"РЕЗУЛЬТАТЫ РАСЧЕТА:\n"
                f"• Размер позиции: {calc.get('position_size', 0):.4f} лотов\n"
                f"• Сумма риска: ${calc.get('risk_amount', 0):.2f}\n"
                f"• Потенциальная прибыль: ${calc.get('potential_profit', 0):.2f}\n"
                f"• Потенциальный убыток: ${calc.get('potential_loss', 0):.2f}\n"
                f"• Соотношение прибыль/риск: {calc.get('reward_risk_ratio', 0):.2f}\n"
                f"• Требуемая маржа: ${calc.get('required_margin', 0):.2f}\n"
                f"• Свободная маржа: ${calc.get('free_margin', 0):.2f}\n\n"
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
            recs.append("Убыточная сделка (TP/SL неверно настроены).")
        if rr < 1:
            recs.append("Соотношение прибыль/риск < 1 — пересмотрите TP/SL.")
        elif rr < 2:
            recs.append("Соотношение 1-2 — можно улучшить TP или уменьшить SL.")
        else:
            recs.append("Хорошее соотношение риск/прибыль (>2).")
        if risk_percent > 0.05:
            recs.append("Риск >5% — высокий риск, рекомендуется 1-3%.")
        elif risk_percent < 0.01:
            recs.append("Риск <1% — можно увеличить до 1-3% при уверенности.")
        else:
            recs.append("Риск в пределах норм (1-5%).")
        if margin_usage > 50:
            recs.append("Загруженность маржи >50% — уменьшите размер позиции.")
        elif margin_usage > 30:
            recs.append("Маржа 30-50% — следите за свободной маржей.")
        else:
            recs.append("Низкая загрузка маржи — есть запас для других позиций.")
        if is_profitable and rr >= 1.5 and risk_percent <= 0.03 and margin_usage <= 40:
            recs.append("ИДЕАЛЬНЫЕ ПАРАМЕТРЫ: можно рассмотреть масштабирование позиции.")
        if not is_profitable or rr < 1 or risk_percent > 0.05:
            recs.append("ВНИМАНИЕ: пересмотрите параметры сделки.")
        return "\n".join(recs)

    @staticmethod
    def generate_portfolio_report(user_id: int) -> str:
        PortfolioManager.ensure_user(user_id)
        p = user_data[user_id]['portfolio']
        perf = p['performance']
        trades = p['trades']
        corr = PortfolioAnalyzer.analyze_correlations(trades)
        vol = PortfolioAnalyzer.analyze_volatility(trades)
        metrics = PortfolioAnalyzer.calculate_portfolio_metrics(trades)
        return (
            f"ПОРТФЕЛЬ ОТЧЕТ\n"
            f"Баланс: ${p['current_balance']:,.2f} | Начальный: ${p['initial_balance']:,.2f}\n"
            f"Сделок: {len(trades)} | Win Rate: {perf['win_rate']:.1f}%\n"
            f"Общий профит: ${perf['total_profit']:,.2f} | Убыток: ${perf['total_loss']:,.2f}\n"
            f"Profit Factor: {perf['profit_factor']:.2f} | Max DD: {perf['max_drawdown']:.1f}%\n\n"
            f"КОРРЕЛЯЦИИ:\n" + "\n".join(corr[:3]) + ("\n..." if len(corr) > 3 else "") + "\n\n"
            f"ВОЛАТИЛЬНОСТЬ:\n" + "\n".join(vol[:3]) + ("\n..." if len(vol) > 3 else "") + "\n\n"
            f"МЕТРИКИ:\n"
            f"• Общий риск: {metrics.get('total_risk', 0)*100:.2f}%\n"
            f"• Средняя волатильность: {metrics.get('avg_volatility', 0):.1f}%\n"
            f"• Баланс направлений: {metrics.get('direction_balance', 0):.2f}\n"
        )

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
            dt = time.time() - start
            if dt > 1.0:
                logger.warning("Slow handler %s: %.2fs", func.__name__, dt)
    return wrapper

@performance_logger
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user = update.effective_user
        user_name = user.first_name if user else "Трейдер"
        user_id = user.id if user else 0
        PortfolioManager.ensure_user(user_id)
        welcome_text = (
            f"Привет, {user_name}!\n\n"
            "PRO Калькулятор Управления Рисками v3.0\n\n"
            "МОИ ВОЗМОЖНОСТИ:\n"
            "• Многопозиционный расчет\n"
            "• Анализ корреляций\n"
            "• Учет волатильности\n"
            "• Портфельные стратегии\n"
            "• Умные рекомендации\n\n"
            "Выберите опцию:"
        )
        keyboard = [
            [InlineKeyboardButton("Профессиональный расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("Мой портфель", callback_data="portfolio")],
            [InlineKeyboardButton("Расширенная аналитика", callback_data="analytics")],
            [InlineKeyboardButton("PRO Инструкции", callback_data="pro_info")]
        ]
        if update.message:
            await update.message.reply_text(welcome_text, reply_markup=InlineKeyboardMarkup(keyboard))
        elif update.callback_query:
            await update.callback_query.message.reply_text(welcome_text, reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        logger.exception("Ошибка в start_command: %s", e)
        await (update.message or update.callback_query.message).reply_text("Ошибка. Попробуйте позже.")

@performance_logger
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        if not query:
            return
        await query.answer()
        data = query.data

        if data == "pro_calculation":
            await start_pro_calculation(query, context); return
        if data == "portfolio":
            await show_portfolio_menu(query, context); return
        if data == "analytics":
            await query.edit_message_text("Аналитика в разработке...", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="main_menu")]]))
            return
        if data == "pro_info":
            await query.edit_message_text("PRO инструкции:\n1) Используйте профессиональные расчеты.\n2) Сохраняйте стратегии.\n3) Следите за корреляциями.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="main_menu")]]))
            return
        if data == "main_menu":
            await start_command(update, context); return

        await handle_main_menu_callbacks(query, context)
    except Exception as e:
        logger.exception("Ошибка в callback_router: %s", e)
        await handle_error_generic(update, context, e)

@performance_logger
async def start_pro_calculation(query, context):
    keyboard = [
        [InlineKeyboardButton("Одна сделка", callback_data="single_trade")],
        [InlineKeyboardButton("Мультипозиция", callback_data="multi_trade")],
        [InlineKeyboardButton("Назад", callback_data="main_menu")]
    ]
    await query.edit_message_text("ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ\nВыберите тип:", reply_markup=InlineKeyboardMarkup(keyboard))

@performance_logger
async def show_portfolio_menu(query, context):
    user_id = query.from_user.id
    PortfolioManager.ensure_user(user_id)
    p = user_data[user_id]['portfolio']
    text = (
        f"PRO ПОРТФЕЛЬ v3.0\n\n"
        f"Баланс: ${p['current_balance']:,.2f}\n"
        f"Сделки: {len(p['trades'])}\n"
        f"Win Rate: {p['performance'].get('win_rate', 0.0):.1f}%\n\n"
        "Выберите опцию:"
    )
    keyboard = [
        [InlineKeyboardButton("Обзор сделок", callback_data="portfolio_trades")],
        [InlineKeyboardButton("Баланс и распределение", callback_data="portfolio_balance")],
        [InlineKeyboardButton("Анализ эффективности", callback_data="portfolio_performance")],
        [InlineKeyboardButton("Сгенерировать отчет", callback_data="portfolio_report")],
        [InlineKeyboardButton("Выгрузить отчет", callback_data="export_portfolio")],
        [InlineKeyboardButton("Добавить сделку", callback_data="portfolio_add_trade")],
        [InlineKeyboardButton("Назад", callback_data="main_menu")]
    ]
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

@performance_logger
async def handle_main_menu_callbacks(query, context):
    data = query.data
    user_id = query.from_user.id
    if data == "single_trade":
        keyboard = [[InlineKeyboardButton(label, callback_data=f"pro_type_{key}")] for key, label in INSTRUMENT_TYPES.items()]
        keyboard.append([InlineKeyboardButton("Назад", callback_data="pro_calculation")])
        await query.edit_message_text("Выберите тип инструмента:", reply_markup=InlineKeyboardMarkup(keyboard))
        return
    if data.startswith("pro_type_"):
        itype = data.replace("pro_type_", "")
        context.user_data['instrument_type'] = itype
        presets = INSTRUMENT_PRESETS.get(itype, [])
        keyboard = [[InlineKeyboardButton(p, callback_data=f"pro_preset_{p}")] for p in presets]
        keyboard.append([InlineKeyboardButton("Ввести свой", callback_data="pro_custom")])
        keyboard.append([InlineKeyboardButton("Назад", callback_data="single_trade")])
        await query.edit_message_text(f"{INSTRUMENT_TYPES.get(itype)} - выберите:", reply_markup=InlineKeyboardMarkup(keyboard))
        return
    if data.startswith("pro_preset_") or data == "pro_custom":
        if data == "pro_custom":
            context.user_data['awaiting'] = 'custom_instrument'
            await query.edit_message_text("Введите инструмент (например EURUSD):")
            return
        context.user_data['instrument'] = data.replace("pro_preset_", "")
        keyboard = [
            [InlineKeyboardButton("BUY", callback_data="BUY"), InlineKeyboardButton("SELL", callback_data="SELL")],
            [InlineKeyboardButton("Назад", callback_data=f"pro_type_{context.user_data.get('instrument_type')}")]
        ]
        await query.edit_message_text(f"Инструмент: {context.user_data['instrument']}\nНаправление:", reply_markup=InlineKeyboardMarkup(keyboard))
        return
    if data in ("BUY", "SELL"):
        context.user_data['direction'] = data
        keyboard = [[InlineKeyboardButton(r, callback_data=f"pro_risk_{r.replace('%','')}")] for r in RISK_LEVELS]
        keyboard.append([InlineKeyboardButton("Назад", callback_data=f"pro_preset_{context.user_data.get('instrument','')}")])
        await query.edit_message_text(f"{context.user_data['instrument']} | {data}\nРиск (%):", reply_markup=InlineKeyboardMarkup(keyboard))
        return
    if data.startswith("pro_risk_"):
        val = float(data.replace("pro_risk_", "")) / 100.0
        context.user_data['risk_percent'] = val
        context.user_data['awaiting'] = 'deposit'
        await query.edit_message_text(f"Риск: {val*100:.2f}%\nВведите депозит (USD):")
        return
    if data.startswith("pro_leverage_"):
        lev = data.replace("pro_leverage_", "")
        context.user_data['leverage'] = lev
        context.user_data['awaiting'] = 'entry_price'
        await query.edit_message_text(f"Плечо: {lev}\nЦена входа:")
        return

    if data == "export_calculation":
        calc = context.user_data.get('last_calculation')
        meta = context.user_data.get('calculation_meta')
        if not calc or not meta:
            await query.edit_message_text("Нет данных.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="main_menu")]]))
            return
        txt = ReportGenerator.generate_calculation_report(calc, meta)
        bio = io.BytesIO(txt.encode('utf-8'))
        bio.name = "calculation_report.txt"
        await query.message.reply_document(document=InputFile(bio), caption="Отчет")
        return

    if data == "save_trade_from_pro":
        calc = context.user_data.get('last_calculation')
        meta = context.user_data.get('calculation_meta')
        if not calc or not meta:
            await query.edit_message_text("Нет данных.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="main_menu")]]))
            return
        trade = {
            'instrument': meta['instrument'], 'direction': meta['direction'],
            'entry_price': meta['entry_price'], 'stop_loss': meta['stop_loss'],
            'take_profit': meta['take_profit'], 'position_size': calc['position_size'],
            'risk_percent': meta['risk_percent'], 'status': 'open', 'profit': 0.0
        }
        tid = PortfolioManager.add_trade(user_id, trade)
        await query.edit_message_text(f"Сделка сохранена (ID: {tid})", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="main_menu")]]))
        return

    if data in ("portfolio_trades", "portfolio_balance", "portfolio_performance", "portfolio_report", "export_portfolio", "portfolio_add_trade"):
        if data == "portfolio_trades":
            trades = user_data[user_id]['portfolio'].get('trades', [])
            lines = ["Последние сделки:"] + [f"ID {t['id']} | {t['instrument']} {t['direction']} | {t['position_size']} лотов" for t in trades[-10:]]
            await query.edit_message_text("\n".join(lines), reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="portfolio")]]))
        elif data == "portfolio_balance":
            p = user_data[user_id]['portfolio']
            await query.edit_message_text(f"Баланс: ${p['current_balance']:,.2f}\nНачальный: ${p['initial_balance']:,.2f}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="portfolio")]]))
        elif data in ("portfolio_performance", "portfolio_report"):
            text = ReportGenerator.generate_portfolio_report(user_id)
            await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("Выгрузить", callback_data="export_portfolio")],
                [InlineKeyboardButton("Назад", callback_data="portfolio")]
            ]))
        elif data == "export_portfolio":
            txt = ReportGenerator.generate_portfolio_report(user_id)
            bio = io.BytesIO(txt.encode('utf-8'))
            bio.name = "portfolio_report.txt"
            await query.message.reply_document(document=InputFile(bio), caption="Отчет")
        elif data == "portfolio_add_trade":
            context.user_data['awaiting'] = 'portfolio_new_instrument'
            await query.edit_message_text("Инструмент (например EURUSD):")
        return

    await query.edit_message_text("Неизвестно. Вернитесь в меню.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="main_menu")]]))

@performance_logger
async def generic_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message or not update.message.text:
            return
        text = update.message.text.strip()
        awaiting = context.user_data.get('awaiting')
        user_id = update.message.from_user.id

        if awaiting == 'custom_instrument':
            ok, inst, msg = InputValidator.validate_instrument(text)
            if not ok:
                await update.message.reply_text(msg); return
            context.user_data['instrument'] = inst
            context.user_data.pop('awaiting', None)
            keyboard = [[InlineKeyboardButton("BUY", callback_data="BUY"), InlineKeyboardButton("SELL", callback_data="SELL")]]
            await update.message.reply_text(f"Инструмент: {inst}\nНаправление:", reply_markup=InlineKeyboardMarkup(keyboard))
            return

        if awaiting == 'deposit':
            ok, val, msg = InputValidator.validate_number(text, 1, 1e9)
            if not ok:
                await update.message.reply_text(msg); return
            context.user_data['deposit'] = val
            context.user_data.pop('awaiting', None)
            keyboard = [[InlineKeyboardButton(l, callback_data=f"pro_leverage_{l}")] for l in LEVERAGES]
            await update.message.reply_text(f"Депозит: ${val:,.2f}\nПлечо:", reply_markup=InlineKeyboardMarkup(keyboard))
            return

        if awaiting == 'entry_price':
            ok, val, msg = InputValidator.validate_price(text)
            if not ok:
                await update.message.reply_text(msg); return
            context.user_data['entry_price'] = val
            context.user_data['awaiting'] = 'stop_loss'
            dir_text = "ниже" if context.user_data['direction'] == 'BUY' else "выше"
            await update.message.reply_text(f"Вход: {val}\nСтоп-лосс ({dir_text}):")
            return

        if awaiting == 'stop_loss':
            ok, val, msg = InputValidator.validate_price(text)
            if not ok:
                await update.message.reply_text(msg); return
            context.user_data['stop_loss'] = val
            context.user_data['awaiting'] = 'take_profit'
            await update.message.reply_text("Тейк-профит:")
            return

        if awaiting == 'take_profit':
            ok, val, msg = InputValidator.validate_price(text)
            if not ok:
                await update.message.reply_text(msg); return
            context.user_data['take_profit'] = val
            context.user_data.pop('awaiting', None)

            calc = FastRiskCalculator.calculate_position_size_fast(
                deposit=context.user_data['deposit'],
                leverage=context.user_data.get('leverage', '1:100'),
                instrument_type=context.user_data.get('instrument_type', 'forex'),
                currency_pair=context.user_data['instrument'],
                entry_price=context.user_data['entry_price'],
                stop_loss=context.user_data['stop_loss'],
                take_profit=context.user_data['take_profit'],
                direction=context.user_data['direction'],
                risk_percent=context.user_data['risk_percent']
            )
            context.user_data['last_calculation'] = calc
            context.user_data['calculation_meta'] = {k: context.user_data[k] for k in ['instrument','direction','deposit','leverage','risk_percent','entry_price','stop_loss','take_profit']}

            result_text = (
                f"РЕЗУЛЬТАТ РАСЧЕТА {'ПРИБЫЛЬНЫЙ' if calc['is_profitable'] else 'УБЫТОЧНЫЙ'}\n\n"
                f"{context.user_data['instrument']} | {context.user_data['direction']}\n"
                f"Депозит: ${context.user_data['deposit']:,.2f} | Плечо: {context.user_data['leverage']}\n"
                f"Риск: {context.user_data['risk_percent']*100:.2f}%\n\n"
                f"Вход: {context.user_data['entry_price']}\nSL: {context.user_data['stop_loss']} ({calc['stop_pips']:.1f}p)\nTP: {context.user_data['take_profit']} ({calc['take_profit_pips']:.1f}p)\n\n"
                f"Позиция: {calc['position_size']:.4f} лотов\nРиск: ${calc['risk_amount']:.2f}\nПрибыль: ${calc['potential_profit']:.2f}\nP/R: {calc['reward_risk_ratio']:.2f}\n"
                f"Маржа: ${calc['required_margin']:.2f} | Свободно: ${calc['free_margin']:.2f}\n\n"
                f"Рекомендации:\n{ReportGenerator.get_professional_recommendations(calc, context.user_data)}"
            )
            keyboard = [
                [InlineKeyboardButton("Выгрузить", callback_data="export_calculation")],
                [InlineKeyboardButton("Сохранить", callback_data="save_trade_from_pro")],
                [InlineKeyboardButton("Новый расчет", callback_data="pro_calculation")],
                [InlineKeyboardButton("Меню", callback_data="main_menu")]
            ]
            await update.message.reply_text(result_text, reply_markup=InlineKeyboardMarkup(keyboard))
            return

        if awaiting and awaiting.startswith('portfolio_new_'):
            new = context.user_data.setdefault('new_trade', {})
            if awaiting == 'portfolio_new_instrument':
                ok, inst, msg = InputValidator.validate_instrument(text)
                if not ok: await update.message.reply_text(msg); return
                new['instrument'] = inst
                context.user_data['awaiting'] = 'portfolio_new_direction'
                await update.message.reply_text("Направление: BUY или SELL")
            elif awaiting == 'portfolio_new_direction':
                d = text.upper()
                if d not in ("BUY", "SELL"): await update.message.reply_text("BUY или SELL"); return
                new['direction'] = d
                context.user_data['awaiting'] = 'portfolio_new_entry'
                await update.message.reply_text("Цена входа:")
            elif awaiting == 'portfolio_new_entry':
                ok, val, msg = InputValidator.validate_price(text)
                if not ok: await update.message.reply_text(msg); return
                new['entry_price'] = val
                context.user_data['awaiting'] = 'portfolio_new_sl'
                await update.message.reply_text("Стоп-лосс:")
            elif awaiting == 'portfolio_new_sl':
                ok, val, msg = InputValidator.validate_price(text)
                if not ok: await update.message.reply_text(msg); return
                new['stop_loss'] = val
                context.user_data['awaiting'] = 'portfolio_new_tp'
                await update.message.reply_text("Тейк-профит:")
            elif awaiting == 'portfolio_new_tp':
                ok, val, msg = InputValidator.validate_price(text)
                if not ok: await update.message.reply_text(msg); return
                new['take_profit'] = val
                context.user_data['awaiting'] = 'portfolio_new_risk'
                await update.message.reply_text("Риск в % (например 1.5):")
            elif awaiting == 'portfolio_new_risk':
                ok, val, msg = InputValidator.validate_number(text, 0.01, 100)
                if not ok: await update.message.reply_text(msg); return
                new['risk_percent'] = val / 100.0
                p = user_data[user_id]['portfolio']
                deposit = p['current_balance'] or p['initial_balance'] or 1000.0
                calc = FastRiskCalculator.calculate_position_size_fast(
                    deposit=deposit, leverage=p['settings']['leverage'],
                    instrument_type='forex', currency_pair=new['instrument'],
                    entry_price=new['entry_price'], stop_loss=new['stop_loss'],
                    take_profit=new['take_profit'], direction=new['direction'],
                    risk_percent=new['risk_percent']
                )
                new['position_size'] = calc['position_size']
                new['status'] = 'open'
                tid = PortfolioManager.add_trade(user_id, new.copy())
                await update.message.reply_text(f"Сделка добавлена (ID: {tid})", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Меню", callback_data="main_menu")]]))
                context.user_data.pop('new_trade', None)
                context.user_data.pop('awaiting', None)
            return

        await update.message.reply_text("Используйте /start или меню.")
    except Exception as e:
        logger.exception("Ошибка в generic_text_handler: %s", e)
        await update.message.reply_text("Ошибка обработки.")

@performance_logger
async def handle_error_generic(update: Update, context: ContextTypes.DEFAULT_TYPE, exc: Exception):
    logger.exception("Ошибка: %s", exc)
    try:
        msg = update.callback_query.message if update.callback_query else update.message
        await msg.reply_text("Ошибка. Вернитесь в меню.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Меню", callback_data="main_menu")]]))
    except:
        pass

# ---------------------------
# HTTP Server (Webhook)
# ---------------------------
async def health_check(request):
    return web.Response(text="OK")

async def handle_webhook(request, application: Application):
    try:
        if not getattr(application, "_initialized", False):
            return web.Response(status=200, text="Initializing...")
        data = await request.json()
        update = Update.de_json(data, application.bot)
        await application.process_update(update)
        return web.Response(status=200)
    except Exception as e:
        logger.exception("Webhook error: %s", e)
        return web.Response(status=500, text="error")

async def start_http_server(application: Application):
    app = web.Application()
    app.router.add_get('/', health_check)
    app.router.add_get('/health', health_check)
    app.router.add_post(WEBHOOK_PATH, lambda r: handle_webhook(r, application))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    logger.info("HTTP server on port %s", PORT)
    return runner

async def set_webhook(application: Application) -> bool:
    if not WEBHOOK_URL:
        return False
    try:
        url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
        await application.bot.set_webhook(url=url, drop_pending_updates=True)
        logger.info("Webhook set: %s", url)
        return True
    except Exception as e:
        logger.exception("Webhook set failed: %s", e)
        return False

# ---------------------------
# Запуск бота
# ---------------------------
async def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(callback_router))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generic_text_handler))
    application.add_error_handler(handle_error_generic)

    if WEBHOOK_URL:
        await application.initialize()
        await application.start()
        if await set_webhook(application):
            runner = await start_http_server(application)
            logger.info("Running in webhook mode.")
            try:
                await asyncio.Event().wait()
            finally:
                await application.stop()
                await application.shutdown()
                await runner.cleanup()
            return
        else:
            logger.warning("Webhook failed. Falling back to polling.")

    logger.info("Running in polling mode.")
    await application.run_polling()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")
