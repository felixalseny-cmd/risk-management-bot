import os
import logging
import asyncio
import re
import time
import functools
import json
import io
import math
import aiohttp
import threading
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
from aiohttp import web

# === НАСТРОЙКИ ДЛЯ RENDER ===
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден! Добавь в переменные окружения.")

PORT = int(os.getenv("PORT", 5000))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_PATH = f"/webhook/{TOKEN}"

# Оптимизированное логирование для Render
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Декоратор для логирования производительности
def log_performance(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            if execution_time > 1.0:
                logger.warning(f"Медленная операция: {func.__name__} заняла {execution_time:.2f}с")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Ошибка в {func.__name__}: {e} (время: {execution_time:.2f}с)")
            raise
    return wrapper

# Состояния диалога
(
    MAIN_MENU, INSTRUMENT_TYPE, CUSTOM_INSTRUMENT, DIRECTION, 
    RISK_PERCENT, DEPOSIT, LEVERAGE, CURRENCY, ENTRY, 
    STOP_LOSS, TAKE_PROFITS, VOLUME_DISTRIBUTION,
    PORTFOLIO_MENU, ADD_TRADE_INSTRUMENT, ADD_TRADE_DIRECTION,
    ADD_TRADE_ENTRY, ADD_TRADE_EXIT, ADD_TRADE_VOLUME, ADD_TRADE_PROFIT,
    DEPOSIT_AMOUNT, WITHDRAW_AMOUNT, SETTINGS_MENU, SAVE_STRATEGY_NAME,
    PRO_DEPOSIT, PRO_LEVERAGE, PRO_RISK, PRO_ENTRY, PRO_STOPLOSS,
    PRO_TAKEPROFIT, PRO_VOLUME, STRATEGY_NAME,
    ANALYTICS_MENU, TAKE_PROFIT_SINGLE, SINGLE_OR_MULTI,
    MULTI_TRADE_MENU, MULTI_INSTRUMENT, MULTI_DIRECTION, MULTI_ENTRY,
    MULTI_STOPLOSS, MULTI_TAKEPROFIT, MULTI_ADD_MORE
) = range(41)

# Константы
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

# Данные по корреляции (упрощенные)
CORRELATION_MATRIX = {
    'EURUSD': {'GBPUSD': 0.8, 'USDJPY': -0.7, 'USDCAD': -0.8, 'AUDUSD': 0.6, 'XAUUSD': 0.3},
    'GBPUSD': {'EURUSD': 0.8, 'USDJPY': -0.6, 'USDCAD': -0.7, 'AUDUSD': 0.5, 'XAUUSD': 0.2},
    'USDJPY': {'EURUSD': -0.7, 'GBPUSD': -0.6, 'USDCAD': 0.9, 'AUDUSD': -0.5, 'XAUUSD': -0.4},
    'USDCAD': {'EURUSD': -0.8, 'GBPUSD': -0.7, 'USDJPY': 0.9, 'AUDUSD': -0.6, 'XAUUSD': -0.3},
    'AUDUSD': {'EURUSD': 0.6, 'GBPUSD': 0.5, 'USDJPY': -0.5, 'USDCAD': -0.6, 'XAUUSD': 0.4},
    'XAUUSD': {'EURUSD': 0.3, 'GBPUSD': 0.2, 'USDJPY': -0.4, 'USDCAD': -0.3, 'AUDUSD': 0.4}
}

# Данные по волатильности (среднегодовая в %)
VOLATILITY_DATA = {
    'EURUSD': 8.5, 'GBPUSD': 9.2, 'USDJPY': 7.8, 'USDCAD': 7.5, 
    'AUDUSD': 10.1, 'NZDUSD': 9.8, 'EURGBP': 6.5,
    'BTCUSD': 65.2, 'ETHUSD': 70.5, 'XRPUSD': 85.3,
    'US30': 15.2, 'NAS100': 18.5, 'SPX500': 16.1,
    'XAUUSD': 14.5, 'XAGUSD': 25.3, 'OIL': 35.2
}

PIP_VALUES = {
    # Forex - основные пары
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
    'OIL': 10, 'NATGAS': 10, 'COPPER': 10,
    # Металлы
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

# Файл для сохранения данных
DATA_FILE = "user_data.json"

# Менеджер данных с сохранением в файл
class DataManager:
    @staticmethod
    def load_data():
        """Загрузка данных из файла"""
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return {}

    @staticmethod
    def save_data():
        """Сохранение данных в файл"""
        try:
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            logger.info("Данные успешно сохранены")
        except Exception as e:
            logger.error(f"Ошибка сохранения данных: {e}")

# Глобальное хранилище данных пользователей
user_data: Dict[int, Dict[str, Any]] = DataManager.load_data()

# Упрощенный кэш с оптимизацией для Render
class FastCache:
    def __init__(self, max_size=100, ttl=300):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            oldest_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])[:10]
            for old_key in oldest_keys:
                del self.cache[old_key]
        self.cache[key] = (value, time.time())

fast_cache = FastCache()

# Анализатор портфеля для многопозиционного анализа
class PortfolioAnalyzer:
    @staticmethod
    def analyze_correlations(trades: List[Dict]) -> List[str]:
        """Анализ корреляций между позициями"""
        if len(trades) < 2:
            return ["ℹ️ Для анализа корреляций нужно минимум 2 позиции"]
        
        analysis = []
        for i, trade1 in enumerate(trades):
            for j, trade2 in enumerate(trades[i+1:], i+1):
                inst1, dir1 = trade1['instrument'], trade1['direction']
                inst2, dir2 = trade2['instrument'], trade2['direction']
                
                if inst1 in CORRELATION_MATRIX and inst2 in CORRELATION_MATRIX[inst1]:
                    corr = CORRELATION_MATRIX[inst1][inst2]
                    
                    if abs(corr) > 0.7:
                        if dir1 == dir2:
                            if corr > 0:
                                analysis.append(f"⚠️ Высокая позитивная корреляция ({corr:.2f}) между {inst1} {dir1} и {inst2} {dir2} - риски удваиваются")
                            else:
                                analysis.append(f"🔄 Высокая негативная корреляция ({corr:.2f}) между {inst1} {dir1} и {inst2} {dir2} - хеджирование позиций")
                        else:
                            if corr > 0:
                                analysis.append(f"⚡ Диверсификация: {inst1} {dir1} vs {inst2} {dir2} (корр: {corr:.2f})")
                            else:
                                analysis.append(f"🎯 Противонаправленные позиции с негативной корреляцией ({corr:.2f})")
        
        return analysis if analysis else ["✅ Корреляционный риск под контролем"]

    @staticmethod
    def analyze_volatility(trades: List[Dict]) -> List[str]:
        """Анализ волатильности позиций"""
        analysis = []
        high_vol_count = 0
        
        for trade in trades:
            instrument = trade['instrument']
            if instrument in VOLATILITY_DATA:
                vol = VOLATILITY_DATA[instrument]
                
                if vol > 20:
                    high_vol_count += 1
                    analysis.append(f"⚡ Высокая волатильность {instrument}: {vol}% (требует осторожности)")
                elif vol > 10:
                    analysis.append(f"📊 Средняя волатильность {instrument}: {vol}%")
                else:
                    analysis.append(f"✅ Низкая волатильность {instrument}: {vol}%")
        
        if high_vol_count >= 3:
            analysis.append("🚨 ВНИМАНИЕ: Много высоковолатильных инструментов - увеличивается общий риск")
        
        return analysis

    @staticmethod
    def generate_portfolio_strategies(trades: List[Dict]) -> List[str]:
        """Генерация стратегий для портфеля"""
        strategies = []
        
        if len(trades) >= 3:
            strategies.append("🎯 СТРАТЕГИЯ 1: Балансировка рисков")
            strategies.append("   • Равномерно распределите капитал между позициями")
            strategies.append("   • Установите стоп-лоссы на основе волатильности")
            strategies.append("   • Ребалансируйте портфель при изменении условий")
            
            strategies.append("")
            strategies.append("📈 СТРАТЕГИЯ 2: Корреляционное хеджирование")
            strategies.append("   • Используйте негативно коррелируемые активы")
            strategies.append("   • Диверсифицируйте по типам инструментов")
            strategies.append("   • Контролируйте общую экспозицию")
            
            strategies.append("")
            strategies.append("⚡ СТРАТЕГИЯ 3: Волатильностное управление")
            strategies.append("   • Уменьшайте размер позиций для волатильных активы")
            strategies.append("   • Используйте ATR для расчета стоп-лоссов")
            strategies.append("   • Адаптируйте риск под текущую волатильность")
        else:
            strategies.append("💡 Для сложных стратегий добавьте больше позиций (рекомендуется 3-5)")
        
        return strategies

    @staticmethod
    def calculate_portfolio_metrics(trades: List[Dict]) -> Dict[str, float]:
        """Расчет метрик портфеля"""
        if not trades:
            return {}
        
        total_risk = sum(trade.get('risk_percent', 0) for trade in trades)
        avg_volatility = sum(VOLATILITY_DATA.get(trade['instrument'], 15) for trade in trades) / len(trades)
        
        # Анализ направлений
        buy_count = sum(1 for trade in trades if trade['direction'] == 'BUY')
        sell_count = len(trades) - buy_count
        
        return {
            'total_risk': total_risk,
            'avg_volatility': avg_volatility,
            'diversity_score': min(len(trades) / 5.0, 1.0),
            'direction_balance': abs(buy_count - sell_count) / len(trades)
        }

# Менеджер портфеля
class PortfolioManager:
    @staticmethod
    def initialize_user_portfolio(user_id: int):
        if user_id not in user_data:
            user_data[user_id] = {}
        
        if 'portfolio' not in user_data[user_id]:
            user_data[user_id]['portfolio'] = {
                'initial_balance': 0,
                'current_balance': 0,
                'trades': [],
                'performance': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0,
                    'total_loss': 0,
                    'win_rate': 0,
                    'average_profit': 0,
                    'average_loss': 0,
                    'profit_factor': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0
                },
                'allocation': {},
                'history': [],
                'settings': {
                    'default_risk': 0.02,
                    'currency': 'USD',
                    'leverage': '1:100'
                },
                'saved_strategies': [],
                'multi_trade_mode': False,
                'multi_trades': []  # Новое поле для многопозиционных сделок
            }
        DataManager.save_data()
    
    @staticmethod
    def add_trade(user_id: int, trade_data: Dict):
        PortfolioManager.initialize_user_portfolio(user_id)
        
        if len(user_data[user_id]['portfolio']['trades']) >= 10:
            raise ValueError("❌ Достигнут лимит в 10 сделок. Удалите старые сделки чтобы добавить новые.")
        
        trade_id = len(user_data[user_id]['portfolio']['trades']) + 1
        trade_data['id'] = trade_id
        trade_data['timestamp'] = datetime.now().isoformat()
        
        user_data[user_id]['portfolio']['trades'].append(trade_data)
        
        profit = trade_data.get('profit', 0)
        user_data[user_id]['portfolio']['current_balance'] += profit
        
        PortfolioManager.update_performance_metrics(user_id)
        
        instrument = trade_data.get('instrument', 'Unknown')
        if instrument not in user_data[user_id]['portfolio']['allocation']:
            user_data[user_id]['portfolio']['allocation'][instrument] = 0
        user_data[user_id]['portfolio']['allocation'][instrument] += 1
        
        user_data[user_id]['portfolio']['history'].append({
            'type': 'trade',
            'action': 'open' if trade_data.get('status') == 'open' else 'close',
            'instrument': instrument,
            'profit': profit,
            'timestamp': trade_data['timestamp']
        })
        DataManager.save_data()
        return trade_id

    @staticmethod
    def add_multi_trade(user_id: int, multi_trade_data: Dict):
        """Добавление многопозиционной сделки"""
        PortfolioManager.initialize_user_portfolio(user_id)
        
        if len(user_data[user_id]['portfolio']['multi_trades']) >= 5:
            raise ValueError("❌ Достигнут лимит в 5 многопозиционных сделок.")
        
        trade_id = len(user_data[user_id]['portfolio']['multi_trades']) + 1
        multi_trade_data['id'] = trade_id
        multi_trade_data['timestamp'] = datetime.now().isoformat()
        
        user_data[user_id]['portfolio']['multi_trades'].append(multi_trade_data)
        DataManager.save_data()
        return trade_id

    @staticmethod
    def update_performance_metrics(user_id: int):
        portfolio = user_data[user_id]['portfolio']
        trades = portfolio['trades']
        
        if not trades:
            return
        
        closed_trades = [t for t in trades if t.get('status') == 'closed']
        if not closed_trades:
            return
            
        winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('profit', 0) < 0]
        breakeven_trades = [t for t in closed_trades if t.get('profit', 0) == 0]
        
        portfolio['performance']['total_trades'] = len(closed_trades)
        portfolio['performance']['winning_trades'] = len(winning_trades)
        portfolio['performance']['losing_trades'] = len(losing_trades)
        portfolio['performance']['total_profit'] = sum(t.get('profit', 0) for t in winning_trades)
        portfolio['performance']['total_loss'] = abs(sum(t.get('profit', 0) for t in losing_trades))
        
        if closed_trades:
            portfolio['performance']['win_rate'] = (len(winning_trades) / len(closed_trades)) * 100
            
            portfolio['performance']['average_profit'] = (
                portfolio['performance']['total_profit'] / len(winning_trades) 
                if winning_trades else 0
            )
            portfolio['performance']['average_loss'] = (
                portfolio['performance']['total_loss'] / len(losing_trades) 
                if losing_trades else 0
            )
            
            if portfolio['performance']['total_loss'] > 0:
                portfolio['performance']['profit_factor'] = (
                    portfolio['performance']['total_profit'] / portfolio['performance']['total_loss']
                )
            else:
                portfolio['performance']['profit_factor'] = float('inf') if portfolio['performance']['total_profit'] > 0 else 0
            
            balance_history = []
            running_balance = portfolio['initial_balance']
            
            for event in sorted(portfolio['history'], key=lambda x: x['timestamp']):
                if event['type'] == 'balance':
                    if event['action'] == 'deposit':
                        running_balance += event['amount']
                    elif event['action'] == 'withdrawal':
                        running_balance -= event['amount']
                elif event['type'] == 'trade' and event['action'] == 'close':
                    running_balance += event['profit']
                
                balance_history.append(running_balance)
            
            if balance_history:
                peak = balance_history[0]
                max_drawdown = 0
                
                for balance in balance_history:
                    if balance > peak:
                        peak = balance
                    drawdown = (peak - balance) / peak * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                
                portfolio['performance']['max_drawdown'] = max_drawdown
        DataManager.save_data()

# Ультра-быстрый калькулятор рисков
class FastRiskCalculator:
    """Оптимизированный калькулятор рисков с упрощенными расчетами"""
    
    @staticmethod
    def calculate_pip_value_fast(instrument_type: str, currency_pair: str, lot_size: float) -> float:
        """Быстрый расчет стоимости пипса"""
        base_pip_value = PIP_VALUES.get(currency_pair, 10)
        
        if instrument_type == 'crypto':
            return base_pip_value * lot_size * 0.1
        elif instrument_type == 'indices':
            return base_pip_value * lot_size * 0.01
        else:
            return base_pip_value * lot_size

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
    ) -> Dict[str, float]:
        """Ультра-быстрый расчет размера позиции с тейк-профитом"""
        try:
            cache_key = f"pos_{deposit}_{leverage}_{instrument_type}_{currency_pair}_{entry_price}_{stop_loss}_{take_profit}_{direction}_{risk_percent}"
            cached_result = fast_cache.get(cache_key)
            if cached_result:
                return cached_result
            
            lev_value = int(leverage.split(':')[1])
            risk_amount = deposit * risk_percent
            
            if instrument_type == 'forex':
                stop_pips = abs(entry_price - stop_loss) * 10000
                take_profit_pips = abs(entry_price - take_profit) * 10000
            elif instrument_type == 'crypto':
                stop_pips = abs(entry_price - stop_loss) * 100
                take_profit_pips = abs(entry_price - take_profit) * 100
            elif instrument_type in ['indices', 'commodities', 'metals']:
                stop_pips = abs(entry_price - stop_loss) * 10
                take_profit_pips = abs(entry_price - take_profit) * 10
            else:
                stop_pips = abs(entry_price - stop_loss) * 10000
                take_profit_pips = abs(entry_price - take_profit) * 10000

            pip_value_per_lot = FastRiskCalculator.calculate_pip_value_fast(
                instrument_type, currency_pair, 1.0
            )
            
            if stop_pips > 0 and pip_value_per_lot > 0:
                max_lots_by_risk = risk_amount / (stop_pips * pip_value_per_lot)
            else:
                max_lots_by_risk = 0
            
            contract_size = CONTRACT_SIZES.get(instrument_type, 100000)
            if entry_price > 0:
                max_lots_by_margin = (deposit * lev_value) / (contract_size * entry_price)
            else:
                max_lots_by_margin = 0
            
            position_size = min(max_lots_by_risk, max_lots_by_margin, 50.0)
            
            if position_size < 0.01:
                position_size = 0.01
            else:
                position_size = round(position_size * 100) / 100
                
            required_margin = (position_size * contract_size * entry_price) / lev_value if lev_value > 0 else 0
            
            if direction == 'BUY':
                potential_profit = (take_profit - entry_price) * pip_value_per_lot * position_size
                potential_loss = (stop_loss - entry_price) * pip_value_per_lot * position_size
            else:
                potential_profit = (entry_price - take_profit) * pip_value_per_lot * position_size
                potential_loss = (entry_price - stop_loss) * pip_value_per_lot * position_size
            
            if potential_profit < 0:
                potential_profit = 0
                reward_risk_ratio = 0
            else:
                reward_risk_ratio = potential_profit / risk_amount if risk_amount > 0 else 0
            
            result = {
                'position_size': position_size,
                'risk_amount': risk_amount,
                'stop_pips': stop_pips,
                'take_profit_pips': take_profit_pips,
                'potential_profit': potential_profit,
                'potential_loss': abs(potential_loss),
                'reward_risk_ratio': reward_risk_ratio,
                'required_margin': required_margin,
                'risk_percent': (risk_amount / deposit) * 100 if deposit > 0 else 0,
                'free_margin': deposit - required_margin,
                'is_profitable': potential_profit > 0
            }
            
            fast_cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Ошибка в быстром расчете размера позиции: {e}")
            return {
                'position_size': 0.01,
                'risk_amount': 0,
                'stop_pips': 0,
                'take_profit_pips': 0,
                'potential_profit': 0,
                'potential_loss': 0,
                'reward_risk_ratio': 0,
                'required_margin': 0,
                'risk_percent': 0,
                'free_margin': deposit,
                'is_profitable': False
            }

    @staticmethod
    def calculate_multi_position(
        deposit: float,
        leverage: str,
        trades: List[Dict],
        total_risk_percent: float = 0.05
    ) -> Dict[str, Any]:
        """Расчет многопозиционного портфеля"""
        try:
            total_risk_amount = deposit * total_risk_percent
            individual_risk = total_risk_amount / len(trades) if trades else 0
            
            results = []
            total_required_margin = 0
            total_potential_profit = 0
            total_potential_loss = 0
            
            for trade in trades:
                calculation = FastRiskCalculator.calculate_position_size_fast(
                    deposit=deposit,
                    leverage=leverage,
                    instrument_type=trade['instrument_type'],
                    currency_pair=trade['instrument'],
                    entry_price=trade['entry_price'],
                    stop_loss=trade['stop_loss'],
                    take_profit=trade['take_profit'],
                    direction=trade['direction'],
                    risk_percent=individual_risk / deposit
                )
                
                results.append({
                    'instrument': trade['instrument'],
                    'direction': trade['direction'],
                    'calculation': calculation
                })
                
                total_required_margin += calculation['required_margin']
                total_potential_profit += calculation['potential_profit']
                total_potential_loss += calculation['potential_loss']
            
            # Анализ корреляций
            correlation_analysis = PortfolioAnalyzer.analyze_correlations(trades)
            volatility_analysis = PortfolioAnalyzer.analyze_volatility(trades)
            portfolio_metrics = PortfolioAnalyzer.calculate_portfolio_metrics(trades)
            
            return {
                'trades': results,
                'portfolio_metrics': {
                    'total_required_margin': total_required_margin,
                    'total_potential_profit': total_potential_profit,
                    'total_potential_loss': total_potential_loss,
                    'margin_usage_percent': (total_required_margin / deposit) * 100 if deposit > 0 else 0,
                    'portfolio_risk_percent': total_risk_percent * 100,
                    'diversity_score': portfolio_metrics.get('diversity_score', 0),
                    'avg_volatility': portfolio_metrics.get('avg_volatility', 0)
                },
                'analysis': {
                    'correlations': correlation_analysis,
                    'volatility': volatility_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка в многопозиционном расчете: {e}")
            return {}

# Валидатор ввода данных
class InputValidator:
    """Класс для валидации вводимых данных"""
    
    @staticmethod
    def validate_number(text: str, min_val: float = 0, max_val: float = None) -> Tuple[bool, float, str]:
        """Валидация числового значения"""
        try:
            value = float(text.replace(',', '.'))
            if value < min_val:
                return False, value, f"❌ Значение не может быть меньше {min_val}"
            if max_val and value > max_val:
                return False, value, f"❌ Значение не может быть больше {max_val}"
            return True, value, "✅ Корректное значение"
        except ValueError:
            return False, 0, "❌ Введите корректное числовое значение"
    
    @staticmethod
    def validate_instrument(instrument: str) -> Tuple[bool, str]:
        """Валидация названия инструмента"""
        instrument = instrument.upper().strip()
        if not instrument:
            return False, "❌ Введите название инструмента"
        if len(instrument) > 20:
            return False, "❌ Название инструмента слишком длинное"
        return True, instrument
    
    @staticmethod
    def validate_price(price: str) -> Tuple[bool, float, str]:
        """Валидация цены"""
        return InputValidator.validate_number(price, 0.0001, 1000000)
    
    @staticmethod
    def validate_percent(percent: str) -> Tuple[bool, float, str]:
        """Валидация процентного значения"""
        return InputValidator.validate_number(percent, 0.01, 100)

# Генератор отчетов
class ReportGenerator:
    @staticmethod
    def generate_calculation_report(calculation_data: Dict, user_data_context: Dict) -> str:
        """Генерация отчета о расчете"""
        try:
            instrument = user_data_context.get('instrument', 'N/A')
            direction = user_data_context.get('direction', 'N/A')
            
            report = f"""
ОТЧЕТ О РАСЧЕТЕ ПОЗИЦИИ
Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}

ПАРАМЕТРЫ СДЕЛКИ:
• Инструмент: {instrument}
• Направление: {direction}
• Депозит: ${user_data_context.get('deposit', 0):,.2f}
• Плечо: {user_data_context.get('leverage', 'N/A')}
• Уровень риска: {user_data_context.get('risk_percent', 0)*100}%

ЦЕНОВЫЕ УРОВНИ:
• Цена входа: {user_data_context.get('entry_price', 0)}
• Стоп-лосс: {user_data_context.get('stop_loss', 0)}
• Тейк-профит: {user_data_context.get('take_profit', 0)}
• Дистанция SL: {calculation_data.get('stop_pips', 0):.2f} пунктов
• Дистанция TP: {calculation_data.get('take_profit_pips', 0):.2f} пунктов

РЕЗУЛЬТАТЫ РАСЧЕТА:
• Размер позиции: {calculation_data.get('position_size', 0):.2f} лотов
• Сумма риска: ${calculation_data.get('risk_amount', 0):.2f}
• Потенциальная прибыль: ${calculation_data.get('potential_profit', 0):.2f}
• Потенциальный убыток: ${calculation_data.get('potential_loss', 0):.2f}
• Соотношение прибыль/риск: {calculation_data.get('reward_risk_ratio', 0):.2f}
• Требуемая маржа: ${calculation_data.get('required_margin', 0):.2f}
• Свободная маржа: ${calculation_data.get('free_margin', 0):.2f}

ПРОФЕССИОНАЛЬНЫЕ РЕКОМЕНДАЦИИ:
{ReportGenerator.get_professional_recommendations(calculation_data, user_data_context)}
"""
            return report
        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")
            return "Ошибка при генерации отчета"

    @staticmethod
    def get_professional_recommendations(calculation_data: Dict, user_data_context: Dict) -> str:
        """Генерация профессиональных рекомендаций"""
        recommendations = []
        
        rr_ratio = calculation_data.get('reward_risk_ratio', 0)
        risk_percent = calculation_data.get('risk_percent', 0)
        position_size = calculation_data.get('position_size', 0)
        free_margin = calculation_data.get('free_margin', 0)
        deposit = user_data_context.get('deposit', 0)
        is_profitable = calculation_data.get('is_profitable', True)
        
        if not is_profitable:
            recommendations.append("🔴 УБЫТОЧНАЯ СДЕЛКА: Тейк-профит ниже/выше цены входа")
            recommendations.append("   💡 Рекомендация: Пересмотрите уровни тейк-профита и стоп-лосса")
        elif rr_ratio < 1:
            recommendations.append("🔴 КРИТИЧЕСКИЙ УРОВЕНЬ: Соотношение прибыль/риск меньше 1")
            recommendations.append("   💡 Рекомендация: Увеличьте дистанцию тейк-профита или уменьшите стоп-лосс")
        elif rr_ratio < 1.5:
            recommendations.append("🟡 НИЗКИЙ УРОВЕНЬ: Соотношение прибыль/риск 1-1.5")
            recommendations.append("   💡 Рекомендация: Стремитесь к соотношению не менее 1:2")
        elif rr_ratio >= 2:
            recommendations.append("🟢 ОТЛИЧНО: Соотношение прибыль/риск более 2:1")
            recommendations.append("   💡 Рекомендация: Оптимальные параметры для сделки")
        
        if risk_percent > 5:
            recommendations.append("🔴 ВЫСОКИЙ РИСК: Более 5% на сделку")
            recommendations.append("   💡 Рекомендация: Уменьшите риск до 1-2% для сохранения капитала")
        elif risk_percent < 1:
            recommendations.append("🟡 НИЗКИЙ РИСК: Менее 1% на сделку")
            recommendations.append("   💡 Рекомендация: Можно увеличить риск до 2-3% для роста")
        else:
            recommendations.append("🟢 ОПТИМАЛЬНЫЙ РИСК: 1-5% на сделку")
            recommendations.append("   💡 Рекомендация: Продолжайте в том же духе")
        
        margin_usage = (calculation_data.get('required_margin', 0) / deposit * 100) if deposit > 0 else 0
        if margin_usage > 50:
            recommendations.append("🔴 ВЫСОКАЯ ЗАГРУЗКА МАРЖИ: Более 50% депозита")
            recommendations.append("   💡 Рекомендация: Уменьшите размер позиции для безопасности")
        elif margin_usage > 30:
            recommendations.append("🟡 УМЕРЕННАЯ ЗАГРРУЗКА МАРЖИ: 30-50% депозита")
            recommendations.append("   💡 Рекомендация: Приемлемый уровень, но следите за рисками")
        else:
            recommendations.append("🟢 НИЗКАЯ ЗАГРУЗКА МАРЖИ: Менее 30% депозита")
            recommendations.append("   💡 Рекомендация: Есть запас для других сделок")
        
        if is_profitable and rr_ratio >= 1.5 and risk_percent <= 3 and margin_usage <= 40:
            recommendations.append("🚀 ИДЕАЛЬНАЯ СДЕЛКА: Все параметры оптимальны!")
        elif not is_profitable or rr_ratio < 1 or risk_percent > 5:
            recommendations.append("⚡ ОПАСНО: Пересмотрите параметры сделки!")
        
        return "\n".join(recommendations)

    @staticmethod
    def generate_multi_trade_report(multi_calculation: Dict, user_data_context: Dict) -> str:
        """Генерация отчета для многопозиционного расчета"""
        try:
            deposit = user_data_context.get('deposit', 0)
            leverage = user_data_context.get('leverage', '1:100')
            total_risk = user_data_context.get('total_risk_percent', 0.05) * 100
            
            report = f"""
📊 ОТЧЕТ ПО МНОГОПОЗИЦИОННОМУ РАСЧЕТУ
Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}

ОБЩИЕ ПАРАМЕТРЫ:
• Депозит: ${deposit:,.2f}
• Плечо: {leverage}
• Общий риск: {total_risk:.1f}%

РЕЗУЛЬТАТЫ РАСЧЕТА:
"""
            
            portfolio_metrics = multi_calculation.get('portfolio_metrics', {})
            report += f"""
📈 МЕТРИКИ ПОРТФЕЛЯ:
• Общая требуемая маржа: ${portfolio_metrics.get('total_required_margin', 0):.2f}
• Использование маржи: {portfolio_metrics.get('margin_usage_percent', 0):.1f}%
• Общая потенциальная прибыль: ${portfolio_metrics.get('total_potential_profit', 0):.2f}
• Общий потенциальный убыток: ${portfolio_metrics.get('total_potential_loss', 0):.2f}
• Диверсификация: {portfolio_metrics.get('diversity_score', 0)*100:.1f}%
• Средняя волатильность: {portfolio_metrics.get('avg_volatility', 0):.1f}%

📋 ДЕТАЛИ СДЕЛОК:
"""
            
            for i, trade in enumerate(multi_calculation.get('trades', []), 1):
                calc = trade['calculation']
                report += f"""
{i}. {trade['instrument']} {trade['direction']}
   • Размер: {calc['position_size']:.2f} лотов
   • Риск: ${calc['risk_amount']:.2f}
   • Прибыль: ${calc['potential_profit']:.2f}
   • Соотношение P/R: {calc['reward_risk_ratio']:.2f}
"""
            
            # Анализ корреляций
            analysis = multi_calculation.get('analysis', {})
            if analysis.get('correlations'):
                report += "\n🔗 АНАЛИЗ КОРРЕЛЯЦИЙ:\n"
                for corr_analysis in analysis['correlations'][:3]:
                    report += f"• {corr_analysis}\n"
            
            if analysis.get('volatility'):
                report += "\n⚡ АНАЛИЗ ВОЛАТИЛЬНОСТИ:\n"
                for vol_analysis in analysis['volatility'][:2]:
                    report += f"• {vol_analysis}\n"
            
            # Рекомендации
            report += "\n💡 СТРАТЕГИЧЕСКИЕ РЕКОМЕНДАЦИИ:\n"
            strategies = PortfolioAnalyzer.generate_portfolio_strategies(
                [{'instrument': t['instrument'], 'direction': t['direction']} for t in multi_calculation.get('trades', [])]
            )
            for strategy in strategies[:5]:
                report += f"{strategy}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Ошибка генерации многопозиционного отчета: {e}")
            return "Ошибка при генерации отчета"

# ========== ОСНОВНЫЕ ОБРАБОТЧИКИ ==========

@log_performance
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Главное меню v3.0"""
    try:
        logger.info(f"Команда /start от пользователя {update.effective_user.id}")
        
        if context.user_data:
            context.user_data.clear()
        
        user = update.message.from_user if update.message else update.callback_query.from_user
        user_name = user.first_name or "Трейдер"
        
        welcome_text = f"""
👋 *Привет, {user_name}!*

🎯 PRO Калькулятор Управления Рисками v3.0

⚡ *МОИ ВОЗМОЖНОСТИ:*
• ✅ Многопозиционный расчет (НОВОЕ!)
• ✅ Анализ корреляций между активами  
• ✅ Учет волатильности инструментов
• ✅ Портфельные стратегии и рекомендации
• ✅ Статистика за 5 лет по основным парам
• ✅ Умные рекомендации для PRO трейдеров

*Выберите опцию:*
"""
        
        user_id = user.id
        PortfolioManager.initialize_user_portfolio(user_id)
        
        keyboard = [
            [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("💼 Мой портфель", callback_data="portfolio")],
            [InlineKeyboardButton("🔮 Расширенная аналитика", callback_data="analytics")],
            [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")]
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
        logger.error(f"Критическая ошибка в start: {e}")
        if update.message:
            await update.message.reply_text(
                "🔄 Произошла ошибка. Попробуйте еще раз или используйте /start",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 Перезапустить", callback_data="main_menu")]
                ])
            )
        return MAIN_MENU

@log_performance
async def start_pro_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало профессионального расчета"""
    try:
        query = update.callback_query
        await query.answer()
        
        keyboard = [
            [InlineKeyboardButton("📈 Одна сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Многопозиционный расчет", callback_data="multi_trade")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            "📊 *ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ*\n\n"
            "🎯 Выберите тип расчета:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return SINGLE_OR_MULTI
    except Exception as e:
        logger.error(f"Ошибка в start_pro_calculation: {e}")
        await handle_error(update, context, e)

@log_performance
async def handle_single_or_multi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора типа расчета"""
    try:
        query = update.callback_query
        await query.answer()
        
        choice = query.data
        
        if choice == "single_trade":
            keyboard = []
            for key, value in INSTRUMENT_TYPES.items():
                keyboard.append([InlineKeyboardButton(value, callback_data=f"pro_type_{key}")])
            keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="pro_calculation")])
            
            await query.edit_message_text(
                "📊 *ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ - ОДНА СДЕЛКА*\n\n"
                "🎯 Выберите тип инструмента:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return INSTRUMENT_TYPE
            
        elif choice == "multi_trade":
            return await start_multi_trade_calculation(update, context)
            
    except Exception as e:
        logger.error(f"Ошибка в handle_single_or_multi: {e}")
        await handle_error(update, context, e)

@log_performance
async def start_multi_trade_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало многопозиционного расчета"""
    try:
        query = update.callback_query
        await query.answer()
        
        # Инициализация данных для многопозиционного расчета
        context.user_data['multi_trades'] = []
        context.user_data['current_trade_index'] = 0
        
        # Запрос общего депозита
        await query.edit_message_text(
            "📊 *МНОГОПОЗИЦИОННЫЙ РАСЧЕТ*\n\n"
            "💰 Введите общий размер депозита:",
            parse_mode='Markdown'
        )
        return MULTI_TRADE_MENU
        
    except Exception as e:
        logger.error(f"Ошибка в start_multi_trade_calculation: {e}")
        await handle_error(update, context, e)

@log_performance
async def handle_multi_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода депозита для многопозиционного расчета"""
    try:
        text = update.message.text
        
        is_valid, deposit, message = InputValidator.validate_number(text, 1, 1000000)
        
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n\n💰 Введите общий размер депозита:",
                parse_mode='Markdown'
            )
            return MULTI_TRADE_MENU
        
        context.user_data['deposit'] = deposit
        
        keyboard = []
        for leverage in LEVERAGES:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"multi_leverage_{leverage}")])
        
        await update.message.reply_text(
            f"💰 *Общий депозит:* ${deposit:,.2f}\n\n"
            "⚖️ Выберите кредитное плечо:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return MULTI_TRADE_MENU
        
    except Exception as e:
        logger.error(f"Ошибка в handle_multi_deposit: {e}")
        await handle_error(update, context, e)

@log_performance
async def handle_multi_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора плеча для многопозиционного расчета"""
    try:
        query = update.callback_query
        await query.answer()
        
        leverage = query.data.replace("multi_leverage_", "")
        context.user_data['leverage'] = leverage
        
        keyboard = []
        for risk in ['3%', '5%', '7%', '10%']:
            keyboard.append([InlineKeyboardButton(risk, callback_data=f"multi_risk_{risk.replace('%', '')}")])
        
        await query.edit_message_text(
            f"⚖️ *Общее плечо:* {leverage}\n\n"
            "🎯 Выберите общий уровень риска для портфеля:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return MULTI_TRADE_MENU
        
    except Exception as e:
        logger.error(f"Ошибка в handle_multi_leverage: {e}")
        await handle_error(update, context, e)

@log_performance
async def handle_multi_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора риска для многопозиционного расчета"""
    try:
        query = update.callback_query
        await query.answer()
        
        risk_percent = float(query.data.replace("multi_risk_", "")) / 100
        context.user_data['total_risk_percent'] = risk_percent
        
        # Начинаем добавление первой сделки
        await query.edit_message_text(
            f"🎯 *Общий риск портфеля:* {risk_percent*100}%\n\n"
            "📊 *Добавление сделки #1*\n\n"
            "Выберите тип инструмента:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💱 Форекс", callback_data="multi_type_forex")],
                [InlineKeyboardButton("₿ Крипто", callback_data="multi_type_crypto")],
                [InlineKeyboardButton("📈 Индексы", callback_data="multi_type_indices")],
                [InlineKeyboardButton("🛢️ Сырье", callback_data="multi_type_commodities")],
                [InlineKeyboardButton("🥇 Металлы", callback_data="multi_type_metals")]
            ])
        )
        return MULTI_INSTRUMENT
        
    except Exception as e:
        logger.error(f"Ошибка в handle_multi_risk: {e}")
        await handle_error(update, context, e)

@log_performance
async def handle_multi_instrument_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора типа инструмента для многопозиционной сделки"""
    try:
        query = update.callback_query
        await query.answer()
        
        instrument_type = query.data.replace("multi_type_", "")
        context.user_data['current_instrument_type'] = instrument_type
        
        presets = INSTRUMENT_PRESETS.get(instrument_type, [])
        
        keyboard = []
        for preset in presets:
            keyboard.append([InlineKeyboardButton(preset, callback_data=f"multi_preset_{preset}")])
        keyboard.append([InlineKeyboardButton("✏️ Ввести свой", callback_data="multi_custom")])
        
        await query.edit_message_text(
            f"📊 *{INSTRUMENT_TYPES[instrument_type]}*\n\n"
            "Выберите инструмент:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return MULTI_INSTRUMENT
        
    except Exception as e:
        logger.error(f"Ошибка в handle_multi_instrument_type: {e}")
        await handle_error(update, context, e)

@log_performance
async def handle_multi_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора инструмента для многопозиционной сделки"""
    try:
        query = update.callback_query
        await query.answer()
        
        if query.data == "multi_custom":
            await query.edit_message_text(
                "✏️ *Введите название инструмента:*\n\n"
                "Пример: EURUSD, BTCUSD, XAUUSD",
                parse_mode='Markdown'
            )
            return MULTI_INSTRUMENT
        else:
            instrument = query.data.replace("multi_preset_", "")
            context.user_data['current_instrument'] = instrument
            
            await query.edit_message_text(
                f"🎯 *Инструмент:* {instrument}\n\n"
                "💎 Введите цену входа:",
                parse_mode='Markdown'
            )
            return MULTI_ENTRY
            
    except Exception as e:
        logger.error(f"Ошибка в handle_multi_instrument: {e}")
        await handle_error(update, context, e)

@log_performance
async def handle_multi_custom_instrument(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка пользовательского инструмента для многопозиционной сделки"""
    try:
        instrument = update.message.text.upper().strip()
        
        is_valid, validated_instrument, message = InputValidator.validate_instrument(instrument)
        
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n\n✏️ Введите название инструмента:",
                parse_mode='Markdown'
            )
            return MULTI_INSTRUMENT
        
        context.user_data['current_instrument'] = validated_instrument
        
        await update.message.reply_text(
            f"🎯 *Инструмент:* {validated_instrument}\n\n"
            "💎 Введите цену входа:",
            parse_mode='Markdown'
        )
        return MULTI_ENTRY
        
    except Exception as e:
        logger.error(f"Ошибка в handle_multi_custom_instrument: {e}")
        await handle_error(update, context, e)

@log_performance
async def handle_multi_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка цены входа для многопозиционной сделки"""
    try:
        text = update.message.text
        
        is_valid, entry_price, message = InputValidator.validate_price(text)
        
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n\n💎 Введите цену входа:",
                parse_mode='Markdown'
            )
            return MULTI_ENTRY
        
        context.user_data['current_entry_price'] = entry_price
        
        keyboard = [
            [InlineKeyboardButton("📈 BUY", callback_data="multi_direction_BUY")],
            [InlineKeyboardButton("📉 SELL", callback_data="multi_direction_SELL")]
        ]
        
        await update.message.reply_text(
            f"💎 *Цена входа:* {entry_price}\n\n"
            "Выберите направление сделки:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return MULTI_DIRECTION
        
    except Exception as e:
        logger.error(f"Ошибка в handle_multi_entry: {e}")
        await handle_error(update, context, e)

@log_performance
async def handle_multi_direction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка направления для многопозиционной сделки"""
    try:
        query = update.callback_query
        await query.answer()
        
        direction = query.data.replace("multi_direction_", "")
        context.user_data['current_direction'] = direction
        
        direction_text = "ниже" if direction == "BUY" else "выше"
        
        await query.edit_message_text(
            f"📊 *Направление:* {direction}\n\n"
            f"🛑 Введите цену стоп-лосса ({direction_text} цены входа):",
            parse_mode='Markdown'
        )
        return MULTI_STOPLOSS
        
    except Exception as e:
        logger.error(f"Ошибка в handle_multi_direction: {e}")
        await handle_error(update, context, e)

@log_performance
async def handle_multi_stoploss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка стоп-лосса для многопозиционной сделки"""
    try:
        text = update.message.text
        
        is_valid, stop_loss, message = InputValidator.validate_price(text)
        
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n\n🛑 Введите цену стоп-лосса:",
                parse_mode='Markdown'
            )
            return MULTI_STOPLOSS
        
        context.user_data['current_stop_loss'] = stop_loss
        
        await update.message.reply_text(
            f"🛑 *Стоп-лосс:* {stop_loss}\n\n"
            "🎯 Введите цену тейк-профита:",
            parse_mode='Markdown'
        )
        return MULTI_TAKEPROFIT
        
    except Exception as e:
        logger.error(f"Ошибка в handle_multi_stoploss: {e}")
        await handle_error(update, context, e)

@log_performance
async def handle_multi_takeprofit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка тейк-профита для многопозиционной сделки"""
    try:
        text = update.message.text
        
        is_valid, take_profit, message = InputValidator.validate_price(text)
        
        if not is_valid:
            await update.message.reply_text(
                f"{message}\n\n🎯 Введите цену тейк-профита:",
                parse_mode='Markdown'
            )
            return MULTI_TAKEPROFIT
        
        # Сохраняем текущую сделку
        current_trade = {
            'instrument': context.user_data['current_instrument'],
            'instrument_type': context.user_data['current_instrument_type'],
            'entry_price': context.user_data['current_entry_price'],
            'direction': context.user_data['current_direction'],
            'stop_loss': context.user_data['current_stop_loss'],
            'take_profit': take_profit
        }
        
        multi_trades = context.user_data.get('multi_trades', [])
        multi_trades.append(current_trade)
        context.user_data['multi_trades'] = multi_trades
        context.user_data['current_trade_index'] = len(multi_trades)
        
        keyboard = [
            [InlineKeyboardButton("➕ Добавить еще сделку", callback_data="multi_add_more")],
            [InlineKeyboardButton("📊 Рассчитать портфель", callback_data="multi_calculate")],
            [InlineKeyboardButton("🔁 Начать заново", callback_data="multi_trade")]
        ]
        
        trades_text = "\n".join([f"{i+1}. {trade['instrument']} {trade['direction']}" for i, trade in enumerate(multi_trades)])
        
        await update.message.reply_text(
            f"✅ *Сделка #{len(multi_trades)} добавлена!*\n\n"
            f"📋 *Текущие сделки:*\n{trades_text}\n\n"
            "Выберите действие:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return MULTI_ADD_MORE
        
    except Exception as e:
        logger.error(f"Ошибка в handle_multi_takeprofit: {e}")
        await handle_error(update, context, e)

@log_performance
async def handle_multi_add_more(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка добавления еще одной сделки"""
    try:
        query = update.callback_query
        await query.answer()
        
        await query.edit_message_text(
            f"📊 *Добавление сделки #{context.user_data['current_trade_index'] + 1}*\n\n"
            "Выберите тип инструмента:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💱 Форекс", callback_data="multi_type_forex")],
                [InlineKeyboardButton("₿ Крипто", callback_data="multi_type_crypto")],
                [InlineKeyboardButton("📈 Индексы", callback_data="multi_type_indices")],
                [InlineKeyboardButton("🛢️ Сырье", callback_data="multi_type_commodities")],
                [InlineKeyboardButton("🥇 Металлы", callback_data="multi_type_metals")]
            ])
        )
        return MULTI_INSTRUMENT
        
    except Exception as e:
        logger.error(f"Обработка в handle_multi_add_more: {e}")
        await handle_error(update, context, e)

@log_performance
async def handle_multi_calculate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Расчет и показ результатов многопозиционного портфеля"""
    try:
        query = update.callback_query
        await query.answer()
        
        multi_trades = context.user_data.get('multi_trades', [])
        deposit = context.user_data.get('deposit', 0)
        leverage = context.user_data.get('leverage', '1:100')
        total_risk_percent = context.user_data.get('total_risk_percent', 0.05)
        
        if not multi_trades:
            await query.edit_message_text(
                "❌ Нет сделок для расчета. Добавьте хотя бы одну сделку.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="multi_trade")]
                ])
            )
            return MULTI_TRADE_MENU
        
        # Расчет многопозиционного портфеля
        multi_calculation = FastRiskCalculator.calculate_multi_position(
            deposit=deposit,
            leverage=leverage,
            trades=multi_trades,
            total_risk_percent=total_risk_percent
        )
        
        if not multi_calculation:
            await query.edit_message_text(
                "❌ Ошибка при расчете портфеля. Попробуйте с другими параметрами.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="multi_trade")]
                ])
            )
            return MULTI_TRADE_MENU
        
        # Генерация отчета
        report = ReportGenerator.generate_multi_trade_report(multi_calculation, context.user_data)
        
        keyboard = [
            [InlineKeyboardButton("💾 Сохранить портфель", callback_data="multi_save_portfolio")],
            [InlineKeyboardButton("📊 Новый расчет", callback_data="multi_trade")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            report,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        # Сохраняем расчет для возможного сохранения
        context.user_data['last_multi_calculation'] = multi_calculation
        
        return ConversationHandler.END
        
    except Exception as e:
        logger.error(f"Ошибка в handle_multi_calculate: {e}")
        await handle_error(update, context, e)

@log_performance
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """PRO Инструкции v3.0"""
    info_text = """
📚 *PRO ИНСТРУКЦИИ v3.0*

🎯 *ОСНОВНЫЕ ВОЗМОЖНОСТИ:*

*1. Многопозиционный расчет (НОВОЕ!)*
   • Единый депозит для группы сделок
   • Общее кредитное плечо для портфеля
   • Автоматический анализ корреляций
   • Умное распределение риска
   • Выгрузка отчетов по группе сделок

*2. Профессиональный анализ*
   • Учет волатильности инструментов
   • Анализ корреляций между активами
   • Статистика за 5 лет по основным парам
   • Рекомендации по портфельным стратегиям

*3. Управление портфелем*
   • Отслеживание всех сделок
   • Анализ эффективности торговли
   • Рекомендации по улучшению
   • Детальная статистика performance

💡 *PRO СОВЕТЫ ДЛЯ ТРЕЙДЕРОВ:*

• *Управление риском*: Не рискуйте более 2-5% депозита на одну сделку
• *Соотношение прибыль/риск*: Стремитесь к 1:2 или выше
• *Диверсификация*: Распределяйте капитал между некоррелируемыми активами
• *Корреляции*: Учитывайте взаимосвязи для снижения общего риска

📊 *МНОГОПОЗИЦИОННАЯ СТРАТЕГИЯ:*
   • Оптимальное количество сделок: 3-5
   • Распределение по типам активов
   • Учет корреляций для хеджирования
   • Балансировка риска по волатильности

🔧 *ТЕХНИЧЕСКАЯ ПОДДЕРЖКА:*
По вопросам работы бота обращайтесь к разработчику.
"""
    
    await update.callback_query.edit_message_text(
        info_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
    )

# ========== ОБРАБОТЧИКИ ОШИБОК И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

@log_performance
async def handle_error(update: Update, context: ContextTypes.DEFAULT_TYPE, error: Exception = None):
    """Обработка ошибок"""
    try:
        error_msg = "❌ Произошла непредвиденная ошибка. Пожалуйста, попробуйте еще раз."
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                error_msg,
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
            )
        elif update.message:
            await update.message.reply_text(
                error_msg,
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
            )
    except Exception as e:
        logger.error(f"Ошибка в обработчике ошибок: {e}")

@log_performance
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена операции"""
    await update.message.reply_text(
        "Операция отменена.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]])
    )
    return ConversationHandler.END

@log_performance
async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка неизвестных команд"""
    await update.message.reply_text(
        "❌ Неизвестная команда. Используйте кнопки меню для навигации.",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Профессиональный расчет", callback_data="pro_calculation")],
            [InlineKeyboardButton("💼 Портфель", callback_data="portfolio")],
            [InlineKeyboardButton("🔮 Аналитика", callback_data="analytics")],
            [InlineKeyboardButton("🚀 Главное меню", callback_data="main_menu")]
        ])
    )

# ========== HTTP СЕРВЕР И WEBHOOKS ==========

async def health_check(request):
    """Health check endpoint для Render"""
    return web.Response(text="OK", status=200)

async def handle_webhook(request, application):
    """Обработчик вебхуков от Telegram"""
    try:
        data = await request.json()
        update = Update.de_json(data, application.bot)
        await application.process_update(update)
        return web.Response(status=200)
    except Exception as e:
        logger.error(f"Ошибка обработки вебхука: {e}")
        return web.Response(status=500)

async def set_webhook(application):
    """Установка вебхука"""
    if not WEBHOOK_URL:
        logger.warning("WEBHOOK_URL не установлен, используем polling")
        return False
    
    try:
        webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
        await application.bot.set_webhook(
            url=webhook_url,
            drop_pending_updates=True
        )
        logger.info(f"Webhook установлен: {webhook_url}")
        return True
    except Exception as e:
        logger.error(f"Ошибка установки webhook: {e}")
        return False

async def start_http_server(application):
    """Запуск HTTP сервера"""
    app = web.Application()
    
    app.router.add_get('/', health_check)
    app.router.add_get('/health', health_check)
    app.router.add_post(WEBHOOK_PATH, lambda request: handle_webhook(request, application))
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    
    logger.info(f"HTTP сервер запущен на порту {PORT}")
    return runner

async def start_webhook_mode(application):
    """Запуск бота в режиме webhook"""
    try:
        webhook_set = await set_webhook(application)
        if not webhook_set:
            logger.error("Не удалось установить вебхук")
            return False
        
        runner = await start_http_server(application)
        
        logger.info("✅ Бот запущен в режиме Webhook!")
        
        while True:
            await asyncio.sleep(3600)
            
    except Exception as e:
        logger.error(f"Ошибка в режиме webhook: {e}")
        return False

def create_application():
    """Создание и настройка приложения"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ Токен бота не найден!")
        return None

    logger.info("🚀 Запуск ПРОФЕССИОНАЛЬНОГО калькулятора рисков v3.0...")
    
    application = Application.builder().token(token).build()

    # Обработчики для профессионального расчета (одиночные сделки)
    pro_calc_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(start_pro_calculation, pattern='^pro_calculation$')],
        states={
            SINGLE_OR_MULTI: [CallbackQueryHandler(handle_single_or_multi)],
            INSTRUMENT_TYPE: [CallbackQueryHandler(pro_select_instrument_type)],
            CUSTOM_INSTRUMENT: [
                CallbackQueryHandler(pro_select_instrument),
                MessageHandler(filters.TEXT & ~filters.COMMAND, pro_handle_custom_instrument)
            ],
            DIRECTION: [CallbackQueryHandler(pro_select_direction)],
            RISK_PERCENT: [CallbackQueryHandler(pro_select_risk)],
            DEPOSIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, pro_handle_deposit)],
            LEVERAGE: [CallbackQueryHandler(pro_select_leverage)],
            ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, pro_handle_entry)],
            STOP_LOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, pro_handle_stop_loss)],
            TAKE_PROFIT_SINGLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, pro_handle_take_profit)],
        },
        fallbacks=[CommandHandler('cancel', cancel), CommandHandler('start', start), CallbackQueryHandler(start, pattern='^main_menu$')]
    )

    # Обработчики для многопозиционного расчета
    multi_calc_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(start_multi_trade_calculation, pattern='^multi_trade$')],
        states={
            MULTI_TRADE_MENU: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_multi_deposit),
                CallbackQueryHandler(handle_multi_leverage, pattern='^multi_leverage_'),
                CallbackQueryHandler(handle_multi_risk, pattern='^multi_risk_')
            ],
            MULTI_INSTRUMENT: [
                CallbackQueryHandler(handle_multi_instrument_type, pattern='^multi_type_'),
                CallbackQueryHandler(handle_multi_instrument, pattern='^multi_preset_|^multi_custom$'),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_multi_custom_instrument)
            ],
            MULTI_ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_multi_entry)],
            MULTI_DIRECTION: [CallbackQueryHandler(handle_multi_direction, pattern='^multi_direction_')],
            MULTI_STOPLOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_multi_stoploss)],
            MULTI_TAKEPROFIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_multi_takeprofit)],
            MULTI_ADD_MORE: [
                CallbackQueryHandler(handle_multi_add_more, pattern='^multi_add_more$'),
                CallbackQueryHandler(handle_multi_calculate, pattern='^multi_calculate$')
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel), CommandHandler('start', start), CallbackQueryHandler(start, pattern='^main_menu$')]
    )

    # Регистрируем обработчики
    application.add_handler(CommandHandler('start', start))
    application.add_handler(pro_calc_conv)
    application.add_handler(multi_calc_conv)
    
    # Упрощенный обработчик состояний
    conv_handler = ConversationHandler(
        entry_points=[],
        states={
            MAIN_MENU: [CallbackQueryHandler(handle_main_menu)],
            PORTFOLIO_MENU: [CallbackQueryHandler(handle_main_menu)],
            ANALYTICS_MENU: [CallbackQueryHandler(handle_main_menu)],
        },
        fallbacks=[CommandHandler('start', start), CommandHandler('cancel', cancel)],
        allow_reentry=True
    )
    application.add_handler(conv_handler)
    
    # Обработчики команд
    application.add_handler(CommandHandler('portfolio', portfolio_command))
    application.add_handler(CommandHandler('analytics', analytics_command))
    application.add_handler(CommandHandler('info', pro_info_command))
    
    # Обработчик неизвестных команд
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    
    # Обработчик неизвестных сообщений
    application.add_handler(MessageHandler(filters.TEXT, unknown_command))
    
    return application

def main():
    """Основная функция запуска"""
    application = create_application()
    if not application:
        return
    
    if WEBHOOK_URL:
        logger.info("🚀 Запуск в режиме Webhook...")
        asyncio.run(start_webhook_mode(application))
    else:
        logger.info("🚀 Запуск в режиме Polling...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
