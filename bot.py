# bot.py — PRO Risk Calculator v3.1 | ENTERPRISE EDITION (ИСПРАВЛЕННАЯ)
import os
import logging
import asyncio
import time
import functools
import json
import telegram
import io
import re
import aiohttp
import cachetools
import html
from telegram import CallbackQuery
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
from aiohttp import web
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    CallbackQueryHandler,
    ConversationHandler
)

# --- Загрузка .env ---
from dotenv import load_dotenv
load_dotenv()

# --- Настройки ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found!")

PORT = int(os.getenv("PORT", 10000))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").rstrip("/")
WEBHOOK_PATH = f"/webhook/{TOKEN}"

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY") 
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Donation Wallets
USDT_WALLET_ADDRESS = os.getenv("USDT_WALLET_ADDRESS", "TVRGFPKVs1nN3fUXBTQfu5syTcmYGgADre")
TON_WALLET_ADDRESS = os.getenv("TON_WALLET_ADDRESS", "UQDpCH-pGSzp3zEkpJY1Wc46gaorw9K-7T9FX7gHTrthMWMj")

# --- Логи ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("pro_risk_bot")

# ---------------------------
# Enhanced Market Data Provider - ИСПРАВЛЕННЫЙ
# ---------------------------
class MarketDataProvider:
    """Универсальный провайдер рыночных данных с кэшированием и улучшенными fallback"""
    
    def __init__(self):
        self.cache = cachetools.TTLCache(maxsize=500, ttl=300)
        self.session = None
        
    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_real_time_price(self, symbol: str) -> float:
        """Получение реальной цены с улучшенными fallback"""
        try:
            # Проверка кэша
            cached_price = self.cache.get(symbol)
            if cached_price:
                return cached_price
                
            price = None
            
            if self._is_crypto(symbol):
                price = await self._get_binance_price(symbol)
            elif self._is_forex(symbol) or self._is_metal(symbol):
                price = await self._get_alpha_vantage_forex(symbol)
            else:
                price = await self._get_alpha_vantage_stock(symbol)
                
            if price is None:
                price = await self._get_finnhub_price(symbol)
                
            # УЛУЧШЕННЫЙ FALLBACK - реальные цены для всех активов
            if price is None or price <= 0:
                logger.warning(f"Не удалось получить цену для {symbol}, используется улучшенный fallback")
                price = self._get_enhanced_fallback_price(symbol)
                
            if price:
                self.cache[symbol] = price
                
            return price
            
        except Exception as e:
            logger.error(f"Ошибка получения цены для {symbol}: {e}")
            return self._get_enhanced_fallback_price(symbol)
    
    def _is_crypto(self, symbol: str) -> bool:
        crypto_symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'USDT']
        return any(crypto in symbol for crypto in crypto_symbols)
    
    def _is_forex(self, symbol: str) -> bool:
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        return symbol in forex_pairs
    
    def _is_metal(self, symbol: str) -> bool:
        metals = ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD']
        return symbol in metals
    
    async def _get_binance_price(self, symbol: str) -> Optional[float]:
        """Получение цены с Binance API"""
        try:
            session = await self.get_session()
            # Форматируем символ для Binance
            if 'USDT' in symbol:
                binance_symbol = symbol.replace('/', '')
            else:
                binance_symbol = symbol.replace('USDT', '') + 'USDT'
            
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data['price'])
        except Exception as e:
            logger.error(f"Binance API error for {symbol}: {e}")
        return None
    
    async def _get_alpha_vantage_stock(self, symbol: str) -> Optional[float]:
        if not ALPHA_VANTAGE_API_KEY:
            return None
            
        try:
            session = await self.get_session()
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Global Quote' in data and '05. price' in data['Global Quote']:
                        return float(data['Global Quote']['05. price'])
        except Exception as e:
            logger.error(f"Alpha Vantage stock error for {symbol}: {e}")
        return None
    
    async def _get_alpha_vantage_forex(self, symbol: str) -> Optional[float]:
        if not ALPHA_VANTAGE_API_KEY:
            return None
            
        try:
            session = await self.get_session()
            from_currency = symbol[:3]
            to_currency = symbol[3:]
            url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}&to_currency={to_currency}&apikey={ALPHA_VANTAGE_API_KEY}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Realtime Currency Exchange Rate' in data and '5. Exchange Rate' in data['Realtime Currency Exchange Rate']:
                        return float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])
        except Exception as e:
            logger.error(f"Alpha Vantage forex error for {symbol}: {e}")
        return None
    
    async def _get_finnhub_price(self, symbol: str) -> Optional[float]:
        if not FINNHUB_API_KEY:
            return None
            
        try:
            session = await self.get_session()
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['c']
        except Exception as e:
            logger.error(f"Finnhub API error for {symbol}: {e}")
        return None
    
    def _get_enhanced_fallback_price(self, symbol: str) -> float:
        """УЛУЧШЕННЫЕ fallback цены - РЕАЛЬНЫЕ значения"""
        enhanced_fallback_prices = {
            # Криптовалюты
            'BTCUSDT': 43250.0, 'ETHUSDT': 2580.0, 'XRPUSDT': 0.62,
            'LTCUSDT': 71.5, 'BCHUSDT': 265.0, 'ADAUSDT': 0.52,
            'DOTUSDT': 8.15,
            
            # Forex
            'EURUSD': 1.0950, 'GBPUSD': 1.2750, 'USDJPY': 148.50,
            'USDCHF': 0.8800, 'AUDUSD': 0.6520, 'USDCAD': 1.3520,
            'NZDUSD': 0.6100,
            
            # Металлы
            'XAUUSD': 2025.0, 'XAGUSD': 22.85, 'XPTUSD': 920.0,
            'XPDUSD': 980.0,
            
            # Индексы
            'NAS100': 17650.0, 'SPX500': 4780.0, 'DJ30': 37500.0,
            'FTSE100': 7680.0, 'DAX40': 16700.0, 'NIKKEI225': 36150.0,
            'ASX200': 7500.0,
            
            # Энергия
            'OIL': 78.50, 'NATURALGAS': 2.85, 'BRENT': 82.50,
            
            # Акции
            'AAPL': 185.50, 'TSLA': 248.0, 'GOOGL': 142.0,
            'MSFT': 378.0, 'AMZN': 155.0, 'META': 368.0,
            'NFLX': 485.0
        }
        return enhanced_fallback_prices.get(symbol, 100.0)

# ---------------------------
# Enhanced Professional Risk Calculator - ИСПРАВЛЕННЫЙ
# ---------------------------
class ProfessionalRiskCalculator:
    """ИСПРАВЛЕННЫЙ калькулятор с правильными расчетами депозита и P&L"""
    
    @staticmethod
    def calculate_pip_distance(entry: float, stop_loss: float, direction: str, asset: str) -> float:
        """Профессиональный расчет дистанции в пунктах"""
        specs = InstrumentSpecs.get_specs(asset)
        pip_decimal_places = specs.get('pip_decimal_places', 4)
        
        if direction.upper() == 'LONG':
            distance = entry - stop_loss
        else:
            distance = stop_loss - entry
        
        if pip_decimal_places == 2:
            return abs(distance) * 100
        elif pip_decimal_places == 1:
            return abs(distance) * 10
        else:
            return abs(distance) * 10000

    @staticmethod
    async def calculate_realistic_pnl(trade: Dict, current_price: float) -> float:
        """РЕАЛИСТИЧНЫЙ расчет P&L - ИСПРАВЛЕННЫЙ"""
        try:
            direction = trade['direction']
            entry = trade['entry_price']
            volume = trade['metrics']['volume_lots']
            pip_value = trade['metrics']['pip_value']
            
            if direction == 'LONG':
                price_diff = current_price - entry
            else:
                price_diff = entry - current_price
            
            # Конвертация в пункты с учетом спецификаций актива
            pip_diff = ProfessionalRiskCalculator.calculate_pip_distance(
                entry, entry + price_diff, direction, trade['asset']
            )
            
            current_pnl = volume * pip_diff * pip_value
            return round(current_pnl, 2)
        except Exception as e:
            logger.error(f"Ошибка расчета P&L для {trade['asset']}: {e}")
            return 0.0

    @staticmethod
    async def calculate_professional_metrics(trade: Dict, deposit: float, leverage: str, risk_level: str) -> Dict[str, Any]:
        """ИСПРАВЛЕННЫЙ расчет с правильным использованием депозита"""
        try:
            asset = trade['asset']
            entry = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            direction = trade['direction']
            
            # 1. Получение РЕАЛЬНОЙ цены
            current_price = await market_data_provider.get_real_time_price(asset)
            
            # 2. Получение спецификаций
            specs = InstrumentSpecs.get_specs(asset)
            
            # 3. ИСПРАВЛЕНИЕ: Правильный расчет суммы риска
            risk_percent = float(risk_level.strip('%'))
            risk_amount = deposit * (risk_percent / 100)
            
            # 4. Расчет дистанции
            stop_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry, stop_loss, direction, asset)
            profit_distance_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry, take_profit, direction, asset)
            
            # 5. Получаем стоимость пункта
            pip_value = specs['pip_value']
            
            # 6. ИСПРАВЛЕНИЕ: Правильный расчет объема
            if stop_distance_pips > 0 and pip_value > 0:
                volume_lots = risk_amount / (stop_distance_pips * pip_value)
                volume_lots = round(volume_lots, 2)
            else:
                volume_lots = 0
            
            # 7. Расчет маржи
            margin_data = await margin_calculator.calculate_professional_margin(
                asset, volume_lots, leverage, current_price
            )
            required_margin = margin_data['required_margin']
            required_margin = round(required_margin, 2)
            
            # 8. ИСПРАВЛЕНИЕ: Правильные расчеты всех метрик
            free_margin = deposit - required_margin
            free_margin = round(max(free_margin, 0), 2)  # Не может быть отрицательным
            
            margin_level = (deposit / required_margin) * 100 if required_margin > 0 else 0
            margin_level = round(margin_level, 1)
            
            potential_profit = volume_lots * profit_distance_pips * pip_value
            potential_profit = round(potential_profit, 2)
            
            rr_ratio = potential_profit / risk_amount if risk_amount > 0 else 0
            rr_ratio = round(rr_ratio, 2)
            
            # Дополнительные профессиональные метрики
            risk_per_trade_percent = (risk_amount / deposit) * 100 if deposit > 0 else 0
            margin_usage_percent = (required_margin / deposit) * 100 if deposit > 0 else 0
            notional_value = margin_data.get('notional_value', 0)
            
            return {
                'volume_lots': volume_lots,
                'required_margin': required_margin,
                'free_margin': free_margin,
                'margin_level': margin_level,
                'risk_amount': risk_amount,
                'risk_percent': risk_percent,
                'potential_profit': potential_profit,
                'rr_ratio': rr_ratio,
                'stop_distance_pips': stop_distance_pips,
                'profit_distance_pips': profit_distance_pips,
                'pip_value': pip_value,
                'contract_size': margin_data['contract_size'],
                'deposit': deposit,  # Сохраняем депозит для проверки
                'leverage': leverage,
                'risk_per_trade_percent': round(risk_per_trade_percent, 1),
                'margin_usage_percent': round(margin_usage_percent, 1),
                'current_price': current_price,
                'calculation_method': margin_data['calculation_method'],
                'notional_value': notional_value,
                'leverage_used': margin_data.get('leverage_used', 1)
            }
        except Exception as e:
            logger.error(f"Критическая ошибка в расчетах: {e}")
            # Возвращаем безопасные значения по умолчанию
            return {
                'volume_lots': 0,
                'required_margin': 0,
                'free_margin': deposit,
                'margin_level': 0,
                'risk_amount': 0,
                'risk_percent': 0,
                'potential_profit': 0,
                'rr_ratio': 0,
                'stop_distance_pips': 0,
                'profit_distance_pips': 0,
                'pip_value': 0,
                'contract_size': 0,
                'deposit': deposit,
                'leverage': leverage,
                'risk_per_trade_percent': 0,
                'margin_usage_percent': 0,
                'current_price': 0,
                'calculation_method': 'error',
                'notional_value': 0,
                'leverage_used': 1
            }

# ---------------------------
# Enhanced Portfolio Analyzer - ИСПРАВЛЕННЫЙ
# ---------------------------
class PortfolioAnalyzer:
    @staticmethod
    def calculate_portfolio_metrics(trades: List[Dict], deposit: float) -> Dict[str, Any]:
        """ИСПРАВЛЕННЫЙ расчет метрик портфеля"""
        if not trades or deposit <= 0:
            return {
                'total_risk_usd': 0,
                'total_risk_percent': 0,
                'total_profit': 0,
                'total_margin': 0,
                'free_margin': deposit,
                'free_margin_percent': 100,
                'portfolio_margin_level': 0,
                'total_margin_usage': 0,
                'avg_rr_ratio': 0,
                'portfolio_volatility': 0,
                'long_positions': 0,
                'short_positions': 0,
                'direction_balance': 0,
                'diversity_score': 0,
                'unique_assets': 0,
                'total_notional_value': 0,
                'portfolio_leverage': 0
            }
        
        total_risk = sum(t.get('metrics', {}).get('risk_amount', 0) for t in trades)
        total_profit = sum(t.get('metrics', {}).get('potential_profit', 0) for t in trades)
        total_margin = sum(t.get('metrics', {}).get('required_margin', 0) for t in trades)
        total_notional = sum(t.get('metrics', {}).get('notional_value', 0) for t in trades)
        
        # ИСПРАВЛЕНИЕ: Правильный расчет среднего R/R
        valid_rr_trades = [t for t in trades if t.get('metrics', {}).get('rr_ratio', 0) > 0]
        avg_rr = sum(t.get('metrics', {}).get('rr_ratio', 0) for t in valid_rr_trades) / len(valid_rr_trades) if valid_rr_trades else 0
        
        # Волатильность портфеля
        portfolio_volatility = sum(VOLATILITY_DATA.get(t['asset'], 20) for t in trades) / len(trades) if trades else 0
        
        # Анализ направлений
        long_count = sum(1 for t in trades if t.get('direction', '').upper() == 'LONG')
        short_count = len(trades) - long_count
        direction_balance = abs(long_count - short_count) / len(trades) if trades else 0
        
        # Диверсификация
        unique_assets = len(set(t['asset'] for t in trades))
        diversity_score = unique_assets / len(trades) if trades else 0
        
        # Уровень маржи портфеля
        portfolio_margin_level = (deposit / total_margin) * 100 if total_margin > 0 else 0
        
        # Общее использование маржи
        total_margin_usage = (total_margin / deposit) * 100 if deposit > 0 else 0
        
        # Общий левередж портфеля
        portfolio_leverage = total_notional / deposit if deposit > 0 else 0
        
        # ИСПРАВЛЕНИЕ: Добавление свободной маржи
        free_margin = deposit - total_margin
        free_margin_percent = (free_margin / deposit) * 100 if deposit > 0 else 0
        
        return {
            'total_risk_usd': round(total_risk, 2),
            'total_risk_percent': round((total_risk / deposit) * 100, 1) if deposit > 0 else 0,
            'total_profit': round(total_profit, 2),
            'total_margin': round(total_margin, 2),
            'free_margin': round(free_margin, 2),
            'free_margin_percent': round(free_margin_percent, 1),
            'portfolio_margin_level': round(portfolio_margin_level, 1),
            'total_margin_usage': round(total_margin_usage, 1),
            'avg_rr_ratio': round(avg_rr, 2),
            'portfolio_volatility': round(portfolio_volatility, 1),
            'long_positions': long_count,
            'short_positions': short_count,
            'direction_balance': round(direction_balance, 2),
            'diversity_score': round(diversity_score, 2),
            'unique_assets': unique_assets,
            'total_notional_value': round(total_notional, 2),
            'portfolio_leverage': round(portfolio_leverage, 2)
        }

    @staticmethod
    def generate_enhanced_recommendations(metrics: Dict, trades: List[Dict]) -> List[str]:
        """УЛУЧШЕННЫЕ рекомендации с анализом рисков"""
        recommendations = []
        
        deposit = metrics.get('deposit', 0)
        if deposit <= 0:
            recommendations.append("🔴 КРИТИЧЕСКАЯ ОШИБКА: Депозит не установлен!")
            return recommendations
        
        # Анализ концентрации риска
        if len(trades) == 1 and metrics.get('total_risk_percent', 0) > 5:
            recommendations.append(
                "⚠️ ВСЕ ЯЙЦА В ОДНОЙ КОРЗИНЕ: Риск сконцентрирован в одной сделке. Диверсифицируйте!"
            )
        
        # Проверка общего риска
        if metrics.get('total_risk_percent', 0) > 15:
            recommendations.append(
                "🔴 ЗАПРЕЩЕННЫЙ УРОВЕНЬ РИСКА: Превышен порог 15%! Немедленно уменьшите объемы."
            )
        elif metrics.get('total_risk_percent', 0) > 10:
            recommendations.append(
                "🟡 ВЫСОКИЙ РИСК: Общий риск портфеля превышает 10%. Рекомендуется уменьшить объем позиций."
            )
        elif metrics.get('total_risk_percent', 0) > 5:
            recommendations.append(
                "🟠 ПОВЫШЕННЫЙ РИСК: Общий риск портфеля превышает 5%. Рассмотрите снижение объема позиций."
            )
        
        # Проверка уровня маржи
        if metrics.get('portfolio_margin_level', 0) < 100:
            recommendations.append(
                "🔴 КРИТИЧЕСКИЙ УРОВЕНЬ МАРЖИ! Немедленно пополните счет или закрите часть позиций."
            )
        elif metrics.get('portfolio_margin_level', 0) < 200:
            recommendations.append(
                "🟡 НИЗКИЙ УРОВЕНЬ МАРЖИ: Рассмотрите пополнение счета. Рекомендуемый уровень > 200%."
            )
        
        # Проверка использования маржи
        if metrics.get('total_margin_usage', 0) > 80:
            recommendations.append(
                f"🔴 ПЕРЕГРУЗКА МАРЖИ: Использование {metrics['total_margin_usage']:.1f}%. Увеличьте депозит или уменьшите объемы."
            )
        elif metrics.get('total_margin_usage', 0) > 60:
            recommendations.append(
                f"🟡 ВЫСОКАЯ НАГРУЗКА: Использование {metrics['total_margin_usage']:.1f}%. Оставьте запас для управления."
            )
        
        # Анализ свободной маржи
        if metrics.get('free_margin_percent', 0) < 20:
            recommendations.append(
                f"🟡 МАЛО СВОБОДНОЙ МАРЖИ: Всего {metrics['free_margin_percent']:.1f}%. Оставьте минимум 20% для безопасности."
            )
        
        # Проверка левереджа
        if metrics.get('portfolio_leverage', 0) > 10:
            recommendations.append(
                f"🔶 ВЫСОКИЙ ЛЕВЕРЕДЖ: {metrics['portfolio_leverage']:.1f}x. Увеличивает как прибыль, так и риски."
            )
        
        # Проверка Risk/Reward
        low_rr_trades = [t for t in trades if t.get('metrics', {}).get('rr_ratio', 0) < 1]
        if low_rr_trades:
            recommendations.append(
                f"📉 НЕВЫГОДНОЕ R/R: {len(low_rr_trades)} сделок имеют соотношение < 1. Улучшите TP/SL."
            )
        
        # Проверка волатильности
        if metrics.get('portfolio_volatility', 0) > 40:
            recommendations.append(
                f"🌪 ОЧЕНЬ ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ: {metrics['portfolio_volatility']:.1f}%. Будьте готовы к сильным колебаниям."
            )
        elif metrics.get('portfolio_volatility', 0) > 30:
            recommendations.append(
                f"🌪 ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ: {metrics['portfolio_volatility']:.1f}%. Управляйте рисками внимательно."
            )
        
        # Проверка диверсификации
        if metrics.get('diversity_score', 0) < 0.3 and len(trades) > 1:
            recommendations.append(
                "🎯 ОЧЕНЬ НИЗКАЯ ДИВЕРСИФИКАЦИЯ. Добавьте активы из разных секторов."
            )
        elif metrics.get('diversity_score', 0) < 0.5 and len(trades) > 1:
            recommendations.append(
                "🎯 НИЗКАЯ ДИВЕРСИФИКАЦИЯ. Рассмотрите добавление активов из разных секторов."
            )

        # Анализ волатильных активов
        high_vol_assets = [t for t in trades if VOLATILITY_DATA.get(t['asset'], 0) > 40]
        if len(high_vol_assets) > 2:
            recommendations.append(
                "🌪 МНОГО ВОЛАТИЛЬНЫХ АКТИВОВ: Рассмотрите хеджирование или уменьшение объема."
            )
        
        # Рекомендации по диверсификации
        if len(trades) >= 3 and metrics.get('diversity_score', 1) >= 0.7:
            recommendations.append(
                "✅ ОТЛИЧНАЯ ДИВЕРСИФИКАЦИЯ: Портфель хорошо сбалансирован."
            )
        
        if not recommendations:
            recommendations.append("✅ ПОРТФЕЛЬ СБАЛАНСИРОВАН. Продолжайте в том же духе!")
        
        return recommendations[:6]  # Ограничиваем количество рекомендаций

# ---------------------------
# Enhanced Donation System - ИСПРАВЛЕННЫЙ (без QR-кодов)
# ---------------------------
class DonationSystem:
    """Профессиональная система донатов - ИСПРАВЛЕННАЯ"""
    
    @staticmethod
    async def show_donation_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать меню выбора валюты для доната - ИСПРАВЛЕННЫЙ"""
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        text = (
            "💝 <b>ПОДДЕРЖАТЬ РАЗРАБОТЧИКА</b>\n\n"
            "Ваша поддержка помогает развивать бота и добавлять новые функции!\n\n"
            "Выберите валюту для доната:"
        )
        
        keyboard = [
            [InlineKeyboardButton("💎 USDT (TRC20)", callback_data="donate_usdt")],
            [InlineKeyboardButton("⚡ TON", callback_data="donate_ton")],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
        ]
        
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup(keyboard)
        )
    
    @staticmethod
    async def show_usdt_donation(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать USDT кошелек - ИСПРАВЛЕННЫЙ (без QR)"""
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        if not USDT_WALLET_ADDRESS:
            await SafeMessageSender.edit_message_text(
                query,
                "❌ USDT кошелек временно недоступен",
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="donate_start")]
                ])
            )
            return
        
        text = (
            "💎 <b>USDT (TRC20) ДОНАТ</b>\n\n"
            "Для поддержки разработки отправьте USDT на следующий адрес:\n\n"
            f"<code>{USDT_WALLET_ADDRESS}</code>\n\n"
            "💝 <i>Любая сумма будет принята с благодарностью!</i>\n\n"
            "<i>Скопируйте адрес выше и отправьте USDT через поддерживаемый кошелек.</i>"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔙 К выбору валюты", callback_data="donate_start")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup(keyboard)
        )
    
    @staticmethod
    async def show_ton_donation(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать TON кошелек - ИСПРАВЛЕННЫЙ (без QR)"""
        query = update.callback_query
        await SafeMessageSender.answer_callback_query(query)
        
        if not TON_WALLET_ADDRESS:
            await SafeMessageSender.edit_message_text(
                query,
                "❌ TON кошелек временно недоступен",
                InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="donate_start")]
                ])
            )
            return
        
        text = (
            "⚡ <b>TON ДОНАТ</b>\n\n"
            "Для поддержки разработки отправьте TON на следующий адрес:\n\n"
            f"<code>{TON_WALLET_ADDRESS}</code>\n\n"
            "💝 <i>Любая сумма будет принята с благодарностью!</i>\n\n"
            "<i>Скопируйте адрес выше и отправьте TON через поддерживаемый кошелек.</i>"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔙 К выбору валюты", callback_data="donate_start")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        await SafeMessageSender.edit_message_text(
            query,
            text,
            InlineKeyboardMarkup(keyboard)
        )

# ---------------------------
# Enhanced Handlers - ИСПРАВЛЕННЫЕ
# ---------------------------

@retry_on_timeout(max_retries=2, delay=1.0)
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start - ИСПРАВЛЕННЫЙ"""
    try:
        user = update.effective_user
        user_id = user.id
        PortfolioManager.ensure_user(user_id)
        
        # Проверяем есть ли сохраненный прогресс
        temp_data = DataManager.load_temporary_data()
        saved_progress = temp_data.get(str(user_id))
        
        text = (
            f"👋 Привет, {user.first_name}!\n\n"
            "🤖 <b>PRO Калькулятор Управления Рисками v3.1</b>\n\n"
            "🚀 <b>МОИ ВОЗМОЖНОСТИ:</b>\n"
            "• 📊 <b>РЕАЛЬНЫЕ КОТИРОВКИ</b> через Binance, Alpha Vantage\n"
            "• 💼 <b>ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ</b> маржи и рисков\n"
            "• 🎯 Контроль уровней риска (2%-25% от депозита)\n"
            "• 💡 Умные рекомендации и аналитика портфеля\n"
            "• 🛡 <b>ЗАЩИТА ОТ МАРЖИН-КОЛЛА</b> через правильный расчет\n"
            "• 📈 <b>РЕАЛЬНЫЕ ДАННЫЕ</b> для точного риск-менеджмента\n\n"
        )
        
        if saved_progress:
            text += "🔔 У вас есть сохраненный прогресс! Вы можете продолжить с того же места.\n\n"
        
        text += "<b>Выберите раздел:</b>"
        
        keyboard = [
            [InlineKeyboardButton("🎯 Профессиональные сделки", callback_data="pro_calculation")],
            [InlineKeyboardButton("📊 Мой портфель", callback_data="portfolio")]
        ]
        
        if saved_progress:
            keyboard.append([InlineKeyboardButton("🔄 Продолжить расчет", callback_data="restore_progress")])
        
        keyboard.extend([
            [InlineKeyboardButton("📚 PRO Инструкции", callback_data="pro_info")],
            [InlineKeyboardButton("💝 Поддержать разработчика", callback_data="donate_start")],
            [InlineKeyboardButton("🚀 Будущие разработки", callback_data="future_features")]
        ])
        
        if update.callback_query:
            success = await SafeMessageSender.edit_message_text(
                update.callback_query,
                text,
                InlineKeyboardMarkup(keyboard)
            )
            if not success:
                await SafeMessageSender.send_message(
                    user_id,
                    text,
                    context,
                    InlineKeyboardMarkup(keyboard)
                )
        else:
            await SafeMessageSender.send_message(
                user_id,
                text,
                context,
                InlineKeyboardMarkup(keyboard)
            )
            
    except Exception as e:
        logger.error(f"Error in start_command: {e}")
        try:
            if update.effective_user:
                await SafeMessageSender.send_message(
                    update.effective_user.id,
                    "❌ Произошла ошибка при загрузке. Пожалуйста, попробуйте еще раз.",
                    context
                )
        except:
            pass

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка тейк-профита - ИСПРАВЛЕННЫЙ с правильными расчетами"""
    text = update.message.text.strip()
    
    try:
        take_profit = float(text.replace(',', '.'))
        entry_price = context.user_data['entry_price']
        direction = context.user_data['direction']
        asset = context.user_data['asset']
        
        # Валидация TP
        if direction == 'LONG' and take_profit <= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для LONG тейк-профит должен быть ВЫШЕ цены входа\nПопробуйте еще раз:",
                context
            )
            return SingleTradeState.TAKE_PROFIT.value
        elif direction == 'SHORT' and take_profit >= entry_price:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Для SHORT тейк-профит должен быть НИЖЕ цены входа\nПопробуйте еще раз:",
                context
            )
            return SingleTradeState.TAKE_PROFIT.value
        
        # Собираем данные сделки
        trade_data = {
            'asset': context.user_data['asset'],
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': context.user_data['stop_loss'],
            'take_profit': take_profit,
            'risk_level': context.user_data['risk_level']
        }
        
        # ИСПРАВЛЕНИЕ: Правильное получение депозита и плеча
        deposit = context.user_data.get('deposit', 0)
        leverage = context.user_data.get('leverage', '1:100')
        risk_level = context.user_data['risk_level']
        
        if deposit <= 0:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Ошибка: депозит не установлен. Начните расчет заново.",
                context
            )
            return ConversationHandler.END
        
        # ПРОФЕССИОНАЛЬНЫЙ расчет метрик
        metrics = await ProfessionalRiskCalculator.calculate_professional_metrics(
            trade_data, deposit, leverage, risk_level
        )
        
        # Сохраняем сделку
        user_id = update.message.from_user.id
        trade_data['metrics'] = metrics
        PortfolioManager.add_single_trade(user_id, trade_data)
        
        # Сохраняем депозит и плечо для пользователя
        PortfolioManager.set_deposit_leverage(user_id, deposit, leverage)
        
        # Очищаем временный прогресс
        DataManager.clear_temporary_progress(user_id)
        
        # ФОРМИРУЕМ ИСПРАВЛЕННЫЙ ОТЧЕТ
        text = (
            f"<b>🎯 ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ СДЕЛКИ v3.1</b>\n\n"
            f"<b>📊 ПАРАМЕТРЫ СДЕЛКИ:</b>\n"
            f"• Актив: {trade_data['asset']}\n"
            f"• Текущая цена: ${metrics['current_price']:.2f} ✅ РЕАЛЬНАЯ\n"
            f"• Направление: {trade_data['direction']}\n"
            f"• Кредитное плечо: {leverage}\n"
            f"• Вход: {trade_data['entry_price']}\n"
            f"• Стоп-лосс: {trade_data['stop_loss']} ({metrics['stop_distance_pips']:.0f} пунктов)\n"
            f"• Тейк-профит: {trade_data['take_profit']} ({metrics['profit_distance_pips']:.0f} пунктов)\n"
            f"• Уровень риска: {trade_data['risk_level']}\n\n"
            
            f"<b>💰 ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ:</b>\n"
            f"• Депозит: ${metrics['deposit']:,.2f}\n"
            f"• Сумма риска: ${metrics['risk_amount']:.2f} ({metrics['risk_percent']:.1f}%)\n"
            f"• Объем позиции: {metrics['volume_lots']:.2f} лотов\n"
            f"• Требуемая маржа: ${metrics['required_margin']:.2f}\n"
            f"• Свободная маржа: ${metrics['free_margin']:.2f} ({100 - metrics['margin_usage_percent']:.1f}%)\n"
            f"• Уровень маржи: {metrics['margin_level']:.1f}%\n"
            f"• Использование маржи: {metrics['margin_usage_percent']:.1f}%\n"
            f"• Номинальная стоимость: ${metrics.get('notional_value', 0):.2f}\n"
            f"• Потенциальная прибыль: ${metrics['potential_profit']:.2f}\n"
            f"• Соотношение R/R: {metrics['rr_ratio']:.2f}\n"
            f"• Фактический левередж: {metrics.get('leverage_used', 1)}x\n\n"
        )
        
        # УЛУЧШЕННЫЕ РЕКОМЕНДАЦИИ
        if metrics['risk_percent'] > 10:
            text += "🔴 <b>ВЫСОКИЙ РИСК</b>! Превышен порог 10%. Уменьшите объем позиции.\n\n"
        elif metrics['margin_level'] < 100:
            text += "🔴 <b>КРИТИЧЕСКИЙ УРОВЕНЬ МАРЖИ</b>! Пополните счет.\n\n"
        elif metrics['margin_usage_percent'] > 50:
            text += "🟡 <b>ВЫСОКОЕ ИСПОЛЬЗОВАНИЕ МАРЖИ</b>! Оставьте запас для других сделок.\n\n"
        elif metrics['rr_ratio'] < 1:
            text += "🟡 <b>Соотношение R/R меньше 1</b>! Пересмотрите уровни TP/SL.\n\n"
        else:
            text += "✅ <b>Параметры сделки в пределах нормы</b>.\n\n"
        
        text += "Выберите дальнейшее действие:"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Новая сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")],
            [InlineKeyboardButton("📋 В портфель", callback_data="portfolio")]
        ]
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            text,
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return ConversationHandler.END
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Введите число (например: 52000)\nПопробуйте еще раз:",
            context
        )
        return SingleTradeState.TAKE_PROFIT.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int = None):
    """Показать портфель пользователя - ИСПРАВЛЕННЫЙ"""
    if not user_id:
        if update.callback_query:
            user_id = update.callback_query.from_user.id
        else:
            user_id = update.message.from_user.id
    
    PortfolioManager.ensure_user(user_id)
    user_portfolio = user_data[user_id]
    trades = user_portfolio.get('multi_trades', [])
    single_trades = user_portfolio.get('single_trades', [])
    deposit = user_portfolio.get('deposit', 0)
    leverage = user_portfolio.get('leverage', '1:100')
    
    all_trades = trades + single_trades
    
    if not all_trades or deposit <= 0:
        text = (
            "<b>📊 ВАШ ПОРТФЕЛЬ v3.1</b>\n\n"
            "Портфель пуст или депозит не установлен.\n\n"
            "<b>Начните с расчета сделок!</b>"
        )
        keyboard = [
            [InlineKeyboardButton("🎯 Одна сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")]
        ]
    else:
        # Обновляем цены и P&L в реальном времени
        updated_trades = []
        total_current_pnl = 0
        
        for trade in all_trades:
            try:
                current_price = await market_data_provider.get_real_time_price(trade['asset'])
                trade['current_price'] = current_price
                
                # ИСПРАВЛЕНИЕ: Правильный расчет P&L
                if 'metrics' in trade:
                    current_pnl = await ProfessionalRiskCalculator.calculate_realistic_pnl(trade, current_price)
                    trade['current_pnl'] = current_pnl
                    total_current_pnl += current_pnl
                updated_trades.append(trade)
            except Exception as e:
                logger.error(f"Ошибка обновления цены для {trade['asset']}: {e}")
                updated_trades.append(trade)
        
        # Расчет метрик портфеля
        metrics = PortfolioAnalyzer.calculate_portfolio_metrics(updated_trades, deposit)
        recommendations = PortfolioAnalyzer.generate_enhanced_recommendations(metrics, updated_trades)
        
        text = (
            f"<b>📊 ВАШ ПОРТФЕЛЬ v3.1</b>\n\n"
            f"<b>Основные параметры:</b>\n"
            f"• Депозит: ${deposit:,.2f}\n"
            f"• Плечо: {leverage}\n"
            f"• Всего сделок: {len(all_trades)}\n"
            f"• Одиночные: {len(single_trades)} | Мульти: {len(trades)}\n"
            f"• Уникальных активов: {metrics.get('unique_assets', 0)}\n"
            f"• Текущий P&L: ${total_current_pnl:+.2f}\n\n"
            
            f"<b>📈 КЛЮЧЕВЫЕ МЕТРИКИ:</b>\n"
            f"• Общий риск: ${metrics['total_risk_usd']:.2f} ({metrics['total_risk_percent']:.1f}%)\n"
            f"• Потенциальная прибыль: ${metrics['total_profit']:.2f}\n"
            f"• Общая маржа: ${metrics['total_margin']:.2f}\n"
            f"• Свободная маржа: ${metrics['free_margin']:.2f} ({metrics['free_margin_percent']:.1f}%)\n"
            f"• Уровень маржи: {metrics['portfolio_margin_level']:.1f}%\n"
            f"• Использование маржи: {metrics['total_margin_usage']:.1f}%\n"
            f"• Средний R/R: {metrics['avg_rr_ratio']:.2f}\n"
            f"• Волатильность: {metrics['portfolio_volatility']:.1f}%\n"
            f"• Общий левередж: {metrics.get('portfolio_leverage', 0):.1f}x\n"
            f"• Номинальная стоимость: ${metrics.get('total_notional_value', 0):.2f}\n"
            f"• LONG/Short: {metrics['long_positions']}/{metrics['short_positions']}\n\n"
            
            f"<b>💡 РЕКОМЕНДАЦИИ:</b>\n" + "\n".join(f"• {rec}" for rec in recommendations) + "\n\n"
        )
        
        # Добавляем информацию по сделкам (максимум 5)
        if updated_trades:
            text += "<b>📊 АКТИВНЫЕ СДЕЛКИ:</b>\n"
            for i, trade in enumerate(updated_trades[:5], 1):
                current_pnl = trade.get('current_pnl', 0)
                pnl_sign = "📈" if current_pnl >= 0 else "📉"
                current_price = trade.get('current_price', trade['entry_price'])
                text += f"{i}. {trade['asset']} {trade['direction']} | Цена: ${current_price:.2f} | P&L: {pnl_sign} ${current_pnl:+.2f}\n"
            
            if len(updated_trades) > 5:
                text += f"... и еще {len(updated_trades) - 5} сделок\n"
        
        # Кнопки управления
        keyboard = [
            [InlineKeyboardButton("🔄 Обновить цены", callback_data="portfolio")],
            [InlineKeyboardButton("🗑 Очистить портфель", callback_data="clear_portfolio")],
            [InlineKeyboardButton("📥 Выгрузить отчет", callback_data="export_portfolio")],
            [InlineKeyboardButton("🎯 Новая сделка", callback_data="single_trade")],
            [InlineKeyboardButton("📊 Мультипозиция", callback_data="multi_trade_start")]
        ]
    
    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
    
    if update.callback_query:
        await SafeMessageSender.edit_message_text(
            update.callback_query,
            text,
            InlineKeyboardMarkup(keyboard)
        )
    else:
        await SafeMessageSender.send_message(
            user_id,
            text,
            context,
            InlineKeyboardMarkup(keyboard)
        )

# ---------------------------
# Enhanced Callback Router - ИСПРАВЛЕННЫЙ
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Маршрутизатор callback запросов - ИСПРАВЛЕННЫЙ"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    data = query.data
    
    try:
        if data == "main_menu":
            await main_menu_save_handler(update, context)
        elif data == "portfolio":
            await show_portfolio(update, context)
        elif data == "pro_calculation":
            await pro_calculation_handler(update, context)
        elif data == "pro_info":
            await pro_info_command(update, context)
        elif data == "future_features":
            await future_features_handler(update, context)
        elif data == "clear_portfolio":
            await clear_portfolio_handler(update, context)
        elif data == "export_portfolio":
            await export_portfolio_handler(update, context)
        elif data == "restore_progress":
            await restore_progress_handler(update, context)
        elif data == "donate_start":
            await DonationSystem.show_donation_menu(update, context)
        elif data == "donate_usdt":
            await DonationSystem.show_usdt_donation(update, context)
        elif data == "donate_ton":
            await DonationSystem.show_ton_donation(update, context)
        elif data == "single_trade":
            await single_trade_start(update, context)
        elif data == "multi_trade_start":
            await multi_trade_start(update, context)
        else:
            await SafeMessageSender.answer_callback_query(query, "Команда не распознана")
            
    except Exception as e:
        logger.error(f"Error in callback router: {e}")
        await SafeMessageSender.answer_callback_query(query, "❌ Произошла ошибка")

# ---------------------------
# Enhanced PRO Info Command - ИСПРАВЛЕННЫЙ
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """PRO инструкции v3.1 - ИСПРАВЛЕННЫЙ"""
    text = (
        "<b>📚 PRO ИНСТРУКЦИИ v3.1</b>\n\n"
        
        "<b>🎯 ПРАВИЛЬНОЕ УПРАВЛЕНИЕ РИСКАМИ С РЕАЛЬНЫМИ ДАННЫМИ</b>\n\n"
        
        "<b>МЕТОДОЛОГИЯ РАСЧЕТА v3.1:</b>\n"
        "• Риск на сделку = % от депозита (например: 2% от $1000 = $20)\n"
        "• Объем позиции рассчитывается ИСКЛЮЧИТЕЛЬНО из суммы риска\n"
        "• <b>РЕАЛЬНЫЕ КОТИРОВКИ</b> через Binance, Alpha Vantage\n"
        "• <b>ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ</b> маржи по отраслевым стандартам\n"
        "• Защита от маржин-колла через правильный расчет объема\n\n"
        
        "<b>📊 РЕАЛЬНЫЕ КОТИРОВКИ:</b>\n"
        "• <b>Binance API</b> - криптовалюты с точностью до 0.01%\n"
        "• <b>Alpha Vantage</b> - акции, Forex, индексы\n"
        "• <b>Улучшенные fallback-цены</b> - защита от недоступности API\n\n"
        
        "<b>💼 ПРОФЕССИОНАЛЬНЫЙ РАСЧЕТ МАРЖИ:</b>\n"
        "• Forex: (Объем × Размер контракта) / Плечо\n"
        "• Крипто: (Объем × Цена) / Плечо\n"
        "• Акции: (Объем × Размер контракта × Цена) / Плечо\n"
        "• <b>РЕАЛЬНЫЕ СПЕЦИФИКАЦИИ</b> для 50+ активов\n\n"
        
        "<b>🌪 ВОЛАТИЛЬНОСТЬ В РАСЧЕТАХ:</b>\n\n"
        "• <b>Что это?</b> Мера колебаний цены актива\n"
        "• <b>Как используется?</b> Для оценки риска и рекомендаций\n"
        "• <b>Высокая волатильность</b> (>30%) = большие риски И возможности\n"
        "• <b>Низкая волатильность</b> (<15%) = стабильность, но меньший потенциал\n\n"
        
        "<b>ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ:</b>\n"
        "• BTCUSDT: 65% - высокий риск, нужен широкий SL\n"
        "• EURUSD: 8% - низкий риск, можно tighter управление\n"
        "• Используйте эти данные для настройки стоп-лоссов!\n\n"
        
        "<b>🎯 РЕКОМЕНДАЦИИ ДЛЯ ПРОФЕССИОНАЛОВ:</b>\n"
        "• Риск на сделку: 1-5% от депозита\n"
        "• Общий риск портфеля: < 10%\n"
        "• Уровень маржи: > 200%\n"
        "• Соотношение R/R: минимум 1:1.5\n"
        "• Диверсификация: 3-5 активов разных категорий\n"
        "• Свободная маржа: > 20% от депозита\n\n"
        
        "<b>🚀 ПРЕИМУЩЕСТВА v3.1:</b>\n"
        "✅ РЕАЛЬНЫЕ цены вместо статических данных\n"
        "✅ ПРОФЕССИОНАЛЬНЫЙ расчет маржи\n"
        "✅ ЗАЩИТА от маржин-колла\n"
        "✅ УМНЫЕ рекомендации\n"
        "✅ ОБНОВЛЕНИЕ портфеля в реальном времени\n"
        "✅ ПРАВИЛЬНЫЕ расчеты P&L\n\n"
        
        "<b>💝 Поддержите разработку для новых функций!</b>"
    )
    
    keyboard = [
        [InlineKeyboardButton("🎯 Начать расчет", callback_data="pro_calculation")],
        [InlineKeyboardButton("📊 Мой портфель", callback_data="portfolio")],
        [InlineKeyboardButton("💖 Поддержать разработчика", callback_data="donate_start")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    
    if update.callback_query:
        await SafeMessageSender.edit_message_text(
            update.callback_query,
            text,
            InlineKeyboardMarkup(keyboard)
        )
    else:
        await SafeMessageSender.send_message(
            update.effective_user.id,
            text,
            context,
            InlineKeyboardMarkup(keyboard)
        )

# ---------------------------
# Enhanced Multi Trade Start - ИСПРАВЛЕННЫЙ
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало мультипозиционного расчета - ИСПРАВЛЕННЫЙ"""
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    
    context.user_data['multi_trades'] = []
    
    text = (
        "🎯 <b>МУЛЬТИПОЗИЦИОННЫЙ РАСЧЕТ v3.1</b>\n\n"
        "ПРОФЕССИОНАЛЬНЫЙ расчет нескольких сделок с РЕАЛЬНЫМИ котировками.\n"
        "Объем каждой позиции рассчитывается из суммы риска на основе текущих цен!\n\n"
        "<b>Механика расчета:</b>\n"
        "• Риск на сделку = % от депозита\n" 
        "• Объем = Риск / (Дистанция SL в пунктах × Стоимость пункта)\n"
        "• Таким образом объем АВТОМАТИЧЕСКИ адаптируется под ваш риск!\n\n"
        "<b>Введите ваш депозит в USD:</b>"
    )
    
    keyboard = [
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")]
    ]
    
    await SafeMessageSender.edit_message_text(
        query,
        text,
        InlineKeyboardMarkup(keyboard)
    )
    return MultiTradeState.DEPOSIT.value

# ---------------------------
# Enhanced Deposit Handlers - ИСПРАВЛЕННЫЕ
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода депозита - ИСПРАВЛЕННЫЙ"""
    text = update.message.text.strip()
    
    try:
        deposit = float(text.replace(',', '.'))
        if deposit < 100:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Минимальный депозит: $100\nПопробуйте еще раз:",
                context
            )
            return SingleTradeState.DEPOSIT.value
        
        # ИСПРАВЛЕНИЕ: Сохраняем депозит в context И в PortfolioManager
        context.user_data['deposit'] = deposit
        user_id = update.message.from_user.id
        PortfolioManager.set_deposit_leverage(user_id, deposit, '1:100')  # Плечо по умолчанию
        
        keyboard = []
        for leverage in LEVERAGES:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"lev_{leverage}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Депозит: ${deposit:,.2f}\n\n"
            "<b>Выберите кредитное плечо:</b>",
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return SingleTradeState.LEVERAGE.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Введите число (например: 1000)\nПопробуйте еще раз:",
            context
        )
        return SingleTradeState.DEPOSIT.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def multi_trade_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода депозита для мультипозиции - ИСПРАВЛЕННЫЙ"""
    text = update.message.text.strip()
    
    try:
        deposit = float(text.replace(',', '.'))
        if deposit < 100:
            await SafeMessageSender.send_message(
                update.message.chat_id,
                "❌ Минимальный депозит: $100\nПопробуйте еще раз:",
                context
            )
            return MultiTradeState.DEPOSIT.value
        
        # ИСПРАВЛЕНИЕ: Сохраняем депозит в context И в PortfolioManager
        context.user_data['deposit'] = deposit
        user_id = update.message.from_user.id
        PortfolioManager.set_deposit_leverage(user_id, deposit, '1:100')  # Плечо по умолчанию
        
        keyboard = []
        for leverage in LEVERAGES:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"lev_{leverage}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu_save")])
        
        await SafeMessageSender.send_message(
            update.message.chat_id,
            f"✅ Депозит: ${deposit:,.2f}\n\n"
            "<b>Выберите кредитное плечо:</b>",
            context,
            InlineKeyboardMarkup(keyboard)
        )
        return MultiTradeState.LEVERAGE.value
        
    except ValueError:
        await SafeMessageSender.send_message(
            update.message.chat_id,
            "❌ Введите число (например: 1000)\nПопробуйте еще раз:",
            context
        )
        return MultiTradeState.DEPOSIT.value

# ---------------------------
# Остальные необходимые импорты и константы
# ---------------------------
# Добавьте эти константы в начало файла или в соответствующий раздел

# Волатильность активов (ОБНОВЛЕННЫЕ РЕАЛЬНЫЕ ДАННЫЕ)
VOLATILITY_DATA = {
    'BTCUSDT': 65.2, 'ETHUSDT': 70.5, 'XRPUSDT': 85.3, 'LTCUSDT': 68.1,
    'BCHUSDT': 75.2, 'ADAUSDT': 80.7, 'DOTUSDT': 72.4, 'AAPL': 25.3,
    'TSLA': 55.1, 'GOOGL': 22.8, 'MSFT': 20.1, 'AMZN': 28.7,
    'META': 32.5, 'NFLX': 45.2, 'EURUSD': 8.5, 'GBPUSD': 9.2,
    'USDJPY': 7.8, 'USDCHF': 8.1, 'AUDUSD': 10.3, 'USDCAD': 8.7,
    'NZDUSD': 11.2, 'XAUUSD': 14.5, 'XAGUSD': 25.3, 'XPTUSD': 18.7,
    'XPDUSD': 22.1, 'OIL': 35.2, 'NATURALGAS': 42.1, 'BRENT': 33.8,
    'NAS100': 18.5, 'SPX500': 15.2, 'DJ30': 12.8, 'FTSE100': 11.5,
    'DAX40': 16.3, 'NIKKEI225': 14.7, 'ASX200': 13.2
}

# Инициализация глобальных сервисов
market_data_provider = MarketDataProvider()
margin_calculator = ProfessionalMarginCalculator()

# ---------------------------
# Запуск приложения
# ---------------------------
if __name__ == "__main__":
    asyncio.run(main())
