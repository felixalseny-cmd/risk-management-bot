# bot.py — PRO Risk Calculator v3.0 | ENTERPRISE EDITION (Fixed)
import os
import logging
import asyncio
import time
import functools
import json
import html as html_module
import re
import io
from datetime import datetime
from typing import Dict, List, Any, Optional

import aiohttp
import cachetools
from dotenv import load_dotenv

# telegram v20 imports
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Bot, CallbackQuery, constants
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters,
    CallbackQueryHandler, ContextTypes, ConversationHandler
)

load_dotenv()

# ---------------------------
# Config
# ---------------------------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is required")

PORT = int(os.getenv("PORT", "10000"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").rstrip("/")
WEBHOOK_PATH = f"/webhook/{TOKEN}"

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

USDT_WALLET_ADDRESS = os.getenv("USDT_WALLET_ADDRESS", "")
TON_WALLET_ADDRESS = os.getenv("TON_WALLET_ADDRESS", "")

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("pro_risk_bot")

# ---------------------------
# Utilities
# ---------------------------
def safe_html_escape(s: Any) -> str:
    """Экранирует динамический контент для использования в parse_mode='HTML'."""
    if s is None:
        return ""
    # Приводим к строке и экранируем <, >, & — безопасно для HTML parse
    return html_module.escape(str(s))

def get_chat_id_from_update(update: Update) -> Optional[int]:
    if update.effective_chat:
        return update.effective_chat.id
    if update.callback_query and update.callback_query.message:
        return update.callback_query.message.chat.id
    if update.message:
        return update.message.chat.id
    return None

# ---------------------------
# Robust Application builder
# ---------------------------
class RobustApplicationBuilder:
    @staticmethod
    def create_application(token: str) -> Application:
        # Use default HTTPXRequest with reasonable pool size
        request = None
        try:
            from telegram.request import HTTPXRequest
            request = HTTPXRequest(connection_pool_size=10, read_timeout=20.0, connect_timeout=10.0)
            app = Application.builder().token(token).request(request).build()
        except Exception:
            # Fallback to default builder
            app = Application.builder().token(token).build()
        return app

# ---------------------------
# Retry decorator
# ---------------------------
def retry_on_timeout(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # handle network/timeouts from telegram lib implicitly
                    logger.warning("Exception in %s attempt %s/%s: %s", func.__name__, attempt+1, max_retries, e)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))
                        continue
                    raise
        return wrapper
    return decorator

# ---------------------------
# SafeMessageSender
# ---------------------------
class SafeMessageSender:
    """Safe send/edit messages with HTML escaping fallback"""
    @staticmethod
    async def _send(bot: Bot, chat_id: int, text: str, reply_markup=None, parse_mode=constants.ParseMode.HTML):
        try:
            return await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup, parse_mode=parse_mode)
        except Exception as e:
            # bubble up
            raise

    @staticmethod
    async def send_message(update_or_chat, text: str, context: ContextTypes.DEFAULT_TYPE = None, reply_markup=None):
        """Helper: accepts either Update or chat_id (int)"""
        # Determine chat_id and bot
        if isinstance(update_or_chat, Update):
            chat_id = get_chat_id_from_update(update_or_chat)
            bot = context.bot if context else Bot(TOKEN)
        else:
            chat_id = update_or_chat
            bot = context.bot if context else Bot(TOKEN)

        if chat_id is None:
            logger.error("No chat_id available to send message")
            return None

        # Try with HTML first
        try:
            return await SafeMessageSender._send(bot, chat_id, text, reply_markup=reply_markup, parse_mode=constants.ParseMode.HTML)
        except Exception as e:
            # If failure due to parse entities -> fallback to escaped HTML (safe) or fallback to plain text
            msg = str(e)
            logger.warning("send_message error: %s. Falling back to escaped text.", msg)
            try:
                escaped = safe_html_escape(text)
                return await SafeMessageSender._send(bot, chat_id, escaped, reply_markup=reply_markup, parse_mode=constants.ParseMode.HTML)
            except Exception as e2:
                logger.warning("escaped HTML failed: %s. Sending plain text.", e2)
                try:
                    return await SafeMessageSender._send(bot, chat_id, text, reply_markup=reply_markup, parse_mode=None)
                except Exception as e3:
                    logger.error("Failed to send fallback plain text: %s", e3)
                    return None

    @staticmethod
    async def edit_message_text(callback_query: CallbackQuery, text: str, reply_markup=None):
        """Edit callback message robustly"""
        try:
            await callback_query.edit_message_text(text=text, reply_markup=reply_markup, parse_mode=constants.ParseMode.HTML)
            return True
        except Exception as e:
            msg = str(e)
            logger.warning("edit_message_text failed: %s. Trying escaped text.", msg)
            try:
                escaped = safe_html_escape(text)
                await callback_query.edit_message_text(text=escaped, reply_markup=reply_markup, parse_mode=constants.ParseMode.HTML)
                return True
            except Exception as e2:
                logger.warning("escaped edit failed: %s. Trying plain text edit.", e2)
                try:
                    await callback_query.edit_message_text(text=text, reply_markup=reply_markup)  # plain
                    return True
                except Exception as e3:
                    logger.error("Final edit_message_text failure: %s", e3)
                    return False

    @staticmethod
    async def answer_callback_query(query: CallbackQuery, text: str = None, show_alert: bool = False):
        try:
            await query.answer(text=text, show_alert=show_alert)
            return True
        except Exception as e:
            logger.error("Failed to answer callback query: %s", e)
            return False

# ---------------------------
# Data persistence helpers
# ---------------------------
class DataManager:
    DATA_FILE = "user_data.json"
    TEMP_FILE = "temp_progress.json"

    @staticmethod
    def load_data() -> Dict[int, Dict[str, Any]]:
        if not os.path.exists(DataManager.DATA_FILE):
            return {}
        try:
            with open(DataManager.DATA_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
        except Exception as e:
            logger.error("Failed to load data: %s", e)
            return {}

    @staticmethod
    def save_data(data: Dict[int, Dict[str, Any]]):
        try:
            with open(DataManager.DATA_FILE, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in data.items()}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save data: %s", e)

    @staticmethod
    def load_temporary_data() -> Dict[str, Any]:
        if not os.path.exists(DataManager.TEMP_FILE):
            return {}
        try:
            with open(DataManager.TEMP_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load temp progress: %s", e)
            return {}

    @staticmethod
    def save_temporary_progress(user_id: int, state_data: Dict[str, Any], state_type: str):
        temp = DataManager.load_temporary_data()
        temp[str(user_id)] = {
            "state_data": state_data,
            "state_type": state_type,
            "saved_at": datetime.utcnow().isoformat()
        }
        try:
            with open(DataManager.TEMP_FILE, "w", encoding="utf-8") as f:
                json.dump(temp, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save temporary progress: %s", e)

    @staticmethod
    def clear_temporary_progress(user_id: int):
        temp = DataManager.load_temporary_data()
        if str(user_id) in temp:
            del temp[str(user_id)]
            try:
                with open(DataManager.TEMP_FILE, "w", encoding="utf-8") as f:
                    json.dump(temp, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error("Failed to clear temp progress: %s", e)

# ---------------------------
# Placeholder Market Data provider with caching and fallbacks
# (Using structure from your source; improved defensive handling)
# ---------------------------
class MarketDataProvider:
    def __init__(self):
        self.cache = cachetools.TTLCache(maxsize=1000, ttl=300)
        self.session: Optional[aiohttp.ClientSession] = None

    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    def _is_crypto(self, symbol: str):
        return any(symbol.upper().endswith(s) or s in symbol.upper() for s in ["USDT", "BTC", "ETH"])

    async def get_price(self, symbol: str) -> float:
        symbol = symbol.upper()
        cached = self.cache.get(symbol)
        if cached:
            return cached
        # Prefer frankfurter for forex pairs
        try:
            if len(symbol) == 6 and symbol.isalpha():
                # Forex pair like EURUSD
                from_c = symbol[:3]
                to_c = symbol[3:]
                session = await self.get_session()
                url = f"https://api.frankfurter.app/latest?from={from_c}&to={to_c}"
                async with session.get(url, timeout=5) as r:
                    if r.status == 200:
                        data = await r.json()
                        rate = data.get("rates", {}).get(to_c)
                        if rate:
                            self.cache[symbol] = float(rate)
                            return float(rate)
        except Exception as e:
            logger.debug("Frankfurter error: %s", e)

        # Binance for crypto (symbol must be like BTCUSDT)
        try:
            if self._is_crypto(symbol):
                s = symbol
                if not s.endswith("USDT"):
                    s = s + "USDT"
                session = await self.get_session()
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={s}"
                async with session.get(url, timeout=8) as r:
                    if r.status == 200:
                        data = await r.json()
                        price = data.get("price")
                        if price:
                            self.cache[symbol] = float(price)
                            return float(price)
        except Exception as e:
            logger.debug("Binance error: %s", e)

        # Finnhub (fallback)
        try:
            if FINNHUB_API_KEY:
                session = await self.get_session()
                url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
                async with session.get(url, timeout=8) as r:
                    if r.status == 200:
                        data = await r.json()
                        price = data.get("c")
                        if price:
                            self.cache[symbol] = float(price)
                            return float(price)
        except Exception as e:
            logger.debug("Finnhub error: %s", e)

        # Final fallback static
        fallback = {
            "EURUSD": 1.0850, "GBPUSD": 1.2650, "USDJPY": 148.50,
            "BTCUSDT": 45000.0, "ETHUSDT": 3000.0, "AAPL": 185.0, "XAUUSD": 1980.0
        }
        val = fallback.get(symbol, 100.0)
        self.cache[symbol] = float(val)
        return float(val)

market_data_provider = MarketDataProvider()

# ---------------------------
# Instrument specs (kept and defensive)
# ---------------------------
class InstrumentSpecs:
    SPECS = {
        "EURUSD": {"type":"forex","contract_size":100000,"pip_value":10.0,"pip_decimal_places":4,"calculation_formula":"forex"},
        "USDJPY": {"type":"forex","contract_size":100000,"pip_value":9.09,"pip_decimal_places":2,"calculation_formula":"forex_jpy"},
        "BTCUSDT":{"type":"crypto","contract_size":1,"pip_value":1.0,"pip_decimal_places":1,"calculation_formula":"crypto"},
        "AAPL":{"type":"stock","contract_size":100,"pip_value":1.0,"pip_decimal_places":2,"calculation_formula":"stocks"},
        "XAUUSD":{"type":"metal","contract_size":100,"pip_value":10.0,"pip_decimal_places":2,"calculation_formula":"metals"},
        "OIL":{"type":"energy","contract_size":1000,"pip_value":10.0,"pip_decimal_places":2,"calculation_formula":"energy"},
    }

    @classmethod
    def get_specs(cls, symbol: str) -> Dict[str, Any]:
        s = symbol.upper()
        if s in cls.SPECS:
            return cls.SPECS[s].copy()
        # default logic
        if "USDT" in s:
            return {"type":"crypto","contract_size":1,"pip_value":1.0,"pip_decimal_places":2,"calculation_formula":"crypto"}
        if len(s) == 6 and s.isalpha():
            return {"type":"forex","contract_size":100000,"pip_value":10.0,"pip_decimal_places":4,"calculation_formula":"forex"}
        return {"type":"stock","contract_size":100,"pip_value":1.0,"pip_decimal_places":2,"calculation_formula":"stocks"}

# ---------------------------
# Margin & Risk calculators (kept from original design, defensive)
# ---------------------------
class ProfessionalMarginCalculator:
    async def calculate_professional_margin(self, symbol: str, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        specs = InstrumentSpecs.get_specs(symbol)
        try:
            lev = 1
            if isinstance(leverage, str) and ":" in leverage:
                lev = int(leverage.split(":")[1])
            contract_size = specs.get("contract_size", 1)
            # Basic formulas:
            if specs.get("calculation_formula") == "forex_jpy":
                required = (volume * contract_size) / (lev * max(current_price, 1e-8))
            elif specs.get("calculation_formula") == "crypto":
                required = (volume * current_price) / max(lev, 1)
            elif specs.get("calculation_formula") == "stocks":
                required = (volume * contract_size * current_price) / max(lev, 1)
            else:
                required = (volume * contract_size * current_price) / max(lev, 1)
            return {"required_margin": max(required, 0.0), "contract_size": contract_size, "calculation_method": specs.get("calculation_formula"), "leverage_used": lev, "notional_value": volume * contract_size * current_price}
        except Exception as e:
            logger.error("margin calc error: %s", e)
            return {"required_margin":0.0,"contract_size":specs.get("contract_size",1),"calculation_method":"error","leverage_used":1,"notional_value":0.0}

margin_calculator = ProfessionalMarginCalculator()

class ProfessionalRiskCalculator:
    @staticmethod
    def calculate_pip_distance(entry: float, target: float, direction: str, asset: str) -> float:
        specs = InstrumentSpecs.get_specs(asset)
        dec = specs.get("pip_decimal_places", 4)
        if (direction or "").upper() == "LONG":
            distance = target - entry
        else:
            distance = entry - target
        if dec == 2:
            return abs(distance) * 100.0
        elif dec == 1:
            return abs(distance) * 10.0
        else:
            return abs(distance) * 10000.0

    @staticmethod
    async def calculate_professional_metrics(trade: Dict[str, Any], deposit: float, leverage: str, risk_level: str) -> Dict[str, Any]:
        try:
            asset = trade.get("asset")
            entry = float(trade.get("entry_price", 0))
            stop = float(trade.get("stop_loss", entry))
            tp = float(trade.get("take_profit", entry))
            direction = trade.get("direction", "LONG").upper()
            current_price = await market_data_provider.get_price(asset)
            specs = InstrumentSpecs.get_specs(asset)
            risk_pct = float(str(risk_level).strip("%")) if risk_level else 2.0
            risk_amount = deposit * (risk_pct/100.0)
            stop_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry, stop, direction, asset)
            profit_pips = ProfessionalRiskCalculator.calculate_pip_distance(entry, tp, direction, asset)
            pip_value = specs.get("pip_value", 1.0)
            volume_lots = 0.0
            if stop_pips > 0 and pip_value > 0:
                volume_lots = round(risk_amount / (stop_pips * pip_value), 2)
            margin_info = await margin_calculator.calculate_professional_margin(asset, volume_lots, leverage, current_price)
            required_margin = round(margin_info.get("required_margin", 0.0), 2)
            free_margin = max(round(deposit - required_margin,2), 0.0)
            margin_level = round((deposit / required_margin) * 100.0, 1) if required_margin > 0 else 0.0
            potential_profit = round(volume_lots * profit_pips * pip_value, 2)
            rr = round(potential_profit / risk_amount, 2) if risk_amount > 0 else 0.0
            current_pnl = round(volume_lots * ( (current_price - entry) * (100 if specs.get("pip_decimal_places",4)==4 else (10 if specs.get("pip_decimal_places",4)==1 else 100))) * pip_value, 2)
            return {
                "volume_lots": volume_lots,
                "required_margin": required_margin,
                "free_margin": free_margin,
                "margin_level": margin_level,
                "risk_amount": risk_amount,
                "risk_percent": risk_pct,
                "potential_profit": potential_profit,
                "rr_ratio": rr,
                "stop_distance_pips": stop_pips,
                "profit_distance_pips": profit_pips,
                "pip_value": pip_value,
                "contract_size": margin_info.get("contract_size", 0),
                "deposit": deposit,
                "leverage": leverage,
                "notional_value": margin_info.get("notional_value", 0),
                "leverage_used": margin_info.get("leverage_used", 1),
                "current_price": current_price,
                "current_pnl": current_pnl,
                "calculation_method": margin_info.get("calculation_method", "unknown")
            }
        except Exception as e:
            logger.error("calculate_professional_metrics error: %s", e)
            return {
                "volume_lots": 0.0, "required_margin":0.0, "free_margin":deposit, "margin_level":0.0,
                "risk_amount":0.0, "risk_percent":0.0, "potential_profit":0.0, "rr_ratio":0.0,
                "stop_distance_pips":0.0, "profit_distance_pips":0.0, "pip_value":0.0, "contract_size":0,
                "deposit":deposit, "leverage":leverage, "notional_value":0.0, "leverage_used":1, "current_price":0.0, "current_pnl":0.0
            }

# ---------------------------
# Portfolio management (simple)
# ---------------------------
user_data = DataManager.load_data()

class PortfolioManager:
    @staticmethod
    def ensure_user(user_id: int):
        if user_id not in user_data:
            user_data[user_id] = {"multi_trades":[],"single_trades":[],"deposit":0.0,"leverage":"1:100","created_at":datetime.utcnow().isoformat()}
            DataManager.save_data(user_data)
    @staticmethod
    def add_single_trade(user_id:int, trade:Dict):
        PortfolioManager.ensure_user(user_id)
        trade["id"] = len(user_data[user_id]["single_trades"]) + 1
        user_data[user_id]["single_trades"].append(trade)
        user_data[user_id]["last_updated"] = datetime.utcnow().isoformat()
        DataManager.save_data(user_data)
    @staticmethod
    def set_deposit_leverage(user_id:int, deposit:float, leverage:str):
        PortfolioManager.ensure_user(user_id)
        user_data[user_id]["deposit"] = deposit
        user_data[user_id]["leverage"] = leverage
        user_data[user_id]["last_updated"] = datetime.utcnow().isoformat()
        DataManager.save_data(user_data)
    @staticmethod
    def clear_portfolio(user_id:int):
        if user_id in user_data:
            user_data[user_id]["multi_trades"]=[]; user_data[user_id]["single_trades"]=[]; user_data[user_id]["deposit"]=0.0
            user_data[user_id]["last_updated"] = datetime.utcnow().isoformat()
            DataManager.save_data(user_data)

# ---------------------------
# Conversation states (simple)
# ---------------------------
from enum import Enum
class SingleTradeState(Enum):
    DEPOSIT = 1; LEVERAGE = 2; ASSET_CATEGORY = 3; ASSET = 4; DIRECTION = 5; ENTRY = 6; STOP_LOSS = 7; RISK_LEVEL = 8; TAKE_PROFIT = 9

ASSET_CATEGORIES = {
    "FOREX": ['EURUSD','GBPUSD','USDJPY','AUDUSD'],
    "CRYPTO": ['BTCUSDT','ETHUSDT'],
    "STOCKS": ['AAPL','TSLA']
}
LEVERAGES = ['1:10','1:20','1:50','1:100','1:200']
RISK_LEVELS = ['2%','5%','7%','10%']

# ---------------------------
# Handlers
# ---------------------------
@retry_on_timeout(max_retries=2, delay=1.0)
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    uid = user.id if user else None
    if uid:
        PortfolioManager.ensure_user(uid)
    name = safe_html_escape(user.first_name if user else "user")
    text = (
        f"👋 Привет, {name}!\n\n"
        "<b>PRO Калькулятор Управления Рисками v3.0</b>\n\n"
        "Выберите раздел:"
    )
    keyboard = [
        [InlineKeyboardButton("🎯 Профессиональные сделки", callback_data="pro_calculation")],
        [InlineKeyboardButton("📊 Мой портфель", callback_data="portfolio")],
        [InlineKeyboardButton("💝 Поддержать разработчика", callback_data="donate_start")]
    ]
    if update.callback_query:
        await SafeMessageSender.edit_message_text(update.callback_query, text, InlineKeyboardMarkup(keyboard))
    else:
        await SafeMessageSender.send_message(update, text, context, InlineKeyboardMarkup(keyboard))

@retry_on_timeout(max_retries=2, delay=1.0)
async def pro_calculation_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    text = "🎯 <b>ПРОФЕССИОНАЛЬНЫЕ СДЕЛКИ</b>\nВыберите режим:"
    keyboard = [
        [InlineKeyboardButton("Одна сделка", callback_data="single_trade")],
        [InlineKeyboardButton("Мультипозиция (сконфигурируйте вручную)", callback_data="multi_trade_start")],
        [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
    ]
    await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(keyboard))

@retry_on_timeout(max_retries=2, delay=1.0)
async def donate_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    text = "💝 Поддержать проект:\n"
    keyboard = [
        [InlineKeyboardButton("USDT", callback_data="donate_usdt"), InlineKeyboardButton("TON", callback_data="donate_ton")],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]
    await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(keyboard))

@retry_on_timeout(max_retries=2, delay=1.0)
async def donate_usdt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await SafeMessageSender.answer_callback_query(q)
    addr = safe_html_escape(USDT_WALLET_ADDRESS or "—")
    text = f"<b>USDT (TRC20)</b>\nАдрес:\n<code>{addr}</code>"
    kb = [[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]]
    await SafeMessageSender.edit_message_text(q, text, InlineKeyboardMarkup(kb))

# Simple single-trade flow (entry -> stop -> risk -> tp)
@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await SafeMessageSender.answer_callback_query(query)
    text = "<b>Введите депозит (USD):</b>"
    kb = [[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]]
    await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(kb))
    return SingleTradeState.DEPOSIT.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # a message handler path
    txt = (update.message.text or "").strip()
    try:
        deposit = float(txt.replace(",", "."))
    except Exception:
        await SafeMessageSender.send_message(update, "Введите корректное число (пример: 1000).", context)
        return SingleTradeState.DEPOSIT.value
    context.user_data['deposit'] = deposit
    # ask leverage
    buttons = [[InlineKeyboardButton(l, callback_data=f"lev_{l}")] for l in LEVERAGES]
    buttons.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
    await SafeMessageSender.send_message(update, "<b>Выберите плечо:</b>", context, InlineKeyboardMarkup(buttons))
    return SingleTradeState.LEVERAGE.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await SafeMessageSender.answer_callback_query(q)
    lev = q.data.replace("lev_", "")
    context.user_data['leverage'] = lev
    # choose category
    kb = [[InlineKeyboardButton(cat, callback_data=f"cat_{cat}")] for cat in ASSET_CATEGORIES.keys()]
    kb.append([InlineKeyboardButton("📝 Ввести вручную", callback_data="asset_manual")])
    kb.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
    await SafeMessageSender.edit_message_text(q, "<b>Выберите категорию актива:</b>", InlineKeyboardMarkup(kb))
    return SingleTradeState.ASSET_CATEGORY.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_asset_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await SafeMessageSender.answer_callback_query(q)
    if q.data == "asset_manual":
        await SafeMessageSender.edit_message_text(q, "✍️ Введите название актива (например: BTCUSDT):", InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]]))
        return SingleTradeState.ASSET.value
    cat = q.data.replace("cat_", "")
    assets = ASSET_CATEGORIES.get(cat, [])
    kb = [[InlineKeyboardButton(a, callback_data=f"asset_{a}")] for a in assets]
    kb.append([InlineKeyboardButton("🔙 Назад", callback_data="pro_calculation")])
    await SafeMessageSender.edit_message_text(q, f"✅ Категория: {safe_html_escape(cat)}\n<b>Выберите актив:</b>", InlineKeyboardMarkup(kb))
    return SingleTradeState.ASSET.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_asset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if q:
        await SafeMessageSender.answer_callback_query(q)
        if q.data.startswith("asset_"):
            asset = q.data.replace("asset_", "")
            context.user_data['asset'] = asset
            kb = [[InlineKeyboardButton("📈 LONG", callback_data="dir_LONG"), InlineKeyboardButton("📉 SHORT", callback_data="dir_SHORT")],
                  [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]]
            await SafeMessageSender.edit_message_text(q, f"✅ Актив: {safe_html_escape(asset)}\n<b>Выберите направление сделки:</b>", InlineKeyboardMarkup(kb))
            return SingleTradeState.DIRECTION.value
    else:
        # message path for manual asset entry
        txt = (update.message.text or "").strip().upper()
        if not re.match(r'^[A-Z0-9\-_]{2,20}$', txt):
            await SafeMessageSender.send_message(update, "Неверный формат актива. Попробуйте снова (например BTCUSDT).", context)
            return SingleTradeState.ASSET.value
        context.user_data['asset'] = txt
        await SafeMessageSender.send_message(update, f"✅ Актив: {safe_html_escape(txt)}\n<b>Выберите направление:</b>", context)
        return SingleTradeState.DIRECTION.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_direction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await SafeMessageSender.answer_callback_query(q)
    direction = q.data.replace("dir_", "")
    context.user_data['direction'] = direction
    await SafeMessageSender.edit_message_text(q, "<b>Введите цену входа (например: 1.0850):</b>", InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]]))
    return SingleTradeState.ENTRY.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip()
    try:
        entry = float(txt.replace(",", "."))
    except Exception:
        await SafeMessageSender.send_message(update, "Введите корректную цену (например: 1.0850).", context)
        return SingleTradeState.ENTRY.value
    context.user_data['entry_price'] = entry
    await SafeMessageSender.send_message(update, "<b>Введите SL (stop loss):</b>", context)
    return SingleTradeState.STOP_LOSS.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip()
    try:
        sl = float(txt.replace(",", "."))
    except Exception:
        await SafeMessageSender.send_message(update, "Введите корректную цену SL (например: 1.0800).", context)
        return SingleTradeState.STOP_LOSS.value
    context.user_data['stop_loss'] = sl
    # ask risk level
    kb = [[InlineKeyboardButton(r, callback_data=f"risk_{r}")] for r in RISK_LEVELS]
    kb.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
    await SafeMessageSender.send_message(update, "<b>Выберите уровень риска:</b>", context, InlineKeyboardMarkup(kb))
    return SingleTradeState.RISK_LEVEL.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_risk_level(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await SafeMessageSender.answer_callback_query(q)
    risk = q.data.replace("risk_", "")
    context.user_data['risk_level'] = risk
    await SafeMessageSender.edit_message_text(q, f"✅ Уровень риска: {safe_html_escape(risk)}\n\n<b>Введите TP (take profit):</b>", InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]]))
    return SingleTradeState.TAKE_PROFIT.value

@retry_on_timeout(max_retries=2, delay=1.0)
async def single_trade_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip()
    try:
        tp = float(txt.replace(",", "."))
    except Exception:
        await SafeMessageSender.send_message(update, "Введите корректную цену TP (например: 1.0900).", context)
        return SingleTradeState.TAKE_PROFIT.value

    context.user_data['take_profit'] = tp
    # Save trade and calculate metrics
    trade = {
        "asset": context.user_data.get("asset"),
        "direction": context.user_data.get("direction"),
        "entry_price": context.user_data.get("entry_price"),
        "stop_loss": context.user_data.get("stop_loss"),
        "take_profit": context.user_data.get("take_profit"),
        "risk_level": context.user_data.get("risk_level"),
        "deposit": context.user_data.get("deposit"),
        "leverage": context.user_data.get("leverage"),
    }
    uid = update.message.from_user.id
    PortfolioManager.ensure_user(uid)
    PortfolioManager.add_single_trade(uid, trade)
    PortfolioManager.set_deposit_leverage(uid, trade['deposit'], trade['leverage'])

    # compute metrics
    metrics = await ProfessionalRiskCalculator.calculate_professional_metrics(trade, trade['deposit'], trade['leverage'], trade['risk_level'])
    trade['metrics'] = metrics

    # Compose output (escape dynamic fields)
    body = (
        "<b>🎯 РЕЗУЛЬТАТЫ РАСЧЕТА v3.0</b>\n\n"
        f"Актив: {safe_html_escape(trade['asset'])}\n"
        f"Направление: {safe_html_escape(trade['direction'])}\n"
        f"Вход: {safe_html_escape(trade['entry_price'])}\n"
        f"SL: {safe_html_escape(trade['stop_loss'])}\n"
        f"TP: {safe_html_escape(trade['take_profit'])}\n\n"
        f"💰 Объем: {metrics['volume_lots']:.2f} лотов\n"
        f"• Риск: ${metrics['risk_amount']:.2f} ({metrics['risk_percent']:.1f}%)\n"
        f"• Потенциальная прибыль: ${metrics['potential_profit']:.2f}\n"
        f"• R/R: {metrics['rr_ratio']:.2f}\n\n"
        f"🛡 Требуемая маржа: ${metrics['required_margin']:.2f}\n"
        f"• Свободная маржа: ${metrics['free_margin']:.2f}\n"
        f"• Уровень маржи: {metrics['margin_level']:.1f}%\n\n"
        f"• Текущая цена: ${metrics['current_price']:.4f}\n"
        f"• Текущий P&L: ${metrics['current_pnl']:.2f}"
    )
    kb = [[InlineKeyboardButton("📊 Портфель", callback_data="portfolio")], [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]]

    await SafeMessageSender.send_message(update, body, context, InlineKeyboardMarkup(kb))
    DataManager.clear_temporary_progress(uid)
    context.user_data.clear()
    return ConversationHandler.END

@retry_on_timeout(max_retries=2, delay=1.0)
async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query if update.callback_query else None
    if query:
        await SafeMessageSender.answer_callback_query(query)
        uid = query.from_user.id
    else:
        uid = update.message.from_user.id
    PortfolioManager.ensure_user(uid)
    p = user_data.get(uid, {"deposit":0.0,"single_trades":[],"multi_trades":[],"leverage":"1:100"})
    trades = p.get("single_trades", []) + p.get("multi_trades", [])
    if not trades:
        txt = "<b>📊 Ваш портфель пуст</b>\n\nСоздайте первую сделку."
        kb = [[InlineKeyboardButton("🎯 Новая сделка", callback_data="single_trade")],[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]]
        if query:
            await SafeMessageSender.edit_message_text(query, txt, InlineKeyboardMarkup(kb))
        else:
            await SafeMessageSender.send_message(update, txt, context, InlineKeyboardMarkup(kb))
        return
    # update metrics
    for t in trades:
        t['metrics'] = await ProfessionalRiskCalculator.calculate_professional_metrics(t, p.get("deposit",0.0), p.get("leverage","1:100"), t.get("risk_level","2%"))
    total_pnl = sum(t['metrics'].get("current_pnl",0) for t in trades)
    text = f"<b>📊 ПОРТФЕЛЬ</b>\nДепозит: ${p.get('deposit',0.0):,.2f}\nСделок: {len(trades)}\nТекущий P&L: ${total_pnl:.2f}\n\n"
    # list trades
    for i,t in enumerate(trades,1):
        pnl = t['metrics'].get("current_pnl",0.0)
        sign = "🟢" if pnl>0 else "🔴" if pnl<0 else "⚪"
        text += f"{sign} <b>#{i}</b> {safe_html_escape(t.get('asset'))} {safe_html_escape(t.get('direction'))}\n   Вход: {safe_html_escape(t.get('entry_price'))} | SL: {safe_html_escape(t.get('stop_loss'))} | TP: {safe_html_escape(t.get('take_profit'))}\n   P&L: ${pnl:.2f}\n\n"
    kb = [[InlineKeyboardButton("🗑 Очистить портфель", callback_data="clear_portfolio")],[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]]
    if query:
        await SafeMessageSender.edit_message_text(query, text, InlineKeyboardMarkup(kb))
    else:
        await SafeMessageSender.send_message(update, text, context, InlineKeyboardMarkup(kb))

@retry_on_timeout(max_retries=2, delay=1.0)
async def clear_portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await SafeMessageSender.answer_callback_query(q)
    uid = q.from_user.id
    PortfolioManager.clear_portfolio(uid)
    await SafeMessageSender.edit_message_text(q, "<b>Портфель очищен</b>", InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]]))

# ---------------------------
# Global error handler registration
# ---------------------------
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    try:
        logger.exception("Unhandled exception: %s", context.error)
    except Exception as e:
        logger.error("Error in error handler: %s", e)

# ---------------------------
# Setup conversation handlers & routing
# ---------------------------
def setup_conversation_handlers(application: Application):
    # ConversationHandler for single trade: we register states with both callback query and message handlers
    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(single_trade_start, pattern="^single_trade$")],
        states={
            SingleTradeState.DEPOSIT.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_deposit)],
            SingleTradeState.LEVERAGE.value: [CallbackQueryHandler(single_trade_leverage, pattern="^lev_")],
            SingleTradeState.ASSET_CATEGORY.value: [CallbackQueryHandler(single_trade_asset_category, pattern="^(cat_|asset_manual)"), MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_asset)],
            SingleTradeState.ASSET.value: [CallbackQueryHandler(single_trade_asset, pattern="^asset_"), MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_asset)],
            SingleTradeState.DIRECTION.value: [CallbackQueryHandler(single_trade_direction, pattern="^dir_")],
            SingleTradeState.ENTRY.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_entry)],
            SingleTradeState.STOP_LOSS.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_stop_loss)],
            SingleTradeState.RISK_LEVEL.value: [CallbackQueryHandler(single_trade_risk_level, pattern="^risk_")],
            SingleTradeState.TAKE_PROFIT.value: [MessageHandler(filters.TEXT & ~filters.COMMAND, single_trade_take_profit)],
        },
        fallbacks=[CommandHandler('start', start_command)]
    )
    application.add_handler(conv)

    # Other handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(lambda u,c: SafeMessageSender.answer_callback_query(u.callback_query) , pattern="^$"), group=100)  # no-op catch
    application.add_handler(CallbackQueryHandler(lambda u,c: logger.info("unrouted callback: %s", u.callback_query.data)))
    # main menu router
    application.add_handler(CallbackQueryHandler(dispatcher_callback_router))

# A compact router for callbacks
async def dispatcher_callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await SafeMessageSender.answer_callback_query(q)
    data = q.data
    try:
        if data == "main_menu":
            await start_command(update, context)
        elif data == "portfolio":
            await show_portfolio(update, context)
        elif data == "pro_calculation":
            await pro_calculation_handler(update, context)
        elif data == "donate_start":
            await donate_start(update, context)
        elif data == "donate_usdt":
            await donate_usdt(update, context)
        elif data == "single_trade":
            await single_trade_start(update, context)
        elif data == "clear_portfolio":
            await clear_portfolio_handler(update, context)
        else:
            await SafeMessageSender.edit_message_text(q, f"Неизвестная команда: {safe_html_escape(data)}", InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]]))
    except Exception as e:
        logger.exception("Error in dispatcher: %s", e)
        try:
            await q.answer("❌ Произошла ошибка")
        except Exception:
            pass

# ---------------------------
# Web server & webhook setup
# ---------------------------
async def set_webhook(application: Application) -> bool:
    if not WEBHOOK_URL:
        return False
    try:
        webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
        logger.info("Setting webhook to %s", webhook_url)
        await application.bot.set_webhook(webhook_url, allowed_updates=[])
        info = await application.bot.get_webhook_info()
        logger.info("Webhook info: %s", info)
        return True
    except Exception as e:
        logger.error("Failed to set webhook: %s", e)
        return False

async def start_http_server(application: Application):
    from aiohttp import web
    app = web.Application()
    async def handle_webhook(request):
        try:
            body = await request.text()
            update = Update.de_json(json.loads(body), application.bot)
            await application.process_update(update)
            return web.Response(status=200)
        except Exception as e:
            logger.exception("Webhook processing error: %s", e)
            return web.Response(status=500)

    async def render_health_check(request):
        return web.Response(text="OK", status=200)

    app.router.add_post(WEBHOOK_PATH, handle_webhook)
    app.router.add_get("/health", render_health_check)
    app.router.add_get("/", render_health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logger.info("HTTP server started on port %s", PORT)
    return runner

# ---------------------------
# MAIN
# ---------------------------
async def main():
    application = RobustApplicationBuilder.create_application(TOKEN)
    # register handlers and conv
    setup_conversation_handlers(application)

    # Register global error handler
    application.add_error_handler(global_error_handler)

    # add fallback message handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u,c: SafeMessageSender.send_message(u, "Используйте меню или /start", c)))

    # Start either webhook or polling
    try:
        if WEBHOOK_URL:
            await application.initialize()
            ok = await set_webhook(application)
            if ok:
                await start_http_server(application)
                logger.info("Bot started in webhook mode.")
                # keep running
                while True:
                    await asyncio.sleep(300)
            else:
                logger.warning("Webhook setup failed, falling back to polling")
                await application.run_polling()
        else:
            logger.info("Starting in polling mode")
            await application.run_polling()
    finally:
        # ensure http session closed
        if market_data_provider.session:
            await market_data_provider.session.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown requested. Exiting...")

