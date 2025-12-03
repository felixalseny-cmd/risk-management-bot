# data_provider.py
# -----------------------------------------
# Быстрый и кэшируемый модуль для:
# - загрузки цен
# - вычисления доходностей
# - хранения mu / sigma
# - работы с asyncio
# - минимизация копий массивов
# -----------------------------------------

import aiohttp
import asyncio
import numpy as np
import time
from cachetools import TTLCache

# -----------------------------------------
# КЭШИ
# -----------------------------------------

# цены: храним 10 минут
_price_cache = TTLCache(maxsize=32, ttl=600)

# статистики: храним 10 минут
_stats_cache = TTLCache(maxsize=32, ttl=600)

# чтобы избежать гонок при одновременной загрузке одного и того же символа
_locks = {}

# -----------------------------------------
# ФУНКЦИИ ПОЛУЧЕНИЯ ДАННЫХ С API
# -----------------------------------------

async def _fetch_prices_binance(symbol: str, limit: int = 1200):
    """
    Получение свечей с Binance.
    Возвращает numpy.array цен close.
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as r:
            data = await r.json()

    # Close = data[][4]
    closes = np.fromiter((float(k[4]) for k in data), dtype=np.float64)
    return closes


async def get_prices(symbol: str) -> np.ndarray:
    """
    Возвращает массив цен (numpy).
    Кэширует на 10 минут.
    """
    if symbol in _price_cache:
        return _price_cache[symbol]

    # создаём lock под конкретный symbol
    lock = _locks.setdefault(symbol, asyncio.Lock())

    async with lock:
        # за время ожидания лок могли уже обновить кэш → проверяем ещё раз
        if symbol in _price_cache:
            return _price_cache[symbol]

        closes = await _fetch_prices_binance(symbol)
        _price_cache[symbol] = closes
        return closes


# -----------------------------------------
# ВОЗВРАТЫ + ПРЕДВЫЧИСЛЕНИЕ MU / SIGMA
# -----------------------------------------

def _compute_stats(returns: np.ndarray):
    """
    mu, sigma для Монте-Карло.
    """
    mu = float(np.mean(returns))
    sigma = float(np.std(returns))
    return mu, sigma


async def get_returns_stats(symbol: str):
    """
    Быстрая выдача:
    - returns (numpy)
    - mu (float)
    - sigma (float)

    Всё кэшируется.
    """
    if symbol in _stats_cache:
        return _stats_cache[symbol]

    prices = await get_prices(symbol)

    # ВАЖНО: делаем разность векторно, без копий
    returns = prices[1:] / prices[:-1] - 1.0

    mu, sigma = _compute_stats(returns)

    bundle = (returns, mu, sigma)
    _stats_cache[symbol] = bundle
    return bundle


# -----------------------------------------
# Быстрая выдача только returns
# -----------------------------------------

async def get_returns(symbol: str) -> np.ndarray:
    returns, _, _ = await get_returns_stats(symbol)
    return returns


# -----------------------------------------
# Быстрая выдача только mu, sigma
# -----------------------------------------

async def get_mu_sigma(symbol: str):
    _, mu, sigma = await get_returns_stats(symbol)
    return mu, sigma


# -----------------------------------------
# Сброс кэшей (при необходимости)
# -----------------------------------------

def clear_caches():
    _price_cache.clear()
    _stats_cache.clear()
