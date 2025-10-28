# Risk Management Calculator Bot 🤖

Профессиональный калькулятор управления рисками для FOREX трейдеров с поддержкой русского и английского языков.

## 🌟 Особенности

- 📊 **Расчет позиции** с учетом риска 2% от депозита
- ⚖️ **Учет кредитного плеча** и маржинальных требований
- 🎯 **Распределение объемов** между тейк-профитами
- 💰 **Расчет прибыли** и рисков в реальном времени
- 🌐 **Поддержка 12+ валютных пар** и металлов
- 💾 **Сохранение пресетов** для быстрого доступа

## 🚀 Быстрый старт

### Русская версия
1. Перейдите в бота: [@fxriskmanagement_bot](https://t.me/fxriskmanagement_bot)
2. Используйте команду `/start`
3. Следуйте инструкциям бота

### English version
1. Go to bot: [@FXWaveRisk_bot](https://t.me/FXWaveRisk_bot)
2. Use command `/start`
3. Follow bot instructions

## 📋 Доступные команды

### Основные команды
- `/start` - начать новый расчет
- `/info` - полная инструкция
- `/help` - краткая справка
- `/presets` - сохраненные стратегии

### Поддерживаемые валютные пары
- **Форекс пары:** EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD, EURGBP, EURJPY, GBPJPY
- **Металлы:** XAUUSD (Gold), XAGUSD (Silver)
- **Криптовалюты:** BTCUSD, ETHUSD

## 🛠 Технические детали

### Структура проекта

risk-management-bot/
├── bot.py # Русская версия бота
├── bot_en.py # Английская версия бота
├── requirements.txt # Зависимости Python
├── render.yaml # Конфигурация русского бота
├── render_en.yaml # Конфигурация английского бота
└── README.md # Документация


### Зависимости
- python-telegram-bot[webhooks]==21.0.1
- python-dotenv==1.0.0

## 🔧 Установка и развертывание

### Локальная разработка
1. Клонируйте репозиторий
2. Установите зависимости: `pip install -r requirements.txt`
3. Создайте файл `.env` с переменными окружения
4. Запустите бота: `python bot.py` или `python bot_en.py`

### Развертывание на Render
1. Создайте два веб-сервиса на Render
2. Используйте соответствующие `.yaml` файлы для конфигурации
3. Настройте переменные окружения в панели Render

## ⚙️ Переменные окружения

### Русский бот
```env
TELEGRAM_BOT_TOKEN=7993604333:AAEA-QABJ5wkSgXkiwZJhqqSbuKdqSF4DDw
RENDER_EXTERNAL_URL=https://risk-management-bot-gvhk.onrender.com

Английский бот
env
TELEGRAM_BOT_TOKEN_EN=7835480792:AAFo7VckIQHLph1grNxknhAXH1_HWph-Q1o
RENDER_EXTERNAL_URL_EN=https://fxwave-risk-bot.onrender.com
🎯 Пример использования
Ввод параметров:

text
Депозит: $1000
Плечо: 1:100
Пара: EURUSD
Вход: 1.0660
SL: 1.0640
TP: 1.0680, 1.0700
Распределение: 50, 50
Результаты:

Размер позиции: 0.50 лота

Риск на сделку: $20 (2% от депозита)

Прибыль по TP1: $100

Прибыль по TP2: $200

🤝 Разработчик
По вопросам и предложениям: @fxfeelgood

📄 Лицензия
MIT License - подробности в файле LICENSE.

Сделано с ❤️ для трейдеров 📈

text

### 2. **bot_en.py** - дублирование валютных пар
Исправьте секцию PIP_VALUES (убрать дубликаты):
```python
PIP_VALUES = {
    'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9, 'USDCHF': 10,
    'USDCAD': 10, 'AUDUSD': 10, 'NZDUSD': 10, 'EURGBP': 10,
    'EURJPY': 9, 'GBPJPY': 9, 'XAUUSD': 10, 'XAGUSD': 50,
    'BTCUSD': 1, 'ETHUSD': 1
}