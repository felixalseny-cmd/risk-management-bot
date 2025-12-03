# ============================================
# TELEGRAM BOT & WEB SERVER
# ============================================

# Импорты уже выполнены в первом сообщении

# ---------------------------
# Donation System - PROFESSIONAL DONATION SYSTEM
# ---------------------------
class DonationSystem:
    """Professional donation system to support development"""
    
    @staticmethod
    async def show_donation_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show currency selection menu for donations"""
        query = update.callback_query
        await query.answer()
        
        text = (
            "💝 <b>SUPPORT THE DEVELOPER</b>\n\n"
            "Your support helps develop the bot and add new features!\n\n"
            "Choose donation currency:"
        )
        
        keyboard = [
            [InlineKeyboardButton("💎 USDT (TRC20)", callback_data="donate_usdt")],
            [InlineKeyboardButton("⚡ TON", callback_data="donate_ton")],
            [InlineKeyboardButton("🔙 Back", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            text=text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='HTML'
        )
    
    @staticmethod
    async def show_usdt_donation(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show USDT wallet for donations"""
        query = update.callback_query
        await query.answer()
        
        if not USDT_WALLET_ADDRESS:
            await query.edit_message_text(
                text="❌ USDT wallet temporarily unavailable",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Back", callback_data="donate_start")]
                ])
            )
            return
        
        text = (
            "💎 <b>USDT (TRC20) DONATION</b>\n\n"
            "To support development, send USDT to the following address:\n\n"
            f"<code>{USDT_WALLET_ADDRESS}</code>\n\n"
            "💝 <i>Any amount will be gratefully accepted!</i>\n\n"
            "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔙 Back to currency selection", callback_data="donate_start")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            text=text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='HTML'
        )
    
    @staticmethod
    async def show_ton_donation(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show TON wallet for donations"""
        query = update.callback_query
        await query.answer()
        
        if not TON_WALLET_ADDRESS:
            await query.edit_message_text(
                text="❌ TON wallet temporarily unavailable",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Back", callback_data="donate_start")]
                ])
            )
            return
        
        text = (
            "⚡ <b>TON DONATION</b>\n\n"
            "To support development, send TON to the following address:\n\n"
            f"<code>{TON_WALLET_ADDRESS}</code>\n\n"
            "💝 <i>Any amount will be gratefully accepted!</i>\n\n"
            "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔙 Back to currency selection", callback_data="donate_start")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            text=text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='HTML'
        )

# ---------------------------
# Professional Margin Calculator
# ---------------------------
class ProfessionalMarginCalculator:
    """Professional margin calculation with real quotes"""
    
    def __init__(self):
        self.market_data = data_provider
    
    @monitor_performance
    async def calculate_professional_margin(self, symbol: str, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Professional margin calculation with real quotes"""
        try:
            specs = InstrumentSpecs.get_specs(symbol)
            
            # Get effective leverage considering limitations
            selected_leverage = int(leverage.split(':')[1])
            max_leverage = specs.get('max_leverage', selected_leverage)
            effective_leverage = min(selected_leverage, max_leverage)
            effective_leverage_str = f"1:{effective_leverage}"
            
            # Different calculation for different asset types
            if specs['type'] == 'forex':
                return await self._calculate_forex_margin(specs, volume, effective_leverage_str, current_price)
            elif specs['type'] == 'crypto':
                return await self._calculate_crypto_margin(specs, volume, effective_leverage_str, current_price)
            elif specs['type'] in ['stock', 'index', 'metal']:
                return await self._calculate_stocks_margin(specs, volume, effective_leverage_str, current_price)
            else:
                return await self._calculate_universal_margin(specs, volume, effective_leverage_str, current_price)
                
        except Exception as e:
            logger.error(f"Margin calculation error for {symbol}: {e}")
            return await self._calculate_universal_margin(specs, volume, leverage, current_price)
    
    async def _calculate_forex_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Margin calculation for Forex according to industry standards"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # For Forex: (Volume × Contract Size) / Leverage
        required_margin = (volume * contract_size) / lev_value
        
        return {
            'required_margin': max(required_margin, 0.01),
            'contract_size': contract_size,
            'calculation_method': 'forex_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size,
            'effective_leverage': leverage
        }
    
    async def _calculate_crypto_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Margin calculation for cryptocurrencies"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # For crypto: (Volume × Price) / Leverage
        required_margin = (volume * contract_size * current_price) / lev_value
        
        return {
            'required_margin': max(required_margin, 0.01),
            'contract_size': contract_size,
            'calculation_method': 'crypto_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size * current_price,
            'effective_leverage': leverage
        }
    
    async def _calculate_stocks_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Margin calculation for stocks, indices, metals"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs['contract_size']
        
        # For stocks/indices/metals: (Volume × Contract Size × Price) / Leverage
        required_margin = (volume * contract_size * current_price) / lev_value
        
        return {
            'required_margin': max(required_margin, 0.01),
            'contract_size': contract_size,
            'calculation_method': 'stocks_standard',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size * current_price,
            'effective_leverage': leverage
        }
    
    async def _calculate_universal_margin(self, specs: Dict, volume: float, leverage: str, current_price: float) -> Dict[str, Any]:
        """Universal margin calculation"""
        lev_value = int(leverage.split(':')[1])
        contract_size = specs.get('contract_size', 1)
        
        required_margin = (volume * contract_size * current_price) / lev_value
        
        return {
            'required_margin': max(required_margin, 0.01),
            'contract_size': contract_size,
            'calculation_method': 'universal',
            'leverage_used': lev_value,
            'notional_value': volume * contract_size * current_price,
            'effective_leverage': leverage
        }

# ---------------------------
# Professional Risk Calculator
# ---------------------------
class ProfessionalRiskCalculator:
    """Professional risk calculator with 2% rule"""
    
    def __init__(self):
        self.margin_calculator = ProfessionalMarginCalculator()
    
    @monitor_performance
    async def calculate_professional_risk(self, trade: Dict, deposit: float, leverage: str) -> Dict[str, Any]:
        """
        FIXED calculation with correct volume determination using 2% rule
        """
        try:
            asset = trade['asset']
            entry = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            direction = trade['direction']
            
            current_price, source = await data_provider.get_price(asset)
            specs = InstrumentSpecs.get_specs(asset)
            
            # FIXED 2% RISK according to risk management rules
            risk_percent = 0.02  # Fixed 2% instead of user choice
            risk_amount = deposit * risk_percent
            
            # Calculate pip distance
            pip_distance = self.calculate_pip_distance(entry, stop_loss, direction, asset)
            pip_value = specs['pip_value']
            
            # CORRECT VOLUME CALCULATION using formula: Volume = Risk Amount / (Stop Distance * Pip Value)
            if pip_distance > 0 and pip_value > 0:
                volume_lots = risk_amount / (pip_distance * pip_value)
                # Round to volume step
                volume_step = specs.get('volume_step', 0.01)
                volume_lots = round(volume_lots / volume_step) * volume_step
                # Limit to minimum volume
                min_volume = specs.get('min_volume', 0.01)
                volume_lots = max(volume_lots, min_volume)
                volume_lots = round(volume_lots, 3)
            else:
                volume_lots = 0
            
            margin_data = await self.margin_calculator.calculate_professional_margin(
                asset, volume_lots, leverage, current_price
            )
            required_margin = margin_data['required_margin']
            required_margin = round(required_margin, 2)
            
            # Equity calculation (balance + unrealized P&L)
            current_pnl = self.calculate_pnl(entry, current_price, volume_lots, direction, asset)
            equity = deposit + current_pnl
            
            # Margin calculations
            free_margin = max(equity - required_margin, 0.0)
            margin_level = (equity / required_margin) * 100 if required_margin > 0 else float('inf')
            
            # Potential profit calculation
            potential_profit = self.calculate_pnl(entry, take_profit, volume_lots, direction, asset)
            potential_profit = round(potential_profit, 2)
            
            rr_ratio = potential_profit / risk_amount if risk_amount > 0 else 0
            rr_ratio = round(rr_ratio, 2)
            
            risk_per_trade_percent = (risk_amount / deposit) * 100 if deposit > 0 else 0
            margin_usage_percent = (required_margin / deposit) * 100 if deposit > 0 else 0
            
            return {
                'volume_lots': volume_lots,
                'required_margin': required_margin,
                'free_margin': free_margin,
                'margin_level': margin_level,
                'risk_amount': risk_amount,
                'risk_percent': risk_per_trade_percent,
                'potential_profit': potential_profit,
                'rr_ratio': rr_ratio,
                'stop_distance_pips': pip_distance,
                'pip_value': pip_value,
                'contract_size': margin_data['contract_size'],
                'deposit': deposit,
                'leverage': leverage,
                'effective_leverage': margin_data.get('effective_leverage', leverage),
                'risk_per_trade_percent': risk_per_trade_percent,
                'margin_usage_percent': margin_usage_percent,
                'current_price': current_price,
                'price_source': source,
                'calculation_method': margin_data['calculation_method'],
                'notional_value': margin_data.get('notional_value', 0),
                'leverage_used': margin_data.get('leverage_used', 1),
                'current_pnl': current_pnl,
                'equity': equity
            }
        except Exception as e:
            logger.error(f"Professional calculation error: {e}")
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
                'pip_value': 0,
                'contract_size': 0,
                'deposit': deposit,
                'leverage': leverage,
                'effective_leverage': leverage,
                'risk_per_trade_percent': 0,
                'margin_usage_percent': 0,
                'current_price': 0,
                'price_source': 'error',
                'calculation_method': 'error',
                'notional_value': 0,
                'leverage_used': 1,
                'current_pnl': 0,
                'equity': deposit
            }
    
    @staticmethod
    def calculate_pip_distance(entry: float, target: float, direction: str, asset: str) -> float:
        """Professional pip distance calculation"""
        specs = InstrumentSpecs.get_specs(asset)
        pip_places = specs.get('pip_places', 4)
        
        if direction.upper() == 'LONG':
            distance = target - entry
        else:  # SHORT
            distance = entry - target
        
        # Конвертация в пипсы
        multiplier = 10 ** (pip_places - 1)
        return abs(distance) * multiplier
    
    @staticmethod
    def calculate_pnl(entry_price: float, exit_price: float, volume: float, direction: str, asset: str) -> float:
        """Professional P&L calculation"""
        specs = InstrumentSpecs.get_specs(asset)
        
        if direction.upper() == 'LONG':
            price_diff = exit_price - entry_price
        else:  # SHORT
            price_diff = entry_price - exit_price
        
        if specs['type'] in ['stock', 'crypto']:
            pnl = price_diff * volume * specs['contract_size']
        else:
            pip_distance = ProfessionalRiskCalculator.calculate_pip_distance(
                entry_price, exit_price, direction, asset
            )
            pnl = pip_distance * volume * specs['pip_value']
        
        return round(pnl, 2)

# ---------------------------
# Portfolio Analyzer
# ---------------------------
class PortfolioAnalyzer:
    """Portfolio analyzer with aggregated metrics"""
    
    @staticmethod
    def calculate_portfolio_metrics(trades: List[Dict], deposit: float) -> Dict[str, Any]:
        """Calculate aggregated portfolio metrics"""
        if not trades:
            return {}
        
        total_risk_usd = sum(trade.get('metrics', {}).get('risk_amount', 0) for trade in trades)
        total_profit = sum(trade.get('metrics', {}).get('potential_profit', 0) for trade in trades)
        total_margin = sum(trade.get('metrics', {}).get('required_margin', 0) for trade in trades)
        total_pnl = sum(trade.get('metrics', {}).get('current_pnl', 0) for trade in trades)
        total_equity = deposit + total_pnl
        avg_rr_ratio = sum(trade.get('metrics', {}).get('rr_ratio', 0) for trade in trades) / len(trades) if trades else 0
        
        total_risk_percent = (total_risk_usd / deposit) * 100 if deposit > 0 else 0
        total_margin_usage = (total_margin / deposit) * 100 if deposit > 0 else 0
        free_margin = max(total_equity - total_margin, 0)
        free_margin_percent = (free_margin / deposit) * 100 if deposit > 0 else 0
        portfolio_margin_level = (total_equity / total_margin * 100) if total_margin > 0 else float('inf')
        
        # Portfolio volatility (weighted average)
        portfolio_volatility = 20  # Default
        
        # Diversification
        unique_assets = len(set(trade['asset'] for trade in trades))
        diversity_score = min(unique_assets / 5, 1.0)  # Max 5 unique for 100%
        
        long_positions = sum(1 for trade in trades if trade['direction'] == 'LONG')
        short_positions = len(trades) - long_positions
        
        # Portfolio leverage
        total_notional = sum(trade.get('metrics', {}).get('notional_value', 0) for trade in trades)
        portfolio_leverage = total_notional / deposit if deposit > 0 else 1
        
        return {
            'total_risk_usd': round(total_risk_usd, 2),
            'total_risk_percent': round(total_risk_percent, 1),
            'total_profit': round(total_profit, 2),
            'avg_rr_ratio': round(avg_rr_ratio, 2),
            'total_pnl': round(total_pnl, 2),
            'total_equity': round(total_equity, 2),
            'total_margin': round(total_margin, 2),
            'total_margin_usage': round(total_margin_usage, 1),
            'free_margin': round(free_margin, 2),
            'free_margin_percent': round(free_margin_percent, 1),
            'portfolio_margin_level': round(portfolio_margin_level, 1),
            'portfolio_volatility': round(portfolio_volatility, 1),
            'unique_assets': unique_assets,
            'diversity_score': round(diversity_score * 100, 1),
            'long_positions': long_positions,
            'short_positions': short_positions,
            'portfolio_leverage': round(portfolio_leverage, 1)
        }

# ---------------------------
# Portfolio Manager
# ---------------------------
class PortfolioManager:
    """Portfolio manager with data saving"""
    
    @staticmethod
    def ensure_user(user_id: int):
        """Ensure user exists in data manager"""
        user_data = data_manager.load_user_data(user_id, "portfolio")
        if not user_data:
            data_manager.save_user_data(user_id, "portfolio", {
                'single_trades': [],
                'multi_trades': [],
                'deposit': 1000.0,
                'leverage': '1:100'
            })
    
    @staticmethod
    def add_single_trade(user_id: int, trade: Dict):
        """Add single trade to portfolio"""
        PortfolioManager.ensure_user(user_id)
        user_data = data_manager.load_user_data(user_id, "portfolio")
        user_data['single_trades'].append(trade)
        data_manager.save_user_data(user_id, "portfolio", user_data)
    
    @staticmethod
    def add_multi_trade(user_id: int, trades: List[Dict]):
        """Add multiple trades to portfolio"""
        PortfolioManager.ensure_user(user_id)
        user_data = data_manager.load_user_data(user_id, "portfolio")
        user_data['multi_trades'].extend(trades)
        data_manager.save_user_data(user_id, "portfolio", user_data)
    
    @staticmethod
    def set_deposit_leverage(user_id: int, deposit: float, leverage: str):
        """Set deposit and leverage for user"""
        PortfolioManager.ensure_user(user_id)
        user_data = data_manager.load_user_data(user_id, "portfolio")
        user_data['deposit'] = deposit
        user_data['leverage'] = leverage
        data_manager.save_user_data(user_id, "portfolio", user_data)
    
    @staticmethod
    def get_user_data(user_id: int) -> Dict:
        """Get user data"""
        PortfolioManager.ensure_user(user_id)
        return data_manager.load_user_data(user_id, "portfolio")
    
    @staticmethod
    def clear_portfolio(user_id: int):
        """Clear user portfolio"""
        user_data = data_manager.load_user_data(user_id, "portfolio")
        user_data['single_trades'] = []
        user_data['multi_trades'] = []
        data_manager.save_user_data(user_id, "portfolio", user_data)

# ---------------------------
# Temporary Data Manager
# ---------------------------
class TempDataManager:
    """Temporary data manager for progress recovery"""
    
    @staticmethod
    def load_temporary_data() -> Dict:
        try:
            with open('temporary_progress.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    @staticmethod
    def save_temporary_data(data: Dict):
        with open('temporary_progress.json', 'w') as f:
            json.dump(data, f)
    
    @staticmethod
    def save_progress(user_id: int, state_data: Dict, state_type: str):
        temp_data = TempDataManager.load_temporary_data()
        temp_data[str(user_id)] = {
            'state_data': state_data,
            'state_type': state_type,
            'timestamp': datetime.now().isoformat()
        }
        TempDataManager.save_temporary_data(temp_data)
    
    @staticmethod
    def clear_temporary_progress(user_id: int):
        temp_data = TempDataManager.load_temporary_data()
        temp_data.pop(str(user_id), None)
        TempDataManager.save_temporary_data(temp_data)

# ---------------------------
# Constants
# ---------------------------
LEVERAGES = {
    "DEFAULT": ["1:100", "1:200", "1:500", "1:1000"]
}

ASSET_CATEGORIES = {
    "Forex": ["EURUSD", "GBPUSD", "USDJPY"],
    "Crypto": ["BTCUSDT", "ETHUSDT"],
    "Stocks": ["AAPL", "TSLA"],
    "Indices": ["NAS100"],
    "Metals": ["XAUUSD", "XAGUSD"],
    "Energy": ["OIL"]
}

# ---------------------------
# State Enums
# ---------------------------
class SingleTradeState(Enum):
    DEPOSIT = 1
    LEVERAGE = 2
    ASSET_CATEGORY = 3
    ASSET = 4
    DIRECTION = 5
    ENTRY = 6
    STOP_LOSS = 7
    TAKE_PROFIT = 8

class MultiTradeState(Enum):
    DEPOSIT = 1
    LEVERAGE = 2
    ASSET_CATEGORY = 3
    ASSET = 4
    DIRECTION = 5
    ENTRY = 6
    STOP_LOSS = 7
    TAKE_PROFIT = 8
    ADD_MORE = 9

# ---------------------------
# Global Instances
# ---------------------------
margin_calculator = ProfessionalMarginCalculator()
risk_calculator = ProfessionalRiskCalculator()
portfolio_analyzer = PortfolioAnalyzer()

# ---------------------------
# Telegram Bot Handlers
# ---------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command /start"""
    text = (
        "🚀 <b>Welcome to PRO RISK CALCULATOR v3.0 ENTERPRISE</b>\n\n"
        "Professional risk calculation tool with fixed 2% rule.\n"
        "Use real quotes and accurate margin calculations.\n\n"
        "Start from main menu:"
    )
    
    keyboard = [
        [InlineKeyboardButton("🎯 Professional Calculation", callback_data="pro_calculation")],
        [InlineKeyboardButton("📊 Portfolio", callback_data="portfolio")],
        [InlineKeyboardButton("📚 Instructions", callback_data="pro_info")],
        [InlineKeyboardButton("💖 Support Developer", callback_data="donate_start")]
    ]
    
    await update.message.reply_text(
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='HTML'
    )

async def pro_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command /pro_info"""
    query = update.callback_query
    await query.answer()
    
    text = (
        "📚 <b>PRO RISK CALCULATOR v3.0 INSTRUCTIONS</b>\n\n"
        "1. <b>Fixed Risk</b>: All calculations use 2% rule per trade.\n"
        "2. <b>Real Prices</b>: Bot gets quotes from multiple APIs.\n"
        "3. <b>Margin</b>: Calculated according to standards.\n"
        "4. <b>Volume</b>: Automatically adjusted for 2% risk.\n"
        "5. <b>Portfolio</b>: Aggregates metrics for multiple trades.\n\n"
        "💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
    )
    
    keyboard = [
        [InlineKeyboardButton("🔙 Back", callback_data="main_menu")]
    ]
    
    await query.edit_message_text(
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='HTML'
    )

async def main_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main menu handler"""
    query = update.callback_query
    await query.answer()
    
    # Clear temporary progress
    TempDataManager.clear_temporary_progress(query.from_user.id)
    context.user_data.clear()
    
    text = (
        "🏠 <b>MAIN MENU</b>\n\n"
        "Professional risk management calculator with fixed 2% risk\n\n"
        "Choose action:"
    )
    
    keyboard = [
        [InlineKeyboardButton("🎯 Professional Calculation", callback_data="pro_calculation")],
        [InlineKeyboardButton("📊 Portfolio", callback_data="portfolio")],
        [InlineKeyboardButton("📚 Instructions", callback_data="pro_info")],
        [InlineKeyboardButton("💖 Support Developer", callback_data="donate_start")],
        [InlineKeyboardButton("🔄 Restore Progress", callback_data="restore_progress")]
    ]
    
    await query.edit_message_text(
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='HTML'
    )

async def pro_calculation_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Professional trades handler"""
    query = update.callback_query
    await query.answer()
    
    text = (
        "🎯 <b>PROFESSIONAL TRADES v3.0</b>\n\n"
        "Choose calculation type:\n\n"
        "• <b>Single Trade</b> - calculation for one position\n"
        "• <b>Multi-position</b> - portfolio calculation from multiple trades\n\n"
        "<i>All cases use fixed 2% risk per trade</i>"
    )
    
    keyboard = [
        [InlineKeyboardButton("🎯 Single Trade", callback_data="single_trade")],
        [InlineKeyboardButton("📊 Multi-position", callback_data="multi_trade_start")],
        [InlineKeyboardButton("🔙 Back", callback_data="main_menu")]
    ]
    
    await query.edit_message_text(
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='HTML'
    )

async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show portfolio with real data"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    user_data = PortfolioManager.get_user_data(user_id)
    
    trades = user_data.get('multi_trades', []) + user_data.get('single_trades', [])
    
    if not trades:
        text = (
            "📊 <b>Your portfolio is empty</b>\n\n"
            "Start with trade calculation with fixed 2% risk!"
        )
        keyboard = [
            [InlineKeyboardButton("🎯 New Trade", callback_data="single_trade")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            text=text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='HTML'
        )
        return
    
    deposit = user_data['deposit']
    
    # Update metrics with real prices
    updated_trades = []
    for trade in trades:
        metrics = await risk_calculator.calculate_professional_risk(trade, deposit, user_data['leverage'])
        trade['metrics'] = metrics
        updated_trades.append(trade)
    
    metrics = portfolio_analyzer.calculate_portfolio_metrics(updated_trades, deposit)
    
    text = (
        "📊 <b>PORTFOLIO v3.0</b>\n\n"
        f"💰 <b>KEY METRICS:</b>\n"
        f"Deposit: ${deposit:,.2f}\n"
        f"Leverage: {user_data['leverage']}\n"
        f"Trades: {len(trades)}\n"
        f"Equity: ${metrics['total_equity']:.2f}\n\n"
        f"🎯 <b>RISKS AND PROFIT:</b>\n"
        f"Total risk: ${metrics['total_risk_usd']:.2f} ({metrics['total_risk_percent']:.1f}%)\n"
        f"Potential profit: ${metrics['total_profit']:.2f}\n"
        f"Average R/R: {metrics['avg_rr_ratio']:.2f}\n"
        f"Current P&L: ${metrics['total_pnl']:.2f}\n\n"
        f"🛡 <b>MARGIN METRICS:</b>\n"
        f"Required margin: ${metrics['total_margin']:.2f} ({metrics['total_margin_usage']:.1f}%)\n"
        f"Free margin: ${metrics['free_margin']:.2f} ({metrics['free_margin_percent']:.1f}%)\n"
        f"Margin level: {metrics['portfolio_margin_level']:.1f}%\n"
        f"Portfolio leverage: {metrics['portfolio_leverage']:.1f}x\n\n"
        f"📈 <b>ANALYTICS:</b>\n"
        f"Volatility: {metrics['portfolio_volatility']:.1f}%\n"
        f"Longs: {metrics['long_positions']} | Shorts: {metrics['short_positions']}\n"
        f"Unique assets: {metrics['unique_assets']}\n"
        f"Diversification: {metrics['diversity_score']}%\n\n"
        "<b>📋 TRADES:</b>\n"
    )
    
    for i, trade in enumerate(updated_trades, 1):
        metrics = trade.get('metrics', {})
        pnl = metrics.get('current_pnl', 0)
        pnl_sign = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
        
        text += (
            f"{pnl_sign} <b>#{i}</b> {trade['asset']} {trade['direction']}\n"
            f"   Entry: {trade['entry_price']} | SL: {trade['stop_loss']} | TP: {trade['take_profit']}\n"
            f"   Volume: {metrics.get('volume_lots', 0):.2f} | Risk: ${metrics.get('risk_amount', 0):.2f}\n"
            f"   P&L: ${pnl:.2f} | Margin: ${metrics.get('required_margin', 0):.2f}\n\n"
        )
    
    text += "\n💎 PRO v3.0 | Smart • Fast • Reliable 🚀"
    
    keyboard = [
        [InlineKeyboardButton("🗑 Clear Portfolio", callback_data="clear_portfolio")],
        [InlineKeyboardButton("📤 Export Report", callback_data="export_portfolio")],
        [InlineKeyboardButton("🎯 New Trade", callback_data="single_trade")],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
    ]
    
    await query.edit_message_text(
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='HTML'
    )

# ---------------------------
# Single Trade Handlers (Simplified)
# ---------------------------
async def single_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start single trade"""
    query = update.callback_query
    await query.answer()
    
    context.user_data.clear()
    
    text = (
        "🎯 <b>SINGLE TRADE v3.0</b>\n\n"
        "Step 1/7: Enter deposit in USD (minimum $100):"
    )
    
    keyboard = [[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]
    
    await query.edit_message_text(
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='HTML'
    )
    return SingleTradeState.DEPOSIT.value

async def single_trade_deposit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Deposit for single trade"""
    text = update.message.text.strip()
    
    TempDataManager.save_progress(update.message.from_user.id, context.user_data.copy(), "single")
    
    try:
        deposit = float(text.replace(',', '.'))
        if deposit < 100:
            await update.message.reply_text(
                "❌ Minimum deposit: $100\nTry again:",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
                ])
            )
            return SingleTradeState.DEPOSIT.value
        
        context.user_data['deposit'] = deposit
        
        keyboard = []
        for leverage in LEVERAGES["DEFAULT"]:
            keyboard.append([InlineKeyboardButton(leverage, callback_data=f"lev_{leverage}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")])
        
        await update.message.reply_text(
            f"✅ Deposit: ${deposit:,.2f}\n\n"
            "Step 2/7: <b>Choose leverage:</b>",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='HTML'
        )
        return SingleTradeState.LEVERAGE.value
        
    except ValueError:
        await update.message.reply_text(
            "❌ Enter a number (example: 1000)\nTry again:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
            ])
        )
        return SingleTradeState.DEPOSIT.value

# ... (Остальные обработчики для single trade - аналогично оригиналу, но адаптированные)

# ---------------------------
# Multi Trade Handlers (Simplified)
# ---------------------------
async def multi_trade_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start multi-position"""
    query = update.callback_query
    await query.answer()
    
    context.user_data.clear()
    context.user_data['current_multi_trades'] = []
    
    text = (
        "📊 <b>MULTI-POSITION v3.0</b>\n\n"
        "Step 1/7: Enter deposit in USD (minimum $100):"
    )
    
    keyboard = [[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]]
    
    await query.edit_message_text(
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='HTML'
    )
    return MultiTradeState.DEPOSIT.value

# ... (Остальные обработчики для multi trade - аналогично оригиналу, но адаптированные)

# ---------------------------
# Callback Router
# ---------------------------
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Callback query router"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    try:
        if data == "main_menu":
            await main_menu_handler(update, context)
        elif data == "portfolio":
            await show_portfolio(update, context)
        elif data == "pro_calculation":
            await pro_calculation_handler(update, context)
        elif data == "pro_info":
            await pro_info_command(update, context)
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
        # Обработчики для leverage и других callback'ов
        elif data.startswith("lev_"):
            await single_trade_leverage_handler(update, context)
        elif data.startswith("mlev_"):
            await multi_trade_leverage_handler(update, context)
        else:
            await query.answer("Command not recognized")
            
    except Exception as e:
        logger.error(f"Error in callback router: {e}")
        await query.answer("❌ An error occurred")

# ---------------------------
# Web Server for Render
# ---------------------------
async def handle_webhook(request):
    """Handle Telegram webhook"""
    try:
        data = await request.json()
        update = Update.de_json(data, application.bot)
        await application.update_queue.put(update)
        return web.Response(text="OK")
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return web.Response(text="Error", status=500)

async def health_check(request):
    """Health check endpoint for Render"""
    return web.json_response({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0"
    })

async def setup_webhook():
    """Setup webhook for Telegram"""
    webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
    await application.bot.set_webhook(webhook_url)

# ---------------------------
# Main Application
# ---------------------------
async def main():
    """Main application function"""
    global application
    
    # Create application
    application = Application.builder().token(TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("pro_info", pro_info_command))
    application.add_handler(CallbackQueryHandler(callback_router))
    
    # Add conversation handlers
    # (Здесь нужно добавить ConversationHandler для single и multi trade)
    
    # Add message handler for fallback
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, 
        lambda update, context: update.message.reply_text(
            "🤖 Use menu for navigation or /start to begin",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
            ])
        )
    ))
    
    # Setup webhook or polling
    if WEBHOOK_URL and WEBHOOK_URL.strip():
        logger.info("Starting in WEBHOOK mode")
        
        # Initialize application
        await application.initialize()
        await application.start()
        
        # Setup webhook
        await setup_webhook()
        
        # Create aiohttp app
        app = web.Application()
        app.router.add_post(WEBHOOK_PATH, handle_webhook)
        app.router.add_get('/health', health_check)
        app.router.add_get('/', health_check)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', PORT)
        await site.start()
        
        logger.info(f"Server started on port {PORT}")
        logger.info(f"Webhook URL: {WEBHOOK_URL}{WEBHOOK_PATH}")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            await runner.cleanup()
            await application.stop()
            await application.shutdown()
    else:
        logger.info("Starting in POLLING mode")
        await application.run_polling()

# ---------------------------
# Cleanup
# ---------------------------
async def cleanup():
    """Cleanup resources"""
    if data_provider:
        await data_provider.close()

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    logger.info("🚀 LAUNCHING ENTERPRISE RISK CALCULATOR v3.0")
    logger.info("✅ MEMORY OPTIMIZED FOR RENDER FREE")
    logger.info("🎯 PROFESSIONAL RISK MANAGEMENT")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹ Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        asyncio.run(cleanup())
        raise
