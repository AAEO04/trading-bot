import os
import hashlib
import ccxt
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
    ConversationHandler,
    AIORateLimiter
)
from aiohttp import web
import asyncio
import logging
import threading
import uuid
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from fuzzywuzzy import process
from typing import List, Tuple
from datetime import datetime
from telegram import Bot
from logging.handlers import RotatingFileHandler

class PriceAlert:
    def __init__(self):
        self.alerts = {}  # user_id -> List[alert_details]
    
    async def add_alert(self, user_id: int, pair: str, price: float, condition: str) -> None:
        """Add a new price alert."""
        if user_id not in self.alerts:
            self.alerts[user_id] = []
        
        self.alerts[user_id].append({
            "pair": pair,
            "price": price,
            "condition": condition,
            "created_at": datetime.now()
        })

    async def check_alerts(self, bot: Bot) -> None:
        """Check all active alerts."""
        while True:
            try:
                for user_id, user_alerts in self.alerts.items():
                    for alert in user_alerts[:]:  # Copy to allow modification
                        current_price = float((await asyncio.to_thread(
                            EXCHANGE.fetch_ticker, alert["pair"]))['last'])
                        
                        if ((alert["condition"] == "above" and current_price > alert["price"]) or
                            (alert["condition"] == "below" and current_price < alert["price"])):
                            await bot.send_message(
                                chat_id=user_id,
                                text=f"🔔 *Price Alert*\n"
                                     f"{alert['pair']} is {alert['condition']} {alert['price']}\n"
                                     f"Current price: {current_price}",
                                parse_mode='Markdown'
                            )
                            user_alerts.remove(alert)
            except Exception as e:
                logging.error(f"Error checking alerts: {e}")
            await asyncio.sleep(60)  # Check every minute

# Initialize global variables
price_alerts: Optional[PriceAlert] = None
price_alerts = None  # Will be initialized in __main__
bot_app = None  # Will be initialized in __main__
START_TIME = datetime.now()

# Define conversation states
AUTH = 0
SELECTING_STRATEGY = 1

# Define logger
logger = logging.getLogger(__name__)

# Initialize handlers list
handlers = []  # Will be populated after handler functions are defined




# ====================== CONFIGURATION ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'bot.log',
            maxBytes=1024*1024,  # 1MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
# Initialize empty handlers list that will be populated later
handlers = []

# Create logger for this module
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    telegram_token: str
    password_hash: str
    port: int = 10000
    webhook_secret: str = ""
    render: bool = False
    environment: str = "development"

    model_config = ConfigDict(
        env_file=".env",
        extra="allow"  # Allow extra fields from environment variables
    )

# Instantiate the Settings class
config = Settings()

# Access configuration values
TELEGRAM_TOKEN = config.telegram_token
PASSWORD_HASH = config.password_hash
PORT = int(os.getenv("PORT", 10000))  # Default to 10000 if PORT is not set
WEBHOOK_SECRET = config.webhook_secret
IS_RENDER = config.render

AUTHORIZED_USERS: Dict[int, bool] = {}
EXCHANGE = ccxt.binance({"enableRateLimit": True})
STRATEGIES = ["MA Crossover", "RSI", "MACD", "ML Enhanced"]
DEFAULT_TRADING_PAIR = "BTC/USDT"

plot_lock = threading.Lock()

AUTH = 0  # Define AUTH as a state for the conversation handler
SELECTING_STRATEGY = 1  # Add this constant

TIMEFRAMES = {
    "1m": "1 minute",
    "5m": "5 minutes",
    "15m": "15 minutes",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
    "1w": "1 week"
}

# ====================== TRADING CORE ======================
class MLTrader:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        if IS_RENDER:
            self._init_new_models()
            logging.info("Render environment: Using fresh models for this session")
        else:
            self._load_or_create_models()

    def _init_new_models(self):
        self.models = {
            "linear": LinearRegression(),
            "svm": SVR(),
            "ann": MLPRegressor(hidden_layer_sizes=(50, 50))
        }

    def _load_or_create_models(self):
        try:
            self.models["linear"] = joblib.load("model_linear.pkl")
        except FileNotFoundError:
            self.models["linear"] = LinearRegression()
        try:
            self.models["svm"] = joblib.load("model_svm.pkl")
        except FileNotFoundError:
            self.models["svm"] = SVR()
        try:
            self.models["ann"] = joblib.load("model_ann.pkl")
        except FileNotFoundError:
            self.models["ann"] = MLPRegressor(hidden_layer_sizes=(50, 50))

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        # Ensure DataFrame has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame missing required columns. Expected {required_columns}, got {df.columns.tolist()}")

        # Calculate indicators
        df = df.copy()  # Create a copy to avoid modifying the original

        # Calculate Moving Averages
        df['MA_50'] = df['close'].rolling(window=50).mean()
        df['MA_200'] = df['close'].rolling(window=200).mean()

        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2

        # Forward fill NaN values
        df.ffill(inplace=True)
        # Backward fill any remaining NaN values at the beginning
        df.bfill(inplace=True)

       # Prevent division by zero in RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        loss = loss.replace(0, np.finfo(float).eps)  # Replace zeros with small value
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    def train(self, df: pd.DataFrame) -> None:
        """Train ML models on the provided data."""
        df = self._add_indicators(df)
        X = df[["MA_50", "MA_200", "RSI", "MACD"]].iloc[:-1]
        y = df["close"].pct_change().shift(-1).dropna()
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.models.items():
            model.fit(X_scaled, y)
            if not IS_RENDER:
                joblib.dump(model, f"model_{name}.pkl")

class TradingStrategy:
    def __init__(self):
        self.ml_trader = MLTrader()
    
    def calculate_signals(self, df: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
        """Calculate trading signals using multiple strategies."""
        signals = pd.DataFrame(index=df.index)
        
        for strategy in strategies:
            if strategy == "MA Crossover":
                signals['ma_signal'] = np.where(df['MA_50'] > df['MA_200'], 1, -1)
            
            elif strategy == "RSI":
                signals['rsi_signal'] = np.where(
                    (df['RSI'] < 30), 1,  # Oversold - Buy
                    np.where(df['RSI'] > 70, -1, 0)  # Overbought - Sell
                )
            
            elif strategy == "MACD":
                signals['macd_signal'] = np.where(
                    df['MACD'] > 0, 1,  # MACD above 0 - Buy
                    np.where(df['MACD'] < 0, -1, 0)  # MACD below 0 - Sell
                )
            
            elif strategy == "ML Enhanced":
                # Prepare features for ML prediction
                X = df[["MA_50", "MA_200", "RSI", "MACD"]].values
                X_scaled = self.ml_trader.scaler.transform(X)
                
                # Get predictions from all models
                predictions = []
                for model in self.ml_trader.models.values():
                    pred = model.predict(X_scaled)
                    predictions.append(np.where(pred > 0, 1, -1))
                
                # Ensemble prediction (majority vote)
                signals['ml_signal'] = np.sign(np.mean(predictions, axis=0))
        
        # Combine signals using weighted average
        weights = {
            'ma_signal': 0.3,
            'rsi_signal': 0.2,
            'macd_signal': 0.2,
            'ml_signal': 0.3
        }
        
        final_signal = pd.Series(0, index=df.index)
        for col, weight in weights.items():
            if (col in signals.columns):
                final_signal += signals[col] * weight
        
        # Threshold for final decision
        return pd.DataFrame({
            'signal': np.where(final_signal > 0.1, 1, 
                             np.where(final_signal < -0.1, -1, 0)),
            'signal_strength': final_signal.abs()
        })

class Backtester:
    def __init__(self, tp: float = 5, sl: float = 3, position_size: float = 0.2) -> None:
        self.tp = tp / 100
        self.sl = sl / 100
        self.position_size = position_size
        self.strategy = TradingStrategy()

    # Fix return type annotation
    def run(self, df: pd.DataFrame, strategies: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Run backtest with multiple strategies."""
        signals = self.strategy.calculate_signals(df, strategies)
        df = df.join(signals)
        
        capital: float = 10000
        position: float = 0
        entry_price: Optional[float] = None
        portfolio = []
        trades = []
        
        for i, row in df.iterrows():
            price = row["close"]
            
            # Close position if take profit or stop loss hit
            if position != 0 and entry_price is not None:
                ret = (price - entry_price) / entry_price
                if (position > 0 and (ret >= self.tp or ret <= -self.sl)) or \
                   (position < 0 and (-ret >= self.tp or -ret <= -self.sl)):
                    profit = position * (price - entry_price)
                    capital += profit
                    trades.append({
                        'exit_time': i,
                        'profit': profit,
                        'roi': ret * 100
                    })
                    position = 0
                    entry_price = None
            
            # Open new position based on signal
            if position == 0 and row['signal'] != 0:
                position_capital = capital * self.position_size
                position = (position_capital / price) * np.sign(row['signal'])
                entry_price = price
                trades.append({
                    'entry_time': i,
                    'position': position,
                    'entry_price': entry_price
                })
            
            portfolio.append(capital + position * price)
        
        df["portfolio"] = portfolio
        
        # Calculate performance metrics
        df['returns'] = df['portfolio'].pct_change()
        metrics = {
            'total_return': (df['portfolio'].iloc[-1] / df['portfolio'].iloc[0] - 1) * 100,
            'sharpe_ratio': df['returns'].mean() / df['returns'].std() * np.sqrt(252),
            'max_drawdown': (df['portfolio'] / df['portfolio'].cummax() - 1).min() * 100,
            'num_trades': len(trades)
        }
        
        logging.info(f"Backtest metrics: {metrics}")
        return df, metrics

class PortfolioAnalytics:
    @staticmethod
    async def generate_report(df: pd.DataFrame, metrics: Dict[str, float]) -> str:
        """Generate detailed portfolio analysis."""
        daily_returns = df['portfolio'].pct_change()
        annualized_return = ((1 + daily_returns.mean()) ** 252 - 1) * 100
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Calculate drawdown series
        drawdown = (df['portfolio'] / df['portfolio'].cummax() - 1) * 100
        avg_drawdown = drawdown[drawdown < 0].mean()
        
        # Calculate win rate
        trades = df['signal'].diff().abs().sum() / 2
        winning_trades = len(daily_returns[daily_returns > 0])
        win_rate = (winning_trades / trades) * 100 if trades > 0 else 0
        
        return (
            "📊 *Detailed Portfolio Analysis*\n\n"
            f"Annual Return: {annualized_return:.2f}%\n"
            f"Volatility: {volatility:.2f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
            f"Avg Drawdown: {avg_drawdown:.2f}%\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Total Trades: {metrics['num_trades']}"
        )

class SignalGenerator:
    def __init__(self, leverage_range: Tuple[int, int] = (20, 75)):
        self.min_leverage, self.max_leverage = leverage_range
        self.ml_trader = MLTrader()
    
    def calculate_entry_range(self, current_price: float, volatility: float) -> Tuple[float, float]:
        """Calculate entry range based on price and volatility."""
        spread = current_price * (volatility * 0.1)  # 10% of volatility
        return (
            round(current_price - spread, 3),
            round(current_price + spread, 3)
        )
    
    def calculate_tp_sl(self, 
                       entry_mid: float, 
                       direction: str, 
                       risk_reward: float = 1.5) -> Tuple[float, float]:
        """Calculate take profit and stop loss levels."""
        if direction == "LONG":
            sl = entry_mid * 0.93  # 7% stop loss
            tp = entry_mid + (entry_mid - sl) * risk_reward
        else:
            sl = entry_mid * 1.07  # 7% stop loss
            tp = entry_mid - (sl - entry_mid) * risk_reward
        
        return round(tp, 3), round(sl, 3)

    async def generate_signal(self, pair: str, timeframe: str = "5m") -> Optional[str]:
        """Generate trading signal with error recovery and locking."""
        lock = asyncio.Lock()  # Add lock to prevent concurrent signal generation
        
        async with lock:
            retries = 3
            for attempt in range(retries):
                try:
                    data = await fetch_data_async(pair, timeframe, limit=100)
                    if not data:
                        continue
                        
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df = self.ml_trader._add_indicators(df)
                    
                    current_price = df['close'].iloc[-1]
                    volatility = df['close'].pct_change().std()
                    
                    signal_strength = self._calculate_signal_strength(df)
                    
                    if abs(signal_strength) >= 2:
                        direction = "LONG" if signal_strength > 0 else "SHORT"
                        entry_low, entry_high = self.calculate_entry_range(current_price, volatility)
                        tp, sl = self.calculate_tp_sl(current_price, direction)
                        
                        return self._format_signal_message(
                            direction, pair, entry_low, entry_high, tp, sl
                        )
                    return None
                    
                except Exception as e:
                    if attempt == retries - 1:
                        logger.error(f"Signal generation failed: {e}")
                        raise
                    await asyncio.sleep(1)

    def _calculate_signal_strength(self, df: pd.DataFrame) -> float:
        """Calculate signal strength based on multiple indicators."""
        signal_strength = 0
        
        # MA Crossover
        if df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1]:
            signal_strength += 1
        else:
            signal_strength -= 1
        
        # RSI
        rsi = df['RSI'].iloc[-1]
        if rsi < 30:
            signal_strength += 1
        elif rsi > 70:
            signal_strength -= 1
        
        # MACD
        if df['MACD'].iloc[-1] > 0:
            signal_strength += 1
        else:
            signal_strength -= 1
        
        return signal_strength

    def _format_signal_message(
        self, direction: str, pair: str, 
        entry_low: float, entry_high: float, 
        tp: float, sl: float
    ) -> str:
        """Format signal message with proper spacing and alignment."""
        return (
            f"{direction}📍\n\n"
            f"{pair} {self.min_leverage}x - {self.max_leverage}x\n\n"
            f"ENTRY : {entry_low:.3f} - {entry_high:.3f}\n\n"
            f"TP : {tp:.3f}\n\n"
            f"SL : {sl:.3f}"
        )

# ====================== DATA FETCHING ======================
async def fetch_data_async(
    trading_pair: str = DEFAULT_TRADING_PAIR,
    timeframe: str = "1d",
    max_retries: int = 5,
    initial_delay: float = 1.0,
    limit: int = 365
) -> List[List[float]]:
    try:
        if trading_pair not in await get_available_pairs():
            raise ValueError(f"Invalid trading pair: {trading_pair}")
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        delay = initial_delay
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                data = await asyncio.to_thread(
                    EXCHANGE.fetch_ohlcv,
                    trading_pair,
                    timeframe,
                    limit=limit
                )
                if not data or len(data) < 10:  # Add minimum data check
                    raise ValueError("Insufficient data received")
                return data
                
            except Exception as e:
                last_error = e
                if attempt == max_retries:
                    logger.error(f"Failed to fetch data after {max_retries} attempts: {e}")
                    raise last_error
                    
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5)
                logger.warning(f"Retry {attempt} failed. New delay: {delay:.1f}s")
                
    except Exception as e:
        logger.error(f"Fatal error in fetch_data_async: {e}")
        raise

async def get_available_pairs() -> List[str]:
    """Fetch available futures trading pairs from Binance."""
    try:
        markets = await asyncio.to_thread(EXCHANGE.load_markets)
        futures_pairs = [
            symbol for symbol in markets.keys()
            if '/USDT' in symbol and markets[symbol].get('future', False)
        ]
        sorted_pairs = sorted(futures_pairs)
        logging.info(f"Found {len(sorted_pairs)} available futures pairs")
        return sorted_pairs
    except Exception as e:
        logging.error(f"Error fetching trading pairs: {e}")
        return ["BTC/USDT", "ETH/USDT"]  # Fallback pairs

# ====================== TELEGRAM HANDLERS ======================

async def show_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a dashboard message to the user."""
    await update.message.reply_text(
        "🤖 *Trading Bot Dashboard*\n\n"
        "Available commands:\n"
        "📊 /backtest - Run trading simulation\n"
        "ℹ️ /status - Check bot status\n"
        "❓ /help - Show all commands\n\n"
        "_Select a command to continue_",
        parse_mode='Markdown'
    )

async def authenticate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle authentication attempts."""
    try:
        user_input = update.message.text.strip()
        logging.info(f"Authentication attempt from user {update.effective_user.id}")
        
        if len(user_input) < 8:
            await update.message.reply_text("❌ Password must be at least 8 characters. Try again:")
            return AUTH
        
        if hashlib.sha256(user_input.encode()).hexdigest() == PASSWORD_HASH:
            AUTHORIZED_USERS[update.effective_user.id] = True
            context.user_data["trading_pair"] = DEFAULT_TRADING_PAIR
            await update.message.reply_text("✅ Authentication successful!")
            await show_dashboard(update, context)
            return ConversationHandler.END
        
        await update.message.reply_text("❌ Invalid password. Try again:")
        return AUTH
    except Exception as e:
        logging.error(f"Authentication error: {e}")
        await update.message.reply_text("❌ An error occurred. Please try /start again.")
        return ConversationHandler.END

async def send_results(message: Any, df: pd.DataFrame, metrics: Dict[str, float]) -> None:
    filename = f"results_{uuid.uuid4().hex}.png"
    try:
        plt.close('all')  # Clear any existing plots
        with plot_lock:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
            
            # Plot portfolio value
            ax1.plot(df.index, df["portfolio"], label="Portfolio Value")
            ax1.set_title("Backtest Results")
            ax1.set_ylabel("USD")
            ax1.grid(True)
            
            # Plot signal strength
            ax2.plot(df.index, df["signal_strength"], label="Signal Strength", color='orange')
            ax2.fill_between(df.index, 0, df["signal_strength"], alpha=0.3)
            ax2.set_ylabel("Signal Strength")
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
        
        caption = (
            f"📈 Backtest Results:\n"
            f"Total Return: {metrics['total_return']:.2f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
            f"Number of Trades: {metrics['num_trades']}"
        )
        
        await message.reply_photo(
            photo=open(filename, "rb"),
            caption=caption
        )
        plt.close('all')  # Ensure all plots are closed
    finally:
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except OSError as e:
            logger.error(f"Error deleting file {filename}: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start command handler."""
    logging.info(f"Processing /start command from user {update.effective_user.id}")
    if update.effective_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text(
            "🔒 Please enter the password to access the bot:"
        )
        return AUTH
    await show_dashboard(update, context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📚 *Available Commands*\n\n"
        "/start - Start the bot\n"
        "/timeframe - Set trading timeframe\n"  # Add this line
        "/search - Search and select trading pairs\n"
        "/strategies - Select trading strategies\n"
        "/backtest - Run backtest simulation\n"
        "/analysis - Get market analysis\n"
        "/alert <price> <above/below> - Set price alert\n"
        "/risk - Manage risk settings\n"
        "/status - Show current settings\n"
        "/help - Show this help message\n\n"
        "_Examples:_\n"
        "• /search BTC - Find BTC pairs\n"
        "• /signal - Get trading signal for current pair",
        parse_mode='Markdown'
    )

async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Backtest command handler."""
    if update.effective_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("❌ Please authenticate first using /start")
        pass  # Replace with 'pass' to avoid syntax errors
    
    trading_pair = context.user_data.get("trading_pair", DEFAULT_TRADING_PAIR)
    timeframe = context.user_data.get("timeframe", "1d")
    active_strategies = context.user_data.get("active_strategies", STRATEGIES.copy())
    
    await update.message.reply_text(
        f"🔄 Running backtest simulation...\n"
        f"Pair: {trading_pair}\n"
        f"Timeframe: {TIMEFRAMES[timeframe]}\n"
        f"Strategies: {', '.join(active_strategies)}"
    )
    
    try:
        data = await fetch_data_async(trading_pair, timeframe)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        df = MLTrader()._add_indicators(df)
        results, metrics = Backtester().run(df, active_strategies)
        
        await send_results(update.message, results, metrics)
        
    except Exception as e:
        logging.error(f"Backtest failed: {e}")
        await update.message.reply_text(f"❌ Error running backtest: {str(e)}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Status command handler."""
    logging.info(f"Processing /status command from user {update.effective_user.id}")
    if update.effective_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("❌ Please authenticate first using /start")
        return
    
    trading_pair = context.user_data.get("trading_pair", DEFAULT_TRADING_PAIR)
    timeframe = context.user_data.get("timeframe", "1d")
    
    await update.message.reply_text(
        f"📊 Bot Status:\n"
        f"Trading Pair: {trading_pair}\n"
        f"Timeframe: {TIMEFRAMES[timeframe]}\n"  # Add this line
        f"Environment: {config.environment}\n"
        f"Models Ready: {len(MLTrader().models)} models"
    )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the telegram bot."""
    logging.error(f"Update {update} caused error {context.error}")
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An error occurred while processing your request."
            )
    except Exception as e:
        logging.error(f"Error in error handler: {e}")

async def handle_webhook(request: web.Request) -> web.Response:
    """Handle webhook updates from Telegram."""
    try:
        if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != WEBHOOK_SECRET:
            logger.warning("Invalid webhook secret token")
            return web.Response(status=403)
        
        update_data = await request.json()
        logger.info("Received webhook update")
        
        update = Update.de_json(update_data, bot_app.bot)
        await bot_app.process_update(update)
        
        return web.Response()
    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        return web.Response(status=500)

async def search_pairs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle pair search."""
    if update.effective_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("❌ Please authenticate first using /start")
        return
    
    query = update.message.text.replace("/search", "").strip().upper()
    if not query:
        await update.message.reply_text(
            "🔍 Enter a trading pair to search (e.g., /search BTC):\n"
            "You can search by token name (BTC) or full pair (BTC/USDT)"
        )
        return
    
    try:
        available_pairs = await get_available_pairs()
        # Find closest matches using fuzzy search
        matches: List[Tuple[str, int]] = process.extract(
            query, 
            available_pairs,
            limit=5
        )
        
        if not matches:
            await update.message.reply_text("❌ No matching pairs found.")
            return
        
        # Create keyboard with matches
        keyboard = []
        for pair, score in matches:
            if score > 60:  # Only show reasonable matches
                keyboard.append([
                    InlineKeyboardButton(pair, callback_data=f"pair_{pair}")
                ])
        
        if keyboard:
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "🎯 Found these matching pairs:\n"
                "Click to select:",
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                "❌ No close matches found.\n"
                "Try searching with a different term."
            )
            
    except Exception as e:
        logging.error(f"Search error: {e}")
        await update.message.reply_text(
            "❌ Error searching pairs. Please try again."
        )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    """Handle button callbacks."""
    query = update.callback_query
    try:
        await query.answer()
        
        if not query.data:
            logger.error("Received callback query without data")
            await query.edit_message_text("❌ Invalid selection")
            return None
            
        if query.data.startswith("tf_"):
            await handle_timeframe_selection(query, context)
        elif query.data.startswith("strat_"):
            return await handle_strategy_selection(query, context)
        elif query.data.startswith("pairs_"):
            await handle_pairs_navigation(query, context)
        elif query.data.startswith("risk_"):
            await handle_risk_settings(query, context)
        else:
            await query.edit_message_text("❌ Invalid selection")
            return None
            
    except Exception as e:
        logger.error(f"Button handler error: {e}", exc_info=True)
        await query.edit_message_text("❌ Error processing selection")
        return None

async def handle_strategy_selection(query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    """Handle strategy selection buttons."""
    strategy = query.data.replace("strat_", "")
    active_strategies = context.user_data.get("active_strategies", STRATEGIES.copy())
    
    if strategy == "all":
        context.user_data["active_strategies"] = STRATEGIES.copy()
        await query.edit_message_text(
            "✅ All strategies selected\n"
            "Use /backtest to run simulation"
        )
        return None
        
    elif strategy == "done":
        if not active_strategies:
            await query.edit_message_text("❌ Please select at least one strategy!")
            return SELECTING_STRATEGY
            
        await query.edit_message_text(
            f"✅ Selected strategies:\n• " + "\n• ".join(active_strategies) +
            "\n\nUse /backtest to run simulation"
        )
        return None
    
    # Toggle strategy
    if strategy in active_strategies:
        active_strategies.remove(strategy)
    else:
        active_strategies.append(strategy)
    context.user_data["active_strategies"] = active_strategies
    
    # Update keyboard
    keyboard = get_strategy_keyboard(active_strategies)
    await query.edit_message_text(
        "🎯 *Strategy Selection*\n\n"
        "Select strategies to use:\n"
        "✅ = Active | ❌ = Inactive\n\n"
        "*Available Strategies:*\n"
        "• MA Crossover - Moving Average strategy\n"
        "• RSI - Relative Strength Index\n"
        "• MACD - Moving Average Convergence Divergence\n"
        "• ML Enhanced - Machine Learning predictions\n\n"
        "Click a strategy to toggle it.\n"
        "Click Done when finished.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )
    return SELECTING_STRATEGY

async def handle_pairs_navigation(query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle pair list navigation."""
    action = query.data.replace("pairs_", "")
    if action == "prev":
        context.user_data['pairs_page'] = max(0, context.user_data.get('pairs_page', 0) - 1)
    elif action == "next":
        context.user_data['pairs_page'] = context.user_data.get('pairs_page', 0) + 1
    elif action == "search":
        await query.edit_message_text(
            "🔍 Use /search followed by the pair name\n"
            "Example: /search BTC"
        )
        return None
    
    await list_pairs(query, context)

async def select_strategies(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle strategy selection."""
    if update.effective_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("❌ Please authenticate first using /start")
        return
    
    # Get currently active strategies
    active_strategies = context.user_data.get("active_strategies", STRATEGIES.copy())
    
    # Create keyboard with strategies
    keyboard = []
    for strategy in STRATEGIES:
        is_active = strategy in active_strategies
        keyboard.append([
            InlineKeyboardButton(
                f"{'✅' if is_active else '❌'} {strategy}",
                callback_data=f"strat_{strategy}"
            )
        ])
    
    # Add "All Strategies" and "Done" buttons
    keyboard.extend([
        [InlineKeyboardButton("📊 All Strategies", callback_data="strat_all")],
        [InlineKeyboardButton("✨ Done", callback_data="strat_done")]
    ])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🎯 *Strategy Selection*\n\n"
        "Select the strategies you want to use:\n"
        "✅ = Active  |  ❌ = Inactive\n\n"
        "*Available Strategies:*\n"
        "• MA Crossover - Moving Average strategy\n"
        "• RSI - Relative Strength Index\n"
        "• MACD - Moving Average Convergence Divergence\n"
        "• ML Enhanced - Machine Learning predictions\n\n"
        "Click a strategy to toggle it.\n"
        "Click Done when finished.",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

def get_strategy_keyboard(active_strategies: List[str]) -> List[List[InlineKeyboardButton]]:
    """Create keyboard for strategy selection."""
    keyboard = []
    
    # Add strategy buttons
    for strategy in STRATEGIES:
        is_active = strategy in active_strategies
        keyboard.append([
            InlineKeyboardButton(
                f"{'✅' if is_active else '❌'} {strategy}",
                callback_data=f"strat_{strategy}"
            )
        ])
    
    # Add control buttons
    keyboard.extend([
        [InlineKeyboardButton("📊 All Strategies", callback_data="strat_all")],
        [InlineKeyboardButton("✨ Done", callback_data="strat_done")]
    ])
    
    return keyboard

class RiskSettings:
    def __init__(self):
        self.default_settings = {
            "position_size": 0.2,  # 20% of capital
            "max_positions": 3,    # Maximum concurrent positions
            "stop_loss": 0.03,    # 3% stop loss
            "take_profit": 0.05,   # 5% take profit
            "trailing_stop": 0.02  # 2% trailing stop
        }

async def risk_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle risk management settings."""
    keyboard = [
        [InlineKeyboardButton("Position Size", callback_data="risk_pos_size")],
        [InlineKeyboardButton("Stop Loss", callback_data="risk_sl")],
        [InlineKeyboardButton("Take Profit", callback_data="risk_tp")],
        [InlineKeyboardButton("Max Positions", callback_data="risk_max_pos")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    current_settings = context.user_data.get("risk_settings", RiskSettings().default_settings)
    await update.message.reply_text(
        "⚠️ *Risk Management Settings*\n\n"
        f"Position Size: {current_settings['position_size']*100}%\n"
        f"Stop Loss: {current_settings['stop_loss']*100}%\n"
        f"Take Profit: {current_settings['take_profit']*100}%\n"
        f"Max Positions: {current_settings['max_positions']}\n\n"
        "Select a setting to modify:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def handle_risk_settings(query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle risk setting updates."""
    setting = query.data.replace("risk_", "")
    settings = context.user_data.get("risk_settings", RiskSettings().default_settings)
    
    try:
        if setting == "pos_size":
            await query.edit_message_text(
                "Enter position size (1-100%):",
                reply_markup=None
            )
            context.user_data["current_risk_setting"] = "position_size"
        elif setting == "sl":
            await query.edit_message_text(
                "Enter stop loss percentage (1-100%):",
                reply_markup=None
            )
            context.user_data["current_risk_setting"] = "stop_loss"
        elif setting == "tp":
            await query.edit_message_text(
                "Enter take profit percentage (1-100%):",
                reply_markup=None
            )
            context.user_data["current_risk_setting"] = "take_profit"
        elif setting == "max_pos":
            await query.edit_message_text(
                "Enter maximum positions (1-10):",
                reply_markup=None
            )
            context.user_data["current_risk_setting"] = "max_positions"
            
    except Exception as e:
        logger.error(f"Error handling risk settings: {e}")
        await query.edit_message_text("❌ Error updating settings")

async def market_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Perform technical and market analysis."""
    if update.effective_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("❌ Please authenticate first using /start")
        return
    
    pair = context.user_data.get("trading_pair", DEFAULT_TRADING_PAIR)
    timeframe = context.user_data.get("timeframe", "1d")
    
    try:
        # Fetch market data
        data = await fetch_data_async(pair, timeframe)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Add technical indicators
        df = MLTrader()._add_indicators(df)
        
        # Calculate market conditions
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        daily_change = ((current_price - prev_price) / prev_price) * 100
        
        rsi = df['RSI'].iloc[-1]
        rsi_signal = "Oversold 🟢" if rsi < 30 else "Overbought 🔴" if rsi > 70 else "Neutral ⚪"
        
        volume_sma = df['volume'].rolling(20).mean().iloc[-1]
        volume_current = df['volume'].iloc[-1]
        volume_trend = "Above Average 📊↗" if volume_current > volume_sma else "Below Average 📊↘"
        
        # Get ML prediction
        X = df[["MA_50", "MA_200", "RSI", "MACD"]].iloc[-1:].values
        X_scaled = MLTrader().scaler.transform(X)
        predictions = []
        for model in MLTrader().models.values():
            pred = model.predict(X_scaled)
            predictions.append(1 if pred > 0 else -1)
        ml_consensus = "Bullish 🤖📈" if np.mean(predictions) > 0 else "Bearish 🤖📉"
        
        # Send analysis
        await update.message.reply_text(
            f"📊 *Market Analysis for {pair}*\n\n"
            f"Current Price: ${current_price:.2f}\n"
            f"24h Change: {daily_change:+.2f}%\n\n"
            f"*Technical Indicators:*\n"
            f"• Trend: {'Bullish 📈' if df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1] else 'Bearish 📉'}\n"
            f"• RSI ({rsi:.1f}): {rsi_signal}\n"
            f"• MACD: {'Positive 📈' if df['MACD'].iloc[-1] > 0 else 'Negative 📉'}\n"
            f"• Volume: {volume_trend}\n\n"
            f"*ML Analysis:*\n"
            f"• Model Consensus: {ml_consensus}\n\n"
            f"Use /backtest to simulate trading with these conditions",
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logging.error(f"Market analysis failed: {e}")
        await update.message.reply_text("❌ Error performing market analysis")

class PriceAlert:
    def __init__(self):
        self.alerts = {}  # user_id -> List[alert_details]
    
    async def add_alert(self, user_id: int, pair: str, price: float, condition: str) -> None:
        """Add a new price alert."""
        if user_id not in self.alerts:
            self.alerts[user_id] = []
        
        self.alerts[user_id].append({
            "pair": pair,
            "price": price,
            "condition": condition,
            "created_at": datetime.now()
        })

    async def check_alerts(self, bot: Bot) -> None:
        """Check all active alerts."""
        while True:
            try:
                for user_id, user_alerts in self.alerts.items():
                    for alert in user_alerts[:]:  # Copy to allow modification
                        current_price = float((await asyncio.to_thread(
                            EXCHANGE.fetch_ticker, alert["pair"]))['last'])
                        
                        if ((alert["condition"] == "above" and current_price > alert["price"]) or
                            (alert["condition"] == "below" and current_price < alert["price"])):
                            await bot.send_message(
                                chat_id=user_id,
                                text=f"🔔 *Price Alert*\n"
                                     f"{alert['pair']} is {alert['condition']} {alert['price']}\n"
                                     f"Current price: {current_price}",
                                parse_mode='Markdown'
                            )
                            user_alerts.remove(alert)
            except Exception as e:
                logging.error(f"Error checking alerts: {e}")
            await asyncio.sleep(60)  # Check every minute

async def set_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set a price alert."""
    if update.effective_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("❌ Please authenticate first using /start")
        return
    
    try:
        args = context.args
        if len(args) != 2:
            await update.message.reply_text(
                "❌ Wrong format. Use:\n"
                "/alert <price> <above/below>\n"
                "Example: /alert 50000 above"
            )
            return
        
        price = float(args[0])
        condition = args[1].lower()
        if condition not in ["above", "below"]:
            await update.message.reply_text("❌ Condition must be 'above' or 'below'")
            return
        
        pair = context.user_data.get("trading_pair", DEFAULT_TRADING_PAIR)
        await price_alerts.add_alert(
            update.effective_user.id,
            pair,
            price,
            condition
        )
        
        await update.message.reply_text(
            f"✅ Alert set for {pair}\n"
            f"Will notify when price goes {condition} ${price:,.2f}"
        )
        
    except ValueError:
        await update.message.reply_text("❌ Invalid price value")
    except Exception as e:
        logging.error(f"Error setting alert: {e}")
        await update.message.reply_text("❌ Error setting alert")

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate and send trading signal."""
    if update.effective_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("❌ Please authenticate first using /start")
        return
    
    processing_message = None
    try:
        pair = context.user_data.get("trading_pair", DEFAULT_TRADING_PAIR)
        processing_message = await update.message.reply_text(
            f"🔄 Generating signal for {pair}..."
        )
        
        signal_gen = SignalGenerator()
        signal_text = await signal_gen.generate_signal(pair)
        
        if signal_text:
            await update.message.reply_text(
                f"🎯 *Signal Generated*\n\n{signal_text}",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(
                "📊 No clear signal at the moment.\n"
                "Try again later or use /analysis for detailed market analysis."
            )
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        await update.message.reply_text("❌ Error generating signal")
    finally:
        if processing_message:
            try:
                await processing_message.delete()
            except Exception as e:
                logger.error(f"Error deleting processing message: {e}")

async def health_check(request: web.Request) -> web.Response:
    """Health check endpoint."""
    try:
        me = await bot_app.bot.get_me()
        return web.json_response({
            "status": "healthy",
            "bot": me.username,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return web.json_response(
            {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status=500
        )

async def warmup_models():
    """Warm up ML models with initial data."""
    try:
        logging.info("Starting model warmup...")
        data = await fetch_data_async(DEFAULT_TRADING_PAIR, "1d")
        df = pd.DataFrame(
            data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        ml_trader = MLTrader()
        ml_trader.train(df)
        logging.info("Model warmup complete")
    except Exception as e:
        logging.error(f"Error during model warmup: {e}")
        raise

async def main():
    """Main application entry point."""
    alert_task = None
    runner = None
    try:
        # Initialize web app
        web_app = web.Application()
        web_app.router.add_post(f"/{TELEGRAM_TOKEN}", handle_webhook)
        web_app.router.add_get("/health", health_check)
        
        # Start alert checker task
        alert_task = asyncio.create_task(price_alerts.check_alerts(bot_app.bot))
        
        # Run web app
        runner = web.AppRunner(web_app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", PORT)
        await site.start()
        
        logging.info(f"Bot started successfully on port {PORT}")
        
        # Keep the application running
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour
            
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        if alert_task:
            alert_task.cancel()
        if runner:
            await runner.cleanup()

async def list_pairs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all available trading pairs."""
    if update.effective_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("❌ Please authenticate first using /start")
        return
    
    try:
        # Fetch available pairs
        pairs = await get_available_pairs()
        
        # Create paginated message
        page_size = 20
        total_pages = (len(pairs) + page_size - 1) // page_size
        current_page = context.user_data.get('pairs_page', 0)
        
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, len(pairs))
        current_pairs = pairs[start_idx:end_idx]
        
        # Format pairs list
        pairs_text = "\n".join(f"• {pair}" for pair in current_pairs)
        
        # Create navigation buttons
        keyboard = []
        nav_row = []
        
        if current_page > 0:
            nav_row.append(InlineKeyboardButton("⬅️ Previous", callback_data="pairs_prev"))
        if current_page < total_pages - 1:
            nav_row.append(InlineKeyboardButton("Next ➡️", callback_data="pairs_next"))
        
        if nav_row:
            keyboard.append(nav_row)
        
        keyboard.append([InlineKeyboardButton("🔍 Search Pairs", callback_data="pairs_search")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"📊 *Available Trading Pairs*\n"
            f"Page {current_page + 1}/{total_pages}\n\n"
            f"{pairs_text}\n\n"
            f"Total pairs: {len(pairs)}",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logging.error(f"Error listing pairs: {e}")
        await update.message.reply_text("❌ Error fetching available pairs")

if __name__ =="__main__":
    try:
        # Initialize bot application
        logging.info("Initializing bot application...")
        bot_app = (
            ApplicationBuilder()
            .token(TELEGRAM_TOKEN)
            .rate_limiter(AIORateLimiter(max_retries=1))
            .build()
        )
        
        # Initialize price alerts
        price_alerts = PriceAlert()
        if not price_alerts:
            raise RuntimeError("Failed to initialize price alerts")
        
        # Register all handlers
        for handler in handlers:
            bot_app.add_handler(handler)
        
        # Add error handler
        bot_app.add_error_handler(error_handler)
        
        logging.info("All handlers registered successfully")
        
        # Start the bot
        if config.environment == "production":
            asyncio.run(main())
        else:
            bot_app.run_polling()
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        raise

# Add timeframe command handler
async def set_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set the trading timeframe."""
    keyboard = []
    for tf, desc in TIMEFRAMES.items():
        keyboard.append([InlineKeyboardButton(f"{desc}", callback_data=f"tf_{tf}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "📊 *Select Trading Timeframe*\n\n"
        "Current timeframes available:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

# Add after all handler function definitions but before __main__

def get_strategy_keyboard(active_strategies: List[str]) -> List[List[InlineKeyboardButton]]:
    """Create keyboard for strategy selection."""
    keyboard = []
    
    # Add strategy buttons
    for strategy in STRATEGIES:
        is_active = strategy in active_strategies
        keyboard.append([
            InlineKeyboardButton(
                f"{'✅' if is_active else '❌'} {strategy}",
                callback_data=f"strat_{strategy}"
            )
        ])
    
    # Add control buttons
    keyboard.extend([
        [InlineKeyboardButton("📊 All Strategies", callback_data="strat_all")],
        [InlineKeyboardButton("✨ Done", callback_data="strat_done")]
    ])
    
    return keyboard

# Define conversation handler
conv_handler = ConversationHandler(
    entry_points=[CommandHandler("start", start)],
    states={
        AUTH: [MessageHandler(filters.TEXT & ~filters.COMMAND, authenticate)],
        SELECTING_STRATEGY: [
            CallbackQueryHandler(handle_strategy_selection, pattern="^strat_")
        ]
    },
    fallbacks=[CommandHandler("start", start)],
    name="auth_conversation"
)

# Update handlers list
handlers = [
    conv_handler,
    CommandHandler("timeframe", set_timeframe),  # Add this line
    CommandHandler("help", help_command),
    CommandHandler("backtest", backtest),
    CommandHandler("status", status),
    CommandHandler("search", search_pairs),
    CommandHandler("risk", risk_settings),
    CommandHandler("analysis", market_analysis),
    CommandHandler("alert", set_alert),
    CommandHandler("signal", signal),
    CommandHandler("list_pairs", list_pairs),
    CommandHandler("pairs", list_pairs),
    CommandHandler("strategies", select_strategies),
    CallbackQueryHandler(button_handler)
]

# 1. First, fix the timeframe selection by adding a command handler
async def set_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set the trading timeframe."""
    keyboard = []
    for tf, desc in TIMEFRAMES.items():
        keyboard.append([InlineKeyboardButton(f"{desc}", callback_data=f"tf_{tf}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "📊 *Select Trading Timeframe*\n\n"
        "Current timeframes available:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

# 2. Add timeframe callback handler
async def handle_timeframe_selection(query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle timeframe selection."""
    timeframe = query.data.replace("tf_", "")
    if timeframe in TIMEFRAMES:
        context.user_data["timeframe"] = timeframe
        await query.edit_message_text(
            f"✅ Timeframe set to: {TIMEFRAMES[timeframe]}\n\n"
            "Use /backtest to run analysis with new timeframe"
        )
    else:
        await query.edit_message_text("❌ Invalid timeframe selection")

# 3. Update the button handler to include timeframe selection
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    """Handle button callbacks."""
    query = update.callback_query
    try:
        await query.answer()
        
        if not query.data:
            logger.error("Received callback query without data")
            await query.edit_message_text("❌ Invalid selection")
            return None
            
        if query.data.startswith("tf_"):
            await handle_timeframe_selection(query, context)
        elif query.data.startswith("strat_"):
            return await handle_strategy_selection(query, context)
        # ... rest of the button handler ...
            
    except Exception as e:
        logger.error(f"Button handler error: {e}", exc_info=True)
        await query.edit_message_text("❌ Error processing selection")
        return None

# 4. Update the handlers list to include timeframe command
handlers = [
    conv_handler,
    CommandHandler("timeframe", set_timeframe),  # Add this line
    CommandHandler("help", help_command),
    # ... rest of the handlers ...
]

# 5. Update help command to include timeframe command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📚 *Available Commands*\n\n"
        "/start - Start the bot\n"
        "/timeframe - Set trading timeframe\n"  # Add this line
        "/search - Search and select trading pairs\n"
        "/strategies - Select trading strategies\n"
        "/backtest - Run backtest simulation\n"
        "/analysis - Get market analysis\n"
        "/alert - Set price alert\n"
        "/risk - Manage risk settings\n"
        "/status - Show current settings\n"
        "/help - Show this help message\n\n"
        "_Examples:_\n"
        "• /search BTC - Find BTC pairs\n"
        "• /signal - Get trading signal for current pair",
        parse_mode='Markdown'
    )

# 6. Update status command to show current timeframe
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Status command handler."""
    if update.effective_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("❌ Please authenticate first using /start")
        return
    
    trading_pair = context.user_data.get("trading_pair", DEFAULT_TRADING_PAIR)
    timeframe = context.user_data.get("timeframe", "1d")
    
    await update.message.reply_text(
        f"📊 Bot Status:\n"
        f"Trading Pair: {trading_pair}\n"
        f"Timeframe: {TIMEFRAMES[timeframe]}\n"  # Add this line
        f"Environment: {config.environment}\n"
        f"Models Ready: {len(MLTrader().models)} models"
    )