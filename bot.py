import asyncio
import logging
import uuid
import random
from typing import List, Dict, Optional, Any, Tuple
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import joblib
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    InlineQueryHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)
from cachetools import TTLCache
from datetime import datetime
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os
from pydantic import BaseSettings
import hashlib

# Configuration Management
class Settings(BaseSettings):
    """Configuration settings for the bot."""
    telegram_token: str
    password_hash: str
    webhook_secret: str
    render: bool = False
    environment: str = "development"
    default_trading_pair: str = "BTC/USDT"
    default_timeframe: str = "1d"
    leverage: str = "5x-400x"
    admin_chat_id: Optional[int] = None
    ml_params: Dict[str, Any] = {"ann_hidden_layers": (50, 50), "svm_kernel": "rbf"}

    class Config:
        env_file = ".env"

config = Settings()
TELEGRAM_TOKEN = config.telegram_token
PASSWORD_HASH = config.password_hash
PORT = int(os.getenv("PORT", 10000))
WEBHOOK_SECRET = config.webhook_secret
IS_RENDER = config.render
AUTHORIZED_USERS: Dict[int, bool] = {}

# Global Settings
EXCHANGE = ccxt.binance({"enableRateLimit": True})
TIMEFRAMES = {"1m": "1 minute", "5m": "5 minutes", "15m": "15 minutes", "1h": "1 hour", "1d": "1 day"}
STRATEGIES = ["MA Crossover", "RSI", "MACD", "ML Enhanced", "Bollinger Bands"]
data_cache = TTLCache(maxsize=500, ttl=3600)  # 1-hour cache

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants for conversation states
AUTH, SELECTING_STRATEGY, POSITION_SIZE, STOP_LOSS, TAKE_PROFIT, SET_CONFIG = range(6)

# Risk Settings Class
class RiskSettings:
    def __init__(self):
        self.default_settings = {
            "position_size": 0.1,  # 10%
            "stop_loss": 0.05,     # 5%
            "take_profit": 0.1     # 10%
        }

# Machine Learning Trader
class MLTrader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.ml_params = config.ml_params
        self._load_or_create_models()

    def _load_or_create_models(self):
        """Load existing models or create new ones."""
        try:
            self.models["linear"] = joblib.load("model_linear.pkl")
            logger.info("Loaded linear regression model.")
        except FileNotFoundError:
            self.models["linear"] = LinearRegression()
            logger.warning("No linear regression model found. Created new model.")
        try:
            self.models["svm"] = joblib.load("model_svm.pkl")
            logger.info("Loaded SVM model.")
        except FileNotFoundError:
            self.models["svm"] = SVR(kernel=self.ml_params.get("svm_kernel", "rbf"))
            logger.warning("No SVM model found. Created new model.")
        try:
            self.models["ann"] = joblib.load("model_ann.pkl")
            logger.info("Loaded ANN model.")
        except FileNotFoundError:
            self.models["ann"] = MLPRegressor(hidden_layer_sizes=self.ml_params.get("ann_hidden_layers", (50, 50)))
            logger.warning("No ANN model found. Created new model.")

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame missing required columns. Expected {required_columns}, got {df.columns.tolist()}")

        df = df.copy()
        df['MA_50'] = df['close'].rolling(window=50).mean()
        df['MA_200'] = df['close'].rolling(window=200).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        loss = loss.replace(0, np.finfo(float).eps)
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    def train(self, df: pd.DataFrame) -> None:
        """Train ML models on the provided data."""
        logger.info("Starting model training...")
        df = self._add_indicators(df)
        X = df[["MA_50", "MA_200", "RSI", "MACD"]].iloc[:-1]
        y = df["close"].pct_change().shift(-1).dropna()
        if len(X) < 1 or len(y) < 1:
            raise ValueError("Insufficient data for training.")
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.models.items():
            model.fit(X_scaled, y)
            joblib.dump(model, f"model_{name}.pkl")
            logger.info(f"Trained and saved {name} model.")

    async def generate_signal(self, pair: str, timeframe: str = "5m") -> Optional[str]:
        """Generate a trading signal based on ML predictions and indicators."""
        logger.info(f"Generating signal for {pair} on {timeframe} timeframe")
        try:
            data = await fetch_data_async(pair, timeframe, limit=200)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = self._add_indicators(df)
            X = df[["MA_50", "MA_200", "RSI", "MACD"]]
            X_scaled = self.scaler.transform(X)
            
            predictions = []
            for model in self.models.values():
                pred = model.predict(X_scaled)
                predictions.append(pred)
            avg_pred = np.mean(predictions, axis=0)
            signal_strength = self._calculate_signal_strength(df)
            
            if abs(signal_strength[-1]) >= 2:
                return "buy" if avg_pred[-1] > 0 else "sell"
            return None
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            await send_admin_alert(None, f"Signal generation failed for {pair}: {str(e)}")
            return None

    def _calculate_signal_strength(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate signal strength based on multiple indicators."""
        signal_strength = np.zeros(len(df))
        signal_strength += np.where(df['MA_50'] > df['MA_200'], 1, -1)
        rsi = df['RSI']
        signal_strength += np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
        signal_strength += np.where(df['MACD'] > 0, 1, np.where(df['MACD'] < 0, -1, 0))
        return signal_strength

# Data Fetching
async def fetch_data_async(
    trading_pair: str = config.default_trading_pair,
    timeframe: str = config.default_timeframe,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    limit: int = 365
) -> List[List[float]]:
    """Fetch OHLCV data from the exchange with in-memory caching."""
    cache_key = f"{trading_pair}_{timeframe}_{limit}"
    if cache_key in data_cache:
        logger.debug(f"Returning cached data for {cache_key}")
        return data_cache[cache_key]
    
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            if trading_pair not in await get_available_pairs():
                raise ValueError(f"Invalid trading pair: {trading_pair}")
            if timeframe not in TIMEFRAMES:
                raise ValueError(f"Invalid timeframe: {timeframe}")
            data = await asyncio.to_thread(
                EXCHANGE.fetch_ohlcv,
                trading_pair,
                timeframe,
                limit=limit
            )
            if not data or len(data) < 10:
                raise ValueError("Insufficient data received")
            data_cache[cache_key] = data
            logger.info(f"Fetched {len(data)} OHLCV candles for {trading_pair}")
            return data
        except ccxt.NetworkError as e:
            logger.warning(f"Network error: {e}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            await send_admin_alert(None, f"Exchange error fetching data for {trading_pair}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if attempt == max_retries:
                raise Exception("Max retries exceeded")
    raise Exception("Max retries exceeded")

async def get_available_pairs() -> List[str]:
    """Fetch available futures trading pairs from Binance."""
    try:
        markets = await asyncio.to_thread(EXCHANGE.load_markets)
        futures_pairs = [
            symbol for symbol in markets.keys()
            if '/USDT' in symbol and markets[symbol].get('future', False)
        ]
        sorted_pairs = sorted(futures_pairs)
        logger.info(f"Found {len(sorted_pairs)} available futures pairs")
        return sorted_pairs
    except Exception as e:
        logger.error(f"Error fetching trading pairs: {e}")
        await send_admin_alert(None, f"Failed to fetch trading pairs: {str(e)}")
        return ["BTC/USDT", "ETH/USDT"]

# Signal Generation
async def get_current_price(pair: str) -> Optional[float]:
    """Fetch the latest closing price for the given trading pair."""
    try:
        data = await fetch_data_async(pair, "1m", limit=1)
        if data and len(data) > 0:
            return data[0][4]  # Close price
        else:
            logger.error(f"No data received for {pair}")
            return None
    except Exception as e:
        logger.error(f"Error fetching current price for {pair}: {e}")
        await send_admin_alert(None, f"Failed to fetch price for {pair}: {str(e)}")
        return None

async def generate_bot_signal(pair: str = config.default_trading_pair, timeframe: str = config.default_timeframe) -> str:
    """
    Generate a trading signal in the format:
    Entry <pair> <position_type>
    <leverage>
    <entry_price>
    Tp- <tp1> - <tp2>
    """
    trader = MLTrader()
    signal = await trader.generate_signal(pair, timeframe)
    if signal is None:
        return "Error: Unable to generate clear signal at this time."
    
    position_type = "LONG" if signal == "buy" else "SHORT"
    leverage = config.leverage
    
    current_price = await get_current_price(pair)
    if current_price is None:
        return f"Error: Unable to fetch current price for {pair}."
    
    entry_price = current_price * (1 + random.uniform(-0.001, 0.001))
    if position_type == "LONG":
        tp1 = entry_price * 1.007
        tp2 = entry_price * 1.012
    else:
        tp1 = entry_price * 0.993
        tp2 = entry_price * 0.988
    
    signal_text = (
        f"Entry {pair} {position_type}\n"
        f"{leverage}\n"
        f"{entry_price:,.0f}\n"
        f"Tp- {tp1:,.0f} - {tp2:,.0f}"
    )
    logger.info(f"Generated signal: {signal_text}")
    return signal_text

# Trading Strategy
class TradingStrategy:
    def __init__(self):
        self.ml_trader = MLTrader()
        self.bb_params = {"period": 20, "std_dev": 2}  # Configurable Bollinger Bands parameters

    def set_bb_params(self, period: int, std_dev: float):
        """Set Bollinger Bands parameters."""
        self.bb_params["period"] = period
        self.bb_params["std_dev"] = std_dev
        logger.info(f"Updated Bollinger Bands parameters: period={period}, std_dev={std_dev}")

    def calculate_signals(self, df: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
        """Calculate trading signals using multiple strategies."""
        df = self.ml_trader._add_indicators(df)
        signals = pd.DataFrame(index=df.index)
        
        for strategy in strategies:
            if strategy == "MA Crossover":
                signals['ma_signal'] = np.where(df['MA_50'] > df['MA_200'], 1, -1)
            elif strategy == "RSI":
                signals['rsi_signal'] = np.where(
                    (df['RSI'] < 30), 1,
                    np.where(df['RSI'] > 70, -1, 0)
                )
            elif strategy == "MACD":
                signals['macd_signal'] = np.where(
                    df['MACD'] > 0, 1,
                    np.where(df['MACD'] < 0, -1, 0)
                )
            elif strategy == "ML Enhanced":
                X = df[["MA_50", "MA_200", "RSI", "MACD"]].values
                X_scaled = self.ml_trader.scaler.transform(X)
                predictions = []
                for model in self.ml_trader.models.values():
                    pred = model.predict(X_scaled)
                    predictions.append(np.where(pred > 0, 1, -1))
                signals['ml_signal'] = np.sign(np.mean(predictions, axis=0))
            elif strategy == "Bollinger Bands":
                period = self.bb_params["period"]
                std_dev = self.bb_params["std_dev"]
                df['middle_band'] = df['close'].rolling(window=period).mean()
                df['std'] = df['close'].rolling(window=period).std()
                df['upper_band'] = df['middle_band'] + (df['std'] * std_dev)
                df['lower_band'] = df['middle_band'] - (df['std'] * std_dev)
                signals['bb_signal'] = np.where(
                    df['close'] < df['lower_band'], 1,
                    np.where(df['close'] > df['upper_band'], -1, 0)
                )
        
        weights = {
            'ma_signal': 0.25,
            'rsi_signal': 0.2,
            'macd_signal': 0.2,
            'ml_signal': 0.25,
            'bb_signal': 0.1
        }
        final_signal = pd.Series(0, index=df.index)
        for col, weight in weights.items():
            if col in signals.columns:
                final_signal += signals[col] * weight
        
        return pd.DataFrame({
            'signal': np.where(final_signal > 0.1, 1,
                            np.where(final_signal < -0.1, -1, 0)),
            'signal_strength': final_signal.abs()
        })

# Backtester
class Backtester:
    def __init__(self, take_profit: float = 0.05, stop_loss: float = 0.03, position_size: float = 0.2) -> None:
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.position_size = position_size
        self.strategy = TradingStrategy()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def run(self, df: pd.DataFrame, strategies: List[str], bot: Optional[Bot] = None, chat_id: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Run backtest with multiple strategies."""
        signals = self.strategy.calculate_signals(df, strategies)
        df = df.join(signals)
        
        capital = 10000
        position = 0
        entry_price = None
        portfolio = []
        trades = []
        
        for i, row in df.iterrows():
            price = row["close"]
            
            if position != 0 and entry_price is not None:
                ret = (price - entry_price) / entry_price
                if (position > 0 and (ret >= self.take_profit or ret <= -self.stop_loss)) or \
                   (position < 0 and (-ret >= self.take_profit or -ret <= -self.stop_loss)):
                    profit = position * (price - entry_price)
                    capital += profit
                    trades.append({
                        'exit_time': i,
                        'profit': profit,
                        'roi': ret * 100
                    })
                    if bot and chat_id:
                        asyncio.create_task(bot.send_message(
                            chat_id=chat_id,
                            text=f"📈 Trade Closed\nProfit: ${profit:,.2f}\nROI: {ret*100:.2f}%",
                            parse_mode='Markdown'
                        ))
                    position = 0
                    entry_price = None
            
            if position == 0 and row['signal'] != 0:
                position_capital = capital * self.position_size
                position = (position_capital / price) * np.sign(row['signal'])
                entry_price = price
                trades.append({
                    'entry_time': i,
                    'position': position,
                    'entry_price': entry_price
                })
                if bot and chat_id:
                    asyncio.create_task(bot.send_message(
                        chat_id=chat_id,
                        text=f"📉 Trade Opened\nPosition: {'Long' if position > 0 else 'Short'}\nEntry: ${entry_price:,.2f}",
                        parse_mode='Markdown'
                    ))
            
            portfolio.append(capital + position * price)
        
        df["portfolio"] = portfolio
        df['returns'] = df['portfolio'].pct_change()
        metrics = {
            'total_return': (df['portfolio'].iloc[-1] / df['portfolio'].iloc[0] - 1) * 100,
            'sharpe_ratio': df['returns'].mean() / df['returns'].std() * np.sqrt(252) if df['returns'].std() != 0 else 0,
            'max_drawdown': (df['portfolio'] / df['portfolio'].cummax() - 1).min() * 100,
            'num_trades': len([t for t in trades if 'exit_time' in t])
        }
        logger.info(f"Backtest metrics: {metrics}")
        
        if bot and chat_id:
            plot_path = self._generate_plot(df, metrics)
            with open(plot_path, 'rb') as f:
                asyncio.create_task(bot.send_photo(
                    chat_id=chat_id,
                    photo=f,
                    caption="Backtest Results Plot"
                ))
            os.remove(plot_path)
        
        return df, metrics

    def _generate_plot(self, df: pd.DataFrame, metrics: Dict[str, float]) -> str:
        """Generate backtest plot in a separate thread."""
        def plot():
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['portfolio'], label='Portfolio Value', color='blue')
            buy_signals = df[df['signal'] > 0]['close']
            sell_signals = df[df['signal'] < 0]['close']
            plt.scatter(buy_signals.index, buy_signals, color='green', marker='^', label='Buy Signal')
            plt.scatter(sell_signals.index, sell_signals, color='red', marker='v', label='Sell Signal')
            plt.title(f"Backtest: {metrics['total_return']:.2f}% Return, {metrics['num_trades']} Trades")
            plt.xlabel("Time")
            plt.ylabel("Portfolio Value")
            plt.legend()
            plt.grid(True)
            plot_path = f"backtest_plot_{uuid.uuid4()}.png"
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        
        future = self.executor.submit(plot)
        return future.result()

# Price Alerts
class AlertManager:
    def __init__(self):
        self.alerts = {}

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
        logger.info(f"Added alert for user {user_id}: {pair} {condition} {price}")

    async def check_alerts(self, bot: Bot) -> None:
        """Check and trigger price alerts with optimized price fetching."""
        while True:
            try:
                pairs = set(alert["pair"] for user_alerts in self.alerts.values() for alert in user_alerts)
                price_cache = {}
                for pair in pairs:
                    ticker = await asyncio.to_thread(EXCHANGE.fetch_ticker, pair)
                    price_cache[pair] = float(ticker['last'])
                
                for user_id, user_alerts in self.alerts.items():
                    for alert in user_alerts[:]:
                        try:
                            current_price = price_cache.get(alert["pair"])
                            if current_price is None:
                                continue
                            if ((alert["condition"] == "above" and current_price > alert["price"]) or
                                (alert["condition"] == "below" and current_price < alert["price"])):
                                await bot.send_message(
                                    chat_id=user_id,
                                    text=f"🔔 *Price Alert*\n{alert['pair']} is {alert['condition']} {alert['price']}\nCurrent price: {current_price}",
                                    parse_mode='Markdown'
                                )
                                user_alerts.remove(alert)
                                logger.info(f"Triggered and removed alert for user {user_id}: {alert['pair']}")
                        except Exception as e:
                            logger.error(f"Error checking alert for {alert['pair']}: {e}")
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error in alert checking loop: {e}")
                await send_admin_alert(bot, f"Alert checking failed: {str(e)}")
                await asyncio.sleep(60)

# Monitoring and Alerts
async def send_admin_alert(bot: Optional[Bot], message: str) -> None:
    """Send an alert to the admin chat."""
    if bot and config.admin_chat_id:
        try:
            await bot.send_message(config.admin_chat_id, f"⚠️ Bot Alert: {message}", parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Failed to send admin alert: {e}")

async def health_check(bot: Bot) -> None:
    """Perform periodic health checks."""
    while True:
        try:
            await asyncio.to_thread(EXCHANGE.load_markets)
            if not MLTrader().models:
                raise ValueError("No ML models loaded")
            logger.debug("Health check passed")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            await send_admin_alert(bot, f"Health check failed: {str(e)}")
        await asyncio.sleep(3600)

# Telegram Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start command handler."""
    user_id = update.effective_user.id
    logger.info(f"User {user_id} started bot")
    if user_id in AUTHORIZED_USERS and AUTHORIZED_USERS[user_id]:
        await update.message.reply_text(
            "✅ You are already authenticated!\n"
            "Use /help to see available commands."
        )
        return ConversationHandler.END
    await update.message.reply_text("🔒 Please enter the password to access the bot:")
    return AUTH

async def authenticate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Authenticate user with SHA-256 and initialize settings."""
    user_id = update.effective_user.id
    user_input = update.message.text.strip().encode('utf-8')
    input_hash = hashlib.sha256(user_input).hexdigest()
    if input_hash == PASSWORD_HASH:
        AUTHORIZED_USERS[user_id] = True
        context.user_data["authenticated"] = True
        context.user_data["trading_pair"] = config.default_trading_pair
        context.user_data["timeframe"] = config.default_timeframe
        context.user_data["active_strategies"] = STRATEGIES.copy()
        context.user_data["risk_settings"] = RiskSettings().default_settings
        await update.message.reply_text(
            "✅ Authenticated successfully!\n"
            "Use /help to see available commands."
        )
        logger.info(f"User {user_id} authenticated successfully")
        return ConversationHandler.END
    await update.message.reply_text("❌ Invalid password. Try again:")
    logger.warning(f"User {user_id} failed authentication attempt")
    return AUTH

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display available commands."""
    if not context.user_data.get("authenticated"):
        await update.message.reply_text("❌ Please /start to authenticate.")
        return
    await update.message.reply_text(
        "📚 *Available Commands*\n\n"
        "/start - Start the bot\n"
        "/timeframe - Set trading timeframe\n"
        "/strategies - Select trading strategies\n"
        "/backtest - Run backtest simulation\n"
        "/set_risk - Set risk parameters\n"
        "/view_risk - View risk settings\n"
        "/set_config - Configure bot settings\n"
        "/alert <price> <above/below> - Set price alert\n"
        "/signal - Generate trading signal\n"
        "/dashboard - View bot summary\n"
        "/status - Show current settings\n"
        "/help - Show this help message\n"
        "/cancel - Cancel current operation\n\n"
        "_Examples:_\n"
        "• /alert 50000 above\n"
        "• /signal\n"
        "• @<BotName> BTCUSDT (inline query)",
        parse_mode='Markdown'
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current bot settings."""
    if not context.user_data.get("authenticated"):
        await update.message.reply_text("❌ Please /start to authenticate.")
        return
    trading_pair = context.user_data.get("trading_pair", config.default_trading_pair)
    timeframe = context.user_data.get("timeframe", config.default_timeframe)
    active_strategies = context.user_data.get("active_strategies", STRATEGIES.copy())
    settings = context.user_data.get("risk_settings", RiskSettings().default_settings)
    bb_params = TradingStrategy().bb_params
    await update.message.reply_text(
        f"📊 Bot Status:\n"
        f"Trading Pair: {trading_pair}\n"
        f"Timeframe: {TIMEFRAMES[timeframe]}\n"
        f"Strategies: {', '.join(active_strategies)}\n"
        f"Bollinger Bands: Period={bb_params['period']}, StdDev={bb_params['std_dev']}\n"
        f"Risk Settings: Position Size={settings['position_size']*100}%, "
        f"SL={settings['stop_loss']*100}%, TP={settings['take_profit']*100}%\n"
        f"Models Loaded: {len(MLTrader().models)}"
    )

async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run a backtest simulation."""
    if not context.user_data.get("authenticated"):
        await update.message.reply_text("❌ Please /start to authenticate.")
        return
    
    trading_pair = context.user_data.get("trading_pair", config.default_trading_pair)
    timeframe = context.user_data.get("timeframe", config.default_timeframe)
    active_strategies = context.user_data.get("active_strategies", STRATEGIES.copy())
    risk_settings = context.user_data.get("risk_settings", RiskSettings().default_settings)
    
    if not active_strategies:
        await update.message.reply_text("❌ No strategies selected. Use /strategies to select.")
        return
    
    processing_message = await update.message.reply_text("🔄 Running backtest...")
    try:
        data = await fetch_data_async(trading_pair, timeframe)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        trader = MLTrader()
        trader.train(df)
        backtester = Backtester(
            take_profit=risk_settings["take_profit"],
            stop_loss=risk_settings["stop_loss"],
            position_size=risk_settings["position_size"]
        )
        results, metrics = backtester.run(df, active_strategies, context.bot, update.effective_chat.id)
        await processing_message.delete()
        await update.message.reply_text(
            f"🔄 Backtest Results:\n"
            f"Pair: {trading_pair}\n"
            f"Timeframe: {TIMEFRAMES[timeframe]}\n"
            f"Strategies: {', '.join(active_strategies)}\n"
            f"Risk Settings: Position Size={risk_settings['position_size']*100}%, "
            f"SL={risk_settings['stop_loss']*100}%, TP={risk_settings['take_profit']*100}%\n"
            f"Total Return: {metrics['total_return']:.2f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
            f"Number of Trades: {metrics['num_trades']}"
        )
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        await send_admin_alert(context.bot, f"Backtest failed for {trading_pair}: {str(e)}")
        await processing_message.delete()
        await update.message.reply_text(f"❌ Error running backtest: {str(e)}")

async def set_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set the trading timeframe."""
    if not context.user_data.get("authenticated"):
        await update.message.reply_text("❌ Please /start to authenticate.")
        return
    current_tf = context.user_data.get("timeframe", config.default_timeframe)
    keyboard = []
    for tf, desc in TIMEFRAMES.items():
        marker = "✅ " if tf == current_tf else ""
        keyboard.append([InlineKeyboardButton(f"{marker}{desc}", callback_data=f"tf_{tf}")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "📊 *Select Trading Timeframe*\n\n"
        f"Current: {TIMEFRAMES[current_tf]}\n\n"
        "Available timeframes:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def handle_timeframe_selection(query, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle timeframe selection."""
    timeframe = query.data.replace("tf_", "")
    if timeframe in TIMEFRAMES:
        context.user_data["timeframe"] = timeframe
        await query.edit_message_text(
            f"✅ Timeframe set to: {TIMEFRAMES[timeframe]}\n\n"
            "Use /backtest or /signal to use new timeframe"
        )
    else:
        await query.edit_message_text("❌ Invalid timeframe selection")

async def select_strategies(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Initiate strategy selection."""
    if not context.user_data.get("authenticated"):
        await update.message.reply_text("❌ Please /start to authenticate.")
        return ConversationHandler.END
    active_strategies = context.user_data.get("active_strategies", STRATEGIES.copy())
    keyboard = get_strategy_keyboard(active_strategies)
    await update.message.reply_text(
        "🎯 *Strategy Selection*\n\n"
        "Select strategies to use:\n"
        "✅ = Active | ❌ = Inactive\n\n"
        "*Available Strategies:*\n"
        "• MA Crossover - Moving Average strategy\n"
        "• RSI - Relative Strength Index\n"
        "• MACD - Moving Average Convergence Divergence\n"
        "• ML Enhanced - Machine Learning predictions\n"
        "• Bollinger Bands - Volatility-based strategy\n\n"
        "Click a strategy to toggle it.\n"
        "Click Done when finished.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )
    return SELECTING_STRATEGY

def get_strategy_keyboard(active_strategies: List[str]) -> List[List[InlineKeyboardButton]]:
    """Create keyboard for strategy selection."""
    keyboard = []
    for strategy in STRATEGIES:
        is_active = strategy in active_strategies
        keyboard.append([
            InlineKeyboardButton(
                f"{'✅' if is_active else '❌'} {strategy}",
                callback_data=f"strat_{strategy}"
            )
        ])
    keyboard.extend([
        [InlineKeyboardButton("📊 All Strategies", callback_data="strat_all")],
        [InlineKeyboardButton("✨ Done", callback_data="strat_done")]
    ])
    return keyboard

async def handle_strategy_selection(query, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
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
    
    if strategy in active_strategies:
        active_strategies.remove(strategy)
    else:
        active_strategies.append(strategy)
    context.user_data["active_strategies"] = active_strategies
    keyboard = get_strategy_keyboard(active_strategies)
    await query.edit_message_text(
        "🎯 *Strategy Selection*\n\n"
        "Select strategies to use:\n"
        "✅ = Active | ❌ = Inactive\n\n"
        "*Available Strategies:*\n"
        "• MA Crossover - Moving Average strategy\n"
        "• RSI - Relative Strength Index\n"
        "• MACD - Moving Average Convergence Divergence\n"
        "• ML Enhanced - Machine Learning predictions\n"
        "• Bollinger Bands - Volatility-based strategy\n\n"
        "Click a strategy to toggle it.\n"
        "Click Done when finished.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )
    return SELECTING_STRATEGY

async def start_set_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the risk settings conversation."""
    if not context.user_data.get("authenticated"):
        await update.message.reply_text("❌ Please /start to authenticate.")
        return ConversationHandler.END
    await update.message.reply_text("Please enter the position size (1-100%):")
    return POSITION_SIZE

async def set_position_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Set position size from user input."""
    try:
        size = float(update.message.text)
        if 1 <= size <= 100:
            context.user_data["risk_settings"]["position_size"] = size / 100
            await update.message.reply_text("Please enter the stop loss percentage (1-100%):")
            return STOP_LOSS
        else:
            await update.message.reply_text("Invalid value. Please enter a number between 1 and 100.")
            return POSITION_SIZE
    except ValueError:
        await update.message.reply_text("Invalid input. Please enter a number.")
        return POSITION_SIZE

async def set_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Set stop loss from user input."""
    try:
        sl = float(update.message.text)
        if 1 <= sl <= 100:
            context.user_data["risk_settings"]["stop_loss"] = sl / 100
            await update.message.reply_text("Please enter the take profit percentage (1-100%):")
            return TAKE_PROFIT
        else:
            await update.message.reply_text("Invalid value. Please enter a number between 1 and 100.")
            return STOP_LOSS
    except ValueError:
        await update.message.reply_text("Invalid input. Please enter a number.")
        return STOP_LOSS

async def set_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Set take profit from user input."""
    try:
        tp = float(update.message.text)
        if 1 <= tp <= 100:
            context.user_data["risk_settings"]["take_profit"] = tp / 100
            settings = context.user_data["risk_settings"]
            await update.message.reply_text(
                f"Risk settings updated:\n"
                f"Position Size: {settings['position_size']*100}%\n"
                f"Stop Loss: {settings['stop_loss']*100}%\n"
                f"Take Profit: {settings['take_profit']*100}%"
            )
            return ConversationHandler.END
        else:
            await update.message.reply_text("Invalid value. Please enter a number between 1 and 100.")
            return TAKE_PROFIT
    except ValueError:
        await update.message.reply_text("Invalid input. Please enter a number.")
        return TAKE_PROFIT

async def view_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View current risk settings."""
    if not context.user_data.get("authenticated"):
        await update.message.reply_text("❌ Please /start to authenticate.")
        return
    settings = context.user_data["risk_settings"]
    await update.message.reply_text(
        f"Current Risk Settings:\n"
        f"Position Size: {settings['position_size']*100}%\n"
        f"Stop Loss: {settings['stop_loss']*100}%\n"
        f"Take Profit: {settings['take_profit']*100}%"
    )

async def set_config(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start configuration setting conversation."""
    if not context.user_data.get("authenticated"):
        await update.message.reply_text("❌ Please /start to authenticate.")
        return ConversationHandler.END
    await update.message.reply_text(
        "⚙️ *Configure Bot*\n\n"
        "Enter configuration in format: key=value\n"
        "Available keys:\n"
        "• pair (e.g., ETH/USDT)\n"
        "• leverage (e.g., 50-200x)\n"
        "• bb_period (e.g., 20)\n"
        "• bb_std_dev (e.g., 2.0)\n\n"
        "Example: pair=ETH/USDT"
    )
    return SET_CONFIG

async def handle_config(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle configuration input."""
    try:
        input_text = update.message.text.strip()
        key, value = input_text.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        
        if key == "pair":
            if value in await get_available_pairs():
                context.user_data["trading_pair"] = value
                await update.message.reply_text(f"✅ Trading pair set to {value}")
            else:
                await update.message.reply_text("❌ Invalid trading pair")
                return SET_CONFIG
        elif key == "leverage":
            context.user_data["leverage"] = value
            await update.message.reply_text(f"✅ Leverage set to {value}")
        elif key == "bb_period":
            period = int(value)
            if 5 <= period <= 100:
                TradingStrategy().set_bb_params(period, TradingStrategy().bb_params["std_dev"])
                await update.message.reply_text(f"✅ Bollinger Bands period set to {period}")
            else:
                await update.message.reply_text("❌ Period must be between 5 and 100")
                return SET_CONFIG
        elif key == "bb_std_dev":
            std_dev = float(value)
            if 1.0 <= std_dev <= 5.0:
                TradingStrategy().set_bb_params(TradingStrategy().bb_params["period"], std_dev)
                await update.message.reply_text(f"✅ Bollinger Bands std_dev set to {std_dev}")
            else:
                await update.message.reply_text("❌ Std_dev must be between 1.0 and 5.0")
                return SET_CONFIG
        else:
            await update.message.reply_text("❌ Invalid configuration key")
            return SET_CONFIG
        
        return ConversationHandler.END
    except ValueError:
        await update.message.reply_text("❌ Invalid format. Use key=value")
        return SET_CONFIG
    except Exception as e:
        logger.error(f"Config setting failed: {e}")
        await update.message.reply_text(f"❌ Error setting config: {str(e)}")
        return SET_CONFIG

async def set_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set a price alert."""
    if not context.user_data.get("authenticated"):
        await update.message.reply_text("❌ Please /start to authenticate.")
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
        pair = context.user_data.get("trading_pair", config.default_trading_pair)
        alert_manager = context.bot_data.get("alert_manager")
        await alert_manager.add_alert(
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
        logger.error(f"Error setting alert: {e}")
        await update.message.reply_text("❌ Error setting alert")

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate and send trading signal."""
    if not context.user_data.get("authenticated"):
        await update.message.reply_text("❌ Please /start to authenticate.")
        return
    processing_message = await update.message.reply_text("🔄 Generating signal...")
    try:
        pair = context.user_data.get("trading_pair", config.default_trading_pair)
        timeframe = context.user_data.get("timeframe", config.default_timeframe)
        signal_text = await generate_bot_signal(pair, timeframe)
        await processing_message.delete()
        await update.message.reply_text(signal_text, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        await send_admin_alert(context.bot, f"Signal generation failed: {str(e)}")
        await processing_message.delete()
        await update.message.reply_text("❌ Error generating signal")

async def dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot dashboard."""
    if not context.user_data.get("authenticated"):
        await update.message.reply_text("❌ Please /start to authenticate.")
        return
    try:
        pair = context.user_data.get("trading_pair", config.default_trading_pair)
        timeframe = context.user_data.get("timeframe", config.default_timeframe)
        signal = await generate_bot_signal(pair, timeframe)
        settings = context.user_data.get("risk_settings", RiskSettings().default_settings)
        active_strategies = context.user_data.get("active_strategies", STRATEGIES.copy())
        alert_manager = context.bot_data.get("alert_manager")
        user_alerts = alert_manager.alerts.get(update.effective_user.id, [])
        alert_summary = "\n".join(
            f"• {alert['pair']} {alert['condition']} ${alert['price']:,.2f}"
            for alert in user_alerts[:3]
        ) or "No active alerts"
        await update.message.reply_text(
            f"📊 *Dashboard*\n\n"
            f"*Current Signal:*\n{signal}\n\n"
            f"*Active Strategies:*\n{', '.join(active_strategies)}\n\n"
            f"*Risk Settings:*\n"
            f"Position Size: {settings['position_size']*100}%\n"
            f"Stop Loss: {settings['stop_loss']*100}%\n"
            f"Take Profit: {settings['take_profit']*100}%\n\n"
            f"*Active Alerts:*\n{alert_summary}",
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Dashboard failed: {e}")
        await send_admin_alert(context.bot, f"Dashboard failed: {str(e)}")
        await update.message.reply_text("❌ Error loading dashboard")

async def inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline queries for quick signal previews."""
    query = update.inline_query.query.strip().upper()
    if not query or '/' not in query:
        return
    try:
        pair = query if query in await get_available_pairs() else config.default_trading_pair
        signal_text = await generate_bot_signal(pair, config.default_timeframe)
        results = [
            InlineQueryResultArticle(
                id=str(uuid.uuid4()),
                title=f"Signal for {pair}",
                input_message_content=InputTextMessageContent(signal_text, parse_mode='Markdown'),
                description="Latest trading signal"
            )
        ]
        await update.inline_query.answer(results)
    except Exception as e:
        logger.error(f"Inline query failed: {e}")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the current operation."""
    await update.message.reply_text("Operation cancelled.")
    return ConversationHandler.END

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
        else:
            await query.edit_message_text("❌ Invalid selection")
    except Exception as e:
        logger.error(f"Button handler error: {e}")
        await send_admin_alert(context.bot, f"Button handler error: {str(e)}")
        await query.edit_message_text("❌ Error processing selection")
    return None

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the Telegram bot."""
    logger.error(f"Update {update} caused error {context.error}")
    await send_admin_alert(context.bot, f"Bot error: {str(context.error)}")
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An error occurred while processing your request."
            )
    except Exception as e:
        logger.error(f"Error in error handler: {e}")

# Main Application
def main():
    """Main entry point for the trading bot."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Conversation handler for authentication and strategy selection
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            AUTH: [MessageHandler(filters.TEXT & ~filters.COMMAND, authenticate)],
            SELECTING_STRATEGY: [CallbackQueryHandler(handle_strategy_selection, pattern="^strat_")]
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        name="auth_conversation"
    )
    
    # Conversation handler for risk settings
    risk_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("set_risk", start_set_risk)],
        states={
            POSITION_SIZE: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_position_size)],
            STOP_LOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_stop_loss)],
            TAKE_PROFIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_take_profit)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    # Conversation handler for configuration
    config_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("set_config", set_config)],
        states={
            SET_CONFIG: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_config)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    # Initialize alert manager
    alert_manager = AlertManager()
    application.bot_data["alert_manager"] = alert_manager
    
    # Register handlers
    handlers = [
        conv_handler,
        risk_conv_handler,
        config_conv_handler,
        CommandHandler("help", help_command),
        CommandHandler("status", status),
        CommandHandler("backtest", backtest),
        CommandHandler("timeframe", set_timeframe),
        CommandHandler("strategies", select_strategies),
        CommandHandler("view_risk", view_risk),
        CommandHandler("set_config", set_config),
        CommandHandler("alert", set_alert),
        CommandHandler("signal", signal),
        CommandHandler("dashboard", dashboard),
        CommandHandler("cancel", cancel),
        InlineQueryHandler(inline_query),
        CallbackQueryHandler(button_handler)
    ]
    for handler in handlers:
        application.add_handler(handler)
    application.add_error_handler(error_handler)
    
    # Start alert checking and health checks
    application.job_queue.run_once(lambda context: alert_manager.check_alerts(application.bot), 0)
    application.job_queue.run_once(lambda context: health_check(application.bot), 0)
    
    # Warm up models
    async def warmup():
        try:
            logger.info("Warming up models...")
            data = await fetch_data_async(config.default_trading_pair, config.default_timeframe)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            MLTrader().train(df)
            logger.info("Model warmup complete")
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
            await send_admin_alert(application.bot, f"Model warmup failed: {str(e)}")
    
    asyncio.run(warmup())
    
    # Start the bot
    if IS_RENDER:
        # Webhook setup for Render
        webhook_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/webhook"
        logger.info(f"Starting webhook on port {PORT} with URL {webhook_url}")
        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path="/webhook",
            webhook_url=webhook_url
        )
    else:
        # Polling for local development
        logger.info("Starting polling mode")
        application.run_polling()

if __name__ == "__main__":
    main()