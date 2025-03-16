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
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
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
import aiohttp.web
import asyncio
import logging
import threading
import uuid
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

# ====================== CONFIGURATION ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)

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

class Backtester:
    def __init__(self, tp: float = 5, sl: float = 3) -> None:
        self.tp = tp / 100
        self.sl = sl / 100

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        capital: float = 10000
        position: int = 0
        entry_price: Optional[float] = None
        portfolio = []
        
        for i, row in df.iterrows():
            price = row["close"]
            if position > 0 and entry_price is None:
                logging.error("Invalid state: Position without entry price!")
                position = 0
                continue
            
            if position > 0:
                ret = (price - entry_price) / entry_price
                if ret >= self.tp or ret <= -self.sl:
                    capital += position * price
                    position = 0
                    entry_price = None
            
            if row["signal"] == 1 and position == 0:
                position = int(capital // price)
                entry_price = price
                capital -= position * price
            elif row["signal"] == -1 and position > 0:
                capital += position * price
                position = 0
                entry_price = None
            
            portfolio.append(capital + position * price)
        
        df["portfolio"] = portfolio
        return df

# ====================== DATA FETCHING ======================
async def fetch_data_async(
    trading_pair: str = DEFAULT_TRADING_PAIR, 
    max_retries: int = 5, 
    initial_delay: float = 1.0
) -> pd.DataFrame:
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return await asyncio.to_thread(
                EXCHANGE.fetch_ohlcv, 
                trading_pair, 
                '1d', 
                limit=365
            )
        except Exception as e:
            if attempt == max_retries:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5)  # Reduce max delay to 5 seconds
            logging.warning(f"Retry {attempt} failed. New delay: {delay:.1f}s")

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

async def send_results(message: Any, df: pd.DataFrame) -> None:
    filename = f"results_{uuid.uuid4().hex}.png"
    try:
        with plot_lock:
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df["portfolio"], label="Portfolio Value")
            plt.title("Backtest Results")
            plt.ylabel("USD")
            plt.grid(True)
            plt.savefig(filename)
            plt.close()
        
        await message.reply_photo(
            photo=open(filename, "rb"),
            caption=f"📈 Final Balance: ${df['portfolio'].iloc[-1]:.2f}"
        )
    finally:
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except OSError as e:
            logging.error(f"Error deleting file {filename}: {e}")

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
    """Help command handler."""
    logging.info(f"Processing /help command from user {update.effective_user.id}")
    await update.message.reply_text(
        "📚 Available commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/backtest - Run backtest simulation\n"
        "/status - Show current status"
    )

async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Backtest command handler."""
    logging.info(f"Processing /backtest command from user {update.effective_user.id}")
    if update.effective_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("❌ Please authenticate first using /start")
        return
    
    await update.message.reply_text("🔄 Running backtest simulation...")
    try:
        data = await fetch_data_async(context.user_data.get("trading_pair", DEFAULT_TRADING_PAIR))
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Add technical indicators and generate signals
        df = MLTrader()._add_indicators(df)
        df['signal'] = np.where(df['MA_50'] > df['MA_200'], 1, -1)
        
        # Run backtest
        results = Backtester().run(df)
        await send_results(update.message, results)
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
    await update.message.reply_text(
        f"📊 Bot Status:\n"
        f"Trading Pair: {trading_pair}\n"
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

async def handle_webhook(request):
    """Process webhook updates from Telegram."""
    try:
        # Verify secret token
        if request.headers.get('X-Telegram-Bot-Api-Secret-Token') != WEBHOOK_SECRET:
            logging.warning("Invalid webhook secret token")
            return aiohttp.web.Response(status=403)
        
        # Get update data
        update_data = await request.json()
        logging.info("Received webhook update")
        
        # Process update
        update = Update.de_json(update_data, bot_app.bot)
        await bot_app.process_update(update)
        
        return aiohttp.web.Response()
    except Exception as e:
        logging.error(f"Webhook error: {e}")
        return aiohttp.web.Response()

# ====================== MAIN EXECUTION ======================
async def warmup_models():
    logging.info("Warming up ML models...")
    raw_data = await fetch_data_async()
    df = pd.DataFrame(
        raw_data,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    MLTrader().train(df)
    logging.info("Model warmup complete")

if __name__ == "__main__":
    # Initialize bot application with more detailed logging
    logging.info("Initializing bot application...")
    bot_app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .rate_limiter(AIORateLimiter(max_retries=1))
        .build()
    )
    
    # Register handlers with logging
    logging.info("Registering command handlers...")
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            AUTH: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, 
                    authenticate
                )
            ],
        },
        fallbacks=[CommandHandler("start", start)],
        name="auth_conversation"
    )
    
    # Add handlers
    bot_app.add_handler(conv_handler)
    bot_app.add_handler(CommandHandler("help", help_command))
    bot_app.add_handler(CommandHandler("backtest", backtest))
    bot_app.add_handler(CommandHandler("status", status))
    bot_app.add_error_handler(error_handler)
    logging.info("All handlers registered successfully")
    
    if config.environment == "production":
        async def main():
            try:
                # Initialize web app
                web_app = aiohttp.web.Application()
                web_app.router.add_get("/", lambda r: aiohttp.web.Response(text="OK"))
                
                # Start bot
                await bot_app.initialize()
                await bot_app.start()
                logging.info("Bot initialized and started")
                
                # Set webhook
                webhook_url = f"https://trading-bot-pn7h.onrender.com/{TELEGRAM_TOKEN}"
                await bot_app.bot.set_webhook(
                    url=webhook_url,
                    allowed_updates=["message", "callback_query"],
                    secret_token=WEBHOOK_SECRET,
                    drop_pending_updates=True
                )
                logging.info(f"Webhook set to: {webhook_url}")
                
                # Add webhook route
                web_app.router.add_post(f"/{TELEGRAM_TOKEN}", handle_webhook)
                
                # Start server
                runner = aiohttp.web.AppRunner(web_app)
                await runner.setup()
                site = aiohttp.web.TCPSite(
                    runner,
                    host="0.0.0.0",
                    port=int(os.getenv("PORT", 10000))
                )
                await site.start()
                logging.info(f"Server started on port {PORT}")
                
                # Keep alive with health checks
                while True:
                    try:
                        me = await bot_app.bot.get_me()
                        logging.info(f"Bot is alive as @{me.username}")
                        await asyncio.sleep(300)
                    except Exception as e:
                        logging.error(f"Health check failed: {e}")
                        await asyncio.sleep(60)
            
            except Exception as e:
                logging.error(f"Fatal error: {e}")
                raise
        
        # Run the bot
        if IS_RENDER:
            asyncio.run(warmup_models())
        asyncio.run(main())
    else:
        bot_app.run_polling()