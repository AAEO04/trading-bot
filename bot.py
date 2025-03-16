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

    def train(self, df: pd.DataFrame) -> None:
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
        "Welcome to the trading bot dashboard! Use the commands to interact with the bot."
    )
async def authenticate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_input = update.message.text
    if len(user_input) < 8:
        await show_dashboard(update, context)
        return AUTH  # Use the defined AUTH variable
    
    if hashlib.sha256(user_input.encode()).hexdigest() == PASSWORD_HASH:
        AUTHORIZED_USERS[update.effective_user.id] = True
        context.user_data["trading_pair"] = DEFAULT_TRADING_PAIR
        await show_dashboard(update)
        return ConversationHandler.END
    
    await update.message.reply_text("❌ Invalid password. Try again:")
    return AUTH  # Use the defined AUTH variable

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

# ====================== MAIN EXECUTION ======================
async def warmup_models():
    logging.info("Warming up ML models...")
    df = pd.DataFrame(await fetch_data_async())
    MLTrader().train(df)
    logging.info("Model warmup complete")

if __name__ == "__main__":
    bot_app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .rate_limiter(AIORateLimiter(max_retries=1))  # Reduce retries
        .build()
    )
    
    if IS_RENDER:
        asyncio.run(warmup_models())
    
    if config.environment == "production":
        # Create the web application
        web_app = aiohttp.web.Application()
        web_app.router.add_get("/", lambda r: aiohttp.web.Response(text="OK"))
        
        # Start the webhook
        bot_app.start()
        
        # Set the webhook
        asyncio.run(bot_app.bot.set_webhook(
            url=f"https://trading-bot-pn7h.onrender.com/{TELEGRAM_TOKEN}",
            secret_token=WEBHOOK_SECRET
        ))
        
        # Run the web application
        aiohttp.web.run_app(
            web_app,
            host="0.0.0.0",
            port=PORT,
            ssl_context=None
        )
    else:
        bot_app.run_polling()