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
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    model_config = SettingsConfigDict(env_file=".env")

config = Settings()  # Critical instantiation

TELEGRAM_TOKEN = config.telegram_token
PASSWORD_HASH = config.password_hash
PORT = config.port
WEBHOOK_SECRET = config.webhook_secret
IS_RENDER = config.render

AUTHORIZED_USERS: Dict[int, bool] = {}
EXCHANGE = ccxt.binance({"enableRateLimit": True})
STRATEGIES = ["MA Crossover", "RSI", "MACD", "ML Enhanced"]
DEFAULT_TRADING_PAIR = "BTC/USDT"

plot_lock = threading.Lock()

# ====================== TRADING CORE ======================
class MLTrader:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        if IS_RENDER:
            self._init_new_models()
            logging.info("Render environment: Using fresh models")
        else:
            self._load_or_create_models()

    def _init_new_models(self):
        self.models = {
            "linear": LinearRegression(),
            "svm": SVR(),
            "ann": MLPRegressor(hidden_layer_sizes=(50, 50),  # Added comma
        }

    def _load_or_create_models(self):
        try:
            self.models["linear"] = joblib.load("model_linear.pkl")
            logging.info("Loaded linear model from disk")
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
        # ... (keep existing indicator code unchanged)

    def train(self, df: pd.DataFrame) -> None:
        # ... (keep existing training code unchanged)

class Backtester:
    # ... (keep existing backtester code unchanged)

# ====================== DATA FETCHING ======================
async def fetch_data_async(
    trading_pair: str = DEFAULT_TRADING_PAIR, 
    max_retries: int = 5, 
    initial_delay: float = 1.0
) -> pd.DataFrame:
    # ... (keep existing fetch code unchanged)

# ====================== TELEGRAM HANDLERS ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (keep existing start handler unchanged)

async def authenticate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (keep existing auth code with password length check)

async def show_dashboard(update_or_query: Any) -> None:
    # ... (keep existing dashboard code unchanged)

async def handle_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (keep existing strategy handling code unchanged)

async def train_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (keep existing training command unchanged)

def apply_strategy(df: pd.DataFrame, strategy: str) -> pd.Series:
    # ... (keep existing strategy application code unchanged)

async def send_results(message: Any, df: pd.DataFrame) -> None:
    # ... (keep existing plot generation with UUID filenames)

# ====================== MAIN EXECUTION ======================
async def warmup_models():
    try:
        logging.info("Initializing models...")
        df = pd.DataFrame(await fetch_data_async())
        MLTrader().train(df)
        logging.info("Model initialization complete")
    except Exception as e:
        logging.error(f"Model warmup failed: {str(e)}")

if __name__ == "__main__":
    # Configure rate limiter for production
    bot_app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .rate_limiter(AIORateLimiter(
            max_retries=3,
            max_delay=30
        ))
        .build()
    )
    
    # Warmup models on Render
    if IS_RENDER:
        asyncio.run(warmup_models())
    
    # Production webhook configuration
    if os.getenv("ENVIRONMENT") == "production":
        web_app = aiohttp.web.Application()
        web_app.router.add_get("/", lambda r: aiohttp.web.Response(text="OK"))
        
        bot_app.run_webhook(
            web_app=web_app,
            host="0.0.0.0",
            port=PORT,
            webhook_url=f"https://trading-bot-pn7h.onrender.com/{TELEGRAM_TOKEN}",
            secret_token=WEBHOOK_SECRET,
            ssl_context=None
        )
    else:
        bot_app.run_polling()