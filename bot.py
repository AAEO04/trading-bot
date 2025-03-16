import os
import hashlib
import ccxt
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Required for server environments
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
    ConversationHandler
)
import aiohttp.web

# ====================== CONFIGURATION ======================
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
PASSWORD_HASH = os.getenv('PASSWORD_HASH')
PORT = int(os.getenv('PORT', 10000))
AUTHORIZED_USERS = {}
EXCHANGE = ccxt.binance({'enableRateLimit': True})
STRATEGIES = ["MA Crossover", "RSI", "MACD", "ML Enhanced"]

# ====================== TRADING CORE ======================
class MLTrader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'linear': LinearRegression(),
            'svm': SVR(),
            'ann': MLPRegressor(hidden_layer_sizes=(50, 50))
        }

    def _add_indicators(self, df):
        """Calculate technical indicators"""
        # Moving Averages
        df['MA_50'] = df['close'].rolling(50).mean()
        df['MA_200'] = df['close'].rolling(200).mean()
        
        # RSI Calculation
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD Calculation
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df.dropna()

    def train(self, df):
        df = self._add_indicators(df)
        X = df[['MA_50', 'MA_200', 'RSI', 'MACD']].iloc[:-1]
        y = df['close'].pct_change().shift(-1).dropna()
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.models.items():
            model.fit(X_scaled, y)

class Backtester:
    def __init__(self, tp=5, sl=3):
        self.tp = tp / 100
        self.sl = sl / 100
    
    def run(self, df):
        capital = 10000
        position = 0
        portfolio = []
        entry_price = 0
        
        for _, row in df.iterrows():
            price = row['close']
            
            if position > 0:
                ret = (price - entry_price) / entry_price
                if ret >= self.tp or ret <= -self.sl:
                    capital += position * price
                    position = 0
            
            if row['signal'] == 1 and position == 0:
                position = capital // price
                entry_price = price
                capital -= position * price
            elif row['signal'] == -1 and position > 0:
                capital += position * price
                position = 0
                
            portfolio.append(capital + position * price)
        
        df['portfolio'] = portfolio
        return df

# ====================== TELEGRAM HANDLERS ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id in AUTHORIZED_USERS:
        return await show_dashboard(update)
    await update.message.reply_text("🔒 Enter access password:")
    return AUTH

async def authenticate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    if hashlib.sha256(user_input.encode()).hexdigest() == PASSWORD_HASH:
        AUTHORIZED_USERS[update.effective_user.id] = True
        await show_dashboard(update)
        return ConversationHandler.END
    await update.message.reply_text("❌ Invalid password. Try again:")
    return AUTH

async def show_dashboard(update: Update):
    keyboard = [[InlineKeyboardButton(s, callback_data=s)] for s in STRATEGIES]
    await update.message.reply_text(
        "🏦 Trading Bot Dashboard",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    df = fetch_data()
    ml = MLTrader()
    ml.train(df)
    
    if query.data == "ML Enhanced":
        predictions = ml.models['ann'].predict(ml.scaler.transform(df[['MA_50', 'MA_200', 'RSI', 'MACD']].iloc[[-1]]))
        df['signal'] = np.where(predictions > 0.5, 1, -1)
    else:
        df['signal'] = apply_strategy(df, query.data)
    
    backtest_results = Backtester().run(df)
    await send_results(query.message, backtest_results)

# ====================== UTILITIES ======================
def fetch_data():
    ohlcv = EXCHANGE.fetch_ohlcv('BTC/USDT', '1d', limit=365)
    return pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])

def apply_strategy(df, strategy):
    """Generate trading signals based on selected strategy"""
    df = df.copy()
    
    if strategy == "MA Crossover":
        df['MA_20'] = df['close'].rolling(20).mean()
        df['MA_50'] = df['close'].rolling(50).mean()
        df['signal'] = np.where(df['MA_20'] > df['MA_50'], 1, -1)
    
    elif strategy == "RSI":
        df['signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    
    elif strategy == "MACD":
        df['signal'] = np.where(df['MACD'] > df['Signal'], 1, -1)
    
    return df['signal'].fillna(0).astype(int)

async def send_results(message, df):
    plt.figure(figsize=(12, 6))
    df['portfolio'].plot(title='Backtest Results')
    plt.ylabel('Portfolio Value (USD)')
    plt.xlabel('Date')
    plt.grid(True)
    plt.savefig('results.png')
    plt.close()
    
    await message.reply_photo(
        photo=open('results.png', 'rb'),
        caption=f"📈 Final Balance: ${df['portfolio'].iloc[-1]:.2f}"
    )

# ====================== MAIN EXECUTION ======================
AUTH = 0

if __name__ == '__main__':
    bot_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            AUTH: [MessageHandler(filters.TEXT & ~filters.COMMAND, authenticate)]
        },
        fallbacks=[]
    )
    
    bot_app.add_handler(conv_handler)
    bot_app.add_handler(CallbackQueryHandler(handle_strategy))
    
    if os.getenv('ENVIRONMENT') == 'production':
        # Configure web server
        web_app = bot_app.updater.http._app
        
        async def health_check(request):
            return aiohttp.web.Response(text="Trading Bot Operational", status=200)
        
        web_app.router.add_get('/', health_check)
        
        # Start webhook
        bot_app.run_webhook(
            host='0.0.0.0',
            port=PORT,
            webhook_url=f"https://trading-bot-pn7h.onrender.com/{TELEGRAM_TOKEN}",
            url_path=TELEGRAM_TOKEN,
            ssl_context=None
        )
    else:
        bot_app.run_polling()