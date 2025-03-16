import os
import json
import hashlib
import ccxt
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.utils import resample
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

# ====================== CONFIGURATION ======================
CONFIG_FILE = "config.json"
AUTHORIZED_USERS = {}
STRATEGIES = ["MA Crossover", "RSI", "MACD", "Combined", "ML Enhanced"]
TP = 5  # Take profit percentage
SL = 3  # Stop loss percentage

# ====================== ML TRADING MODELS ======================
class MLTrader:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'svm': SVR(),
            'ann': MLPRegressor(hidden_layer_sizes=(50, 50))
        }
        self.confidence_intervals = {}
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        # Technical Indicators
        df['MA_50'] = df['close'].rolling(50).mean()
        df['MA_200'] = df['close'].rolling(200).mean()
        df['RSI'] = self._calculate_rsi(df)
        df['MACD'] = self._calculate_macd(df)
        return df.dropna()
        
    def train_models(self, df, n_bootstrap=100):
        df = self.prepare_features(df)
        X = df[['MA_50', 'MA_200', 'RSI', 'MACD']].iloc[:-1]
        y = df['close'].pct_change().shift(-1).dropna()
        
        for name, model in self.models.items():
            predictions = []
            for _ in range(n_bootstrap):
                X_sample, y_sample = resample(X, y)
                model.fit(X_sample, y_sample)
                predictions.append(model.predict(X.iloc[[-1]]))
            self.confidence_intervals[name] = np.percentile(predictions, [5, 95])
            
    def get_predictions(self, df):
        df = self.prepare_features(df)
        X = self.scaler.transform(df[['MA_50', 'MA_200', 'RSI', 'MACD']].iloc[[-1]])
        return {name: model.predict(X)[0] for name, model in self.models.items()}

# ====================== TRADING STRATEGIES ======================
class TradingStrategies:
    @staticmethod
    def ma_crossover(df, short=50, long=200):
        df['MA_Short'] = df['close'].rolling(short).mean()
        df['MA_Long'] = df['close'].rolling(long).mean()
        return np.where(df['MA_Short'] > df['MA_Long'], 1, -1)

    @staticmethod
    def rsi_strategy(df, period=14, overbought=70, oversold=30):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        rs = gain.rolling(period).mean() / loss.rolling(period).mean()
        rsi = 100 - (100 / (1 + rs))
        return np.where(rsi > overbought, -1, np.where(rsi < oversold, 1, 0))

    @staticmethod
    def macd_strategy(df):
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        return np.where(macd > signal, 1, -1)

# ====================== BACKTESTER WITH TP/SL ======================
class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        
    def run_backtest(self, df, tp=TP, sl=SL):
        capital = self.initial_capital
        position = 0
        entry_price = 0
        portfolio = []
        
        for idx, row in df.iterrows():
            current_price = row['close']
            
            # Check TP/SL
            if position > 0:
                returns_pct = (current_price - entry_price) / entry_price
                if returns_pct >= tp/100 or returns_pct <= -sl/100:
                    capital += position * current_price
                    position = 0
            
            # Execute signals
            if row['signal'] == 1 and position == 0:
                position = capital // current_price
                entry_price = current_price
                capital -= position * current_price
            elif row['signal'] == -1 and position > 0:
                capital += position * current_price
                position = 0
                
            portfolio.append(capital + position * current_price)
            
        df['portfolio'] = portfolio
        return self._calculate_metrics(df)
    
    def _calculate_metrics(self, df):
        returns = df['portfolio'].pct_change().dropna()
        return {
            'Total Return': f"{(df['portfolio'].iloc[-1]/self.initial_capital-1):.2%}",
            'Sharpe Ratio': f"{returns.mean()/returns.std()*np.sqrt(252):.2f}",
            'Max Drawdown': f"{(df['portfolio']/df['portfolio'].cummax()-1).min():.2%}"
        }

# ====================== TELEGRAM BOT HANDLERS ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id in AUTHORIZED_USERS:
        return await show_dashboard(update)
    
    await update.message.reply_text("🔒 Enter access password:")
    return AUTH

async def authenticate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    config = json.load(open(CONFIG_FILE))
    
    if hashlib.sha256(user_input.encode()).hexdigest() == config['password_hash']:
        AUTHORIZED_USERS[update.effective_user.id] = True
        await show_dashboard(update)
        return ConversationHandler.END
    await update.message.reply_text("❌ Invalid password. Try again:")
    return AUTH

async def show_dashboard(update: Update):
    keyboard = [
        [InlineKeyboardButton(s, callback_data=s)] for s in STRATEGIES
    ] + [[InlineKeyboardButton("📊 Backtest", callback_data="backtest")]]
    
    await update.message.reply_text(
        "🏦 Trading Bot Dashboard\nSelect Strategy:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    strategy = query.data
    df = fetch_market_data()
    
    if strategy == "ML Enhanced":
        ml = MLTrader()
        ml.train_models(df)
        predictions = ml.get_predictions(df)
        # Add ML strategy implementation
    else:
        df['signal'] = execute_strategy(df, strategy)
    
    context.user_data['df'] = df.to_dict()
    await query.edit_message_text(f"{strategy} signals generated!")

async def run_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = pd.DataFrame(context.user_data['df'])
    results = Backtester().run_backtest(df)
    
    plt.figure(figsize=(12,6))
    df['portfolio'].plot(title='Backtest Results')
    plt.savefig('backtest.png')
    
    await update.callback_query.message.reply_photo(
        photo=open('backtest.png', 'rb'),
        caption=f"📊 Results:\n" + "\n".join([f"{k}: {v}" for k,v in results.items()])
    )

# ====================== UTILITY FUNCTIONS ======================
def fetch_market_data():
    exchange = ccxt.binance({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=365)
    return pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])

def execute_strategy(df, strategy):
    if strategy == "MA Crossover":
        return TradingStrategies.ma_crossover(df)
    elif strategy == "RSI":
        return TradingStrategies.rsi_strategy(df)
    elif strategy == "MACD":
        return TradingStrategies.macd_strategy(df)
    elif strategy == "Combined":
        return np.sign(
            TradingStrategies.ma_crossover(df) + 
            TradingStrategies.rsi_strategy(df) + 
            TradingStrategies.macd_strategy(df)
        )

# ====================== MAIN APPLICATION ======================
AUTH = 0


if __name__ == '__main__':
    # For production (Render), use environment variables directly
    if os.getenv('ENVIRONMENT') == 'production':
        config = {
            'telegram_token': os.getenv('TELEGRAM_TOKEN'),
            'password_hash': os.getenv('PASSWORD_HASH')
        }
    else:
        # For local development
        if not os.path.exists(CONFIG_FILE):
            print("First-time setup required! Run 'python setup.py'")
            exit(1)
        config = json.load(open(CONFIG_FILE))        
    config = json.load(open(CONFIG_FILE))
    
    app = ApplicationBuilder().token(config['telegram_token']).build()
    
    # Conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={AUTH: [MessageHandler(filters.TEXT & ~filters.COMMAND, authenticate)]},
        fallbacks=[]
    )
    
    # Handlers
    app.add_handler(conv_handler)
    app.add_handler(CallbackQueryHandler(handle_strategy, pattern="^(" + "|".join(STRATEGIES) + ")$"))
    app.add_handler(CallbackQueryHandler(run_backtest, pattern="^backtest$"))
    
    # Start bot
    if os.getenv('ENVIRONMENT') == 'production':
        app.run_webhook(
            listen='0.0.0.0',
            port=int(os.getenv('PORT', 8443)),
            url_path=config['telegram_token'],
            webhook_url=f"https://your-render-app.onrender.com/{config['telegram_token']}"
        )
    else:
        app.run_polling()