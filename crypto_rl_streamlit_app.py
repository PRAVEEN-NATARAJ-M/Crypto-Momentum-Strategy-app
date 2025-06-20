# crypto_rl_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import plotly.graph_objs as go
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

# 📥 User Inputs
st.set_page_config(page_title="Crypto Momentum RL", layout="wide")

# Sidebar: Info and Settings
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    **Crypto Momentum Strategy** is a simple reinforcement learning (RL) based trading bot that uses RSI and MACD to make Buy/Hold/Sell decisions on cryptocurrencies.

    🔍 **Terms Explained:**
    - **RSI (Relative Strength Index):** Detects overbought or oversold conditions.
    - **MACD (Moving Average Convergence Divergence):** Trend momentum indicator.

    📈 Select your favorite crypto and see the RL model's prediction.
    """)
    api_key = st.text_input("🔑 Enter TwelveData API Key", type="password")
    symbol = st.selectbox("📉 Select Trading Pair", ["BTC/USD", "ETH/USD", "ADA/USD"])
    timeframe = st.selectbox("🕒 Timeframe", ["1min", "5min", "15min", "1h", "4h", "1day"], index=2)
    theme = st.radio("🎨 Theme", ["Light", "Dark"], index=0)

# 📈 1. Data Fetching
def fetch_history(symbol="BTC/USD", interval="1min", api_key=""):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={api_key}"
    response = requests.get(url).json()
    if "values" not in response:
        raise ValueError(f"Error fetching data: {response}")
    df = pd.DataFrame(response["values"])
    df = df.rename(columns={"datetime": "ts"})
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
        else:
            raise KeyError(f"Missing column: {col}")
    df["vol"] = pd.to_numeric(df.get("volume", pd.Series([0.0]*len(df))), errors="coerce")
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd_diff()
    return df.dropna()

# 🧠 2. Gym Environment
class CryptoEnv(Env):
    def __init__(self, df):
        self.df = df.reset_index()
        self.n = len(df)
        self.ptr = 0
        self.action_space = Discrete(3)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.balance = 1_000
        self.crypto = 0.0

    def reset(self, seed=None, options=None):
        self.ptr = 0
        self.balance, self.crypto = 1000, 0
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.loc[self.ptr]
        return np.array([row.rsi, row.macd, row.close], dtype=np.float32)

    def step(self, action):
        price = self.df.loc[self.ptr, 'close']
        if action == 2:
            self.crypto += self.balance / price
            self.balance = 0
        elif action == 0:
            self.balance += self.crypto * price
            self.crypto = 0
        self.ptr += 1
        done = self.ptr >= self.n - 1
        net = self.balance + self.crypto * price
        reward = net - 1000
        return self._get_observation(), reward, done, False, {}

# 🚀 Fetch & train model
if api_key:
    try:
        df = fetch_history(symbol=symbol, interval=timeframe, api_key=api_key)
        env = DummyVecEnv([lambda: CryptoEnv(df)])
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=10000)

        # 🔮 Prediction
        latest = df.iloc[-1]
        obs = np.array([[latest.rsi, latest.macd, latest.close]], dtype=np.float32)
        action, _ = model.predict(obs)
        label = ["Sell", "Hold", "Buy"][int(action)]

        # 🎨 Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["close"], mode="lines", name="Close"))
        fig.update_layout(
            title=f"{symbol} ({timeframe}) - Price Chart",
            xaxis_title="Time", yaxis_title="Price (USD)",
            template="plotly_dark" if theme == "Dark" else "plotly_white",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)
        st.subheader("🤖 Model Recommendation")
        st.info(f"**Action:** {label}  |  **RSI:** {latest.rsi:.2f}  |  **MACD:** {latest.macd:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.warning("🔑 Please enter a valid API key to proceed.")