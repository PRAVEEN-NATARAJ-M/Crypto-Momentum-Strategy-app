
# crypto_rl_streamlit_app.py

# ========================
# 📦 Import Libraries
# ========================
import streamlit as st
import pandas as pd
import requests
import numpy as np
import ta
import plotly.graph_objs as go
from gym import Env
from gym.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ========================
# 🔧 Helper Function: Fetch Historical Data
# ========================
def fetch_history(symbol="BTC/USD", interval="1min", api_key="YOUR_API_KEY_HERE"):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={api_key}"
    response = requests.get(url).json()
    if "values" not in response:
        st.error("API Error: Check symbol or API key.")
        return None
    df = pd.DataFrame(response["values"])
    df = df.rename(columns={"datetime": "ts"})
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["vol"] = df["volume"].astype(float) if "volume" in df.columns else 0.0
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["macd"] = ta.trend.macd_diff(df["close"])
    return df.dropna()

# ========================
# 🧠 Custom Gym Environment
# ========================
class CryptoEnv(Env):
    def __init__(self, df):
        self.df = df.reset_index()
        self.n = len(df)
        self.ptr = 0
        self.action_space = Discrete(3)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.balance = 1000.0
        self.crypto = 0.0

    def reset(self):
        self.ptr = 0
        self.balance, self.crypto = 1000.0, 0.0
        return self._get_observation()

    def _get_observation(self):
        row = self.df.loc[self.ptr]
        return np.array([row.rsi, row.macd, row.close], dtype=np.float32)

    def step(self, action):
        price = self.df.loc[self.ptr, 'close']
        if action == 2:  # Buy
            self.crypto += self.balance / price
            self.balance = 0
        elif action == 0:  # Sell
            self.balance += self.crypto * price
            self.crypto = 0
        self.ptr += 1
        done = self.ptr >= self.n - 1
        net = self.balance + self.crypto * price
        reward = net - 1000
        return self._get_observation(), reward, done, {}

# ========================
# 🎯 Action Mapping
# ========================
ACTION_LABELS = {0: "Sell", 1: "Hold", 2: "Buy"}

# ========================
# 🚀 Streamlit App UI
# ========================
st.set_page_config(layout="wide", page_title="Crypto RL Agent")
st.title("📈 Crypto Price Momentum Strategy (Reinforcement Learning)")

# Sidebar - Explanation
st.sidebar.header("📚 About This App")
st.sidebar.markdown("""
**What is Cryptocurrency?**
Cryptocurrency is a digital or virtual form of money based on blockchain technology.

**What is a Crypto Momentum Strategy?**
A trading strategy that leverages indicators like RSI and MACD to ride on short-term trends.

**Key Terms**
- **RSI**: Measures overbought or oversold conditions.
- **MACD**: Indicates momentum and trend direction.
- **PPO Agent**: A Reinforcement Learning model that learns to maximize profit.
""")

# Sidebar - Symbol selection and API key input
api_key = st.sidebar.text_input("🔑 Enter TwelveData API Key", type="password")
symbol = st.sidebar.selectbox("💱 Select Trading Pair", ["BTC/USD", "ETH/USD", "ADA/USD"])

# Fetch & visualize
if api_key:
    df = fetch_history(symbol, api_key=api_key)
    if df is not None:
        # PPO Training (short episode for demo)
        env = DummyVecEnv([lambda: CryptoEnv(df)])
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=10000)

        # Latest data input for prediction
        latest = df.iloc[-1]
        obs = np.array([[latest.rsi, latest.macd, latest.close]], dtype=np.float32)
        action, _ = model.predict(obs)
        action_label = ACTION_LABELS[int(action)]

        # Price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["close"], mode='lines', name="Close Price"))
        fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Timestamp", yaxis_title="Price (USD)")

        # Show predictions and chart
        st.subheader("📊 Market Indicators")
        st.metric("RSI", f"{latest.rsi:.2f}")
        st.metric("MACD", f"{latest.macd:.4f}")
        st.metric("Price", f"${latest.close:.2f}")
        st.success(f"🤖 RL Agent Action: **{action_label}**")

        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please enter your API key in the sidebar to begin.")

