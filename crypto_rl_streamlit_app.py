
import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import plotly.graph_objs as go
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

ACTION_LABELS = {0: "Sell", 1: "Hold", 2: "Buy"}

def fetch_history(symbol="BTC/USD", interval="1min", api_key="76d8f8e054464b98bb0228deb84f19b0"):
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
    if "volume" in df.columns:
        df["vol"] = df["volume"].astype(float)
    else:
        df["vol"] = pd.Series([0.0] * len(df))
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["macd"] = ta.trend.macd_diff(df["close"])
    return df.dropna()

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

# Sidebar content
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
**What is Cryptocurrency?**
A digital or virtual currency that uses cryptography for security.

**Momentum Strategy**
Uses indicators like RSI and MACD to identify market trends and make buy/sell decisions.

**Actions Explained**
- 🟥 Sell: Agent recommends selling your crypto assets.
- ⏸ Hold: Agent suggests holding current positions.
- 🟩 Buy: Agent suggests purchasing crypto assets.
""")

st.title("📊 Crypto Momentum Strategy using Reinforcement Learning")
symbol = st.selectbox("Select Trading Pair", ["BTC/USD", "ETH/USD", "ADA/USD"])

df = fetch_history(symbol=symbol)
env = DummyVecEnv([lambda: CryptoEnv(df)])
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=10000)

latest = df.iloc[-1]
obs = np.array([[latest.rsi, latest.macd, latest.close]], dtype=np.float32)
action, _ = model.predict(obs)
action_label = ACTION_LABELS[int(action)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["ts"], y=df["close"], mode='lines', name='Close Price'))
fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Time", yaxis_title="USD", height=400)

st.plotly_chart(fig, use_container_width=True)
st.subheader("📬 Agent Decision:")
st.success(f"The agent recommends: **{action_label}**")
