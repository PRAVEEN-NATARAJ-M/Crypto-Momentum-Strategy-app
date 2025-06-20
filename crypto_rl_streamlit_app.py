
import streamlit as st
import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import plotly.graph_objs as go

# 📥 Fetch historical data
def fetch_history(symbol="BTC/USD", interval="1h", api_key=None):
    if not api_key:
        return None
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": 100,
        "format": "JSON"
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    if "values" not in data:
        return None

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "ts"})
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df.get("volume", 0.0)
    if isinstance(df["volume"], pd.Series):
        df["volume"] = df["volume"].astype(float)
    else:
        df["volume"] = 0.0
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    return df

# 🎮 Custom Environment
from gym import Env
from gym.spaces import Discrete, Box

class CryptoEnv(Env):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.max_step = len(df) - 1
        self.action_space = Discrete(3)  # Buy, Hold, Sell
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 14
        return self._next_observation()

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([row["close"], row["rsi"], row["volume"]], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_step
        reward = 1 if action == 0 else -1 if action == 2 else 0
        obs = self._next_observation()
        return obs, reward, done, {}

# 🌐 Streamlit UI
st.set_page_config(page_title="Crypto Momentum RL Strategy", layout="wide")

st.sidebar.markdown("### ✅ Convergence Divergence:")
st.sidebar.info("Trend momentum indicator.\n\n☑️ Select your favorite crypto and see the RL model's prediction.")

api_key = st.sidebar.text_input("🔐 Enter TwelveData API Key", type="password")

symbol = st.sidebar.selectbox("📈 Select Trading Pair", ["BTC/USD", "ETH/USD", "ADA/USD"])
timeframe = st.sidebar.selectbox("🕒 Timeframe", ["1min", "5min", "15min", "1h", "4h", "1day"])
theme = st.sidebar.radio("🖌 Theme", ["Light", "Dark"])

if api_key:
    df = fetch_history(symbol=symbol, interval=timeframe, api_key=api_key)

    if df is not None and not df.empty:
        st.title("📊 Crypto Momentum Strategy using Reinforcement Learning")
        st.write(f"Showing data for **{symbol}** with **{timeframe}** timeframe.")

        # 🧠 Train RL model
        env = DummyVecEnv([lambda: CryptoEnv(df)])
        model = PPO("MlpPolicy", env, verbose=0)
        with st.spinner("Training model, please wait..."):
            model.learn(total_timesteps=10000)
        st.success("✅ Model training complete!")

        # 📈 Plot price & RSI
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df["ts"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
        fig.update_layout(title="Candlestick Chart", xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        st.line_chart(df[["rsi"]].set_index(df["ts"]))

        # 🤖 Show prediction
        obs = env.reset()
        action, _ = model.predict(obs)
        action_label = {0: "Buy", 1: "Hold", 2: "Sell"}[int(action)]
        st.subheader("🤖 RL Model Prediction")
        st.info(f"Predicted Action: **{action_label}**")
    else:
        st.warning("⚠️ Failed to fetch data. Please check your API key and selection.")
else:
    st.warning("Please enter a valid API key to proceed.")
