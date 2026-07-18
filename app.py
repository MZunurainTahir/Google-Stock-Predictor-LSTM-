import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Price LSTM", page_icon="📈", layout="wide")


class StockLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


@st.cache_resource
def load_artifacts():
    model = StockLSTM()
    model.load_state_dict(torch.load('models/stock_lstm.pt', map_location='cpu', weights_only=True))
    model.eval()
    scaler = joblib.load('models/scaler.pkl')
    with open('models/config.json') as f:
        config = json.load(f)
    with open('models/test_metrics.json') as f:
        metrics = json.load(f)
    return model, scaler, config, metrics


@st.cache_data
def load_data():
    df = pd.read_csv('data/GOOG.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)


st.title("📈 Google Stock Price Predictor — LSTM")
st.caption("2-layer LSTM trained from scratch in PyTorch on 19 years of real Google (GOOG) stock data")

model, scaler, config, metrics = load_artifacts()
df = load_data()
WINDOW = config['window']

col1, col2, col3 = st.columns(3)
col1.metric("Test RMSE", f"${metrics['rmse']:.2f}")
col2.metric("Test MAE", f"${metrics['mae']:.2f}")
col3.metric("Test MAPE", f"{metrics['mape']:.1f}%")

st.warning("⚠️ Educational project only — stock prices are influenced by countless real-world factors this model doesn't see. Not financial advice.")

st.divider()

tab1, tab2 = st.tabs(["🔮 Predict Next Day", "📊 Full History"])

with tab1:
    st.subheader("Predict tomorrow's closing price from the last 60 trading days")

    n_days_ahead = st.slider("Days to forecast ahead (recursive)", 1, 10, 5)

    close_prices = df['Close'].values.reshape(-1, 1)
    last_window = close_prices[-WINDOW:]
    scaled_window = scaler.transform(last_window).flatten().tolist()

    predictions = []
    with torch.no_grad():
        for _ in range(n_days_ahead):
            x = torch.tensor(scaled_window[-WINDOW:], dtype=torch.float32).view(1, WINDOW, 1)
            pred_scaled = model(x).item()
            predictions.append(pred_scaled)
            scaled_window.append(pred_scaled)

    predictions_actual = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    last_date = df['Date'].iloc[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=n_days_ahead)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        recent = df.tail(90)
        ax.plot(recent['Date'], recent['Close'], label='Recent Actual', color='#264653')
        ax.plot(future_dates, predictions_actual, label='Forecast', color='#E76F51', marker='o', linestyle='--')
        ax.axvline(last_date, color='gray', linestyle=':')
        ax.legend()
        ax.set_title('Recent History + Forecast', fontweight='bold')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.write("**Forecast values:**")
        forecast_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Predicted Close': [f"${p:.2f}" for p in predictions_actual]
        })
        st.dataframe(forecast_df, hide_index=True, use_container_width=True)

    st.caption("⚠️ Multi-day forecasts are recursive (each prediction feeds the next) — uncertainty compounds quickly beyond 1-2 days.")

with tab2:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df['Date'], df['Close'], color='#457B9D')
    ax.set_title('Google Stock Closing Price — Full History', fontweight='bold')
    ax.set_xlabel('Date'); ax.set_ylabel('Price ($)')
    st.pyplot(fig)

    st.image('images/actual_vs_predicted.png', caption='Model performance on held-out test period', use_container_width=True)

st.sidebar.header("About")
st.sidebar.info(
    "2-layer LSTM (64 hidden units) trained from scratch in PyTorch on 19 years of real "
    "GOOG closing prices. Uses a 60-day sliding window. See the notebook for full methodology."
)
