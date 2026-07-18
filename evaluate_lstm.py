import numpy as np, pandas as pd, torch, torch.nn as nn, joblib, json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from train_lstm import StockLSTM, make_sequences

with open('models/config.json') as f:
    config = json.load(f)
WINDOW, split_idx = config['window'], config['split_idx']

df = pd.read_csv('data/GOOG.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
close = df['Close'].values.reshape(-1, 1)
dates = df['Date'].values

train_raw, test_raw = close[:split_idx], close[split_idx:]
test_dates = dates[split_idx:]

scaler = joblib.load('models/scaler.pkl')
test_context = np.concatenate([train_raw[-WINDOW:], test_raw])
test_scaled = scaler.transform(test_context)
X_test, y_test = make_sequences(test_scaled, WINDOW)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

model = StockLSTM()
model.load_state_dict(torch.load('models/stock_lstm.pt', map_location='cpu', weights_only=True))
model.eval()

with torch.no_grad():
    preds_scaled = model(X_test_t).numpy()

preds = scaler.inverse_transform(preds_scaled)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(actual, preds))
mae = mean_absolute_error(actual, preds)
mape = np.mean(np.abs((actual - preds) / actual)) * 100
print(f"Test RMSE: ${rmse:.2f}")
print(f"Test MAE:  ${mae:.2f}")
print(f"Test MAPE: {mape:.2f}%")

with open('models/test_metrics.json', 'w') as f:
    json.dump({'rmse': float(rmse), 'mae': float(mae), 'mape': float(mape)}, f)

# ---- Training curves ----
with open('models/history.json') as f:
    history = json.load(f)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(history['train_loss'], label='Train Loss (MSE, scaled)')
ax.plot(history['test_loss'], label='Test Loss (MSE, scaled)')
ax.set_yscale('log')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log scale)')
ax.set_title('LSTM Training Loss', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('images/training_curves.png', dpi=120, bbox_inches='tight')
plt.close()

# ---- Actual vs Predicted ----
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(test_dates, actual, label='Actual Price', color='#264653', linewidth=1.5)
ax.plot(test_dates, preds, label='LSTM Predicted', color='#E76F51', linewidth=1.5, linestyle='--')
ax.set_title(f'Google Stock Price — Actual vs Predicted (Test Set, RMSE=${rmse:.2f})', fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Price ($)')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/actual_vs_predicted.png', dpi=120, bbox_inches='tight')
plt.close()

# ---- Full history plot ----
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(dates[:split_idx], train_raw, label='Train Period', color='#457B9D')
ax.plot(test_dates, actual, label='Test Period (Actual)', color='#264653')
ax.axvline(dates[split_idx], color='gray', linestyle=':', label='Train/Test Split')
ax.set_title('Google Stock Closing Price — Full History (2004-2023)', fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Price ($)')
ax.legend()
plt.tight_layout()
plt.savefig('images/full_history.png', dpi=120, bbox_inches='tight')
plt.close()

print("Saved: training_curves.png, actual_vs_predicted.png, full_history.png")
