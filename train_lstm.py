import numpy as np, pandas as pd, torch, torch.nn as nn, joblib, json, time
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)
np.random.seed(42)

class StockLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last time step


def make_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def main():
    WINDOW = 60
    df = pd.read_csv('data/GOOG.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    close = df['Close'].values.reshape(-1, 1)

    split_idx = int(len(close) * 0.85)
    train_raw, test_raw = close[:split_idx], close[split_idx:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    test_context = np.concatenate([train_raw[-WINDOW:], test_raw])
    test_scaled = scaler.transform(test_context)

    X_train, y_train = make_sequences(train_scaled, WINDOW)
    X_test, y_test = make_sequences(test_scaled, WINDOW)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    print(f"Train sequences: {X_train_t.shape}, Test sequences: {X_test_t.shape}")

    model = StockLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    EPOCHS = 60
    history = {'train_loss': [], 'test_loss': []}

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t)

        history['train_loss'].append(loss.item())
        history['test_loss'].append(test_loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - train_loss: {loss.item():.6f} - test_loss: {test_loss.item():.6f}")

    print(f"Training took {time.time()-t0:.1f}s")

    torch.save(model.state_dict(), 'models/stock_lstm.pt')
    joblib.dump(scaler, 'models/scaler.pkl')
    with open('models/history.json', 'w') as f:
        json.dump(history, f)
    with open('models/config.json', 'w') as f:
        json.dump({'window': WINDOW, 'split_idx': split_idx}, f)

    print("Saved model, scaler, history, config.")


if __name__ == '__main__':
    main()
