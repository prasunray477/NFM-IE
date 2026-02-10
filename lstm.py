#LSTM Model (for comparison)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from load_dataset import load_data

def lstm_forecast(
    ticker,
    start_date,
    end_date,
    look_back=60,
    epochs=50
):
    # Reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    data = load_data(ticker, start_date, end_date)
    if data is None or data.empty:
        return None

    prices = data['Close'].values.reshape(-1, 1)

    # Train-test split
    split = int(len(prices) * 0.8)
    train, test = prices[:split], prices[split - look_back:]

    # Scale ONLY on training data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    # Create sequences
    def create_sequences(series):
        X, y = [], []
        for i in range(look_back, len(series)):
            X.append(series[i - look_back:i, 0])
            y.append(series[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled)
    X_train = X_train.reshape(X_train.shape[0], look_back, 1)

    # Model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Recursive multi-step forecast
    last_seq = test_scaled[-look_back:]
    predictions = []

    for _ in range(30):
        pred = model.predict(
            last_seq.reshape(1, look_back, 1),
            verbose=0
        )[0, 0]
        predictions.append(pred)
        last_seq = np.vstack([last_seq[1:], [[pred]]])

    return scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()



# EVALUATION METRICS
def calculate_metrics(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    return {
        'RMSE': np.sqrt(mse),
        'MAE': np.mean(np.abs(actual - predicted)),
        'MAPE': np.mean(np.abs((actual - predicted) / actual)) * 100,
        'Theil_U': np.sqrt(mse) / (np.sqrt(np.mean(predicted ** 2)) + np.sqrt(np.mean(actual ** 2)))
    }