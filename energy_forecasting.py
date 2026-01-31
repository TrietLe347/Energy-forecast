import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM



def create_sequences(x,y,window_size=24):
    x_seq,y_seq = [],[]
    for i in range(len(x) - window_size):
        x_seq.append(x[i:i+window_size])
        y_seq.append(y[i+window_size])

    return np.array(x_seq),np.array(y_seq)


EPOCHS =20
HORIZON = 6

dataset = fetch_ucirepo(id=235)

df = dataset.data.features.copy()



df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],format='%d/%m/%Y %H:%M:%S', dayfirst=True)
df = df.drop(columns = ['Date', 'Time'])
df = df.set_index('Datetime')



df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

df = df.resample('h').mean()
df = df.dropna()


features = [
    'Global_active_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3'
]

df = df[features]

df['Target'] = df['Global_active_power'].shift(-HORIZON)
df = df.dropna()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

df_scaled = pd.DataFrame(scaled, columns=df.columns, index=df.index)

x = df_scaled.drop(columns=['Target']).values
y = df_scaled['Target'].values


# MLP data (NO sequences)
X_mlp = df_scaled.drop(columns=['Target']).values
y_mlp = df_scaled['Target'].values

split_mlp = int(0.8 * len(X_mlp))
X_mlp_train, X_mlp_test = X_mlp[:split_mlp], X_mlp[split_mlp:]
y_mlp_train, y_mlp_test = y_mlp[:split_mlp], y_mlp[split_mlp:]

X_lstm, y_lstm = create_sequences(X_mlp, y_mlp, window_size=24)

split_lstm = int(0.8 * len(X_lstm))
X_lstm_train, X_lstm_test = X_lstm[:split_lstm], X_lstm[split_lstm:]
y_lstm_train, y_lstm_test = y_lstm[:split_lstm], y_lstm[split_lstm:]

print(X_lstm_train.shape)  # (samples, 24, 6)


#normal model
mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_mlp_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

mlp_model.compile(optimizer='adam', loss='mse')


#lstm_model

lstm_model = Sequential([
    LSTM(64,return_sequences = True, input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])),
    LSTM(32),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')


history = mlp_model.fit(
    X_mlp_train,
    y_mlp_train,
    epochs=EPOCHS,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
history_lstm = lstm_model.fit(
    X_lstm_train,
    y_lstm_train,
    epochs=EPOCHS,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

predictions = mlp_model.predict(X_mlp_test)
mse = mean_squared_error(y_mlp_test,predictions)
mae = mean_absolute_error(y_mlp_test,predictions)

naive_predictions = X_mlp_test[:,0] #Global_active_power (Scaled)

naive_mse = mean_squared_error(y_mlp_test,naive_predictions)
naive_mae = mean_absolute_error(y_mlp_test,naive_predictions)


lstm_predictions = lstm_model.predict(X_lstm_test)

lstm_mse = mean_squared_error(y_lstm_test, lstm_predictions)
lstm_mae = mean_absolute_error(y_lstm_test, lstm_predictions)

print("MODEL COMPARISON")
print("----------------")
print(f"Naive MSE: {naive_mse:.6f} | MAE: {naive_mae:.6f}")
print(f"MLP   MSE: {mse:.6f} | MAE: {mae:.6f}")
print(f"LSTM  MSE: {lstm_mse:.6f} | MAE: {lstm_mae:.6f}")



# Flatten predictions
mlp_preds = predictions.flatten()
lstm_preds = lstm_predictions.flatten()

# Align lengths (LSTM starts later due to windowing)
min_len = min(len(y_mlp_test), len(lstm_preds))

y_plot = y_mlp_test[:min_len]
mlp_plot = mlp_preds[:min_len]
naive_plot = naive_predictions[:min_len]
lstm_plot = lstm_preds[:min_len]


plt.figure(figsize=(12, 5))

plt.plot(y_plot[:200], label='Actual', linewidth=2)
plt.plot(naive_plot[:200], label='Naive', linestyle='--')
plt.plot(mlp_plot[:200], label='MLP')
plt.plot(lstm_plot[:200], label='LSTM')

plt.title('6-Hour Ahead Energy Consumption Forecast (Model Comparison)')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Power Consumption')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/experiment4.png", dpi=150)
plt.show()

plt.figure(figsize=(12, 4))

plt.plot(np.abs(y_plot - naive_plot)[:200], label='Naive Error', linestyle='--')
plt.plot(np.abs(y_plot - mlp_plot)[:200], label='MLP Error')
plt.plot(np.abs(y_plot - lstm_plot)[:200], label='LSTM Error')

plt.title('Absolute Prediction Error (6-Hour Horizon)')
plt.xlabel('Time Steps')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/experiment4-error.png", dpi=150)
plt.show()

