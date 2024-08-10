import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('top7.csv')
close_prices = data['Close'].values

# Feature Engineering: Create feature matrix 'X' using lagged prices
X = np.array([close_prices[i-10:i] for i in range(10, len(close_prices))])
y = close_prices[10:]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(10, 1)))
model.add(LSTM(50))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1)

# Predictions
y_pred = model.predict(X_test_scaled).flatten()

# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual Prices')
plt.plot(y_pred, label='Predicted Prices', color='red')
plt.title('LSTM Prediction vs Actual Prices')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
