import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('top7.csv')
close_prices = data['Close'].values.reshape(-1, 1)  # Reshape for scaler

# Feature Engineering: Create a feature matrix 'X' using lagged prices
X = np.hstack([close_prices[:-i] for i in range(1, 6)])  # Using last 5 days as features
y = close_prices[5:]  # Target variable, shifted by 5 days

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM Regression Model
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_rbf.fit(X_train_scaled, y_train.ravel())  # Fit model

# Predictions
y_pred = svr_rbf.predict(X_test_scaled)

# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual Prices')
plt.plot(y_pred, label='Predicted Prices', color='red')
plt.title('Support Vector Regression Prediction vs Actual Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
