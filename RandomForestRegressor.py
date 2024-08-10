import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('top7.csv')
close_prices = data['Close'].values

# Feature Engineering: Create feature matrix 'X' using lagged prices
X = np.array([close_prices[i-5:i] for i in range(5, len(close_prices))])
y = close_prices[5:]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Predictions
y_pred = random_forest.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual Prices')
plt.plot(y_pred, label='Predicted Prices', color='red')
plt.title('Random Forest Regression Prediction vs Actual Prices')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
