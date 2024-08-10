import yfinance as yf
import pandas as pd

# Define the ticker symbols for the top seven IT companies
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "BABA", "TCEHY"]

# Create an empty DataFrame to store concatenated data
combined_data = pd.DataFrame()

for ticker in tickers:
    # Fetch historical data for each ticker
    data = yf.Ticker(ticker).history(period="1y")  # Adjust period as needed
    data['Ticker'] = ticker  # Add a column to identify the stock in the combined data
    combined_data = pd.concat([combined_data, data])

# Reset index to make the date a column and better organize the combined data
combined_data.reset_index(inplace=True)

# Save the combined data to a CSV file
combined_data.to_csv('top_seven_IT_companies_stock_data.csv', index=False)

print("Data downloaded and saved to top_seven_IT_companies_stock_data.csv")
