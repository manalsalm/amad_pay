### personal_ai_saving_assistant/main.py
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from app.categorize import categorize
from app.forecast import forecast_category , forecast_all_categories , check_saving_target , check_saving_target_yearly , forecast_with_arima, sarimax_forecast, arima_forecast, lstm_forecast, load_and_prepare_data , create_generator, train_lstm, forecast_future, plot_forecast
from app.recommender import generate_suggestions
from app.generate_data import generate_low_transaction_data , generate_concise_transaction_data
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


filename = "data/consistent_transactions_data_2025-07-26 14-38-16.csv"
# Generate data
generation_date = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

# Save to CSV
#csv_filename = f"data/low_transaction_data_{generation_date}.csv"
csv_cfilename =f"data/consistent_transactions_data_{generation_date}.csv"

#low_data_df = generate_low_transaction_data(num_transactions=10000, start_date='2024-01-01',end_date='2025-07-25',output_csv=csv_filename)
#generation_date = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
low_data_df = generate_concise_transaction_data(num_transactions=10000, start_date='2024-01-01',end_date='2025-07-25',output_csv=csv_cfilename)

# Save to CSV
#csv_filename = f"data/low_transaction_data_{generation_date}.csv"
#low_data_df.to_csv(csv_filename, index=False)

print(f"Low transaction data saved to {csv_cfilename}")

# Load your data
df = pd.read_csv(filename)
df['Category'] = df['Description'].apply(categorize)

#forecast = forecast_with_arima(csv_cfilename, forecast_days=300)
#print(forecast)
sarimax_forecast(csv_file=filename,forecast_days=300,save_csv=f"forcast_result/csv/arima_forcast_{generation_date}.csv",save_plot=f"forcast_result/images/arima_forcast_{generation_date}.png")
# Read and clean CSV
df = pd.read_csv(filename, parse_dates=["Date"])

# Ensure Amount is clean numeric
df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

# Check for NaNs (missing or bad entries)
print(df['Amount'].isna().sum(), "bad amount entries found")

# Drop NaNs
#df = df.dropna(subset=['Amount'])

#print(df)

# Aggregate daily total spending
daily_spend = df.groupby("Date")["Amount"].sum().reset_index()

# Prepare the dataframe for Prophet
daily_spend = daily_spend.rename(columns={"Date": "ds", "Amount": "y"})

full_range = pd.date_range(start=daily_spend.index.min(), end=daily_spend.index.max())
daily_sum = daily_spend.reindex(full_range, fill_value=0)
#daily_sum.index.name = 'Date'

# Prophet expects positive values, so ensure no negatives
daily_spend = daily_spend[daily_spend["y"] > 0]

# Initialize and fit Prophet
model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
model.fit(daily_spend)

# Create a future dataframe
future = model.make_future_dataframe(periods=30)  # Forecast 30 days into the future

# Make predictions
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("Forecast of Daily Spending")
plt.xlabel("Date")
plt.ylabel("Spending")
plt.tight_layout()
plt.show()

'''''
look_back_days = 300
forecast_days = 30

# Load and process data
daily_data = load_and_prepare_data(csv_cfilename)

# Prepare sequence and model
generator, scaler, scaled_data = create_generator(daily_data, look_back=look_back_days)
model = train_lstm(generator, look_back_days)
# Forecast future
forecast_vals = forecast_future(model, scaled_data, scaler, look_back_days, future_days=forecast_days)

# Plot results
plot_forecast(daily_data, forecast_vals)
'''
lstm_forecast(filename, look_back=200, forecast_days=30, epochs=20,save_csv=f"forcast_result/csv/lstm_forcast_{generation_date}.csv",save_plot=f"forcast_result/images/lstm_forcast_{generation_date}.png")

# Forecast Dining 
forecast = forecast_category(filename, 'Groceries',periods=30*5)
#print(forecast.tail())

#df['Date'] = pd.to_datetime(df['Date'])
forecast_all_categories(filename, periods=30*5)

message = check_saving_target_yearly(filename, monthly_income=22000, saving_target=15000)
print(message)

# Print saving suggestions
suggestions = generate_suggestions(df)
print("\n\U0001F4A1 Money Saving Tips:")
for tip in suggestions:
    print("-", tip)

'''''
forecast_df = forecast_category(df, category="Groceries")  # forecast next 7 days

for _, row in forecast_df.tail(7).iterrows():  # show last 7 days (future)
    date = row['ds'].date()
    predicted = row['yhat']
    lower = row['yhat_lower']
    upper = row['yhat_upper']
    print(f"Date: {date}")
    print(f"  Predicted spending: ${predicted:.2f}")
    print(f"  Likely range: ${lower:.2f} to ${upper:.2f}")
    print(f"  Explanation: You are expected to spend around ${predicted:.2f} on Dining on {date}.")
    print(f"               The spending could be as low as ${lower:.2f} or as high as ${upper:.2f}.")
    print()
'''
