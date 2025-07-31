### personal_ai_saving_assistant/app/forecast.py
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def forecast_category(csv_file, category, periods=7):
    df = pd.read_csv(csv_file, parse_dates=["Date"])

    # Ensure Amount is clean numeric
    df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Check for NaNs (missing or bad entries)
    print(df['Amount'].isna().sum(), "bad amount entries found")
    df_cat = df[df['Category'] == category]
    df_cat = df_cat.groupby('Date')['Amount'].sum().reset_index()
    df_cat.columns = ['ds', 'y']

    if len(df_cat) < 2:
        # Not enough data to forecast
        return None
    
    model = Prophet()
    model.fit(df_cat)

    future = model.make_future_dataframe(periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def forecast_category1(df, category, periods=7):

    # Ensure Amount is clean numeric
    df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Check for NaNs (missing or bad entries)
    print(df['Amount'].isna().sum(), "bad amount entries found")
    df_cat = df[df['Category'] == category]
    df_cat = df_cat.groupby('Date')['Amount'].sum().reset_index()
    df_cat.columns = ['ds', 'y']

    if len(df_cat) < 2:
        # Not enough data to forecast
        return None
    
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_cat)

    future = model.make_future_dataframe(periods)
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]



def forecast_all_categories(csv_file, periods=7):
    df = pd.read_csv(csv_file, parse_dates=["Date"])

    # Ensure Amount is clean numeric
    df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    categories = df['Category'].unique()
    all_forecasts = {}
    
    for category in categories:
        forecast_df = forecast_category(csv_file, category, periods)
        if forecast_df is not None:
            all_forecasts[category] = forecast_df
        else:
            print(f"Not enough data to forecast for category: {category}")
    
    # Print forecast results for all categories
    for category, forecast_df in all_forecasts.items():
        print(f"--- Forecast for category: {category} ---")
        for _, row in forecast_df.iterrows():
            date = row['ds'].date()
            predicted = row['yhat']
            lower = row['yhat_lower']
            upper = row['yhat_upper']
            print(f"Date: {date}")
            print(f"  Predicted spending: ${predicted:.2f}")
            print(f"  Likely range: RS{lower:.2f} to ${upper:.2f}")
            print(f"  Explanation: You are expected to spend around RS{predicted:.2f} on {category} on {date}.")
            print(f"               The spending could be as low as RS{lower:.2f} or as high as ${upper:.2f}.")
            print()


def forecast_all_categories1(df, periods=7):

    # Ensure Amount is clean numeric
    df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    categories = df['Category'].unique()
    all_forecasts = {}
    
    for category in categories:
        forecast_df = forecast_category1(df, category, periods)
        if forecast_df is not None:
            all_forecasts[category] = forecast_df
        else:
            print(f"Not enough data to forecast for category: {category}")
    
    # Print forecast results for all categories
    for category, forecast_df in all_forecasts.items():
        print(f"--- Forecast for category: {category} ---")
        for _, row in forecast_df.iterrows():
            date = row['ds'].date()
            predicted = row['yhat']
            lower = row['yhat_lower']
            upper = row['yhat_upper']
            print(f"Date: {date}")
            print(f"  Predicted spending: RS{predicted:.2f}")
            print(f"  Likely range: RS{lower:.2f} to ${upper:.2f}")
            print(f"  Explanation: You are expected to spend around RS{predicted:.2f} on {category} on {date}.")
            print(f"               The spending could be as low as RS{lower:.2f} or as high as ${upper:.2f}.")
            print()


def check_saving_target(csv_file, monthly_income, saving_target, forecast_period=30):
    """
    Args:
        df: DataFrame with past transactions (with 'Date', 'Amount', 'Category').
        monthly_income: User's monthly income as float.
        saving_target: Desired saving amount per month as float.
        forecast_period: Days ahead to forecast (default 30).

    Returns:
        A string suggestion or alert about saving target status.
    """
    df = pd.read_csv(csv_file, parse_dates=["Date"])

    # Ensure Amount is clean numeric
    df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Import inside function if needed
    from prophet import Prophet

    # Aggregate daily total spending
    df_daily = df.groupby('Date')['Amount'].sum().reset_index()
    df_daily.columns = ['ds', 'y']

    # Clean data
    df_daily = df_daily.dropna(subset=['y'])
    if len(df_daily) < 2:
        return "Not enough data to forecast your spending."

    # Fit Prophet model
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_daily)

    # Forecast future spending for forecast_period days
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)

    # Sum predicted spending over forecast period (next month)
    predicted_spending = forecast['yhat'].tail(forecast_period).sum()

    # Calculate budget left after savings
    budget_after_savings = monthly_income - saving_target

    # Check if predicted spending exceeds budget after savings
    if predicted_spending > budget_after_savings:
        excess = predicted_spending - budget_after_savings
        return (f"Warning: Your predicted spending (RS{predicted_spending:.2f}) exceeds your "
                f"budget after savings (RS{budget_after_savings:.2f}) by RS{excess:.2f}. "
                "Consider reducing expenses to meet your saving goal.")
    else:
        spare = budget_after_savings - predicted_spending
        return (f"Good job! Your predicted spending (${predicted_spending:.2f}) is within your budget "
                f"after savings (${budget_after_savings:.2f}). You have around ${spare:.2f} spare for extra expenses.")


def check_saving_target1(df_daily, monthly_income, saving_target, forecast_period=30):
    """
    Args:
        df: DataFrame with past transactions (with 'Date', 'Amount', 'Category').
        monthly_income: User's monthly income as float.
        saving_target: Desired saving amount per month as float.
        forecast_period: Days ahead to forecast (default 30).

    Returns:
        A string suggestion or alert about saving target status.
    """
    #df = pd.read_csv(csv_file, parse_dates=["Date"])

    # Ensure Amount is clean numeric
    #df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
    #df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Import inside function if needed
    from prophet import Prophet

    # Aggregate daily total spending
    #df_daily = df.groupby('Date')['Amount'].sum().reset_index()
    #df_daily.columns = ['ds', 'y']

    # Clean data
    df_daily = df_daily.dropna(subset=['y'])
    if len(df_daily) < 2:
        return "Not enough data to forecast your spending."
    
    # Fit Prophet model
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_daily)

    # Forecast future spending for forecast_period days
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)

    # Sum predicted spending over forecast period (next month)
    predicted_spending = forecast['yhat'].tail(forecast_period).sum()

    # Calculate budget left after savings
    budget_after_savings = monthly_income - saving_target
    
    # Check if predicted spending exceeds budget after savings
    if predicted_spending > budget_after_savings:
        excess = predicted_spending - budget_after_savings
        
        return (f"Warning: Your predicted spending (RS{predicted_spending:.2f}) exceeds your "
                f"budget after savings (RS{budget_after_savings:.2f}) by RS{excess:.2f}. "
                "Consider reducing expenses to meet your saving goal.")
    else:
        spare = budget_after_savings - predicted_spending
        return (f"Good job! Your predicted spending (RS{predicted_spending:.2f}) is within your budget "
                f"after savings (RS{budget_after_savings:.2f}). You have around RS{spare:.2f} spare for extra expenses.")


# Example usage:
# message = check_saving_target(df, monthly_income=5000, saving_target=1000, forecast_period=30)
# print(message)


def check_saving_target_yearly(csv_file, monthly_income, saving_target, forecast_days=365):
    df = pd.read_csv(csv_file, parse_dates=["Date"])

    # Ensure Amount is clean numeric
    df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    from prophet import Prophet

    generation_date = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

    if saving_target * 12 > monthly_income * 12:
        return ("Error: Your annual saving target exceeds your annual income. "
                "Please lower your saving target or increase your income.")

    # Aggregate daily total spending
    df_daily = df.groupby('Date')['Amount'].sum().reset_index()
    df_daily.columns = ['ds', 'y']
    df_daily = df_daily.dropna(subset=['y'])
    if len(df_daily) < 2:
        return "Not enough data to forecast your spending."

    # Fit Prophet model
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_daily)

    # Forecast future spending for next 365 days
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Identify spikes: values above 1.5x standard deviation
    threshold = forecast['yhat'].mean() + 1.5 * forecast['yhat'].std()
    spikes = forecast[forecast['yhat'] > threshold]

    # Plot with spikes
    fig = model.plot(forecast)
    plt.title("📈 Monthly Spending Forecast with Spikes")
    plt.xlabel("Month")
    plt.ylabel("Spending")

    # Highlight spikes in red
    plt.scatter(spikes['ds'], spikes['yhat'], color='red', label='Spike', zorder=3)
    plt.legend()
    plt.tight_layout()
    # 🔽 Save the plot as PNG
    fig.savefig(f"forcast_result/images/forecast_yearly_with_spikes_{generation_date}.png", dpi=300)
    plt.show()
    
    # Optional: Save results
    forecast.to_csv(f"forcast_result/csv/forecast_yearly_with_spikes_{generation_date}.csv", index=False)
    # Sum predicted spending over the year
    predicted_spending = forecast['yhat'].tail(forecast_days).sum()
    predicted_spending = forecast['yhat'].tail(forecast_days).sum()

    # Calculate budget left after savings (annualized)
    annual_income = monthly_income * 12
    annual_saving_target = saving_target * 12
    budget_after_savings = annual_income - annual_saving_target

    if predicted_spending > budget_after_savings:
        excess = predicted_spending - budget_after_savings
        return (f"Warning: Your predicted annual spending (${predicted_spending:.2f}) "
                f"exceeds your budget after savings (${budget_after_savings:.2f}) by ${excess:.2f}. "
                "Consider reducing expenses to meet your saving goal.")
    else:
        spare = budget_after_savings - predicted_spending
        return (f"Good job! Your predicted annual spending (${predicted_spending:.2f}) "
                f"is within your budget after savings (${budget_after_savings:.2f}). "
                f"You have around ${spare:.2f} spare for extra expenses.")


def check_saving_target_yearly1(df, monthly_income, saving_target, forecast_days=365):

    # Ensure Amount is clean numeric
    df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    from prophet import Prophet

    generation_date = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

    if saving_target * 12 > monthly_income * 12:
        return ("Error: Your annual saving target exceeds your annual income. "
                "Please lower your saving target or increase your income.")

    # Aggregate daily total spending
    df_daily = df.groupby('Date')['Amount'].sum().reset_index()
    df_daily.columns = ['ds', 'y']
    df_daily = df_daily.dropna(subset=['y'])
    if len(df_daily) < 2:
        return "Not enough data to forecast your spending."

    # Fit Prophet model
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_daily)

    # Forecast future spending for next 365 days
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Identify spikes: values above 1.5x standard deviation
    threshold = forecast['yhat'].mean() + 1.5 * forecast['yhat'].std()
    spikes = forecast[forecast['yhat'] > threshold]

    # Plot with spikes
    fig = model.plot(forecast)
    plt.title("📈 Monthly Spending Forecast with Spikes")
    plt.xlabel("Month")
    plt.ylabel("Spending")

    # Highlight spikes in red
    plt.scatter(spikes['ds'], spikes['yhat'], color='red', label='Spike', zorder=3)
    plt.legend()
    plt.tight_layout()
    # 🔽 Save the plot as PNG
    fig.savefig(f"forcast_result/images/forecast_yearly_with_spikes_{generation_date}.png", dpi=300)
    plt.show()
    
    # Optional: Save results
    forecast.to_csv(f"forcast_result/csv/forecast_yearly_with_spikes_{generation_date}.csv", index=False)
    # Sum predicted spending over the year
    predicted_spending = forecast['yhat'].tail(forecast_days).sum()
    predicted_spending = forecast['yhat'].tail(forecast_days).sum()

    # Calculate budget left after savings (annualized)
    annual_income = monthly_income * 12
    annual_saving_target = saving_target * 12
    budget_after_savings = annual_income - annual_saving_target

    if predicted_spending > budget_after_savings:
        excess = predicted_spending - budget_after_savings
        return (f"Warning: Your predicted annual spending (${predicted_spending:.2f}) "
                f"exceeds your budget after savings (${budget_after_savings:.2f}) by ${excess:.2f}. "
                "Consider reducing expenses to meet your saving goal.")
    else:
        spare = budget_after_savings - predicted_spending
        return (f"Good job! Your predicted annual spending (${predicted_spending:.2f}) "
                f"is within your budget after savings (${budget_after_savings:.2f}). "
                f"You have around ${spare:.2f} spare for extra expenses.")


# Example:
# message = check_saving_target_yearly(df, monthly_income=5000, saving_target=1000)
# print(message)

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

def forecast_with_arima(csv_file='consistent_transactions.csv', forecast_days=300, plot_file='arima_forecast.png'):
    # Load and prepare data
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Sum amount per day
    daily_sum = df.groupby('Date')['Amount'].sum().reset_index()

    # Set date as index and reindex for continuity
    daily_sum.set_index('Date', inplace=True)
    full_range = pd.date_range(start=daily_sum.index.min(), end=daily_sum.index.max())
    daily_sum = daily_sum.reindex(full_range, fill_value=0.0)
    daily_sum.index.name = 'Date'

    # Fit ARIMA model (auto-order can be used with pmdarima for tuning)
    model = ARIMA(daily_sum['Amount'], order=(1, 1, 1))  # ARIMA(p,d,q)
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=forecast_days)

    # Build forecast index
    forecast_index = pd.date_range(start=daily_sum.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sum.index, daily_sum['Amount'], label='Actual Spending')
    plt.plot(forecast_index, forecast, label='Forecasted Spending', color='orange')
    plt.title('Daily Spending Forecast (ARIMA)')
    plt.xlabel('Date')
    plt.ylabel('Amount ($)')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('arima_spending_forecast.png')
    plt.show()

    # Save forecast to CSV
    forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecasted_Amount': forecast})
    forecast_df.to_csv('forcast_result/csv/forecasted_spending_arima.csv', index=False)
    print("Forecast saved to forecasted_spending_arima.csv and plot saved as arima_spending_forecast.png")

    return forecast_df

def sarimax_forecast(csv_file='transactions.csv', forecast_days=30, save_csv='arima_forecast.csv', save_plot='arima_forecast.png'):
    # Load and prepare data
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)

    # Aggregate daily spending
    daily_sum = df.groupby('Date')['Amount'].sum()

    # Fill missing dates with 0
    full_range = pd.date_range(start=daily_sum.index.min(), end=daily_sum.index.max())
    daily_sum = daily_sum.reindex(full_range, fill_value=0)
    daily_sum.index.name = 'Date'

    # Fit ARIMA model (tune order as needed)
    #model = ARIMA(daily_sum, order=(2, 1, 2))
    #model_fit = model.fit()
    model_fit = sm.tsa.statespace.SARIMAX(daily_sum, order=(1, 1, 1), 
                                      seasonal_order=(1, 1, 1, 7)).fit(disp=False)
    # Forecast future spending
    forecast = model_fit.forecast(steps=forecast_days)
    forecast_index = pd.date_range(start=daily_sum.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sum.index, daily_sum, label='Historical Spending')
    plt.plot(forecast_index, forecast, label='Forecast', color='orange')
    plt.title('Daily Spending Forecast (ARIMA)')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_plot)
    plt.show()

    # Save forecast to CSV
    forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecasted_Amount': forecast})
    forecast_df.to_csv(save_csv, index=False)
    print("Forecast saved to arima_forecast.csv and plot saved to arima_forecast.png")

    return forecast_df

def arima_forecast(csv_file='transactions.csv', forecast_days=30, save_csv='arima_forecast.csv', save_plot='arima_forecast.png'):
    # Load and prepare data
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)

    # Aggregate daily spending
    daily_sum = df.groupby('Date')['Amount'].sum()

    # Fill missing dates with 0
    full_range = pd.date_range(start=daily_sum.index.min(), end=daily_sum.index.max())
    daily_sum = daily_sum.reindex(full_range, fill_value=0)
    daily_sum.index.name = 'Date'

    # Fit ARIMA model (tune order as needed)
    model = ARIMA(daily_sum, order=(2, 1, 2))
    model_fit = model.fit()

    # Forecast future spending
    forecast = model_fit.forecast(steps=forecast_days)
    forecast_index = pd.date_range(start=daily_sum.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sum.index, daily_sum, label='Historical Spending')
    plt.plot(forecast_index, forecast, label='Forecast', color='orange')
    plt.title('Daily Spending Forecast (ARIMA)')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_plot)
    plt.show()

    # Save forecast to CSV
    forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecasted_Amount': forecast})
    forecast_df.to_csv(save_csv, index=False)
    print("Forecast saved to arima_forecast.csv and plot saved to arima_forecast.png")

    return forecast_df

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Load and prepare transaction data
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df.dropna(subset=['Amount'])

    # Aggregate daily totals
    daily_total = df.groupby('Date').sum().reset_index()

    # Fill missing dates
    all_days = pd.date_range(daily_total['Date'].min(), daily_total['Date'].max())
    daily_total = daily_total.set_index('Date').reindex(all_days, fill_value=0).rename_axis("Date").reset_index()
    return daily_total

# Step 2: Normalize and create time series generator
def create_generator(data, look_back=30):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['Amount']])
    generator = TimeseriesGenerator(scaled, scaled, length=look_back, batch_size=16)
    return generator, scaler, scaled

# Step 3: Build and train LSTM model
def train_lstm(generator, look_back):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=20, verbose=1)
    return model

# Step 4: Forecast future days
def forecast_future(model, scaled_data, scaler, look_back, future_days=30):
    input_seq = scaled_data[-look_back:].reshape(1, look_back, 1)
    predictions = []

    for _ in range(future_days):
        next_val = model.predict(input_seq, verbose=0)[0]
        predictions.append(next_val)
        input_seq = np.append(input_seq[:, 1:, :], [[next_val]], axis=1)

    return scaler.inverse_transform(predictions)

def save_forecast_to_csv(forecast_values, last_date, output_csv="lstm_forecast.csv"):
    forecast_days = len(forecast_values)
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Amount': forecast_values.flatten()
    })
    
    forecast_df.to_csv(output_csv, index=False)
    print(f"Forecast saved to: {output_csv}")

# Step 5: Plot and save forecast
def plot_forecast(daily_total, forecast_values, output_path="lstm_forecast.png"):
    forecast_days = len(forecast_values)
    future_dates = pd.date_range(daily_total['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    save_forecast_to_csv(forecast_values, daily_total['Date'].max())
    plt.figure(figsize=(12, 6))
    plt.plot(daily_total['Date'], daily_total['Amount'], label="Historical")
    plt.plot(future_dates, forecast_values, label="Forecast (LSTM)", color='orange')
    plt.title("Daily Spending Forecast with LSTM")
    plt.xlabel("Date")
    plt.ylabel("Amount")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import Sequential

def lstm_forecast(csv_file, look_back=30, forecast_days=30, epochs=20,
                  save_csv='lstm_forecast.csv', save_plot='lstm_forecast.png'):
    # Load and prepare data
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    daily_sum = df.groupby('Date')['Amount'].sum()
    full_dates = pd.date_range(daily_sum.index.min(), daily_sum.index.max())
    daily_sum = daily_sum.reindex(full_dates, fill_value=0)
    daily_sum.index.name = 'Date'

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_sum.values.reshape(-1, 1))

    # Create time series generator
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=look_back, batch_size=16)

    # Build LSTM model
    #model = Sequential([
    #    LSTM(50, activation='relu', input_shape=(look_back, 1)),
    #    Dense(1)
    #])

    params = {
	"loss": "mean_squared_error",
	"optimizer": "adam",
	"dropout": 0.2,
	"lstm_units": 90,
	"epochs": 30,
	"batch_size": 128,
	"es_patience" : 10
}
    
    model = Sequential()
    
    model.add(LSTM(units=90, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(rate=params["dropout"]))
    model.add(LSTM(units=params["lstm_units"], return_sequences=True))
    model.add(Dropout(rate=params["dropout"]))
    model.add(LSTM(units=params["lstm_units"], return_sequences=True))
    model.add(Dropout(rate=params["dropout"]))
    model.add(LSTM(units=params["lstm_units"], return_sequences=False))
    model.add(Dropout(rate=params["dropout"]))
    model.add(Dense(1))

    model.compile(loss=params["loss"],
              	optimizer=params["optimizer"],
              	metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    # Train model
    model.fit(generator, epochs=epochs, verbose=1)

    # Forecast future values
    input_seq = scaled_data[-look_back:]
    predictions = []
    for _ in range(forecast_days):
        input_reshaped = input_seq.reshape((1, look_back, 1))
        pred = model.predict(input_reshaped, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq[1:], pred)

    # Inverse scale predictions
    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Prepare forecast dates
    future_dates = pd.date_range(daily_sum.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

    # Save forecast CSV
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Amount': forecast})
    forecast_df.to_csv(save_csv, index=False)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sum.index, daily_sum.values, label='Historical Spending')
    plt.plot(future_dates, forecast, label='LSTM Forecast', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.title('Daily Spending Forecast (LSTM)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_plot)
    plt.show()

    print(f"Forecast saved to '{save_csv}' and plot saved to '{save_plot}'")

# Example usage:
# lstm_forecast('transactions.csv', look_back=30, forecast_days=30, epochs=20)
