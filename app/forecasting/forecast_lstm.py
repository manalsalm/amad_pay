
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

def csv_lstm_forecast(csv_file, look_back=30, forecast_days=30, epochs=20,
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


def lstm_forecast(df, look_back=30, forecast_days=30, epochs=20,
                  save_csv='lstm_forecast.csv', save_plot='lstm_forecast.png'):
    # Load and prepare data
    #df = pd.read_csv(csv_file, parse_dates=['Date'])
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
# lstm_forecast(df, look_back=30, forecast_days=30, epochs=20)
