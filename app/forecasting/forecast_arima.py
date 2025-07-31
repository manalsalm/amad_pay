from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import pandas as pd

def forecast_with_arima(df, forecast_days=300, plot_file='arima_forecast.png'):
    # Load and prepare data
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

def sarimax_forecast(df, forecast_days=30, save_csv='arima_forecast.csv', save_plot='arima_forecast.png'):
    # Load and prepare data
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

def arima_forecast(df, forecast_days=30, save_csv='arima_forecast.csv', save_plot='arima_forecast.png'):
    # Load and prepare data
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