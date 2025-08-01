### personal_ai_saving_assistant/app/forecast.py
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from app.Module.ForcastCategories import ForecastCategories

def prophet_forecast_category(df, category, periods=7):

    # Ensure Amount is clean numeric
    df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Check for NaNs (missing or bad entries)
    #print(df['Amount'].isna().sum(), "bad amount entries found")
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



def prophet_forecast_all_categories(df, periods=7):

    categories_list = []
    # Ensure Amount is clean numeric
    df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    categories = df['Category'].unique()
    all_forecasts = {}
    
    for category in categories:
        forecast_df = prophet_forecast_category(df, category, periods)
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
            #print(f"Date: {date}")
            #print(f"  Predicted spending: RS{predicted:.2f}")
            #print(f"  Likely range: RS{lower:.2f} to ${upper:.2f}")
            #print(f"  Explanation: You are expected to spend around RS{predicted:.2f} on {category} on {date}.")
            #print(f"               The spending could be as low as RS{lower:.2f} or as high as RS{upper:.2f}.")
            #print()
            categories_list.append(ForecastCategories(category,date,predicted,lower,upper) )

    return categories_list

def prophet_check_saving_target(df, monthly_income, saving_target, forecast_period=30):
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
    df = df.dropna(subset=['y'])
    if len(df) < 2:
        return "Not enough data to forecast your spending."
    
    # Fit Prophet model
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)

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


def prophet_check_saving_target_yearly(df, monthly_income, saving_target, forecast_days=365):

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

    # Calculate budget left after savings (annualized)
    annual_income = monthly_income * 12
    annual_saving_target = saving_target * 12
    budget_after_savings = annual_income - annual_saving_target

    if predicted_spending > budget_after_savings:
        excess = predicted_spending - budget_after_savings
        return (f"Warning: Your predicted annual spending (RS{predicted_spending:.2f}) "
                f"exceeds your budget after savings (RS{budget_after_savings:.2f}) by RS{excess:.2f}. "
                "Consider reducing expenses to meet your saving goal.")
    else:
        spare = budget_after_savings - predicted_spending
        return (f"Good job! Your predicted annual spending (RS{predicted_spending:.2f}) "
                f"is within your budget after savings (RS{budget_after_savings:.2f}). "
                f"You have around RS{spare:.2f} spare for extra expenses.")

# Example:
# message = check_saving_target_yearly(df, monthly_income=5000, saving_target=1000)
# print(message)

