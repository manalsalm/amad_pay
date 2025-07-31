import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')
from app.categorize import categories, categorize_text
### personal_ai_saving_assistant/main.py
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from app.categorize import categorize
#from app.forecast import forecast_category , check_saving_target1, forecast_all_categories , forecast_all_categories1 , check_saving_target , check_saving_target_yearly , forecast_with_arima, sarimax_forecast, arima_forecast, lstm_forecast, load_and_prepare_data , create_generator, train_lstm, forecast_future, plot_forecast
from app.recommender import generate_suggestions
from app.generate_data import generate_low_transaction_data , generate_concise_transaction_data
import matplotlib.pyplot as plt
from app.forecasting.forecast_lstm import lstm_forecast
from app.forecasting.forecast_prophet import prophet_forecast_all_categories
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# English categories with Arabic keywords
categories = {
    "Evaluation & Audit": ["تقييم", "مراجعة", "اكتوارية", "حساب ختامي"],
    "Supply & Procurement": ["توريد", "تأمين", "قطع غيار", "لوحات", "بنرات", "مطبوعات", "فرش", "تجهيزات", "دهانات", "بشت"],
    "Construction & Development": ["تنفيذ", "سور", "معمل", "إنشاء", "فحص", "نظام", "صبة خرسانية"],
    "Security Services": ["حراسات", "أمنية", "فحص الحاويات"],
    "Maintenance": ["صيانة", "برمجة", "فحص", "إصلاح"],
    "Events & Catering": ["مهرجان", "خيام", "مسارح", "وجبة"],
    "Fuel & Energy": ["وقود", "ديزل", "بنزين", "محطة وقود", "تغذية"],
    "Consulting Services": ["استشارية", "دراسة", "تقييم"],
    "Logistics & Operations": ["نقل", "تركيب", "تشغيل", "فحص"],
    "Other": []  # Default category
}
'''''
df1 = pd.read_excel('Tender/Tenders Data by  jan 2025.xlsx',  usecols='A,D,E,N')
#print(df.head())

df1.to_csv("Tender/Tenders Data by  jan 2025.csv", index=False, encoding="utf-8-sig")

# Load your data
df1 = pd.read_csv("Tender/Tenders Data by  jan 2025.csv")

df1.columns = ['Name', 'Date', 'Description', 'Amount']

df2 = pd.read_excel('Tender/Tenders Data by Feb 2025.xlsx',  usecols='A,D,E,N')
#print(df.head())

df2.to_csv("Tender/Tenders Data by Feb 2025.csv", index=False, encoding="utf-8-sig")

# Load your data
df2 = pd.read_csv("Tender/Tenders Data by Feb 2025.csv")

df2.columns = ['Name', 'Date', 'Description', 'Amount']

df3 = pd.read_excel('Tender/Tenders Data by March 2025.xlsx',  usecols='A,D,E,N')
#print(df.head())

df3.to_csv("Tender/Tenders Data by March 2025.csv", index=False, encoding="utf-8-sig")

# Load your data
df4 = pd.read_csv("Tender/Tenders Data by March 2025.csv")

df4 = pd.read_excel('Tender/Tenders Data by  jun 2025.xlsx',  usecols='A,D,E,N')
#print(df.head())

df4.to_csv("Tender/Tenders Data by  jun 2025.csv", index=False, encoding="utf-8-sig")

# Load your data
df4 = pd.read_csv("Tender/Tenders Data by  jun 2025.csv")
'''''

df4 = pd.read_excel('Tender/Tenders Data by Sector 2022.xlsx',  usecols='B,C,G,H')
#print(df.head())

df4.to_csv("Tender/Tenders Data by Sector 2022.csv", index=False, encoding="utf-8-sig")

# Load your data
df4 = pd.read_csv("Tender/Tenders Data by Sector 2022.csv")

df4.columns = ['Name', 'Description', 'Date', 'Amount']

df5 = pd.read_excel('Tender/Tenders Data by Sector 2023.xlsx',  usecols='B,C,G,H')
#print(df.head())

df5.to_csv("Tender/Tenders Data by Sector 2023.csv", index=False, encoding="utf-8-sig")

# Load your data
df5 = pd.read_csv("Tender/Tenders Data by Sector 2023.csv")

df5.columns = ['Name', 'Description', 'Date', 'Amount']

df6 = pd.read_excel('Tender/Tenders 2020.xlsx',  usecols='B,C,G,H')
#print(df.head())

df6.to_csv("Tender/Tenders 2020.csv", index=False, encoding="utf-8-sig")

# Load your data
df6 = pd.read_csv("Tender/Tenders 2020.csv")

df6.columns = ['Name', 'Description', 'Date', 'Amount']

df7 = pd.read_excel('Tender/Tenders 2021.xlsx',  usecols='B,C,G,H')
#print(df.head())

df7.to_csv("Tender/Tenders 2021.csv", index=False, encoding="utf-8-sig")

# Load your data
df7 = pd.read_csv("Tender/Tenders 2021.csv")

df7.columns = ['Name', 'Description', 'Date', 'Amount']

frames = [df4,df6,df7]

df = pd.concat(frames)
#df.to_csv("Tender/Tenders Data by Sector 2020 - 2021 - 2022 - 2023.csv", index=False, encoding="utf-8-sig")

print(df.head())

df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

# Apply categorization
if 'Description' in df.columns:
    df['Category'] = df['Description'].apply(categorize_text)

#df.to_csv("Tender/Tenders Data by Sector Categories 2020 - 2021 - 2022 - 2023.csv", index=False, encoding="utf-8-sig")

#print(df.head())

daily_spend = df.groupby("Date")["Amount"].sum().reset_index()


# 3) Group by Name then Date
daily_spend = (
    df.groupby(["Name", "Date"], as_index=False)["Amount"]
      .sum()
      .sort_values(["Name", "Date"])
)

print(daily_spend.head())
spend1 = daily_spend[daily_spend["Name"] == "الهيئة السعودية للبيانات والذكاء الاصطناعي"]# "الهيئة السعودية للبيانات والذكاء الاصطناعي"] #"وزارة التعليم"]

# Prepare the dataframe for Prophet
spend = spend1.rename(columns={"Date": "ds", "Amount": "y"})
#result = check_saving_target1(spend,monthly_income=100000000,saving_target=20000,forecast_period=30)
#print(result)
full_range = pd.date_range(start=spend.index.min(), end=spend.index.max())
#daily_sum = spend.reindex(full_range, fill_value=0)
#daily_sum.index.name = 'Date'

# Prophet expects positive values, so ensure no negatives
spend = spend[spend["y"] > 0]

# Initialize and fit Prophet
model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
model.fit(spend)

# Create a future dataframe
future = model.make_future_dataframe(periods=30*12)  # Forecast 30 days into the future

# Make predictions
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("Forecast of Daily Spending")
plt.xlabel("Date")
plt.ylabel("Spending")
plt.tight_layout()
plt.show()


prophet_forecast_all_categories(df, periods=30)

lstm_forecast(spend1)