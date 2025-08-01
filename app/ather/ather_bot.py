import os
import pandas as pd
import requests
from app.forecasting.forecast_prophet import prophet_forecast_all_categories
from app.recommender import generate_suggestions

def get_sample_data(df, max_rows=436580):
    return df.head(max_rows).to_string(index=False)

def ask_llama3(query, history):
    folder = os.path.join(os.path.dirname(__file__), "..", "..", "data", "consistent_transactions_data_2025-07-26 14-38-16.csv")
    folder = os.path.abspath(folder)
    # all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    dfs = []

    # for file in all_files:
    #     df = pd.read_csv(file)
    #     if 'Date' in df.columns:
    #         df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    #     dfs.append(df)
    data = pd.read_csv(folder)
    # data = pd.concat(dfs, ignore_index=True)
    # data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
    income = 2000

    forcast = prophet_forecast_all_categories(data)
    sample = get_sample_data(data)
    recom =  generate_suggestions(data)   
    
    try:
        full_prompt = (
            "You are a financial assistant analyzing the following tabular spending data and represent it in the decision:\n\n"
            f"{sample}\n\n"
            f"Use this as monthly income is:\n{income}\n\n"
            f"Use this forcasting result for the decision:\n{forcast}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"User's next question:\n{query}"
            f"Give budgeting tips based on spending patterns and use also this recommendation for the tips :\n{recom}"
            # "IMPORTANT: Your entire response must be valid JSON. Use ONLY the following format:\n"
            # "{\n"
            # '  "response": "your advice here"\n'
            # "}\n"
            # "Do not include any extra text outside the JSON. No explanations, no markdown, no headers — just JSON."
        )

        payload = {
            "model": "llama3",         
            "prompt": full_prompt,
            "stream": False
        }
        

        response = requests.post("http://localhost:11434/api/generate", json=payload)
        result = response.json()["response"]
        history += f"\nUser: {query}\nAssistant: {result}\n"
        return (result, history)

    except Exception as e:
        print("\nError:", e)

