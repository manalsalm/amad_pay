import os
import pandas as pd
import requests

folder = "C:/Users/Nora-Basalamah/Documents/Amad Pay/amad_pay/data"
all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
dfs = []

for file in all_files:
    df = pd.read_csv(file)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

def get_sample_data(df, max_rows=436580):
    return df.head(max_rows).to_string(index=False)

history = ""
income = 20000

while True:
    query = input("\nYou: ")
    if query.lower() == 'quit':
        break
    try:
        sample = get_sample_data(data)

        full_prompt = (
            "You are a financial assistant analyzing the following tabular spending data:\n\n"
            f"{sample}\n\n"
            f"Income is:\n{income}\n\n"
            f"Conversation so far:\n{history}\n\n"
            f"User's next question:\n{query}"
            f"Give budgeting tips based on spending patterns.\n\n"
        )

        # full_prompt = (
        #     "[INST] You are a helpful financial assistant.\n\n"
        #     "Here is a table of the user's recent financial transactions:\n"
        #     f"{sample}\n\n"
        #     "Conversation so far:\n"
        #     f"{history}\n\n"
        #     "Now, answer the following user question clearly and concisely based on the data provided:\n"
        #     f"{query}\n\n"
        #     "[/INST]"
        # )

        # payload = {
        #     "model": "deepseek-llm",
        #     "prompt": full_prompt,
        #     "stream": False
        # }

        payload = {
            "model": "llama3",         
            "prompt": full_prompt,
            "stream": False
        }
        

        response = requests.post("http://localhost:11434/api/generate", json=payload)
        result = response.json()["response"]

        print("\nAther:", result)

        history += f"\nUser: {query}\nAssistant: {result}\n"

    except Exception as e:
        print("\nError:", e)

