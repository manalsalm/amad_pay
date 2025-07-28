import pandas as pd
import numpy as np
from random import choice, randint, uniform
from datetime import datetime, timedelta

def generate_low_transaction_data(
        num_transactions=100, 
        start_date='2025-01-01', 
        end_date='2025-06-30',
        output_csv='consistent_transactions.csv'
):

    categories = ['Dining', 'Subscription', 'Transport', 'Groceries', 'Shopping']
    descriptions = {
        'Dining': ['Starbucks', 'McDonald\'s', 'Pizza Hut'],
        'Subscription': ['Netflix', 'Spotify', 'Amazon Prime'],
        'Transport': ['Uber', 'Bus Ticket', 'Taxi'],
        'Groceries': ['Walmart', 'Costco', 'Whole Foods'],
        'Shopping': ['Amazon', 'eBay', 'Local Store']
    }
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    delta_days = (end_dt - start_dt).days

    data = []
    for _ in range(num_transactions):
        random_days = randint(0, delta_days)
        date = start_dt + timedelta(days=random_days)
        category = choice(categories)
        description = choice(descriptions[category])
        amount = round(np.random.uniform(1, 50), 2)  # Random amount between $10 and $200
        
        data.append({
            'Date': date,
            'Description': description,
            'Amount': amount,
            'Category': category
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    return df

def generate_concise_transaction_data(
    start_date='2024-07-01',
    end_date='2025-07-01',
    num_transactions=250,
    output_csv='consistent_transactions.csv'
):
    categories = ['Groceries', 'Dining', 'Shopping', 'Subscription', 'Transport']
    descriptions = {
        'Groceries': ['Walmart', 'Costco', 'Whole Foods'],
        'Dining': ['Starbucks', 'Pizza Hut', 'McDonald\'s'],
        'Shopping': ['Amazon', 'eBay', 'Local Store'],
        'Subscription': ['Netflix', 'Spotify', 'Amazon Prime'],
        'Transport': ['Taxi', 'Uber', 'Bus Ticket']
    }

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    data = []

    for _ in range(num_transactions):
        category = choice(categories)
        description = choice(descriptions[category])
        amount = round(uniform(1, 10), 2)
        date = start_dt + timedelta(days=randint(0, (end_dt - start_dt).days))
        data.append([date.strftime('%Y-%m-%d'), description, amount, category])

    df = pd.DataFrame(data, columns=['Date', 'Description', 'Amount', 'Category'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    return df