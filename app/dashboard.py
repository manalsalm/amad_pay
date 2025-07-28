
### personal_ai_saving_assistant/app/dashboard.py
import streamlit as st
import pandas as pd
from app.categorize import categorize
from app.recommender import generate_suggestions

st.title("\U0001F9E0 Personal AI Saving Assistant")

uploaded = st.file_uploader("Upload your transaction CSV", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    df['Category'] = df['Description'].apply(categorize)
    st.write("\U0001F4CA Categorized Transactions", df)

    st.bar_chart(df.groupby('Category')['Amount'].sum())

    suggestions = generate_suggestions(df)
    st.subheader("\U0001F4A1 Saving Suggestions")
    for tip in suggestions:
        st.markdown(f"- {tip}")