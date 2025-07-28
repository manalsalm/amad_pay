def generate_suggestions(df):
    tips = []
    cat_totals = df.groupby('Category')['Amount'].sum()
    total_spending = df['Amount'].sum()

    # Calculate percentages
    cat_percent = cat_totals / total_spending * 100

    # Heuristic tips using percentage thresholds
    if cat_percent.get('Dining', 0) > 20:
        tips.append("You're spending over 20% on Dining. Try reducing takeout or cooking at home.")

    if cat_percent.get('Subscription', 0) > 10:
        tips.append("Subscriptions take up over 10% of your budget – review and cancel unused ones.")

    if cat_percent.get('Shopping', 0) > 15:
        tips.append("Shopping is over 15% of your expenses. Consider prioritizing needs over wants.")

    if df['Amount'].max() > 0.2 * total_spending:
        tips.append("You have a large transaction that’s over 20% of your total spending. Make sure it was essential.")

    if cat_percent.get('Groceries', 0) > 30:
        tips.append("Groceries make up over 30% of your expenses – consider using coupons or buying in bulk.")

    if cat_percent.get('Transport', 0) > 15:
        tips.append("Transport exceeds 15% of your budget. Carpooling or public transit might help save money.")

    if cat_percent.get('Entertainment', 0) > 10:
        tips.append("Entertainment is over 10% of your spending. Explore free or low-cost alternatives.")

    # Identify category with highest spending
    highest_category = cat_totals.idxmax()

    # Top 3 highest expenses in that category
    top_expenses = df[df['Category'] == highest_category].sort_values(by='Amount', ascending=False).head(3)

    tips.append(f"Your highest spending category is '{highest_category}'. Here are your top expenses there:")

    for _, row in top_expenses.iterrows():
        tips.append(f"  - '{row['Description']}' cost you ${row['Amount']:.2f}. Consider if it was necessary or can be avoided.")

    return tips
