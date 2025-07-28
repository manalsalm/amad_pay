### personal_ai_saving_assistant/app/categorize.py
def categorize(description):
    desc = description.lower()
    if any(keyword in desc for keyword in ['netflix', 'spotify', 'hulu', 'disney+', 'amazon prime']):
        return 'Subscription'
    elif any(keyword in desc for keyword in ['uber', 'lyft', 'taxi', 'transport', 'bus', 'metro']):
        return 'Transport'
    elif any(keyword in desc for keyword in ['starbucks', 'mcdonald', 'kfc', 'burger king', 'pizza hut', 'restaurant', 'dine']):
        return 'Dining'
    elif any(keyword in desc for keyword in ['walmart', 'costco', 'grocery', 'supermarket', 'whole foods', 'aldi']):
        return 'Groceries'
    elif any(keyword in desc for keyword in ['amazon', 'shein', 'fatfetch', 'onaas']):
        return 'Shopping'
    elif any(keyword in desc for keyword in ['cinema', 'movie', 'theatre', 'concert', 'game', 'net cafe']):
        return 'Entertainment'
    elif any(keyword in desc for keyword in ['gym', 'fitness', 'yoga', 'workout']):
        return 'Health'
    elif any(keyword in desc for keyword in ['pharmacy', 'clinic', 'hospital', 'medic']):
        return 'Medical'
    elif any(keyword in desc for keyword in ['school', 'college', 'university', 'tuition', 'course']):
        return 'Education'
    elif any(keyword in desc for keyword in ['insurance', 'policy', 'premium']):
        return 'Insurance'
    elif any(keyword in desc for keyword in ['electric', 'gas', 'water', 'utility', 'bill']):
        return 'Utilities'
    return 'Other'