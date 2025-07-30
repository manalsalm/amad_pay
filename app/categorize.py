import re
from collections import defaultdict


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

def categories(expense):
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
    """Categorize an expense based on keywords in its description"""
    expense_lower = expense.lower()
    for category, keywords in categories.items():
        if category == "Other":
            continue
        for keyword in keywords:
            if re.search(rf"\b{keyword.lower()}\b", expense_lower):
                return category
    return "Other"
import pandas as pd

def categorize_text(text):
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
    """Categorize Arabic text based on keywords"""
    if pd.isna(text) or not isinstance(text, str):
        return "Other"
    
    text_lower = text.lower()
    for category, keywords in categories.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword.lower())}\b", text_lower):
                return category
    return "Other"