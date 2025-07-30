import json
from datetime import datetime

with open("purchases.json") as f:
    purchases = json.load(f)

with open("offers.json") as f:
    offers = json.load(f)    

def get_matching_offers(purchases, offers, top_n=3):
    today = datetime.today().date()
    matches = []
    items = []
    for p in purchases:
        pid = p["product_id"]
        if p["quantity"] > 2:
            items.append(pid)
    for offer in offers:
        if offer["product_id"] in items and datetime.strptime(offer["valid_until"], "%Y-%m-%d").date() >= today:
            matches.append(offer)
    return matches

def get_offers():
    offerList = []
    for offer in offers:
        offerList.append(offer)
    return offerList

if __name__ == "__main__":
    matched_offers = get_matching_offers(purchases, offers)
    general_offers  = get_offers()
    print(f"\nMatched Offers: {matched_offers}")
    print(f"\nGeneral Offers: {general_offers}")
    user_input = input("\nYou: ")
