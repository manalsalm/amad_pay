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
        for item in p["items"]:
            pid = item["name"]
            if int(item["quantity"]) > 2:
                items.append(pid)
    for offer in offers:
        if offer["name"] in items:
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