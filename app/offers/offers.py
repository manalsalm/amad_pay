import json
from datetime import datetime

def get_matching_offers(purchases, offers, top_n=3):
    today = datetime.today().date()
    products = []
    matches = []
    for p in purchases:
        for item in p["items"]:
            pid = item["name"]
            if int(item["quantity"]) > 2:
                products.append(pid)
    for offer in offers:
        if offer["name"] in products:
            matches.append(offer)
    return matches

def get_offers(offers):
    offerList = []
    for offer in offers:
        offerList.append(offer)
    return offerList