import requests
from bs4 import BeautifulSoup
import json

import requests
from bs4 import BeautifulSoup



def parse_products(json_data):
    """
    Parses the product JSON data and returns a cleaned list of products
    
    Args:
        json_data: The JSON product data (list of product dictionaries)
        
    Returns:
        list: Cleaned list of product dictionaries with key details
    """
    product_list = []
    
    for product in json_data:
        # Extract basic product info
        product_info = {
            'name': product.get('name'),
            'image': "",
            'original_price':"",
            'discount':"",
            'price_after':""
        }
        
        # Parse variants
        for variant in product.get('variants', []):
            '''''
            variant_info = {
                'variant_id': variant.get('id'),
                'name': variant.get('name'),
                'full_name': variant.get('fullName'),
                'image': variant.get('images', [None])[0] if variant.get('images') else None,
                'pricing': [],
                
            }
            '''
            product_info['image'] = variant.get('images', [None])[0] if variant.get('images')[0] else None,
            # Parse pricing for each store
            for store_data in variant.get('storeSpecificData', []):
                '''''
                pricing = {
                    'store_id': store_data.get('storeId'),
                    'mrp': store_data.get('mrp'),
                    'discount': store_data.get('discount'),
                    'price_after_discount': float(store_data.get('mrp', 0)) - float(store_data.get('discount', 0)),
                    'stock': store_data.get('stock'),
                    'unit': store_data.get('unit'),
                    'tax': store_data.get('tax', {}).get('VAT') if store_data.get('tax') else None
                }
                '''
                product_info['original_price'] = store_data.get('mrp')
                product_info['discount'] = store_data.get('discount')
                product_info['price_after'] = float(store_data.get('mrp', 0)) - float(store_data.get('discount', 0))
                #variant_info['pricing'].append(pricing)
            
            #product_info['variants'].append(variant_info)
        product_info["image"] = product_info["image"][0]
        product_list.append(product_info)
    
    return product_list

def get_script_tags_from_url(url):
    """
    Fetches HTML from a URL and extracts all script tags.
    
    Args:
        url (str): The target URL to fetch.
    
    Returns:
        list: A list of dictionaries containing:
            - 'src' (str or None): The `src` attribute (for external scripts)
            - 'content' (str): The script's inner content (for inline scripts)
    """
    try:
        # Default headers (similar to Postman)
        default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        headers=None
        # Merge user headers with defaults (user headers take priority)
        final_headers = {**default_headers, **(headers or {})}
        # 1. Fetch HTML
        response = requests.get(url, headers=final_headers)
        response.raise_for_status()  # Raise error if request fails

        # 2. Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        script_tags = soup.find_all('script')
        
        # 3. Extract script details
        scripts = []
        for script in script_tags:
            src = script.get('src')
            content = script.string or ''.join(str(c) for c in script.contents)
            if content !='' and content.startswith('{') :
                scripts.append({
                    'content': content.strip() if content else ''
                })

        return scripts
    
    except requests.RequestException as e:
        print(f"HTTP Request Error: {e}")
        return []
    except Exception as e:
        print(f"Parsing Error: {e}")
        return []

def extract_product_data(scripts):
    """AI is creating summary for extract_product_data

    Args:
        scripts (str): list of html script tags content

    Returns:
        [type]: [description]
    """
    parsed_products = []

    for script in scripts:
        
        content = script["content"]
        content = json.loads(content )

        product_data = content["props"]["pageProps"]["layouts"]["data"]["page"]["layouts"][0]["value"]["collection"]["product"]

        parsed_products = parse_products(product_data)
        print(json.dumps(parsed_products, indent=2))

                # Print first product details as example
        if parsed_products:
            print("First product:")
            print(json.dumps(parsed_products[0], indent=2))
            print(parsed_products[0]["image"][0])
                                    
        with open("offers/tamimi-product.json", "w", encoding="utf-8") as f:
            json.dump(parsed_products, f, indent=2, ensure_ascii=False)        
    
    return parsed_products

def get_tamimi_supermarket_offer():
    url = "https://shop.tamimimarkets.com/tag/weekly-offers-1"
    scripts = get_script_tags_from_url(url)
    products = extract_product_data(scripts)
    return products
    