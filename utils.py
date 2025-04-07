import re
import requests

import re

def extract_products(data):
    products = []

    for item in data:
        for image_path, txt_path in item.items():
            name = None
            price = None
            link = None

            try:
                with open(txt_path, 'r') as file:
                    for line in file:
                        if line.startswith("Name:"):
                            name = line.split("Name:")[1].strip()
                        elif line.startswith("Price:"):
                            match = re.search(r'\$?([\d,.]+)', line)
                            if match:
                                price = float(match.group(1).replace(',', ''))
                        elif line.startswith("Link:"):
                            link = line.split("Link:")[1].strip()

                if name and price is not None and link:
                    products.append({
                        'image': image_path,
                        'Name': name,
                        'Price': price,
                        'Link': link
                    })
            except FileNotFoundError:
                print(f"Warning: File not found: {txt_path}")
            except Exception as e:
                print(f"Error processing {txt_path}: {e}")

    return products


 
