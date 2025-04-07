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






def upload_to_imgbb(image_path, api_key="72dffb425d527280be597176e0e22dab"):
    with open(image_path, "rb") as file:
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": api_key,
        }
        files = {
            "image": file,
        }
        response = requests.post(url, data=payload, files=files)
        if response.status_code == 200:
            print(response.json()["data"]["url"])
            return response.json()["data"]["url"]
        else:
            print("❌ Upload failed:", response.text)
            return None

 