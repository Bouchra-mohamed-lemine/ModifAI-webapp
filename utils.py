import re
import requests

# Extract product information (name, price, link) from metadata text files
def extract_products(data):
    products = []

    # Iterate through each item in the data list
    for item in data:
        for image_path, txt_path in item.items():  # Each item is a dict: {image_path: metadata_path}
            name = None
            price = None
            link = None

            try:
                # Open and read the associated metadata text file
                with open(txt_path, 'r') as file:
                    for line in file:
                        # Extract product name
                        if line.startswith("Name:"):
                            name = line.split("Name:")[1].strip()
                        # Extract product price using regex to find numeric value
                        elif line.startswith("Price:"):
                            match = re.search(r'\$?([\d,.]+)', line)
                            if match:
                                price = float(match.group(1).replace(',', ''))  # Convert to float
                        # Extract product link
                        elif line.startswith("Link:"):
                            link = line.split("Link:")[1].strip()

                # Only add to list if all required fields are present
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

    return products  # Return list of extracted product info
