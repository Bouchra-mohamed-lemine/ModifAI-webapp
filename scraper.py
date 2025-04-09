# Import standard libraries
import os
import time
import requests
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sklearn.metrics.pairwise import cosine_similarity
import torch
import cv2
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import shutil

# Set retailer search URLs (currently only IKEA)
RETAILERS = {
    "ikea": "https://www.ikea.com/us/en/search/?q="
}

# Load pretrained ResNet50 model for feature extraction
weights = ResNet50_Weights.DEFAULT
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match model input
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
resnet_model = resnet50(weights=weights)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # Remove classifier layer
resnet_model.eval().to(device)  # Set model to eval mode and move to device

# Extract image embedding using ResNet50
def get_image_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet_model(image).squeeze().cpu().numpy()
    return embedding.reshape(1, -1)

# Calculate color histogram in HSV space
def get_color_histogram(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten().reshape(1, -1)

# Combine cosine similarity scores of ResNet embeddings and color histograms
def combined_similarity_score(embed1, embed2, hist1, hist2, alpha=0.7):
    resnet_sim = cosine_similarity(embed1, embed2)[0][0]
    color_sim = cosine_similarity(hist1, hist2)[0][0]
    return alpha * resnet_sim + (1 - alpha) * color_sim

# Setup and return a headless Selenium Chrome driver
def get_driver():
    options = Options()
    options.add_argument("--headless")  # Run Chrome in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.binary_location = os.environ.get("CHROME_BIN", "/usr/bin/chromium")
    service = Service(executable_path=os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver"))
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# Parse float price from text, returns inf on failure
def parse_price(price_text):
    try:
        return float(''.join(filter(lambda x: x.isdigit() or x == '.', price_text)))
    except:
        return float('inf')

# Main scraping logic for finding similar products
def search_products(output_dir, reference_img_path, budget, style, room_type, product_name, similarity_threshold=0.7, alpha=0.7):
    driver = get_driver()
    try:
        # Extract features from the reference image
        reference_embedding = get_image_embedding(reference_img_path)
        reference_hist = get_color_histogram(reference_img_path)
    except Exception as err:
        print(f"[ERROR] Could not load reference image: {err}")
        return []

    # Format search query
    query = f"{style} {room_type} {product_name}".replace(" ", "+")
    product_candidates = []
    seen_links = set()  # Avoid duplicates

    # Iterate through all retailers (currently only IKEA)
    for retailer, base_url in RETAILERS.items():
        try:
            url = f"{base_url}{query}"
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div")))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
            time.sleep(2)

            # Parse page HTML using BeautifulSoup
            soup = BeautifulSoup(driver.page_source, "html.parser")
            print(f"Scraping {retailer}...")

            products = soup.select('div.plp-product-list__products > div.plp-fragment-wrapper')
            for p in products:
                try:
                    # Extract product details
                    name = p.select_one('span.plp-price-module__product-name').text.strip()
                    price_text = p.select_one('span.plp-price__integer').text.strip()
                    price = parse_price(price_text)
                    image = p.select_one('img')['src']
                    link = p.select_one('a')['href']

                    if link in seen_links:
                        continue
                    if price > budget or not image.startswith("http"):
                        continue

                    seen_links.add(link)

                    # Download and save image
                    img_data = requests.get(image).content
                    os.makedirs("temp_images", exist_ok=True)
                    temp_path = f"temp_images/{hash(image)}.jpg"
                    with open(temp_path, 'wb') as handler:
                        handler.write(img_data)

                    # Extract features from product image
                    ikea_embedding = get_image_embedding(temp_path)
                    image_hist = get_color_histogram(temp_path)

                    # Compute similarity score
                    similarity = combined_similarity_score(
                        ikea_embedding, reference_embedding,
                        image_hist, reference_hist,
                        alpha=alpha
                    )

                    # Store product if similarity threshold is met
                    if similarity >= similarity_threshold:
                        product_candidates.append({
                            "name": name,
                            "price": price,
                            "image": image,
                            "link": link,
                            "similarity": similarity,
                            "img_data": img_data
                        })

                except Exception as e:
                    print(f"IKEA parsing error: {str(e)}")
                    continue
        except Exception as e:
            print(f"Error processing {retailer}: {str(e)}")
            continue

    driver.quit()

    # Keep top 3 most similar products
    top_products = sorted(product_candidates, key=lambda x: x['similarity'], reverse=True)[:3]
    output_dict = {}

    for i, product in enumerate(top_products, 1):
        # Generate safe filenames
        safe_name = "".join(c for c in product['name'] if c.isalnum() or c in (' ', '_')).rstrip().replace(" ", "_")
        image_path = f"{output_dir}/{i:02d}_{safe_name}.jpg"
        meta_path = f"{output_dir}/{i:02d}_{safe_name}.txt"

        # Save image and metadata
        with open(image_path, 'wb') as img_file:
            img_file.write(product['img_data'])

        with open(meta_path, 'w', encoding='utf-8') as meta_file:
            meta_file.write(f"Name: {product['name']}\n")
            meta_file.write(f"Price: ${product['price']:.2f}\n")
            meta_file.write(f"Retailer: ikea\n")
            meta_file.write(f"Link: {product['link']}\n")
            meta_file.write(f"Similarity: {product['similarity']:.4f}\n")

        print(f"Saved Top-{i}: {safe_name} (Similarity: {product['similarity']:.2f})")
        output_dict[image_path] = meta_path

    return output_dict

# Wrapper function for batch processing multiple product reference images
def ikea_scraper(product_paths, budget, style, room_type):
    output_dir = "static/filtered_results"

    # Clear previous results
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    products_paths = []
    seen_names = set()  # Track product names to avoid duplicates

    # Iterate through each reference image
    for img_path in product_paths:
        # Extract product name from filename
        product_name = img_path.split("/")[-1].split(".")[0].split("_")[0]
        print("****************** Scraper looking for ", product_name)

        # Search and filter results for current product
        output = search_products(
            output_dir=output_dir,
            reference_img_path=img_path,
            budget=int(budget),
            style=style,
            room_type=room_type,
            product_name=product_name,
            similarity_threshold=0.6,
            alpha=0.4
        )

        # Filter out duplicates by name
        filtered_output = {}
        for img_path, meta_path in output.items():
            base_name = os.path.basename(img_path)
            name_key = "_".join(base_name.split("_")[1:]).replace(".jpg", "")
            if name_key not in seen_names:
                seen_names.add(name_key)
                filtered_output[img_path] = meta_path

        if filtered_output:
            products_paths.append(filtered_output)

    return products_paths
