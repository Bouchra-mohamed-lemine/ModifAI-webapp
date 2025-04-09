import os
import re
import time
import uuid
import requests
import traceback
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from urllib.parse import urljoin
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
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
 


# Configure retailer URLs
RETAILERS = {
    "ikea": "https://www.ikea.com/us/en/search/?q="
}

# Image processing setup
weights = ResNet50_Weights.DEFAULT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = resnet50(weights=weights)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1]).eval().to(device)

def get_image_embedding(img_path):
    """Generate ResNet50 embedding for an image"""
    try:
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet_model(image).squeeze().cpu().numpy()
        return embedding.reshape(1, -1)
    except Exception as e:
        print(f"❌ Image embedding failed: {traceback.format_exc()}")
        raise

def get_color_histogram(image_path):
    """Generate color histogram features for an image"""
    try:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten().reshape(1, -1)
    except Exception as e:
        print(f"❌ Color histogram failed: {traceback.format_exc()}")
        raise

def combined_similarity_score(embed1, embed2, hist1, hist2, alpha=0.7):
    """Calculate combined similarity score"""
    resnet_sim = cosine_similarity(embed1, embed2)[0][0]
    color_sim = cosine_similarity(hist1, hist2)[0][0]
    return alpha * resnet_sim + (1 - alpha) * color_sim



def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.binary_location = os.environ.get("CHROME_BIN", "/usr/bin/chromium")

    driver = webdriver.Chrome(
        executable_path=os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver"),
        options=options
    )

    return driver



def parse_price(price_text):
    """Improved price parsing with international format support"""
    try:
        # Handle various price formats
        price_str = re.sub(r'[^\d.,]', '', price_text)
        if ',' in price_str and '.' in price_str:
            return float(price_str.replace('.', '').replace(',', '.'))
        elif ',' in price_str:
            return float(price_str.replace(',', ''))
        return float(price_str)
    except Exception as e:
        print(f"⚠️ Price parsing failed for '{price_text}': {str(e)}")
        return float('inf')

def search_products(output_dir, reference_img_path, budget, style, room_type, product_name, similarity_threshold=0.7, alpha=0.7):
    """Main scraping function with enhanced error handling"""
    driver = None
    try:
        driver = get_driver()
        print(f"🔍 Starting search for '{product_name}' (${budget} budget)")
        
        # Load reference features
        reference_embedding = get_image_embedding(reference_img_path)
        reference_hist = get_color_histogram(reference_img_path)
        
        product_candidates = []
        seen_links = set()
        query = f"{style} {room_type} {product_name}".replace(" ", "+")

        for retailer, base_url in RETAILERS.items():
            try:
                url = f"{base_url}{query}"
                print(f"🌐 Navigating to {url}")
                driver.get(url)
                
                # Wait for product container
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.plp-product-list__products"))
                
                # Simulate scrolling
                for _ in range(3):
                    driver.execute_script("window.scrollBy(0, window.innerHeight)")
                    time.sleep(1.5)

                soup = BeautifulSoup(driver.page_source, "html.parser")
                products = soup.select('div.plp-product-list__products > div.plp-fragment-wrapper')
                print(f"🏬 Found {len(products)} products on {retailer.upper()}")

                for idx, p in enumerate(products, 1):
                    try:
                        print(f"  📦 Processing product {idx}/{len(products)}")
                        
                        # Extract product details
                        name = p.select_one('span.plp-price-module__product-name').text.strip()
                        price_text = p.select_one('span.plp-price__integer').text.strip()
                        image_elem = p.select_one('img[src^="http"]')
                        link_elem = p.select_one('a.plp-product__image-link[href]')

                        if not all([name, price_text, image_elem, link_elem]):
                            print("⚠️ Missing required product elements")
                            continue

                        price = parse_price(price_text)
                        image_url = image_elem['src']
                        link = urljoin(base_url, link_elem['href'])

                        # Filtering logic
                        if link in seen_links:
                            print("⏭️ Duplicate product, skipping")
                            continue
                        if price > budget:
                            print(f"🚫 Price exceeds budget (${price:.2f})")
                            continue

                        seen_links.add(link)
                        
                        # Download product image
                        print(f"  📥 Downloading image from {image_url}")
                        try:
                            img_response = requests.get(image_url, timeout=10)
                            img_response.raise_for_status()
                        except Exception as e:
                            print(f"⚠️ Image download failed: {str(e)}")
                            continue

                        # Process image
                        temp_dir = "temp_images"
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.jpg")
                        
                        try:
                            with open(temp_path, 'wb') as f:
                                f.write(img_response.content)
                            
                            # Calculate features
                            product_embedding = get_image_embedding(temp_path)
                            product_hist = get_color_histogram(temp_path)
                            similarity = combined_similarity_score(
                                product_embedding, reference_embedding,
                                product_hist, reference_hist,
                                alpha=alpha
                            )
                            print(f"  📊 Similarity score: {similarity:.2f}")
                            
                            if similarity >= similarity_threshold:
                                product_candidates.append({
                                    'name': name,
                                    'price': price,
                                    'image': image_url,
                                    'link': link,
                                    'similarity': similarity,
                                    'img_data': img_response.content
                                })
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                    except Exception as e:
                        print(f"⚠️ Product processing error: {traceback.format_exc()}")
                        continue

            except Exception as e:
                print(f"❌ Retailer processing failed: {traceback.format_exc()}")
                continue

        # Process and save results
        print(f"🎯 Found {len(product_candidates)} qualifying products")
        top_products = sorted(product_candidates, key=lambda x: (-x['similarity'], x['price']))[:3]
        output_dict = {}

        for i, product in enumerate(top_products, 1):
            try:
                safe_name = re.sub(r'[^\w_]', '', product['name']).strip('_').replace(' ', '_')
                image_path = os.path.join(output_dir, f"{i:02d}_{safe_name}.jpg")
                meta_path = os.path.join(output_dir, f"{i:02d}_{safe_name}.txt")

                # Save image
                with open(image_path, 'wb') as f:
                    f.write(product['img_data'])
                
                # Save metadata
                with open(meta_path, 'w', encoding='utf-8') as f:
                    f.write(f"Name: {product['name']}\n")
                    f.write(f"Price: ${product['price']:.2f}\n")
                    f.write(f"Retailer: {retailer}\n")
                    f.write(f"Link: {product['link']}\n")
                    f.write(f"Similarity: {product['similarity']:.4f}\n")

                print(f"💾 Saved result {i}: {safe_name} (Score: {product['similarity']:.2f})")
                output_dict[image_path] = meta_path
            except Exception as e:
                print(f"❌ Failed to save result {i}: {traceback.format_exc()}")

        return output_dict

    except Exception as e:
        print(f"❌ Critical error in search_products: {traceback.format_exc()}")
        return {}
    finally:
        if driver:
            driver.quit()
            print("🧹 Chrome driver closed")




def ikea_scraper(product_paths, budget, style, room_type):
    """Main entry point for the scraper"""
    output_dir = os.path.abspath("static/filtered_results")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    seen_names = set()

    for img_path in product_paths:
        try:
            print(f"\n{'#' * 40}")
            print(f"Processing {os.path.basename(img_path)}")
            print(f"{'#' * 40}")
            
            product_name = os.path.basename(img_path).split('_')[0]
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

            # Filter duplicates
            filtered = {}
            for img_path, meta_path in output.items():
                name_key = os.path.basename(img_path).split('_', 1)[1].rsplit('.', 1)[0]
                if name_key not in seen_names:
                    seen_names.add(name_key)
                    filtered[img_path] = meta_path
            
            if filtered:
                results.append(filtered)
                print(f"✅ Added {len(filtered)} new products")
            else:
                print("🟡 No new products found for this image")

        except Exception as e:
            print(f"❌ Failed to process {img_path}: {traceback.format_exc()}")

    return results
