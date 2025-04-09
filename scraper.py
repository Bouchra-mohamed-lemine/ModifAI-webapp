# import os
# import time
# import requests
# import numpy as np
# from PIL import Image
# from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from sklearn.metrics.pairwise import cosine_similarity
# import torch
# import cv2
# from torchvision import transforms
# from torchvision.models import resnet50, ResNet50_Weights
# import shutil

# # Retailer Search URLs
# RETAILERS = {
#     "ikea": "https://www.ikea.com/us/en/search/?q="
# }

# # Image transform and model setup
# weights = ResNet50_Weights.DEFAULT
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# resnet_model = resnet50(weights=weights)
# resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # Remove classifier
# resnet_model.eval().to(device)

# # Embedding and color histogram functions
# def get_image_embedding(img_path):
#     image = Image.open(img_path).convert("RGB")
#     image = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = resnet_model(image).squeeze().cpu().numpy()
#     return embedding.reshape(1, -1)

# def get_color_histogram(image_path):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (224, 224))
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
#                         [0, 180, 0, 256, 0, 256])
#     return cv2.normalize(hist, hist).flatten().reshape(1, -1)

# def combined_similarity_score(embed1, embed2, hist1, hist2, alpha=0.7):
#     resnet_sim = cosine_similarity(embed1, embed2)[0][0]
#     color_sim = cosine_similarity(hist1, hist2)[0][0]
#     return alpha * resnet_sim + (1 - alpha) * color_sim

# # Web driver setup
# def get_driver():
#     options = Options()
#     options.add_argument("--headless")
#     options.add_argument("--disable-gpu")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--window-size=1920,1080")
#     return webdriver.Chrome(service=Service(), options=options)

# def parse_price(price_text):
#     try:
#         return float(''.join(filter(lambda x: x.isdigit() or x == '.', price_text)))
#     except:
#         return float('inf')

# # Main product search function
# def search_products(output_dir, reference_img_path, budget, style, room_type, product_name, similarity_threshold=0.7, alpha=0.7):
    
#     driver = get_driver()
#     try:
#         reference_embedding = get_image_embedding(reference_img_path)
#         reference_hist = get_color_histogram(reference_img_path)
#     except Exception as err:
#         print(f"[ERROR] Could not load reference image: {err}")
#         return []

#     query = f"{style} {room_type} {product_name}".replace(" ", "+")
#     product_candidates = []

#     output_dict = {}

#     for retailer, base_url in RETAILERS.items():
#         try:
#             url = f"{base_url}{query}"
#             driver.get(url)
#             WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div")))
#             driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
#             time.sleep(2)

#             soup = BeautifulSoup(driver.page_source, "html.parser")
#             print(f"Scraping {retailer}...")

#             products = soup.select('div.plp-product-list__products > div.plp-fragment-wrapper')
#             for p in products:
#                 try:
#                     name = p.select_one('span.plp-price-module__product-name').text.strip()
#                     price_text = p.select_one('span.plp-price__integer').text.strip()
#                     price = parse_price(price_text)
#                     image = p.select_one('img')['src']
#                     link = p.select_one('a')['href']

#                     if price > budget or not image.startswith("http"):
#                         continue

#                     img_data = requests.get(image).content
#                     os.makedirs("temp_images", exist_ok=True)
#                     temp_path = f"temp_images/{hash(image)}.jpg"
#                     with open(temp_path, 'wb') as handler:
#                         handler.write(img_data)

#                     ikea_embedding = get_image_embedding(temp_path)
#                     image_hist = get_color_histogram(temp_path)
#                     similarity = combined_similarity_score(
#                         ikea_embedding, reference_embedding,
#                         image_hist, reference_hist,
#                         alpha=alpha
#                     )

#                     if similarity >= similarity_threshold:
#                         product_candidates.append({
#                             "name": name,
#                             "price": price,
#                             "image": image,
#                             "link": link,
#                             "similarity": similarity,
#                             "img_data": img_data
#                         })

#                 except Exception as e:
#                     print(f"IKEA parsing error: {str(e)}")
#                     continue
#         except Exception as e:
#             print(f"Error processing {retailer}: {str(e)}")
#             continue

#     driver.quit()

#     top_products = sorted(product_candidates, key=lambda x: x['similarity'], reverse=True)[:3]

#     for i, product in enumerate(top_products, 1):
#         safe_name = "".join(c for c in product['name'] if c.isalnum() or c in (' ', '_')).rstrip().replace(" ", "_")
#         image_path = f"{output_dir}/{i:02d}_{safe_name}.jpg"
#         meta_path = f"{output_dir}/{i:02d}_{safe_name}.txt"

#         with open(image_path, 'wb') as img_file:
#             img_file.write(product['img_data'])

#         with open(meta_path, 'w', encoding='utf-8') as meta_file:
#             meta_file.write(f"Name: {product['name']}\n")
#             meta_file.write(f"Price: ${product['price']:.2f}\n")
#             meta_file.write(f"Retailer: ikea\n")
#             meta_file.write(f"Link: {product['link']}\n")
#             meta_file.write(f"Similarity: {product['similarity']:.4f}\n")

#         print(f"Saved Top-{i}: {safe_name} (Similarity: {product['similarity']:.2f})")

    
#         output_dict[image_path] = meta_path

#     return output_dict






# def ikea_scraper(product_paths, budget, style, room_type):

#     # Remove the filtered_results folder if it exists
#     output_dir = "static/filtered_results"
#     if os.path.exists(output_dir) and os.path.isdir(output_dir):
#         shutil.rmtree(output_dir)

#     os.makedirs(output_dir, exist_ok=True)
    
#     products_paths = []

#     for img_path in product_paths:
#         output = search_products(
#             output_dir=output_dir,
#             reference_img_path=img_path,
#             budget=int(budget),
#             style=style,
#             room_type=room_type,
#             product_name=img_path.split("/")[-1].split(".")[0].split("_")[0],
#             similarity_threshold=0.6,  # we tweak this for more/less results
#             alpha=0.4  # we tweak this to weigh shape vs color
#         )

#         products_paths.append(output)

#     return products_paths




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

# Retailer Search URLs
RETAILERS = {
    "ikea": "https://www.ikea.com/us/en/search/?q="
}

# Image transform and model setup
weights = ResNet50_Weights.DEFAULT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = resnet50(weights=weights)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # Remove classifier
resnet_model.eval().to(device)

def get_image_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet_model(image).squeeze().cpu().numpy()
    return embedding.reshape(1, -1)

def get_color_histogram(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten().reshape(1, -1)

def combined_similarity_score(embed1, embed2, hist1, hist2, alpha=0.7):
    resnet_sim = cosine_similarity(embed1, embed2)[0][0]
    color_sim = cosine_similarity(hist1, hist2)[0][0]
    return alpha * resnet_sim + (1 - alpha) * color_sim

def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(service=Service(), options=options)

def parse_price(price_text):
    try:
        return float(''.join(filter(lambda x: x.isdigit() or x == '.', price_text)))
    except:
        return float('inf')

def search_products(output_dir, reference_img_path, budget, style, room_type, product_name, similarity_threshold=0.7, alpha=0.7):
    
    driver = get_driver()
    try:
        reference_embedding = get_image_embedding(reference_img_path)
        reference_hist = get_color_histogram(reference_img_path)
    except Exception as err:
        print(f"[ERROR] Could not load reference image: {err}")
        return []

    query = f"{style} {room_type} {product_name}".replace(" ", "+")
    product_candidates = []

    seen_links = set()  # NEW: track seen products

    for retailer, base_url in RETAILERS.items():
        try:
            url = f"{base_url}{query}"
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div")))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
            time.sleep(2)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            print(f"Scraping {retailer}...")

            products = soup.select('div.plp-product-list__products > div.plp-fragment-wrapper')
            for p in products:
                try:
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

                    img_data = requests.get(image).content
                    os.makedirs("temp_images", exist_ok=True)
                    temp_path = f"temp_images/{hash(image)}.jpg"
                    with open(temp_path, 'wb') as handler:
                        handler.write(img_data)

                    ikea_embedding = get_image_embedding(temp_path)
                    image_hist = get_color_histogram(temp_path)
                    similarity = combined_similarity_score(
                        ikea_embedding, reference_embedding,
                        image_hist, reference_hist,
                        alpha=alpha
                    )

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

    top_products = sorted(product_candidates, key=lambda x: x['similarity'], reverse=True)[:3]
    output_dict = {}

    for i, product in enumerate(top_products, 1):
        safe_name = "".join(c for c in product['name'] if c.isalnum() or c in (' ', '_')).rstrip().replace(" ", "_")
        image_path = f"{output_dir}/{i:02d}_{safe_name}.jpg"
        meta_path = f"{output_dir}/{i:02d}_{safe_name}.txt"

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





def ikea_scraper(product_paths, budget, style, room_type):
    output_dir = "static/filtered_results"
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    products_paths = []
    seen_names = set()  # <- NEW: to track seen product names (e.g., "PARADISISK")

    for img_path in product_paths:
        product_name = img_path.split("/")[-1].split(".")[0].split("_")[0]
        print("******************", product_name)

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

        # Filter out any already seen products
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

 
