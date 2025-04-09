from flask import Flask, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
from image_generator import furnish_image    
import requests
from yolo import extract_items               
from scraper import ikea_scraper             
from utils import extract_products           
import json

app = Flask(__name__)

# Configuration: set and create folder for uploaded images
upload_folder = 'static/uploads'
os.makedirs(upload_folder, exist_ok=True)

# Route: Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route: Handles room image upload and pipeline execution
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('room-image')

    # Collect form values (style, marketplace, budget, room type)
    selected_style = request.form.get('selected-style')
    selected_marketplace = request.form.get('selected-marketplace')
    budget = request.form.get('budget')
    room_type = request.form.get('room-type')

    if not file or file.filename == '':
        # Use default image if no upload
        filename = 'default.png'
        filepath = os.path.join('static/uploads', filename)
    else:
        # Save uploaded file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join('static/uploads', filename)
        file.save(filepath)
        print(f"Image saved in: {filepath}")

    # Convert local path to public URL (based on Render deployment)
    img_url = f"https://modifai-website.onrender.com/{filepath}"

    # Generate furnished version of the room using AI
    furnished_img_path = furnish_image(filename.split(".")[0], img_url, style=selected_style, room_type=room_type)

    # Detect furniture items in the furnished image using YOLO
    img_paths = extract_items(furnished_img_path)
    print("✅ Final products:", img_paths)

    # Scrape matching IKEA products for each detected item
    products_paths = ikea_scraper(img_paths, budget, selected_style, room_type)

    # Extract product metadata from text files
    products = extract_products(products_paths)  

    # Render results page with before/after images and product matches
    return render_template(
        'results.html',
        before_image=filepath[filepath.index('static'):],
        after_image=furnished_img_path,
        products=products
    )



# Route: View selected items in basket (after results.html form submission)
@app.route('/view_basket', methods=['POST'])
def view_basket():
    selected_products_json = request.form.get('selected_products')
    selected_products = json.loads(selected_products_json) if selected_products_json else []
    
    # Render basket view
    return render_template('basket.html', items=selected_products)

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render port if available, else default to 5000
    app.run(host="0.0.0.0", port=port)
