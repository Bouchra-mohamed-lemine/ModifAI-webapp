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

# Configuration
upload_folder = 'static/uploads'
os.makedirs(upload_folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload():


    # products = [{'image': 'static/filtered_results/01_HYLTARP.jpg', 'Name': 'HYLTARP', 'Price': 1449.0}, {'image': 'static/filtered_results/02_HYLTARP.jpg', 'Name': 'HYLTARP', 'Price': 1449.0}, {'image': 'static/filtered_results/03_UPPLAND.jpg', 'Name': 'UPPLAND', 'Price': 849.0}]
    # return render_template('results.html', before_image="static/uploads/input.png", after_image="static/generated_images/output_0.png", products=products)


    if 'room-image' not in request.files:
        return "No file part", 400

    file = request.files['room-image']
    if file.filename == '':
        return "No selected file", 400
    
    selected_style = request.form.get('selected-style')
    selected_marketplace = request.form.get('selected-marketplace')
    budget = request.form.get('budget')
    room_type = request.form.get('room-type')

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        print(f"Image saved in: {filepath}")
        img_url = f"https://modifai-yc0w.onrender.com/{filepath}"


        # Call the other script's function and pass the image path
        furnished_img_path = furnish_image(filename.split(".")[0], img_url, style=selected_style, room_type=room_type)

        # pass furnished_img_path to yolo.py to extract the products images
        img_paths = extract_items(furnished_img_path)

        print("✅ Final products:", img_paths)


        # pass output of yolo to scraper.py
        products_paths = ikea_scraper(img_paths, budget, selected_style, room_type)
        products = extract_products(products_paths)  

        return render_template('results.html', before_image=filepath[filepath.index('static'):], after_image=furnished_img_path, products=products)

    return "Something went wrong", 500




@app.route('/view_basket', methods=['POST'])
def view_basket():
    selected_products_json = request.form.get('selected_products')
    selected_products = json.loads(selected_products_json) if selected_products_json else []
    
    return render_template('basket.html', items=selected_products)






if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
