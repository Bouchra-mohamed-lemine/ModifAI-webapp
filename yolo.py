from ultralytics import YOLO
import cv2
import numpy as np
import os 
import shutil

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")


def extract_items(img_path):

    output_dir = "static/products"

    # Remove the products folder if it exists
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    img_paths = []

    # Load and resize the image to 400x400
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (400, 400))

    # Run inference on resized image
    results = model(resized_img)[0]

    # Loop through detected objects
    for i, cls_id in enumerate(results.boxes.cls):
        class_name = results.names[int(cls_id)]
        print(f"✅ {class_name} found!")

        # Get mask and bounding box
        mask = results.masks.data[i].cpu().numpy()
        box = results.boxes.xyxy[i].cpu().numpy().astype(int)

        # Prepare mask
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (resized_img.shape[1], resized_img.shape[0]))

        # Create white background
        white_background = np.ones_like(resized_img, dtype=np.uint8) * 255

        # Apply the mask (object where mask is 255, background where mask is 0)
        object_only = cv2.bitwise_and(resized_img, resized_img, mask=mask)
        background_only = cv2.bitwise_and(white_background, white_background, mask=cv2.bitwise_not(mask))
        combined = cv2.add(object_only, background_only)

        # Crop to bounding box
        x1, y1, x2, y2 = box
        cropped = combined[y1:y2, x1:x2]

        # Save result
        # Base path
        base_path = f"static/products/{class_name}.png"
        output_path = base_path

        # Increment suffix until the file name is unique
        counter = 1
        while os.path.exists(output_path):
            output_path = f"static/products/{class_name}_{counter}.png"
            counter += 1

        # Save the image
        cv2.imwrite(output_path, cropped)
        print(f"🖼️ Saved as {output_path}")

        img_paths.append(output_path)



    return img_paths
