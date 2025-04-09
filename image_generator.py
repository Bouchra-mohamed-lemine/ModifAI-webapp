import replicate
import os
import requests

# Set the API token
os.environ["REPLICATE_API_TOKEN"] = "r8_03FfNMLg3qEHJKFRS0eP7kCxvPjktNi1yT3Bs"


def furnish_image(img_file_name, image_url, style, room_type):

    prompt = f"A beutiful furnished {room_type} with {style} touch"

    os.makedirs("static/generated_images", exist_ok=True)

    # Run the replicate model
    output = replicate.run(
        "jschoormans/comfyui-interior-remodel:2a360362540e1f6cfe59c9db4aa8aa9059233d40e638aae0cdeb6b41f3d0dcce",
        input={
            "image": image_url,
            "prompt": prompt,
            "output_quality": 80,
            "output_format": "png"
        }
    )

    # Save the result(s)
    for index, item in enumerate(output):
        image_path = f"static/generated_images/output_{img_file_name}_{index}.png"
        with open(image_path, "wb") as f:
            f.write(item.read())
        print(f"✅ Saved: output_{index}.png")

    return image_path
