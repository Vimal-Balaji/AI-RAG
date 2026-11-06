import base64

def encode_image(image_path):
    image_path = image_path.replace("\\", "/")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()