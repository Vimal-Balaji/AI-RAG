import requests
from utils.constants import LM_STUDIO_URL,DESCRIPTION_DIR
from utils.encodeBase64 import encode_image
import os

def get_description_path(video_name, image_name):
    pdf_dir = os.path.join(DESCRIPTION_DIR, video_name)
    os.makedirs(pdf_dir, exist_ok=True)
    txt_name = os.path.splitext(image_name)[0] + ".txt"
    return os.path.join(pdf_dir, txt_name)

def generate_description(path,text=""):
    img_b64 = encode_image(path)
    payload = {
        "model": "google/gemma-4-4b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":
                        f"""
                        This is the general description of the image extracted from the pdf.\"{text}\".
                        Extract all the text.From the text extracted describe this image in short .Restrict the output with 100 words.
                        Include:
                        - Objects(if any)
                        - Charts(if any)
                        - Axes labels(if any)
                        - Graph trends(if any)
                        - Tables(if any)
                        - Technical content(if any)
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0
    }

    r = requests.post(LM_STUDIO_URL, json=payload)
    return r.json()["choices"][0]["message"]["content"]


def load_or_generate_description(image_path, video_name):
    image_name = os.path.basename(image_path)
    desc_path = get_description_path(video_name, image_name)
    if os.path.exists(desc_path):
        print(f"Loading cached description: {desc_path}")
        with open(desc_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    print(f"Generating description for: {image_name}")
    description = generate_description(image_path)
    with open(desc_path, "w", encoding="utf-8") as f:
        f.write(description)

    return description
