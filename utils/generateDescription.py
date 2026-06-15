import requests
from utils.constants import LM_STUDIO_URL
from utils.encodeBase64 import encode_image

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