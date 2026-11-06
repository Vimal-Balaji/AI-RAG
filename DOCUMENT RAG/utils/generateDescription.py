import boto3
import os
from utils.constants import DESCRIPTION_DIR
from utils.encodeBase64 import encode_image

# ── Bedrock client ────────────────────────────────────────────────────────────
REGION   = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "google.gemma-3-4b-it")

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_description_path(pdf_name, image_name):
    pdf_dir = f"{DESCRIPTION_DIR}/{pdf_name}"
    os.makedirs(pdf_dir, exist_ok=True)
    txt_name = os.path.splitext(image_name)[0] + ".txt"
    return f"{pdf_dir}/{txt_name}"


def _read_image_bytes(path: str) -> bytes:
    """Return raw bytes for an image file."""
    path = path.replace("\\", "/")
    with open(path, "rb") as f:
        return f.read()


def _image_format(path: str) -> str:
    """Infer Bedrock image format from file extension."""
    ext = os.path.splitext(path)[-1].lower()
    return {"jpg": "jpeg", ".jpg": "jpeg", ".jpeg": "jpeg",
            ".png": "png", ".gif": "gif", ".webp": "webp"}.get(ext, "png")


# ── Core generation ───────────────────────────────────────────────────────────

def generate_description(path: str, text: str = "") -> str:
    """
    Call AWS Bedrock (Converse API) with an image and optional context text.
    Returns a short description (≤ 100 words).
    """
    image_bytes  = _read_image_bytes(path)
    image_format = _image_format(path)

    prompt_text = (
        f'This is the general description of the image extracted from the pdf. "{text}". '
        "Extract all the text. From the text extracted describe this image in short. "
        "Restrict the output with 100 words.\n"
        "Include:\n"
        "- Objects (if any)\n"
        "- Charts (if any)\n"
        "- Axes labels (if any)\n"
        "- Graph trends (if any)\n"
        "- Tables (if any)\n"
        "- Technical content (if any)"
    )

    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": prompt_text},
                    {
                        "image": {
                            "format": image_format,
                            "source": {"bytes": image_bytes},
                        }
                    },
                ],
            }
        ],
        inferenceConfig={"maxTokens": 512, "temperature": 0},
    )

    return response["output"]["message"]["content"][0]["text"]


# ── Cache-aware entry point ───────────────────────────────────────────────────

def load_or_generate_description(image_path: str, context_text: str, pdf_name: str) -> str:
    image_name = os.path.basename(image_path)
    desc_path  = get_description_path(pdf_name, image_name)

    if os.path.exists(desc_path):
        print(f"Loading cached description: {desc_path}")
        with open(desc_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    print(f"Generating description for: {image_name}")
    description = generate_description(image_path, context_text)

    with open(desc_path, "w", encoding="utf-8") as f:
        f.write(description)

    return description