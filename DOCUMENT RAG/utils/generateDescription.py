import os
import boto3
from utils.constants import (
    DESCRIPTION_DIR,
    S3_BUCKET_NAME,
    S3_DESCRIPTIONS_PREFIX,
    AWS_REGION,
    BEDROCK_MODEL_ID,
)
from utils.s3_storage import (
    read_image_bytes,
    object_exists,
    read_text,
    upload_text,
)

# ── Bedrock client ────────────────────────────────────────────────────────────
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_description_s3_key(pdf_name: str, image_name: str) -> str:
    """Return S3 object key for the description text file."""
    txt_name = os.path.splitext(image_name)[0] + ".txt"
    return f"{S3_DESCRIPTIONS_PREFIX}/{pdf_name}/{txt_name}"

def get_description_path(pdf_name: str, image_name: str) -> str:
    """Return local filesystem path for description caching."""
    pdf_dir = os.path.join(DESCRIPTION_DIR, pdf_name).replace("\\", "/")
    os.makedirs(pdf_dir, exist_ok=True)
    txt_name = os.path.splitext(image_name)[0] + ".txt"
    return f"{pdf_dir}/{txt_name}"

def _image_format(path: str) -> str:
    """Infer Bedrock image format from file extension."""
    ext = os.path.splitext(path)[-1].lower()
    return {
        "jpg": "jpeg",
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".png": "png",
        ".gif": "gif",
        ".webp": "webp",
    }.get(ext, "png")


# ── Core generation ───────────────────────────────────────────────────────────

def generate_description(path_or_s3_key: str, text: str = "") -> str:
    """
    Call AWS Bedrock (Converse API) with an image and optional context text.
    Reads image bytes from either S3 or local storage.
    Returns a short description (<= 100 words).
    """
    image_bytes = read_image_bytes(path_or_s3_key)
    image_format = _image_format(path_or_s3_key)

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
        modelId=BEDROCK_MODEL_ID,
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


# ── Cache-aware entry point (S3-first) ─────────────────────────────────────────

def load_or_generate_description(
    image_path: str, context_text: str, pdf_name: str
) -> str:
    """
    Checks if description exists in S3; if not, generates it via Bedrock
    and uploads the generated description to the S3 bucket.
    """
    image_name = os.path.basename(image_path)
    s3_key = get_description_s3_key(pdf_name, image_name)
    local_path = get_description_path(pdf_name, image_name)

    # 1. Check if cached in S3
    try:
        if object_exists(s3_key, bucket=S3_BUCKET_NAME):
            print(f"Loading cached description from S3: s3://{S3_BUCKET_NAME}/{s3_key}")
            return read_text(s3_key, bucket=S3_BUCKET_NAME)
    except Exception as e:
        print(f"Note: Could not check S3 for description ({e}). Checking local cache...")

    # 2. Check local cache fallback
    if os.path.exists(local_path):
        print(f"Loading cached description from local file: {local_path}")
        with open(local_path, "r", encoding="utf-8") as f:
            description = f.read().strip()
        # Sync up to S3 in the background
        try:
            upload_text(description, s3_key, bucket=S3_BUCKET_NAME)
            print(f"Uploaded cached description to S3: s3://{S3_BUCKET_NAME}/{s3_key}")
        except Exception:
            pass
        return description

    # 3. Generate new description
    print(f"Generating description for: {image_name}")
    description = generate_description(image_path, context_text)

    # 4. Store in S3 bucket
    try:
        upload_text(description, s3_key, bucket=S3_BUCKET_NAME)
        print(f"Stored description in S3: s3://{S3_BUCKET_NAME}/{s3_key}")
    except Exception as e:
        print(f"Warning: Failed to save description to S3: {e}")

    # 5. Save local backup copy
    try:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(description)
    except Exception:
        pass

    return description
