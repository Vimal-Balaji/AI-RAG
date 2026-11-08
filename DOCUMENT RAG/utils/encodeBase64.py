import base64
from utils.s3_storage import read_image_bytes

def encode_image(image_path: str) -> str:
    """Encode an image to base64, supporting both local filesystem and S3 keys."""
    raw_bytes = read_image_bytes(image_path)
    return base64.b64encode(raw_bytes).decode("utf-8")
