import os
import sys
from pathlib import Path

# Ensure root folder is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import boto3
from botocore.exceptions import ClientError
from typing import Optional, List, Tuple
from utils.constants import (
    S3_BUCKET_NAME,
    AWS_REGION,
    OUTPUT_DIR,
    DESCRIPTION_DIR,
    EMBEDDED_DB_DIR,
    S3_IMAGES_PREFIX,
    S3_DESCRIPTIONS_PREFIX,
    S3_EMBEDDED_DB_PREFIX,
)

_s3_client = None

def get_s3_client():
    """Lazily initialize and return boto3 S3 client."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client

def parse_s3_uri(uri: str, default_bucket: str = S3_BUCKET_NAME) -> Tuple[str, str]:
    """
    Parse an S3 URI (s3://bucket/key) or relative key into (bucket, key).
    """
    clean_uri = uri.replace("\\", "/")
    if clean_uri.startswith("s3://"):
        parts = clean_uri[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key
    return default_bucket, clean_uri

def object_exists(s3_key_or_uri: str, bucket: str = S3_BUCKET_NAME) -> bool:
    """Check if an object exists in S3."""
    b, key = parse_s3_uri(s3_key_or_uri, bucket)
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=b, Key=key)
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ("404", "NoSuchKey", "NotFound"):
            return False
        # Re-raise permissions or connection errors
        raise

def upload_bytes(
    data: bytes,
    s3_key: str,
    content_type: Optional[str] = None,
    bucket: str = S3_BUCKET_NAME,
) -> str:
    """Upload raw bytes directly to S3 and return the S3 key."""
    b, key = parse_s3_uri(s3_key, bucket)
    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type

    s3 = get_s3_client()
    s3.put_object(Bucket=b, Key=key, Body=data, **extra_args)
    return key

def read_bytes(s3_key_or_uri: str, bucket: str = S3_BUCKET_NAME) -> bytes:
    """Read raw bytes directly from S3."""
    b, key = parse_s3_uri(s3_key_or_uri, bucket)
    s3 = get_s3_client()
    response = s3.get_object(Bucket=b, Key=key)
    return response["Body"].read()

def upload_text(
    text: str,
    s3_key: str,
    encoding: str = "utf-8",
    content_type: str = "text/plain; charset=utf-8",
    bucket: str = S3_BUCKET_NAME,
) -> str:
    """Upload a string as UTF-8 encoded text to S3."""
    return upload_bytes(
        data=text.encode(encoding),
        s3_key=s3_key,
        content_type=content_type,
        bucket=bucket,
    )

def read_text(
    s3_key_or_uri: str,
    encoding: str = "utf-8",
    bucket: str = S3_BUCKET_NAME,
) -> str:
    """Read UTF-8 encoded text directly from S3."""
    raw_bytes = read_bytes(s3_key_or_uri, bucket=bucket)
    return raw_bytes.decode(encoding)

def upload_file(
    local_path: str,
    s3_key: str,
    bucket: str = S3_BUCKET_NAME,
) -> str:
    """Upload a local file to S3."""
    b, key = parse_s3_uri(s3_key, bucket)
    s3 = get_s3_client()
    s3.upload_file(local_path, b, key)
    return key

def download_file(
    s3_key_or_uri: str,
    local_path: str,
    bucket: str = S3_BUCKET_NAME,
) -> str:
    """Download an S3 object to local disk, creating parent directories if needed."""
    b, key = parse_s3_uri(s3_key_or_uri, bucket)
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    s3 = get_s3_client()
    s3.download_file(b, key, local_path)
    return local_path

def list_objects(prefix: str = "", bucket: str = S3_BUCKET_NAME) -> List[str]:
    """List all S3 object keys matching a given prefix."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for item in page["Contents"]:
                keys.append(item["Key"])
    return keys

def read_image_bytes(path_or_s3_key: str, bucket: str = S3_BUCKET_NAME) -> bytes:
    """
    Read image bytes seamlessly from either:
    1. Local filesystem path (if file exists locally)
    2. S3 bucket (if provided as an S3 key or s3:// URI)
    """
    clean_path = path_or_s3_key.replace("\\", "/")

    # Check local filesystem first
    if os.path.isfile(clean_path):
        with open(clean_path, "rb") as f:
            return f.read()

    # Otherwise read directly from S3
    return read_bytes(clean_path, bucket=bucket)

def sync_local_directory_to_s3(
    local_dir: str,
    s3_prefix: str,
    bucket: str = S3_BUCKET_NAME,
):
    """
    Recursively upload all files from a local directory to an S3 prefix.
    """
    if not os.path.exists(local_dir):
        print(f"Local directory does not exist: {local_dir}")
        return

    print(f"Syncing local directory '{local_dir}' to s3://{bucket}/{s3_prefix} ...")
    count = 0
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file = os.path.join(root, file)
            rel_path = os.path.relpath(local_file, local_dir).replace("\\", "/")
            s3_key = f"{s3_prefix}/{rel_path}".lstrip("/")
            upload_file(local_file, s3_key, bucket=bucket)
            print(f"  Uploaded: {local_file} -> s3://{bucket}/{s3_key}")
            count += 1
    print(f"Sync complete. Uploaded {count} file(s).")

def sync_all_local_to_s3():
    """Upload all existing local images, descriptions, and embedded DBs to S3."""
    sync_local_directory_to_s3(OUTPUT_DIR, S3_IMAGES_PREFIX)
    sync_local_directory_to_s3(DESCRIPTION_DIR, S3_DESCRIPTIONS_PREFIX)
    sync_local_directory_to_s3(EMBEDDED_DB_DIR, S3_EMBEDDED_DB_PREFIX)

if __name__ == "__main__":
    import sys
    if "--sync" in sys.argv:
        sync_all_local_to_s3()
    else:
        print(f"S3 Helper loaded. Target bucket: s3://{S3_BUCKET_NAME}/ (Region: {AWS_REGION})")
        print("To sync local files to S3, run with: python utils/s3_storage.py --sync")
