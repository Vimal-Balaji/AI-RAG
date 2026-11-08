import os
import fitz  # PyMuPDF
from utils.constants import S3_BUCKET_NAME, S3_IMAGES_PREFIX, OUTPUT_DIR
from utils.s3_storage import upload_bytes

def image_extraction(pdf_path: str, output_dir: str = OUTPUT_DIR, upload_to_s3: bool = True):
    """
    Extract images from a PDF file, upload them directly to the S3 bucket,
    and optionally save a local copy.

    Returns a context dictionary mapping both S3 keys and local paths to
    [closest_text, page_number].
    """
    context = {}
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_subdir = os.path.join(output_dir, pdf_name).replace("\\", "/")
    os.makedirs(output_subdir, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            rects = page.get_image_rects(xref)
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_name = f"{pdf_name}_img_{image_count}.{image_ext}"
            image_path = f"{output_subdir}/{image_name}"
            s3_key = f"{S3_IMAGES_PREFIX}/{pdf_name}/{image_name}"

            # Save locally for cache / local inspection
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            # Store in S3 bucket
            if upload_to_s3:
                try:
                    upload_bytes(
                        data=image_bytes,
                        s3_key=s3_key,
                        content_type=f"image/{image_ext}",
                        bucket=S3_BUCKET_NAME,
                    )
                    print(f"Uploaded extracted image to S3: s3://{S3_BUCKET_NAME}/{s3_key}")
                except Exception as e:
                    print(f"Warning: Failed to upload {s3_key} to S3: {e}")

            image_count += 1

            if not rects:
                print(f"Extracted image {image_name} without position info.")
                closest_text = ""
            else:
                img_rect = rects[0]
                blocks = page.get_text("blocks")
                closest_text = ""
                min_distance = float("inf")

                for block in blocks:
                    x0, y0, x1, y1, text, *_ = block
                    if y0 > img_rect.y1:
                        distance = y0 - img_rect.y1
                        if distance < min_distance:
                            min_distance = distance
                            closest_text = text.strip()

            entry = [closest_text, page_num + 1]
            # Map both S3 key and local path for maximum compatibility
            context[s3_key] = entry
            context[image_path] = entry

    return context
