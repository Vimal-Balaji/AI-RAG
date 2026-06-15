from multiprocessing import context
import fitz  # PyMuPDF
import os

pdf_path = "document.pdf"
output_dir = "extracted_images"
def image_extraction(pdf_path,output_dir):

    context={}
    os.makedirs(output_dir, exist_ok=True)
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

            image_name = (
                f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
            )

            image_path = os.path.join(output_dir, image_name)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_count += 1

            if not rects:
                print(f"Extracted image {image_name} without position info.")
                continue

            img_rect = rects[0]
            print(f"\nImage {page_num+1}_{img_index+1}")

            # Extract all text blocks
            blocks = page.get_text("blocks")
            closest_text = None
            min_distance = float("inf")

            for block in blocks:
                x0, y0, x1, y1, text, *_ = block
                if y0 > img_rect.y1:
                    distance = y0 - img_rect.y1
                    if distance < min_distance:
                        min_distance = distance
                        closest_text = text.strip()
            context[image_path]=[closest_text,page_num+1]
    return context


           
