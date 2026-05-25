import fitz # PyMuPDF
import os

pdf_path = r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\Know_It_s_Absent__Yet_Point_Anyway__Existence_Localization_Contradictions_in_Vision_Language_Models (2).pdf"
output_dir = r"f:\OneDrive\Phuong_2025\VIN\NLP\Project"

print(f"Opening PDF: {pdf_path}")
doc = fitz.open(pdf_path)

image_count = 0
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    image_list = page.get_images(full=True)
    print(f"Page {page_num+1} has {len(image_list)} images.")
    
    for img_idx, img in enumerate(image_list):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        
        image_name = f"extracted_p{page_num+1}_img{img_idx+1}.{image_ext}"
        image_path = os.path.join(output_dir, image_name)
        
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        print(f"Saved: {image_name} (size={len(image_bytes)} bytes)")
        image_count += 1

print(f"Total extracted images: {image_count}")
