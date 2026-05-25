import os
from PIL import Image

output_dir = r"f:\OneDrive\Phuong_2025\VIN\NLP\Project"

def inspect_image(path):
    if os.path.exists(path):
        try:
            img = Image.open(path)
            print(f"{os.path.basename(path)}: format={img.format}, size={img.size}, mode={img.mode}")
        except Exception as e:
            print(f"Error inspecting {path}: {e}")
    else:
        print(f"{path} does not exist.")

inspect_image(os.path.join(output_dir, "fig1.jpg"))
inspect_image(os.path.join(output_dir, "extracted_p3_img2.jpeg"))
inspect_image(os.path.join(output_dir, "extracted_p5_img1.png"))
