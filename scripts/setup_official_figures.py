import os
from PIL import Image

project_dir = r"f:\OneDrive\Phuong_2025\VIN\NLP\Project"

def setup_figures():
    # 1. Setup figure1.png
    fig1_jpg_path = os.path.join(project_dir, "fig1.jpg")
    fig1_png_path = os.path.join(project_dir, "figure1.png")
    extracted_p3_path = os.path.join(project_dir, "extracted_p3_img2.jpeg")
    
    if os.path.exists(fig1_jpg_path):
        print(f"Converting {fig1_jpg_path} to {fig1_png_path}...")
        img = Image.open(fig1_jpg_path)
        img.save(fig1_png_path, "PNG")
        print("figure1.png created from fig1.jpg successfully.")
    elif os.path.exists(extracted_p3_path):
        print(f"Converting extracted pdf image {extracted_p3_path} to {fig1_png_path}...")
        img = Image.open(extracted_p3_path)
        img.save(fig1_png_path, "PNG")
        print("figure1.png created from PDF successfully.")
    else:
        print("Warning: figure1.png source not found!")

    # 2. Setup figure2.png
    extracted_p5_path = os.path.join(project_dir, "extracted_p5_img1.png")
    fig2_png_path = os.path.join(project_dir, "figure2.png")
    
    if os.path.exists(extracted_p5_path):
        print(f"Copying {extracted_p5_path} to {fig2_png_path}...")
        img = Image.open(extracted_p5_path)
        img.save(fig2_png_path, "PNG")
        print("figure2.png created from PDF successfully.")
    else:
        print("Warning: figure2.png source not found!")

    print("Figures setup completed.")

if __name__ == "__main__":
    setup_figures()
