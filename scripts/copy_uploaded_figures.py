import shutil
import os

conv_dir = r"C:\Users\ADMIN\.gemini\antigravity-ide\brain\815d4631-404b-4e0a-b068-f193c452b961"
project_dir = r"f:\OneDrive\Phuong_2025\VIN\NLP\Project"

def copy_figures():
    mappings = {
        "media__1779691412328.png": "figure1.png",
        "media__1779691399524.png": "figure2.png",
        "media__1779691384799.png": "figure3.png"
    }
    
    for src_name, dest_name in mappings.items():
        src_path = os.path.join(conv_dir, src_name)
        dest_path = os.path.join(project_dir, dest_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"Copied {src_name} to {dest_name} (size={os.path.getsize(dest_path)} bytes)")
        else:
            print(f"Error: source {src_path} not found.")

if __name__ == "__main__":
    copy_figures()
