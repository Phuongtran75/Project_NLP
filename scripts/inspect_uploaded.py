import os
from PIL import Image

conv_dir = r"C:\Users\ADMIN\.gemini\antigravity-ide\brain\815d4631-404b-4e0a-b068-f193c452b961"

def inspect(filename):
    path = os.path.join(conv_dir, filename)
    if os.path.exists(path):
        img = Image.open(path)
        print(f"{filename}: size={img.size}, format={img.format}")
    else:
        print(f"{filename} does not exist.")

inspect("media__1779691412328.png")
inspect("media__1779691399524.png")
inspect("media__1779691384799.png")
