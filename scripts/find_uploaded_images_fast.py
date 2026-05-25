import os
import time

conv_dir = r"C:\Users\ADMIN\.gemini\antigravity-ide\brain\815d4631-404b-4e0a-b068-f193c452b961"
temp_dir = os.environ.get("TEMP", "C:\\tmp")

def search_images_fast(directory):
    matches = []
    if not os.path.exists(directory):
        return matches
    for file in os.listdir(directory):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(directory, file)
            matches.append((path, os.path.getmtime(path)))
    return matches

print("Search in conversation directory:")
for path, mtime in sorted(search_images_fast(conv_dir), key=lambda x: x[1], reverse=True):
    print(f"  {path} (size={os.path.getsize(path)}, modified {time.time() - mtime:.1f}s ago)")

print("Search in temp directory (flat):")
for path, mtime in sorted(search_images_fast(temp_dir), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {path} (size={os.path.getsize(path)}, modified {time.time() - mtime:.1f}s ago)")
