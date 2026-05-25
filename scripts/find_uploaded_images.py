import os
import time

appdata_dir = r"C:\Users\ADMIN\.gemini\antigravity-ide"
temp_dir = os.environ.get("TEMP", "C:\\tmp")

def search_new_images(root_dir):
    matches = []
    # Search for files modified in the last 10 minutes
    now = time.time()
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, file)
                try:
                    mtime = os.path.getmtime(path)
                    if now - mtime < 600:  # 10 minutes
                        matches.append((path, mtime))
                except Exception:
                    pass
    return matches

print("Searching AppData for new images:")
for path, mtime in sorted(search_new_images(appdata_dir), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {path} (modified {time.time() - mtime:.1f}s ago)")

print("Searching Temp for new images:")
for path, mtime in sorted(search_new_images(temp_dir), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {path} (modified {time.time() - mtime:.1f}s ago)")
