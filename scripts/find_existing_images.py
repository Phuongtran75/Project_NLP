import os

workspace_root = r"f:\OneDrive\Phuong_2025\VIN\NLP"
matches = []
for root, dirs, files in os.walk(workspace_root):
    for file in files:
        if "figure" in file.lower() and file.lower().endswith((".png", ".jpg", ".jpeg")):
            matches.append(os.path.join(root, file))

print("Found image files matching 'figure':")
for m in matches:
    print(m)
