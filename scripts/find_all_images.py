import os

workspace_root = r"f:\OneDrive\Phuong_2025\VIN\NLP"
matches = []
for root, dirs, files in os.walk(workspace_root):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            matches.append(os.path.join(root, file))

print("Found ALL image files in workspace:")
for m in matches:
    print(m)
