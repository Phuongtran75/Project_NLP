with open(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\report\report.md", "r", encoding="utf-8") as f:
    text = f.read()

# We look for Section 2.2.3 and insert the figure right after the ASCII drawing and before Section 2.2.4 Discussion
target = "#### 2.2.4 Discussion"
replacement = """![Figure 2: PCA projection of hidden states at the final token position.](figure2.png)

#### 2.2.4 Discussion"""

if target in text:
    text = text.replace(target, replacement)
    print("Successfully inserted figure2.png!")
else:
    print("Failed to find target in report.md.")

with open(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\report\report.md", "w", encoding="utf-8") as f:
    f.write(text)
