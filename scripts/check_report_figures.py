with open(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\report\report.md", "r", encoding="utf-8") as f:
    text = f.read()

print("Is figure1 in report:", "figure1.png" in text)
print("Is figure2 in report:", "figure2.png" in text)
print("Is figure3 in report:", "figure3.png" in text)

# Print lines around PCA Section
import re
match = re.search(r'### 2\.2 Task 2: Latent Representation Probing.*', text, re.DOTALL)
if match:
    print("\n--- Task 2 Section Beginning ---")
    print(match.group(0)[:1200])
