with open(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\report\report.md", "r", encoding="utf-8") as f:
    text = f.read()

# Insert Figure 1 before 1.2 Dataset Origins
target_1 = "### 1.2 Dataset Origins and Description"
replacement_1 = """![Figure 1: An example of Existence-Localization Contradiction.](figure1.png)

### 1.2 Dataset Origins and Description"""

# Insert Figure 2 in Section 2.2.3 after the ASCII drawing
target_2 = """          |      [ x x x x x x ]
          |      [ x x x x x x x ] (Contradictory BBox Cluster: l=1)
          +---------------------------------------------> PC1
```"""
replacement_2 = """          |      [ x x x x x x ]
          |      [ x x x x x x x ] (Contradictory BBox Cluster: l=1)
          +---------------------------------------------> PC1
```

![Figure 2: PCA projection of hidden states at the final token position.](figure2.png)"""

# Insert Figure 3 in Section 2.3.3 after the ASCII curve drawing
target_3 = """     0 |---+---+---+---+---+---+---+---+---+---> Scale α
         -0.8 -0.5 -0.2 0.0 0.4 0.8 1.2 1.6
```"""
replacement_3 = """     0 |---+---+---+---+---+---+---+---+---+---> Scale α
         -0.8 -0.5 -0.2 0.0 0.4 0.8 1.2 1.6
```

![Figure 3: Model performance under different steering coefficients.](figure3.png)"""

if target_1 in text:
    text = text.replace(target_1, replacement_1)
if target_2 in text:
    text = text.replace(target_2, replacement_2)
if target_3 in text:
    text = text.replace(target_3, replacement_3)
    print("Figures markup inserted in report.md successfully!")
else:
    print("Failed to insert figures markup.")

with open(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\report\report.md", "w", encoding="utf-8") as f:
    f.write(text)
