with open(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\VLM_Contradictions_Presentation.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Replace the typo \end{alertblock> with \end{alertblock}
old_text = "            \\end{alertblock>"
new_text = "            \\end{alertblock}"

if old_text in text:
    text = text.replace(old_text, new_text)
    print("Typos fixed!")
else:
    print("Typos NOT found!")
    
with open(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\VLM_Contradictions_Presentation.tex", "w", encoding="utf-8") as f:
    f.write(text)
