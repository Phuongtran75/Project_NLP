with open(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\VLM_Contradictions_Presentation.tex", "r", encoding="utf-8") as f:
    text = f.read()

import re
matches = re.findall(r'.{0,30}alertblock.{0,30}', text)
for m in matches:
    print(m)
