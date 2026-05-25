import os
pdf_path = r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\VLM_Contradictions_Presentation.pdf"
print("File exists:", os.path.exists(pdf_path))
if os.path.exists(pdf_path):
    print("File size:", os.path.getsize(pdf_path))
    try:
        with open(pdf_path, "rb") as f:
            data = f.read(100)
            print("First 100 bytes of PDF:", data)
    except Exception as e:
        print("Error reading PDF:", e)
