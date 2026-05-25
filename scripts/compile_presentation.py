import subprocess
import os

project_dir = r"f:\OneDrive\Phuong_2025\VIN\NLP\Project"

def compile_pdf():
    print("Compiling Beamer slides using pdflatex (Pass 1)...")
    result1 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "VLM_Contradictions_Presentation.tex"],
        cwd=project_dir,
        capture_output=True,
        text=True
    )
    print("Pass 1 return code:", result1.returncode)
    
    print("Compiling Beamer slides using pdflatex (Pass 2)...")
    result2 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "VLM_Contradictions_Presentation.tex"],
        cwd=project_dir,
        capture_output=True,
        text=True
    )
    print("Pass 2 return code:", result2.returncode)
    
    pdf_path = os.path.join(project_dir, "VLM_Contradictions_Presentation.pdf")
    if os.path.exists(pdf_path):
        print(f"Success! Compiled presentation PDF is located at: {pdf_path} (size={os.path.getsize(pdf_path)} bytes)")
    else:
        print("Error: Presentation PDF was not generated.")
        print("Stderr log from pdflatex:")
        print(result2.stderr)

if __name__ == "__main__":
    compile_pdf()
