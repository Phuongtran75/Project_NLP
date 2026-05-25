import json
import os

def test_notebook_execution():
    notebook_path = r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\VLM_Contradictions_Interactive.ipynb"
    print(f"Testing execution of notebook: {notebook_path}")
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    print(f"Found {len(code_cells)} code cells.")
    
    global_env = {}
    
    for idx, cell in enumerate(code_cells):
        code_text = "".join(cell["source"])
        print(f"Executing Code Cell {idx+1}...")
        try:
            # We execute it inside a unified global environment so that cell variables persist
            exec(code_text, global_env)
            print(f"Cell {idx+1} executed successfully!")
        except Exception as e:
            print(f"Error in Cell {idx+1}: {e}")
            raise e
            
    print("All notebook cells executed without errors! Verified.")

if __name__ == "__main__":
    test_notebook_execution()
