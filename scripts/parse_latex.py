import re
with open(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\VLM_Contradictions_Presentation.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

stack = []
for idx, line in enumerate(lines):
    line_num = idx + 1
    # Find all \begin{...} and \end{...}
    begins = re.findall(r'\\begin\{([a-zA-Z*]+)\}', line)
    ends = re.findall(r'\\end\{([a-zA-Z*]+)\}', line)
    
    for b in begins:
        stack.append((b, line_num))
    for e in ends:
        if not stack:
            print(f"Error: \\end{{{e}}} on line {line_num} has no matching \\begin")
        else:
            top_env, top_line = stack.pop()
            if top_env != e:
                print(f"Mismatch: \\begin{{{top_env}}} on line {top_line} closed by \\end{{{e}}} on line {line_num}")

while stack:
    top_env, top_line = stack.pop()
    print(f"Unclosed: \\begin{{{top_env}}} on line {top_line}")
