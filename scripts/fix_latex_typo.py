with open(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\VLM_Contradictions_Presentation.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Replace the specific mismatched end block in Key Observation
old_text = r"""        \begin{column}{0.38\textwidth}
            \begin{alertblock}{Key Observation}
                While Existence Accuracy (EA) is consistently high ($80\% - 100\%$), \textbf{CELF scores are strikingly low}. 
                \vskip 0.15cm
                InternVL models achieve \textbf{0.0\% CELF} across all datasets, meaning they \textbf{never abstain} under localization queries, even when they just correctly recognized object absence.
            \end{block}
        \end{column}"""

new_text = r"""        \begin{column}{0.38\textwidth}
            \begin{alertblock}{Key Observation}
                While Existence Accuracy (EA) is consistently high ($80\% - 100\%$), \textbf{CELF scores are strikingly low}. 
                \vskip 0.15cm
                InternVL models achieve \textbf{0.0\% CELF} across all datasets, meaning they \textbf{never abstain} under localization queries, even when they just correctly recognized object absence.
            \end{alertblock}
        \end{column}"""

if old_text in text:
    text = text.replace(old_text, new_text)
    print("Successfully replaced!")
else:
    # If the text has different spacing, let's use a regex replacement
    import re
    # We find \begin{alertblock}{Key Observation} ... \end{block} and replace the \end{block} with \end{alertblock}
    pattern = r'(\\begin\{alertblock\}\{Key Observation\}.*?)\\end\{block\}'
    text, count = re.subn(pattern, r'\1\\end{alertblock}', text, flags=re.DOTALL)
    print(f"Substituted {count} times using regex.")

with open(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\VLM_Contradictions_Presentation.tex", "w", encoding="utf-8") as f:
    f.write(text)
