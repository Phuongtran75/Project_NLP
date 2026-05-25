import os

report_path = r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\report\report.md"

with open(report_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Swap cover page roles
old_cover_bac = "1. **Nguyen Ba Thanh Bac** (Student ID: V202502001) – *Role: Preprocessing & Pipeline Architecture*"
old_cover_phuong = "3. **Tran Thi Hoai Phuong** (Student ID: V202502962) – *Role: Activation Steering & Mitigation*"

new_cover_bac = "1. **Nguyen Ba Thanh Bac** (Student ID: V202502001) – *Role: Activation Steering & Mitigation*"
new_cover_phuong = "3. **Tran Thi Hoai Phuong** (Student ID: V202502962) – *Role: Preprocessing & Pipeline Architecture*"

text = text.replace(old_cover_bac, new_cover_bac)
text = text.replace(old_cover_phuong, new_cover_phuong)

# 2. Swap Team Contribution Statement (Section 5)
old_contrib_bac = "* **Nguyen Ba Thanh Bac (Preprocessing & Pipeline)**: Set up the repository architecture (`/data/`, `/scripts/`, `/report/`). Developed the regex coordinate parser and dataset loaders for POPE, AMBER, MME, and DASH-B. Coded the 2-Stage Conditional Protocol and evaluated baseline performance."
old_contrib_phuong = "* **Tran Thi Hoai Phuong (Activation Steering)**: Designed the latent steering algorithm. Integrated EasySteer to construct contrastive steer directions. Conducted the grid search over scaling factor $\\alpha$ and layers, and built the final interactive Jupyter Playground."

new_contrib_bac = "* **Nguyen Ba Thanh Bac (Activation Steering)**: Designed the latent steering algorithm. Integrated EasySteer to construct contrastive steer directions. Conducted the grid search over scaling factor $\\alpha$ and layers, and built the final interactive Jupyter Playground."
new_contrib_phuong = "* **Tran Thi Hoai Phuong (Preprocessing & Pipeline)**: Set up the repository architecture (`/data/`, `/scripts/`, `/report/`). Developed the regex coordinate parser and dataset loaders for POPE, AMBER, MME, and DASH-B. Coded the 2-Stage Conditional Protocol and evaluated baseline performance."

text = text.replace(old_contrib_bac, new_contrib_bac)
text = text.replace(old_contrib_phuong, new_contrib_phuong)

# 3. Swap Individual Reflections headers & contents (Section 6)
old_refl_bac_header = "### 6.1 Reflection by Nguyen Ba Thanh Bac (Preprocessing & Pipeline)"
old_refl_phuong_header = "### 6.3 Reflection by Tran Thi Hoai Phuong (Activation Steering)"

new_refl_bac_header = "### 6.1 Reflection by Nguyen Ba Thanh Bac (Activation Steering)"
new_refl_phuong_header = "### 6.3 Reflection by Tran Thi Hoai Phuong (Preprocessing & Pipeline)"

text = text.replace(old_refl_bac_header, new_refl_bac_header)
text = text.replace(old_refl_phuong_header, new_refl_phuong_header)

# Swap the reflection text blocks
old_refl_bac_text = '"In this project, my primary role was setting up the pipeline architecture and pre-processing the diverse datasets. I designed the unified data loader and implemented regular expression patterns to extract bounding boxes from diverse output formats. A major challenge was parsing Qwen’s natural coordinates compared to InternVL’s `<ref>` XML tag format, which frequently caused string-index errors. Overcoming this taught me the importance of robust data sanitization and output validation in VLM evaluation. I gained deep knowledge of visual grounding benchmarks and learned how structural formatting impacts generative models. In the future, I plan to research structured schema constraints (like Instructor or Outlines) during decoding to completely prevent malformed coordinate generations." (120 words)'

old_refl_phuong_text = '"As the steering and mitigation lead, I developed the lightweight inference-time steering engine. I integrated the EasySteer framework, computed the contrastive difference vectors, and injected the steering vector. My largest obstacle was managing the trade-off between increasing CELF on negative samples and preventing over-abstention on positive grounding tasks. Conducting a systematic grid search over layers 7-13 and coefficient $\\alpha$ allowed me to identify the optimal layer-scale trade-off. This project deepened my understanding of representation engineering and causal latent manipulation without expensive fine-tuning. In the future, I will focus on developing dynamic, input-adaptive steering mechanisms that adjust $\\alpha$ automatically based on model confidence." (115 words)'

# Replace them swapped
text = text.replace(old_refl_bac_text, "PLACEHOLDER_TEMP_REFL")
text = text.replace(old_refl_phuong_text, old_refl_bac_text)
text = text.replace("PLACEHOLDER_TEMP_REFL", old_refl_phuong_text)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(text)
print("Role swap completed in report.md successfully!")
