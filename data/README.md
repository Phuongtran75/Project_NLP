# Dataset Information: VLM Existence-Localization Contradiction Evaluation

This folder contains the metadata, format specifications, and links for the evaluation datasets used in Phase 2: Final Report & Code.

## Overview of Evaluation Benchmarks

We evaluate four leading vision-language model families across four standard multimodal and object-existence/hallucination benchmarks:

1. **POPE (Polite Object Presence Evaluation)** (Li et al., 2023b)
   * **Purpose**: Evaluates object-level hallucination by querying the presence of objects in COCO images.
   * **Negative Subset Definition**: Focuses on "Absent" objects (co-occurring but not present, or completely random) to isolate where the model correctly answers "No" but might fail downstream grounding.
   * **Data Format**: `{"image": "coco_image_id.png", "object": "cat", "exists": false}`

2. **AMBER (Object Existence Subset)** (Wang et al., 2024)
   * **Purpose**: An LLM-free multi-dimensional hallucination evaluation benchmark. We isolate the *Object Existence* subset containing negative pairs (objects absent in the images).
   * **Data Format**: `{"image_id": "amber_001.png", "query_object": "chair", "exists": 0}`

3. **MME (Object Existence Subset)** (Fu et al., 2026)
   * **Purpose**: A comprehensive evaluation benchmark for MLLMs. The Object Existence subtask specifically tests binary existence on negative and positive questions. We isolate all 300 negative samples.
   * **Data Format**: `{"image": "mme_existence_01.png", "object": "person", "exists": 0}`

4. **DASH-B (Detection and Assessment of Systematic Hallucinations)** (Augustin et al., 2025)
   * **Purpose**: Specifically designed to assess systematic hallucinations where VLMs hallucinate co-occurring objects. Provides a rigorous test of whether a model can abstain under localization of highly plausible but absent referents.
   * **Data Format**: `{"image": "dash_b_105.jpg", "object": "fork", "exists": 0}`

---

## Dataset Format Specification & Data Schema

To ensure perfect reproducibility, all evaluation datasets are formatted in standard JSON lines (JSONL) with the following unified schema:

```json
{
  "image_path": "data/images/coco_001.png",
  "query_object": "person",
  "existence_ground_truth": 0,
  "vqa_prompt": "Is there a person in the image?",
  "grounding_prompt": "Locate a person in the image and identify its bounding box if it exists."
}
```

* `existence_ground_truth`: `0` for absent objects, `1` for present objects.
* `vqa_prompt`: Standard binary existence query.
* `grounding_prompt`: Model-specific grounding instruction designed to elicit coordinate outputs (e.g. natural language coordinates or `<ref>` tags).

---

## Reconstruction and Reproducibility

For compliance with licensing and file size limits, the full raw image sets are not included directly in this repository. 
To replicate the exact figures in the report:
1. Download the images from the official COCO 2017 val set (for POPE), AMBER, MME, and DASH-B websites.
2. Place the images in `data/images/`.
3. The evaluation script in `scripts/evaluate_tasks.py` (or the `VLM_Contradictions_Interactive.ipynb` notebook) will automatically match the image paths and run the evaluation pipeline.
