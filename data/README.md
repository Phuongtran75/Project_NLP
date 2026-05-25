# Dataset Information

This folder contains metadata, format specifications, and download links for the evaluation datasets used in the paper:

> **Know It's Absent, Yet Point Anyway: Cross-Task Semantic Contradictions in Vision-Language Models**

---

## Evaluated Models

We evaluate four open-source vision-language models (VLMs):

| Model | Parameters | Source |
|:---|:---:|:---|
| Qwen2.5-VL-7B-Instruct | 7B | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Qwen3-VL-8B-Instruct | 8B | [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| InternVL3-8B | 8B | [HuggingFace](https://huggingface.co/OpenGVLab/InternVL3-8B) |
| InternVL3.5-8B | 8B | [HuggingFace](https://huggingface.co/OpenGVLab/InternVL3.5-8B) |

---

## Evaluation Benchmarks

We evaluate on four object-existence and hallucination-related benchmarks. For each benchmark, we isolate the **negative subset** (objects confirmed absent from the image) to study the Existence–Localization Contradiction.

### 1. POPE (Polling-based Object Probing Evaluation)

- **Reference**: Li et al. (2023b). *Evaluating object hallucination in large vision-language models.* EMNLP 2023, pp. 292–305.
- **Image Source**: MS COCO 2017 validation set
- **Purpose**: Queries models on the presence/absence of objects under random, popular, and adversarial sampling settings. We isolate the negative subsets where the queried object is confirmed absent.
- **Download**: [COCO 2017 val images](https://cocodataset.org/#download) | [POPE annotations](https://github.com/AoiDragon/POPE)

### 2. AMBER (Object Existence Subset)

- **Reference**: Wang et al. (2024). *AMBER: An LLM-free multi-dimensional benchmark for MLLMs hallucination evaluation.* arXiv:2311.07397.
- **Purpose**: A comprehensive hallucination evaluation benchmark. We isolate the **object existence subset** containing binary presence/absence queries and their corresponding negative pairs.
- **Download**: [AMBER GitHub](https://github.com/junyangwang0616/AMBER)

### 3. MME (Object Existence Subset)

- **Reference**: Fu et al. (2026). *MME: A comprehensive evaluation benchmark for multimodal large language models.* NeurIPS Datasets and Benchmarks Track.
- **Purpose**: A comprehensive evaluation benchmark for MLLMs. We utilize the **object existence subtask**, selecting all negative query pairs.
- **Download**: [MME GitHub](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)

### 4. DASH-B (Detection and Assessment of Systematic Hallucinations)

- **Reference**: Augustin et al. (2025). *DASH: Detection and assessment of systematic hallucinations of VLMs.* arXiv:2503.23573.
- **Purpose**: Specifically designed to assess systematic hallucinations where VLMs hallucinate highly co-occurring but absent items (e.g., a fork next to a plate). Tests whether a model can abstain under localization of highly plausible but absent referents.
- **Download**: [DASH-B GitHub](https://github.com/YuxiXie/DASH)

---

## Evaluation Protocol

We use a **2-Stage Conditional Evaluation Protocol** over the negative subset N = {(v_i, o_i) : e_i = 0}:

1. **Stage 1 — Existence Recognition**: Query the model with an existence template (e.g., *"Is there a {object} in the image?"*). Parse the response into a binary prediction ê_i ∈ {0, 1}.
2. **Stage 2 — Localization under Absence**: Query the model with a localization template (e.g., *"Locate the {object} in the image"*). Parse the response into a binary localization decision l̂_i ∈ {0, 1}, where l̂=1 if the model returns a bounding box and l̂=0 if the model abstains.

### Metrics

| Metric | Definition |
|:---|:---|
| **EA** (Existence Accuracy) | Fraction of negative samples where the model correctly predicts absence |
| **NLP** (Null Localization Precision) | Fraction of localization abstentions on truly negative examples |
| **CELF** (Conditional Existence-Localization Faithfulness) | P(l̂=0 \| e=0, ê=0) — primary metric measuring logical consistency |

### Steering Intervention

For each model, we construct a paired set of **100 contrastive examples** (hallucinated bbox vs. correct abstention) to estimate the steering direction. The steering vector is computed as:

```
s(l) = mean(h_abs) - mean(h_loc)
```

Grid search is performed over layers l ∈ {7, 8, ..., 13} and model-specific scaling coefficients α.

---

## Data Schema

All evaluation data follows this unified format:

```json
{
  "image_path": "data/images/coco_001.png",
  "query_object": "person",
  "existence_ground_truth": 0,
  "vqa_prompt": "Is there a person in the image?",
  "grounding_prompt": "Locate a person in the image and identify its bounding box if it exists."
}
```

- `existence_ground_truth`: `0` = absent, `1` = present
- `vqa_prompt`: Standard binary existence query (Stage 1)
- `grounding_prompt`: Model-specific localization instruction (Stage 2). Uses natural language for Qwen and `<ref>` tags for InternVL.

---

## Reproduction

Due to licensing and file size constraints, raw images are not included in this repository.

To reproduce the evaluation:

1. Download images from the official sources listed above
2. Place the images in `data/images/`
3. Run the evaluation pipeline in `notebooks/inference.ipynb`

For steering vector construction:
1. Run `notebooks/create_hidden_state.ipynb` to extract hidden states
2. Run `notebooks/create_steer_vectors.ipynb` to compute steering directions
3. Run `notebooks/inference.ipynb` with the steering vector to evaluate steered performance
