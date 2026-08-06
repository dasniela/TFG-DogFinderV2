# DogFinder V2

Visual similarity search for dog images using MobileNetV2 feature extraction and FAISS indexing.

Final thesis project — Computer Engineering, UNIR 2025. Grade: 9/10.
[Published in UNIR's institutional repository](https://reunir.unir.net/items/543bc492-bf4e-483e-a6c0-fc56d8a0cc6e)

---

## What it does

Given a query image of a dog, the system returns the most visually similar dogs already registered in the database, ranked by cosine similarity and filtered by a configurable threshold.

The motivating use case is lost-dog identification: matching a photo of a found dog against a database of reported missing ones, where breed labels are unreliable but visual appearance is not.

MobileNetV2 (pre-trained, used as a frozen feature extractor) generates embeddings; FAISS indexes them for fast approximate search; SQLite persists the dog records alongside their extracted features.

![Demo](docs/demo.png)

---

## Results

Cosine similarity thresholds swept from 0.20 to 0.95 against a manually defined ground-truth set.

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.20 | 0.486 | 0.981 | 0.631 |
| 0.30 | 0.526 | 0.981 | 0.665 |
| 0.40 | 0.652 | 0.971 | 0.765 |
| **0.50** | **0.931** | **0.835** | **0.859** |
| 0.55 | 0.916 | 0.639 | 0.715 |
| 0.60 | 0.745 | 0.435 | 0.516 |
| 0.70 | 0.286 | 0.089 | 0.128 |
| 0.80 | 0.048 | 0.014 | 0.021 |

**0.50 is the operating threshold** — highest F1 (0.859), against a design target of ≥ 0.85.

Below 0.50, recall stays near 100% but precision collapses to roughly 50%: the system returns almost everything relevant, buried in false positives. Above 0.55, recall falls sharply (0.64 at 0.55, 0.43 at 0.60) as genuine matches get filtered out. Past 0.90 the system returns nothing.

![F1-score vs similarity threshold](docs/f1_vs_threshold.png)

### Performance

Requirements were defined before evaluation, then measured against.

| Metric | Result | Target |
|---|---|---|
| Search latency | 1.58 ms ± 0.07 ms | < 5 ms |
| Index build | 20,755 images in 11.88 min | < 20 min |
| Duplicate detection precision | 100% | ≥ 95% |
| Similarity search F1 | 0.859 | ≥ 0.85 |

Latency measured across 500 individual queries against a 20,755-image index, returning top-15 results at threshold 0.50.

Duplicate detection tested on a controlled set of 20 images (6 originals, 6 exact duplicates, 6 visually similar, 2 unrelated). All 6 exact duplicates were correctly identified and skipped; 14 images registered, 6 omitted.

### Dataset

| Source | Images | Classes | Notes |
|---|---|---|---|
| Dog_Mx_Dataset | 314 | — | Own collection; 313 unique (one intentional duplicate) |
| Stanford Dogs Dataset | 20,580 | 120 breeds | Public dataset |
| Duplicate_test_subset | 20 | — | Built for duplicate-detection testing |

---

## How it works

**Feature extraction** — MobileNetV2 is used as a black box; its internals are not modified. Each image is converted into an embedding vector representing its visual characteristics.

**Storage** — SQLite holds each dog's name, location, date, image path and extracted features.

**Duplicate prevention** — On insertion, the system checks both by file path and by feature similarity, so near-identical images don't create redundant records.

**Similarity search** — FAISS indexes all embeddings and returns the nearest neighbours to a query vector, filtered by the similarity threshold.

**Evaluation** — Tooling to sweep thresholds and compute Precision, Recall and F1 against a ground-truth grouping.

---

## Project structure

| File | Purpose |
|---|---|
| `add_dogs_to_db.py` | Main script: processes and registers images, using FAISS to skip duplicates |
| `common_dog_finder_config.py` | Global configuration — models, paths, shared utilities |
| `dog_features.faiss` | FAISS index for similarity search |
| `dog_finder_demo_v4.db` | SQLite database of processed dog records |
| `dog_id_map.json` | Maps FAISS IDs to image paths |
| `dog_groups.json` | Ground-truth groupings of visually similar dogs, used for evaluation |
| `TFG_DogFinder_FeatureExtraction.ipynb` | Experimentation and feature extraction notebook |

---

## Getting started

```bash
pip install -r requirements.txt
```

Adjust image directories and paths in `common_dog_finder_config.py`, then populate the database and FAISS index:

```bash
python add_dogs_to_db.py
```

For quick testing, point the config at the reduced dataset (`TEST_IMAGE_DIRS`) instead of the full one.

---

## Notes

- MobileNetV2 is used as a frozen feature extractor; no fine-tuning or architectural changes.
- The project prioritises efficient integration and resource optimisation over model complexity — the target was a system that runs in constrained environments.
- The Stanford Dogs Dataset is subject to its own licence and is not redistributed here.

---

## Author

Daniela Díaz — Computer Engineering (TFG), UNIR 2025
[LinkedIn](https://www.linkedin.com/in/daniela-alejandra-d%C3%ADaz-ru%C3%ADz/)
