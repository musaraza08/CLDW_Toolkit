# CLDW Toolkit – System Documentation

This document explains the overall architecture of the CLDW Trajectory Extraction & Visualization Toolkit. It describes what each package does and how the pipeline runs end-to-end in simple terms.

---

## 1) Overview

The toolkit turns historical texts from the Corpus of Lake District Writing (CLDW) into travel trajectories and interactive maps. It offers two extractors:
- An OpenAI-based LLM extractor
- A spaCy-based rule/dependency extractor

Results can be visualized on maps and compared against manual ground truth (GT). Summary statistics and CSVs are generated for analysis.

---

## 2) Core Packages

### `data_loader`
- Purpose: Load metadata and geoparsed TEI XML for the LD80 corpus.
- Input paths:
  - `data/LakeDistrictCorpus/LD80_metadata/LD80_metadata.csv`
  - `data/LakeDistrictCorpus/LD80_geoparsed/`
- Output: A Python dict (a “record”) containing fields like `ID`, `Author`, `Title_short`, `Year_Pub`, and a `geoparsed` object with `text` and `entities` (name, lat, long, type).
- Notes: Metadata is filtered to relevant genres. If a geoparsed file is missing, the record is still returned with `geoparsed=None`.

### `llm_extractor`
- Purpose: Extract a structured travel trajectory using an OpenAI model.
- Requirements: `OPENAI_API_KEY` in environment or `.env`.
- How it works (simple):
  - Reads text from the record.
  - Uses chunking for long inputs (overlapping windows) and then merges results.
  - Returns a JSON-like dict with `trajectory` (ordered places), `route_summary`, and `confidence`.
- Notes: Uses JSON mode. Deduplicates places and enforces 1..N ordering.

### `nlp_extractor`
- Purpose: Extract a trajectory using spaCy rules and dependency patterns (no LLM).
- Requirements: `en_core_web_lg` spaCy model.
- How it works (simple):
  - Detects movement clues like FROM/TO/VIA in sentences.
  - Chains movement events into an ordered path.
  - Applies basic plausibility checks (e.g., very long jumps with weak cues).
- Output: Same shape as LLM extractor for easy comparison.

### `visualizer`
- Purpose: Make interactive Folium maps.
- Layers: GT (manual), LLM, NLP. Missing coordinates are geocoded when possible.
- Output files per ID: `map_gt_vs_llm.html`, `map_gt_vs_nlp.html`, `map_combined.html`.

### `validator`
- Purpose: Compare predictions to manual ground truth and compute metrics.
- GT source: `results/manual_trajectories.json`.
- Matching order: gazetteer ref (if present) → geo proximity → name similarity.
- Metrics include:
  - Node precision/recall/F1, TP/FP/FN
  - Edge (route) precision/recall/F1, Kendall tau
  - Sequence LCS, edit distance (norm), sequence/positional accuracy
  - Geospatial mean/median km and % within 1/5/10 km
  - Error indicators (e.g., hallucinations, mis-ordering)
- Output per ID: `results/<ID>/validation.json`.

### `analysis`
- Purpose: Aggregate results across IDs, making CSVs and a summary JSON.
- Typical outputs: `analysis/*.csv`, `analysis/summary.json`.

---

## 3) Orchestration Scripts

### `compare_trajectories.py` (end-to-end)
Coordinates the full pipeline for one or more records.
- Select records: by `--id` (metadata `ID`) or `--num-records` (first N).
- Steps per record:
  1. Load with `data_loader`.
  2. Run `llm_extractor` (unless `--no-llm`).
  3. Run `nlp_extractor` (unless `--no-nlp`).
  4. Save per-method results to `results/<ID>/llm_results.json` and/or `results/<ID>/nlp_results.json`.
  5. Create maps via `visualizer` (`map_gt_vs_llm.html`, `map_gt_vs_nlp.html`, `map_combined.html`).
  6. Validate against GT with `validator`, writing `results/<ID>/validation.json`.
- Also writes an aggregate file: `data/trajectories_comparison.json`.

### `run_generate_maps.py`
Rebuilds maps for each `results/<ID>/` folder based on existing `llm_results.json` / `nlp_results.json`.

### `run_validate_all.py`
Validates all IDs in `results/` that have ground truth in `results/manual_trajectories.json`.

### `run_analysis.py`
Aggregates validations across IDs and writes CSV/JSON summaries to `analysis/`.

---

## 4) Data Flow (simple view)

```
DataLoader → Record → [LLM Extractor] → results/<ID>/llm_results.json
                         [NLP Extractor] → results/<ID>/nlp_results.json
                              ↓
                           Visualizer → maps in results/<ID>/
                              ↓
                            Validator → results/<ID>/validation.json
                              ↓
                             Analysis → CSVs and summary in analysis/
```

---

## 5) Inputs and Outputs

- Inputs:
  - Metadata CSV: `data/LakeDistrictCorpus/LD80_metadata/LD80_metadata.csv`
  - Geoparsed XML: `data/LakeDistrictCorpus/LD80_geoparsed/`
  - Manual GT (for validation): `results/manual_trajectories.json`
- Per-ID outputs (`results/<ID>/`):
  - `llm_results.json`, `nlp_results.json`
  - `map_gt_vs_llm.html`, `map_gt_vs_nlp.html`, `map_combined.html`
  - `validation.json`
- Aggregate outputs:
  - `data/trajectories_comparison.json`
  - `analysis/*.csv`, `analysis/summary.json`

---

## 6) Notes and Configuration

- LLM extractor needs `OPENAI_API_KEY` in `.env` or environment.
- NLP extractor needs `en_core_web_lg` installed.
- If some geoparsed files are missing, the record is still processed (with `geoparsed=None`), but some steps may be skipped.

---

## 7) Extending the System

- You can add another extractor that returns the same output shape as the LLM/NLP extractors:
  - A list `trajectory` with items like `{ "place": str, "order": int, ... }`
  - A string `route_summary`
  - A string `confidence` in {"low", "medium", "high"}
- If you follow this shape, the validator and visualizer will work with minimal changes. 