# CLDW Trajectory Extraction & Visualization Toolkit

A concise toolkit for extracting, comparing, validating, and visualizing travel trajectories from the Corpus of Lake District Writing (CLDW). It provides an LLM pipeline, a rule-based spaCy pipeline, side-by-side comparison, evaluation against manual ground truth, and interactive Folium maps.

---

## Requirements

- Python 3.10+
- OS: macOS, Linux, or Windows

### Python libraries
Install via `requirements.txt`:

```
lxml
spacy==3.7.2
pydantic>=2.0.0
openai
python-dotenv>=1.0.0
folium>=0.14.0
geopy>=2.3.0
pandas>=1.5.0
tqdm>=4.64.0 
```

Additional model download required for spaCy NLP pipeline:

```
python -m spacy download en_core_web_lg
```

---

## Setup

1) Clone and create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

2) Environment variables (.env)

- Create a file named `.env` in the project root and add your OpenAI key:

```bash
OPENAI_API_KEY="sk-..."
```

- Alternatively, export it in your shell:

```bash
export OPENAI_API_KEY="sk-..."
```

The LLM extractor reads `.env` automatically via `python-dotenv`.

---

## Data Layout

Expected paths (subset):

```
├── data/
│   ├── LakeDistrictCorpus/
│   │   ├── LD80_metadata/LD80_metadata.csv         # metadata (filtered to Travelogue/Journal/Guide/Survey/Epistle)
│   │   ├── LD80_geoparsed/                         # CQP-style XML with lat/long
│   │   └── ...
└── results/                                        # per-ID outputs, maps, validation
```

---

## Python APIs (optional)

- `data_loader.DataLoader` loads metadata and geoparsed XML. It filters to genres Travelogue, Journal, Guide, Survey, Epistle.
- `llm_extractor.LLMTrajectoryExtractor(model="gpt-4o")` extracts trajectories using OpenAI Chat Completions with JSON output.
- `nlp_extractor.NLPTrajectoryExtractor(model="en_core_web_lg")` performs rule-based extraction using spaCy dep-matcher.
- `visualizer.TrajectoryVisualizer` builds/saves Folium maps for LLM/NLP vs GT layers.
- `validator.TrajectoryValidator` evaluates predictions vs manual ground truth and writes `validation.json`.

---

## Command-Line Utilities

### 1) End-to-end comparison – `compare_trajectories.py`
Runs LLM and/or NLP extractors on a subset of texts, saves per-ID results and maps, and writes a combined JSON.

Parameters:
- `--num-records INT` (default: 1): number of records to process (ignored if `--id` provided)
- `--id STRING`: specific metadata `ID` to process (overrides `--num-records`)
- `--no-llm` (flag): skip LLM extraction
- `--no-nlp` (flag): skip NLP extraction
- `--llm-model STRING` (default: `gpt-4o`): OpenAI model name
- `--output PATH` (default: `data/trajectories_comparison.json`): combined JSON output
- `--results-dir DIR` (default: `results`): directory for per-ID outputs (JSON + maps)

Examples:
```bash
# Process first 3 records with both pipelines
python compare_trajectories.py --num-records 3

# Process a specific ID (as listed in metadata CSV)
python compare_trajectories.py --id 1767_a

# Run only NLP (no OpenAI required)
python compare_trajectories.py --num-records 5 --no-llm

# Use a specific OpenAI model and custom outputs
python compare_trajectories.py --num-records 2 \
  --llm-model gpt-4o-mini --output data/trajectories_comparison.json --results-dir results
```

Outputs per ID in `results/{ID}/`:
- `llm_results.json` and/or `nlp_results.json`
- `map_combined.html`, `map_gt_vs_llm.html`, `map_gt_vs_nlp.html`
- `validation.json` (auto-written if manual GT present)

### 2) Validate against ground truth – `run_validate_all.py`
Reads `results/` and validates IDs present in `results/manual_trajectories.json`.

Parameters:
- `--results-dir DIR` (default: `results`)
- `--manual-gt PATH` (default: `results/manual_trajectories.json`)

Examples:
```bash
python run_validate_all.py
python run_validate_all.py --results-dir results --manual-gt results/manual_trajectories.json
```

Writes `results/{ID}/validation.json` per validated ID.

### 3) Generate/refresh maps – `run_generate_maps.py`
Rebuilds all maps for any ID folders in `results/` using existing `llm_results.json` / `nlp_results.json` files.

Examples:
```bash
python run_generate_maps.py
```

Outputs per ID in `results/{ID}/`:
- `map_combined.html`, `map_gt_vs_llm.html`, `map_gt_vs_nlp.html`

### 4) Aggregate analysis – `run_analysis.py`
Runs pairwise/summary analysis over all `results/{ID}/` folders and writes CSV/JSON summaries into `analysis/`.

Parameters:
- `--results-dir DIR` (default: `results`)
- `--analysis-dir DIR` (default: `analysis`)

Examples:
```bash
python run_analysis.py
python run_analysis.py --results-dir results --analysis-dir analysis
```

---