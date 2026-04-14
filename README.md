---
title: Spotify Genre and Recommendation Engine
emoji: 🎵
colorFrom: green
colorTo: black
sdk: streamlit
sdk_version: "1.56.0"
app_file: app/streamlit_app.py
pinned: false
---

# Spotify Genre & Recommendation Engine

**Content-based track recommender for 113,000+ Spotify tracks across 114 genres.**  
A two-stage ML pipeline: LightGBM genre classifier → cosine-similarity nearest-neighbour
recommender. Genre probability vectors serve as musical embeddings, enabling
cross-genre recommendations that a simple category filter cannot produce.

> **Live demo:** [*Deploy to Streamlit Community Cloud and add URL here*]  
> **Dataset:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — Kaggle

---

## Project Overview

| | |
|---|---|
| **Problem** | Given a track's audio features, find the k most musically similar tracks |
| **End user** | Music listeners, playlist curators, Spotify-style recommendation systems |
| **Data** | 113,534 tracks × 21 columns · 114 genres · ~995 tracks per genre |
| **Model output** | 114-dimensional probability embedding → cosine-similarity ranking |
| **Key design decision** | Use `predict_proba()` as embeddings, not hard genre labels |

The core insight: a jazz-influenced hip-hop track produces a mixed probability vector
(e.g., 40% jazz, 35% hip-hop, 15% soul). A hard-label system would file it under one
genre and never surface it for listeners of the other. The probability embedding places
it between both clusters, so it surfaces for either audience.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Two-Stage Pipeline                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  data/dataset.csv  (113,534 tracks × 21 columns)                 │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────┐                         │
│  │  Stage 0: Data Cleaning             │  src/data/cleaner.py    │
│  │  · Quality gate (7 checks)          │  src/data/quality.py    │
│  │  · Duration anomalies removed        │                        │
│  │  · Tempo octave doubling corrected   │                        │
│  │  · Zero time signatures imputed      │                        │
│  └──────────────────┬──────────────────┘                         │
│                     │ data/cleaned.csv                           │
│                     ▼                                            │
│  ┌─────────────────────────────────────┐                         │
│  │  Feature Engineering                │  src/features/          │
│  │  · 15 audio features selected       │  engineer.py            │
│  │  · 12 new features created          │  engineering.py         │
│  │  · Correlation filter (r > 0.95)    │  run_features.py        │
│  └──────────────────┬──────────────────┘                         │
│                     │ 15-feature matrix                          │
│                     ▼                                            │
│  ┌─────────────────────────────────────┐                         │
│  │  Stage 1: Genre Classifier          │  src/models/            │
│  │  · Baseline (Dummy + LogReg)         │  baseline.py            │
│  │  · 3-model comparison               │  compare.py             │
│  │  · Optuna tuning (50 trials, TPE)   │  tuning.py              │
│  │  · Winner: LightGBM (tuned)         │  run_training.py        │
│  └──────────────────┬──────────────────┘                         │
│                     │ predict_proba() → 114-dim embedding        │
│                     ▼                                            │
│  ┌─────────────────────────────────────┐                         │
│  │  Stage 2: Track Recommender         │  src/models/            │
│  │  · NearestNeighbors (cosine metric) │  recommender.py         │
│  │  · Brute-force over all embeddings  │  train.py               │
│  │  · Top-k most similar tracks        │                         │
│  └──────────────────┬──────────────────┘                         │
│                     │                                            │
│                     ▼                                            │
│  Streamlit Portfolio App   app/streamlit_app.py                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Results

All models evaluated on the same 80/20 stratified train/test split.
Random baseline = 1/114 ≈ 0.88% (uniform random guess across 114 classes).

| Model | Top-1 Accuracy | F1 Macro | Top-5 Accuracy | vs Random | Train Time |
|---|---|---|---|---|---|
| DummyClassifier | 0.88% | 0.01% | 4.4% | 1× | <1s |
| LogisticRegression | 20.4% | 19.7% | 58.9% | 23× | ~19s |
| RandomForest | 34.6% | 33.2% | 71.4% | 39× | ~64s |
| **LightGBM (default)** | **42.1%** | **40.7%** | **80.1%** | **48×** | **~25s** |
| **LightGBM (tuned) ★** | **~44%** | **~43%** | **~82%** | **~50×** | — |

> **Why 44% top-1 is a good result:** with 114 equally-represented classes, a perfect
> random classifier scores 0.88%. Our model is ~50× better than that. More importantly,
> **top-5 accuracy of ~82%** means the correct genre is in the model's top-5 predictions
> 82% of the time — the embedding vector is well-placed in genre space for 4 in 5 tracks,
> which is what the recommender needs.

Experiment runs tracked in MLflow — view with:
```bash
mlflow ui --port 5001
# → http://localhost:5001
```

---

## Tech Stack

| Tool | Role |
|---|---|
| **Python 3.11** | Primary language |
| **pandas / NumPy** | Data manipulation and numerical computing |
| **LightGBM** | Gradient-boosted genre classifier (114 classes) |
| **scikit-learn** | Train/test split, evaluation metrics, NearestNeighbors |
| **Optuna** | Bayesian hyperparameter tuning (TPE sampler, 50 trials) |
| **MLflow** | Experiment tracking (file-based, no server required) |
| **Streamlit** | Interactive portfolio web app |
| **Plotly** | Interactive charts (heatmaps, bar charts, box plots) |
| **pytest** | Test suite (38 tests across data, features, and model layers) |
| **Docker** | Containerised deployment |
| **GitHub Actions** | CI/CD (test + lint on every push) |
| **joblib** | Model serialisation |

---

## Setup & Installation

**Prerequisites:** Python 3.11+, pip

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/spotify-recsys.git
cd spotify-recsys

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the project package (enables src/ imports)
pip install -e .
```

**Download the dataset:**
1. Download [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) from Kaggle
2. Place `dataset.csv` in `data/`

> **Or skip straight to the app:** `data/sample.csv` (5,016 tracks, all 114 genres)
> is already in the repo. The Streamlit app and EDA pages work without running the
> full pipeline.

---

## How to Run

### Full training pipeline

```bash
# Step 1 — Clean raw data (saves data/cleaned.csv)
python src/data/cleaner.py

# Step 2 — Engineer features (saves data/features.csv)
python src/features/run_features.py

# Step 3 — Train all models + log to MLflow (saves models/production_model.pkl)
python src/models/run_training.py

# Step 4 — Build recommendation index (saves models/recommender.pkl)
python src/models/train.py
```

### Streamlit portfolio app

```bash
streamlit run app/streamlit_app.py
# → http://localhost:8501
```

The app works **without running the training pipeline** — it falls back to
`data/sample.csv` and demo model metrics so all four pages are always navigable.
The sidebar shows a green/red indicator for classifier and recommender status.

### Interactive recommendation (Python API)

```python
from models.recommender import TrackRecommender

rec = TrackRecommender.load()

# By track name
rec.recommend_by_name("Clair de Lune", k=10)

# By Spotify track ID
rec.recommend_by_id("5SuOikwiRyPMVoIQDJUgSV", k=10)

# By integer index
rec.recommend(idx=42, k=5)
```

### View MLflow experiment runs

```bash
mlflow ui --port 5001
# → http://localhost:5001
```

### Docker

```bash
# Build the image
docker build -t spotify-recsys .

# Run (mounts data/ and models/ so the container sees trained artifacts)
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  spotify-recsys

# Or use docker-compose
docker-compose up
```

### Tests

```bash
# Run full test suite
pytest tests/ -v

# Run a single test file
pytest tests/test_model.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Feature Engineering

### Features used by the classifier (15 total)

| Feature | Type | Range | Musical meaning |
|---|---|---|---|
| `danceability` | Float | 0–1 | Rhythmic regularity and beat strength |
| `energy` | Float | 0–1 | Perceived intensity and activity |
| `valence` | Float | 0–1 | Musical positiveness (happy vs. sad) |
| `speechiness` | Float | 0–1 | Presence of spoken words |
| `acousticness` | Float | 0–1 | Confidence the track is acoustic |
| `instrumentalness` | Float | 0–1 | Absence of vocal content |
| `liveness` | Float | 0–1 | Presence of live audience |
| `loudness` | Float | −60–5 dB | Overall loudness |
| `tempo` | Float | BPM | Beats per minute (tempo octave-corrected) |
| `popularity` | Integer | 0–100 | Spotify stream-count proxy |
| `duration_ms` | Integer | ms | Track length |
| `key` | Integer | 0–11 | Musical key (C=0 … B=11) |
| `mode` | Integer | 0–1 | Major (1) or minor (0) |
| `time_signature` | Integer | 3–5 | Beats per bar |
| `explicit` | Boolean → 0/1 | 0–1 | Explicit content flag |

### Features created (12 new, 2 dropped by filter)

| Feature | Category | Created from | Why |
|---|---|---|---|
| `is_spoken_word` | Domain | `speechiness > 0.66` | Spotify's own threshold for speech-dominant tracks |
| `is_live_recording` | Domain | `liveness > 0.80` | Above 0.8 = strong live signal |
| `is_instrumental` | Domain | `instrumentalness > 0.50` | Bimodal split; binarised not scaled |
| `is_acoustic` | Domain | `acousticness > 0.50` | Same bimodal split |
| `audio_brightness` | Statistical | `energy + valence + danceability` | Composite "bright energy" score |
| `audio_atypicality` | Statistical | `speechiness + liveness + instrumentalness` | Tracks unusual in multiple dimensions |
| `danceability_x_energy` | Interaction | `danceability × energy` | High on both = dance music signal |
| `valence_x_energy` | Interaction | `valence × energy` | Separates happy-loud from sad-quiet |
| `acoustic_vocal` | Interaction | `acousticness × (1 − instrumentalness)` | Acoustic + vocal = singer-songwriter |
| `tempo_x_danceability` | Interaction | `tempo × danceability` | Fast + danceable = specific genre cluster |
| ~~`loudness_norm`~~ | *(dropped)* | `(loudness + 60) / 65` | r = 1.0 with `loudness` — linear rescale adds no information |
| ~~`tempo_norm`~~ | *(dropped)* | `tempo / 250` | r = 1.0 with `tempo` — same reason |

> `loudness_norm` and `tempo_norm` were created as normalised building blocks for the
> interaction features. The correlation filter (threshold r > 0.95) correctly removes
> them — their interaction-derived siblings carry the new signal and are retained.

---

## Key Decisions & Lessons

**1. Genre probabilities as embeddings (not hard labels)**  
There are three levels of content-based recommendation, and this system sits at the
most expressive one:

| Approach | Similarity over | What it captures |
|---|---|---|
| Genre filter | Hard label | Same category only — misses everything cross-genre |
| Raw audio features | 15-dim cosine | Direct acoustic similarity, but no learned structure |
| **Learned embeddings** ★ | **114-dim predict_proba()** | **Non-linear genre relationships learned from data** |

Using `predict_proba()` produces a 114-dim soft representation. Two tracks with similar
probability distributions are musically similar even if filed under different genre labels.
A track that scores 40% jazz / 30% blues / 15% soul is described more richly than just
"jazz" — and the embedding places it near both clusters in recommendation space.

This is also why a hybrid approach (content + collaborative filtering) wasn't needed:
the learned embedding already captures latent musical relationships the way CF captures
latent taste relationships — just from genre structure instead of user behaviour.
Pure collaborative filtering would require a user–item interaction matrix that this
dataset doesn't contain.

**2. File-based MLflow tracking**  
`mlflow.set_tracking_uri(mlruns_path)` writes to a local directory with no server process.
No port conflicts, no background daemon. `mlflow ui --port 5001` when you need to browse.
`mlruns/` is gitignored so experiment history stays local.

**3. Tempo octave-doubling correction**  
580 tracks appeared above 200 BPM — trip-hop, blues, children's music, piano — none of
which have real tempos anywhere near 200 BPM. Spotify's BPM detector sometimes locks onto
eighth-note subdivisions instead of the main beat, reporting 2× the true tempo. Fix:
halve all values > 200 BPM. Nine piano tracks were hand-verified by a trained musician;
all confirmed in the right range after correction.

**4. What didn't work: dropping loudness_norm and tempo_norm**  
Early versions of the pipeline kept `loudness_norm` and `tempo_norm` in the final feature
set. Both have r = 1.0 with their source columns — they are exact linear rescalings, so
they add zero information while inflating the feature count and misleading feature
importance rankings. The correlation filter correctly removes them. Lesson: build
intermediate features explicitly as building blocks, then let the filter do its job.

**5. Cosine similarity over Euclidean distance**  
Probability vectors live on a simplex (they sum to 1). Cosine similarity measures the
angle between vectors, not magnitude — so a track with high confidence (sharp peak in
its distribution) and a similar track with lower confidence (broader distribution) still
score as similar. Euclidean distance would penalise the magnitude difference.

---

## Project Structure

```
spotify-recsys/
│
├── app/
│   └── streamlit_app.py      # 4-page Streamlit portfolio app
│
├── data/
│   ├── sample.csv            # Representative 5,016-row sample (committed)
│   ├── dataset.csv           # Full raw data (gitignored — download from Kaggle)
│   └── cleaned.csv           # Cleaned data (gitignored — generated by cleaner.py)
│
├── models/                   # Trained model artefacts (gitignored)
│   ├── production_model.pkl  # LightGBM classifier (tuned)
│   ├── recommender.pkl       # Fitted TrackRecommender
│   ├── best_params.json      # Optuna best hyperparameters
│   └── comparison.csv        # 3-model comparison results
│
├── notebooks/
│   ├── eda.ipynb             # 7-section exploratory data analysis
│   └── overview.md           # Data walkthrough with cleaning decisions
│
├── src/
│   ├── data/
│   │   ├── loader.py         # Load and inspect raw CSV
│   │   ├── quality.py        # 7-check data quality gate
│   │   └── cleaner.py        # Cleaning pipeline (7 steps)
│   │
│   ├── features/
│   │   ├── engineer.py       # Feature selection for the ML pipeline
│   │   ├── engineering.py    # Feature creation and correlation filter
│   │   └── run_features.py   # End-to-end feature pipeline script
│   │
│   └── models/
│       ├── classifier.py     # LightGBM genre classifier
│       ├── recommender.py    # Cosine-similarity NearestNeighbors recommender
│       ├── baseline.py       # Dummy + LogisticRegression baselines
│       ├── compare.py        # 3-model comparison
│       ├── tuning.py         # Optuna hyperparameter search
│       ├── run_training.py   # MLflow experiment orchestration
│       └── train.py          # Full training → recommendation pipeline
│
├── tests/
│   ├── conftest.py           # sys.path setup for all tests
│   ├── test_data_quality.py  # 11 tests for quality gate
│   ├── test_features.py      # 14 tests for feature engineering
│   └── test_model.py         # 13 tests for classifier interface
│
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions: test + lint on push
│
├── Dockerfile                # python:3.11-slim image
├── docker-compose.yml        # App service with data/models volumes
├── requirements.txt          # Python dependencies
├── setup.py                  # Package install (enables src/ imports)
└── .gitignore
```

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click **New app** → select repo, branch `main`, file path `app/streamlit_app.py`
5. Click **Deploy**

The app works **without the full dataset or trained models** — `data/sample.csv` is
committed to the repo and provides real Spotify data for all four pages. The sidebar
shows component status; the "Try it yourself" prediction section falls back to a
heuristic demo when `models/production_model.pkl` is not present.

---

## Running the Tests

```bash
pytest tests/ -v
```

```
38 passed in 5.76s
├── test_data_quality.py   11 tests  (quality gate pass/fail, warnings)
├── test_features.py       14 tests  (shape, dtype, NaN, stratification)
└── test_model.py          13 tests  (train, predict, proba, embeddings, load)
```

All tests use synthetic DataFrames — no dependency on data files, fully CI-compatible.

---

## Acknowledgements

Dataset: [Maharshi Pandya — Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset), Kaggle.  
Spotify audio features documented in the [Spotify for Developers API reference](https://developer.spotify.com/documentation/web-api/reference/get-audio-features).
