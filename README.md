# AI-Powered Credit Risk Intelligence Platform

> Production-grade credit risk assessment platform powered by LightGBM, SHAP, and an LLM-based Talk-to-Data interface — built on the **Home Credit Default Risk** dataset.

---

## Architecture

```
project/
├── app/
│   ├── eda/                   # Phase 1: EDA, profiling, visualisations
│   │   ├── data_loader.py
│   │   ├── profiler.py
│   │   ├── visualizations.py
│   │   ├── insights.py
│   │   └── report_generator.py
│   ├── utils/                 # Shared utilities (logger, helpers)
│   ├── ml/                    # Phase 2: ML pipeline (future)
│   ├── explainability/        # Phase 4: SHAP (future)
│   ├── rules/                 # Phase 5: Business rules (future)
│   ├── chatbot/               # Phase 3: NL→SQL chatbot (future)
│   └── ui/                    # Phase 6: Streamlit UI (future)
├── data/
│   ├── raw/                   # ← Place Kaggle CSVs here
│   └── processed/
├── documents/
│   └── eda/                   # Generated EDA reports & plots
├── models/                    # Trained model artefacts (future)
├── tests/                     # pytest test suite
├── config.py                  # Central configuration
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [Kaggle Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk/data)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env as needed (LOG_LEVEL, EDA_SAMPLE_SIZE, etc.)
```

### 4. Place dataset files

```
data/raw/
    application_train.csv   ← required
    bureau.csv              ← optional (recommended)
    previous_application.csv← optional
```

### 5. Run Phase 1 — EDA Pipeline

```bash
python -m app.eda.report_generator
```

**Output files:**
- `documents/eda/eda_report.md` — Full Markdown EDA report
- `documents/eda/eda_profile.json` — Machine-readable profile
- `documents/eda/01_default_distribution.png`
- `documents/eda/02_income_vs_default.png`
- `documents/eda/03_credit_amount_vs_default.png`
- `documents/eda/04_age_vs_default.png`
- `documents/eda/05_employment_vs_default.png`
- `documents/eda/06_correlation_heatmap.png`

---

## Configuration

All settings are driven by environment variables (via `.env`).  No hardcoded values.

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `EDA_SAMPLE_SIZE` | _(none)_ | Row limit for EDA (blank = full dataset) |
| `EDA_FIGURE_DPI` | `150` | Plot image DPI |
| `EDA_CORRELATION_TOP_N` | `30` | Features in correlation heatmap |
| `RANDOM_SEED` | `42` | Reproducibility seed |
| `OPENAI_API_KEY` | — | LLM chatbot API key (Phase 3) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Provider-agnostic base URL |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model name |

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Tests are fully isolated via fixtures and `unittest.mock` — no real CSV files required.

---

## Development Phases

| Phase | Status | Description |
|---|---|---|
| **1** | ✅ **Complete** | EDA: data loading, profiling, visualisations, insights, report |
| **2** | 🔜 Planned | ML: LightGBM preprocessing, training, evaluation |
| **3** | 🔜 Planned | Chatbot: NL→SQL with OpenAI-compatible API |
| **4** | 🔜 Planned | Explainability: TreeSHAP waterfall & summary plots |
| **5** | 🔜 Planned | Business rules: surrogate tree extraction + policy engine |
| **6** | 🔜 Planned | Streamlit UI: 7-page application |
| **7** | 🔜 Planned | Docker deployment |

---

## LLM Provider Configuration (Phase 3 — Chatbot)

The chatbot uses an **OpenAI-compatible API** — no provider lock-in:

```env
# OpenAI
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Ollama (local)
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=llama3

# Groq
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama3-70b-8192
```

---

## License

Internal research project — not for production deployment without regulatory review.