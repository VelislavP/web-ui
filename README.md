# VerifyBG

Bulgarian fake-news detector. Paste a news URL or raw article text and get an AI verdict powered by a fine-tuned mBERT classifier, LIME explanations, RAG retrieval, and Gemini-generated context.

## Requirements

- Python 3.11
- The API keys listed below

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd web-ui-test

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```
SAFE_BROWSING_API_KEY=   # Google Safe Browsing API v4 key
GEMINI_API_KEY=          # Google Gemini API key
HF_TOKEN=                # HuggingFace token (needed to download the model)
```

| Key | Where to get it |
|-----|----------------|
| `SAFE_BROWSING_API_KEY` | [Google Cloud Console](https://console.cloud.google.com/) → Enable *Safe Browsing API* → Credentials |
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/) |
| `HF_TOKEN` | [HuggingFace Settings](https://huggingface.co/settings/tokens) → New token (read) |

### 4. Build the RAG database (one-time)

The `rag_bg_news_db/` folder is already included in the repo, so you can skip this step unless you want to rebuild it from scratch:

```bash
python build_rag_db.py
```

## Running the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project structure

```
app.py              — Streamlit UI
classifier.py       — mBERT fake-news classifier
explainer.py        — LIME explainability
rag.py              — ChromaDB retrieval + Gemini explanation
scraper.py          — Article scraper (newspaper4k)
domain_checker.py   — WHOIS + Safe Browsing checks
trust_score.py      — Composite trust score (ML 70pt + safety 20pt + domain 10pt)
build_rag_db.py     — One-time script to build the local vector database
rag_bg_news_db/     — ChromaDB vector store (pre-built)
```
