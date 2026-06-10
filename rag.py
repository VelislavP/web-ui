import os
import re
import requests
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()

_DB_PATH = Path(__file__).parent / "rag_bg_news_db"
_COLLECTION_NAME = "credible_bg_news"
_EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_GEMINI_MODEL = "gemini-3.1-flash-lite"  # 15 RPM · 250K TPM · 500 RPD

_WIKI_API = "https://bg.wikipedia.org/w/api.php"
_WIKI_HEADERS = {"User-Agent": "VerifyBG/1.0 (university project)"}

_embedding_model = None
_collection = None
_gemini_client = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embedding_model


def _get_collection():
    global _collection
    if _collection is None:
        if not _DB_PATH.exists():
            return None
        client = chromadb.PersistentClient(path=str(_DB_PATH))
        try:
            _collection = client.get_collection(_COLLECTION_NAME)
        except Exception:
            return None
    return _collection


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY не е зададен в .env")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def retrieve_relevant_facts(article_text: str, n_results: int = 3) -> list:
    collection = _get_collection()
    if collection is None:
        return []

    model = _get_embedding_model()
    query_embedding = model.encode([_clean_text(article_text)])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
    )

    retrieved = []
    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "title":    metadata["title"],
            "url":      metadata["url"],
            "source":   metadata["source"],
            "distance": distance,
            "text":     doc[:1200],
        })
    return retrieved


def search_wikipedia_bg(query: str, n_results: int = 2) -> list:
    """Search Bulgarian Wikipedia and return article summaries."""
    try:
        # Step 1: search for matching titles
        search_resp = requests.get(
            _WIKI_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": n_results,
                "format": "json",
                "utf8": 1,
            },
            headers=_WIKI_HEADERS,
            timeout=5,
        )
        search_resp.raise_for_status()
        hits = search_resp.json().get("query", {}).get("search", [])
        if not hits:
            return []

        titles = [h["title"] for h in hits]

        # Step 2: fetch extracts for those titles
        extract_resp = requests.get(
            _WIKI_API,
            params={
                "action": "query",
                "titles": "|".join(titles),
                "prop": "extracts|info",
                "exintro": True,
                "explaintext": True,
                "inprop": "url",
                "format": "json",
                "utf8": 1,
            },
            headers=_WIKI_HEADERS,
            timeout=5,
        )
        extract_resp.raise_for_status()
        pages = extract_resp.json().get("query", {}).get("pages", {})

        results = []
        for page in pages.values():
            extract = (page.get("extract") or "").strip()
            if not extract:
                continue
            results.append({
                "title":    page.get("title", ""),
                "url":      page.get("fullurl", f"https://bg.wikipedia.org/wiki/{page.get('title','')}"),
                "source":   "Wikipedia BG",
                "distance": None,
                "text":     extract[:1200],
            })
        return results

    except Exception:
        return []


def _build_prompt(
    article_text: str,
    model_prediction: str,
    fake_probability: float,
    news_facts: list,
    wiki_facts: list,
) -> str:
    news_text = ""
    for i, fact in enumerate(news_facts, 1):
        news_text += f"\n[Новинарски източник {i}]\nЗаглавие: {fact['title']}\nОткъс:\n{fact['text']}\n"

    wiki_text = ""
    for i, fact in enumerate(wiki_facts, 1):
        wiki_text += f"\n[Wikipedia {i}]\nЗаглавие: {fact['title']}\nОткъс:\n{fact['text']}\n"

    sources_section = ""
    if news_text:
        sources_section += f"\nРелевантни новинарски източници:\n{news_text}"
    if wiki_text:
        sources_section += f"\nРелевантни Wikipedia статии:\n{wiki_text}"
    if not sources_section:
        sources_section = "\nНе са намерени релевантни източници."

    return f"""Ти си асистент за обяснение на класификация на новини на български език.

Класификаторът е дал следния резултат:
- Клас: {model_prediction}
- Вероятност за fake: {fake_probability:.2f}

Входна статия:
{article_text[:2000]}
{sources_section}

Задача:
Напиши кратко обяснение на български език (3–5 изречения).

Обяснението трябва да съдържа:
1. Какво е предсказал моделът.
2. Кои елементи от статията изглеждат подозрителни или достоверни.
3. Дали намерените източници подкрепят, не подкрепят или не са достатъчни за проверка.
4. Кратък финален извод.

Не измисляй факти извън предоставените източници.
Ако източниците не са достатъчни, кажи ясно, че няма достатъчно доказателства."""


def _parse_retry_delay(exc: Exception) -> float | None:
    try:
        match = re.search(r"'retryDelay':\s*'(\d+)s'", str(exc))
        return float(match.group(1)) if match else None
    except Exception:
        return None


_BG_STOPWORDS = {
    "и","в","на","се","за","е","да","от","с","по","не","го","но","до","при",
    "а","или","като","това","той","тя","те","ще","са","има","една","един",
    "едно","всички","също","ако","когато","тъй","защото","след","преди",
    "без","под","над","между","може","трябва","бе","беше","бяха","бил",
    "нова","нов","ново","стар","стара","старо","голям","голяма","голямо",
}

def _extract_wiki_query(text: str, max_words: int = 6) -> str:
    """Extract the most informative words from the article for use as a Wikipedia search query."""
    # Use the first 300 chars — usually the title + first sentence
    snippet = text[:300]
    # Keep only words ≥4 chars that aren't stopwords
    words = re.findall(r'\b[а-яА-Яa-zA-Z]{4,}\b', snippet)
    seen, keywords = set(), []
    for w in words:
        wl = w.lower()
        if wl not in _BG_STOPWORDS and wl not in seen:
            seen.add(wl)
            keywords.append(w)
        if len(keywords) == max_words:
            break
    return " ".join(keywords) if keywords else snippet[:80]


def rag_explain(
    article_text: str,
    model_prediction: str,
    fake_probability: float,
) -> dict:
    """Retrieve facts from ChromaDB + Wikipedia BG, then generate a Gemini explanation."""
    # ChromaDB local news facts
    news_facts = retrieve_relevant_facts(article_text, n_results=3) if _DB_PATH.exists() else []

    # Wikipedia BG on-demand: build a short keyword query from the article
    wiki_query = _extract_wiki_query(article_text)
    wiki_facts = search_wikipedia_bg(wiki_query, n_results=2)

    all_sources = news_facts + wiki_facts

    prompt = _build_prompt(article_text, model_prediction, fake_probability, news_facts, wiki_facts)

    try:
        client = _get_gemini_client()
        response = client.models.generate_content(model=_GEMINI_MODEL, contents=prompt)
        return {
            "rag_explanation": response.text,
            "rag_sources":     all_sources,
            "status":          "ok",
        }
    except Exception as exc:
        retry_delay = _parse_retry_delay(exc)
        if retry_delay:
            status = f"Gemini rate limit — опитайте след {int(retry_delay)} секунди."
        else:
            status = f"Грешка при Gemini: {exc}"
        return {"rag_explanation": None, "rag_sources": all_sources, "status": status}
