import random

# STUB — replace function body with LIME/SHAP + Gemini API when ready.
#
# Real implementation will:
#   1. Run LIME (or SHAP) on the classifier to get actual token/word importances
#   2. Build the LIME HTML visualization and store in lime_html
#   3. Query ChromaDB for relevant RAG context (optional)
#   4. Send features + label + confidence to Gemini API for a Bulgarian explanation
#   5. Return the same dict structure below

_FAKE_EXPLANATION = (
    "Статията показва признаци на невярна информация. "
    "Използвани са емоционално натоварени изрази като \"{w1}\" и \"{w2}\", "
    "характерни за сензационно съдържание. "
    "Липсват конкретни източници и цитати."
)

_REAL_EXPLANATION = (
    "Статията показва признаци на достоверно съдържание. "
    "Тонът е неутрален, използвани са конкретни факти "
    "и заглавието съответства на съдържанието."
)


def explain(text: str, label: str, confidence: float) -> dict:
    """Return top word features with weights and a Bulgarian explanation string."""
    words = [w for w in text.split() if len(w) > 4]
    sample = random.sample(words, min(8, len(words))) if words else ["дума"] * 8

    features = [(w, round(random.uniform(-0.2, 0.2), 4)) for w in sample]
    features.sort(key=lambda x: abs(x[1]), reverse=True)

    if label == "Fake" and len(features) >= 2:
        explanation = _FAKE_EXPLANATION.format(w1=features[0][0], w2=features[1][0])
    elif label == "Fake":
        explanation = _FAKE_EXPLANATION.format(w1="сензация", w2="шок")
    else:
        explanation = _REAL_EXPLANATION

    return {
        "top_features": features,
        "explanation": explanation,
        "lime_html": None,
    }
