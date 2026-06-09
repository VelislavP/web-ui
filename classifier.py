# STUB — replace function body with XLM-RoBERTa inference when model is ready.
#
# Real implementation will:
#   1. Load a fine-tuned XLM-RoBERTa model from a saved checkpoint
#   2. Tokenize the input text with its tokenizer
#   3. Run a forward pass and read logits
#   4. Return the same dict structure below

_FAKE_KEYWORDS = ["ШОКИРАЩО", "сензация", "няма да повярвате", "тайната", "скрита истина"]


def classify(text: str) -> dict:
    """Return label ('Fake'|'Real') and confidence (0.0–1.0)."""
    if any(kw in text for kw in _FAKE_KEYWORDS):
        return {"label": "Fake", "confidence": 0.89}
    return {"label": "Real", "confidence": 0.76}
