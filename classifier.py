import re

import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_HF_REPO = "velislavp/ml-bert-new-detect-bg"
# Best threshold from notebook threshold sweep (maximises F1 on test set)
FAKE_THRESHOLD = 0.31

_tokenizer = None
_model = None
_device = None


def _load():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(_HF_REPO)
    _model = AutoModelForSequenceClassification.from_pretrained(_HF_REPO)
    _model.to(_device)
    _model.eval()


def clean_text(text: str) -> str:
    """Mirrors the preprocessing used during training."""
    text = text.lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'\b\S+\.(bg|com|net|org|eu|info)\b', ' ', text)
    text = re.sub(r'\bhtml\b', ' ', text)
    text = re.sub(r'\bd0\b', ' ', text)
    for term in ["petel", "bradva", "skafeto", "fakti", "pik", "blitz"]:
        text = re.sub(rf'\b{term}\b', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def predict_proba(texts: list) -> np.ndarray:
    """Return softmax probabilities shape (n, 2): col-0=credible, col-1=fake.

    Uses max_length=256 — intentional memory/speed optimisation for LIME's
    100 perturbation calls (mirrors notebook cell-27).
    """
    _load()
    all_probs = []
    for i in range(0, len(texts), 4):
        batch = [clean_text(t) for t in texts[i : i + 4]]
        enc = _tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(_device) for k, v in enc.items()}
        with torch.no_grad():
            logits = _model(**enc).logits
        all_probs.append(softmax(logits.cpu().numpy(), axis=1))
        if _device == "cuda":
            torch.cuda.empty_cache()
    return np.vstack(all_probs)


def classify(text: str) -> dict:
    """Return label ('Fake'|'Real') and confidence probability (0.0–1.0).

    Uses max_length=512 to match the training tokenisation (notebook cell-6).
    """
    _load()
    enc = _tokenizer(
        clean_text(text),
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )
    enc = {k: v.to(_device) for k, v in enc.items()}
    with torch.no_grad():
        logits = _model(**enc).logits
    raw_logits = logits.cpu().numpy()[0]
    probs = softmax(raw_logits)
    p_fake = float(probs[1])
    p_real = float(probs[0])
    label = "Fake" if p_fake >= FAKE_THRESHOLD else "Real"
    confidence = p_fake if label == "Fake" else p_real
    return {
        "label": label,
        "confidence": confidence,
        "p_fake": p_fake,
        "p_real": p_real,
        "logit_real": float(raw_logits[0]),
        "logit_fake": float(raw_logits[1]),
        "threshold": FAKE_THRESHOLD,
    }
