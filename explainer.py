from lime.lime_text import LimeTextExplainer

from classifier import clean_text, predict_proba

_explainer = LimeTextExplainer(class_names=["credible", "fake"])


def explain(text: str, label: str, confidence: float) -> dict:
    """Run LIME and return top word features with weights plus a Bulgarian summary."""
    cleaned = clean_text(text)

    exp = _explainer.explain_instance(
        cleaned,
        predict_proba,
        num_features=10,
        labels=[1],
        num_samples=300,
    )

    features = [(str(w), round(float(v), 4)) for w, v in exp.as_list(label=1)]

    return {
        "top_features": features,
        "explanation": _build_explanation(label, confidence, features),
        "lime_html": None,
    }


def _build_explanation(label: str, confidence: float, features: list) -> str:
    fake_words = [w for w, v in features if v > 0.05]
    real_words = [w for w, v in features if v < -0.05]

    if label == "Fake":
        indicators = (
            ", ".join(f'„{w}"' for w in fake_words[:3])
            if fake_words
            else "LIME не откри ясни текстови индикатори (всички тегла < 0.05)"
        )
        return (
            f"Статията показва признаци на невярна информация с увереност {confidence:.0%}. "
            f"Ключови думи, насочващи към фалшиво съдържание: {indicators}."
        )
    else:
        indicators = (
            ", ".join(f'„{w}"' for w in real_words[:3])
            if real_words
            else "LIME не откри ясни текстови индикатори (всички тегла < 0.05)"
        )
        return (
            f"Статията показва признаци на достоверно съдържание с увереност {confidence:.0%}. "
            f"Индикатори за достоверност: {indicators}."
        )
