def _deduct(reason: str, loss: float) -> dict | None:
    pts = round(loss * 100)
    return {"reason": reason, "points": pts} if pts > 0 else None


def calculate_domain_score(domain_info: dict) -> dict:
    """Score a domain using only Safe Browsing + age + privacy signals (no ML)."""
    is_safe = domain_info["safe_browsing"]["is_safe"]
    age_days = domain_info.get("domain_age_days")
    privacy = domain_info.get("privacy_protected", False)

    safety_score = 1.0 if is_safe else 0.0
    age_score = min(age_days / 180, 1.0) if age_days is not None else 0.5
    privacy_penalty = 0.10 if privacy else 0.0

    final = max(0.0, 0.65 * safety_score + 0.35 * age_score - privacy_penalty)

    deductions = []
    if not is_safe:
        d = _deduct("Открити заплахи в Safe Browsing", (1.0 - safety_score) * 0.65)
        if d:
            deductions.append(d)
    if age_score < 1.0:
        label = "Непознат домейн (няма WHOIS дата)" if age_days is None else "Млад домейн (под 180 дни)"
        d = _deduct(label, (1.0 - age_score) * 0.35)
        if d:
            deductions.append(d)
    if privacy:
        d = _deduct("Скрит регистрант (WHOIS поверителност)", privacy_penalty)
        if d:
            deductions.append(d)

    if final >= 0.75:
        verdict = "Домейнът изглежда надежден"
        verdict_color = "green"
    elif final >= 0.45:
        verdict = "Домейнът изисква внимание"
        verdict_color = "yellow"
    else:
        verdict = "Домейнът е подозрителен"
        verdict_color = "red"

    return {
        "domain_score": round(final * 100),
        "safety_score": round(safety_score * 100),
        "age_score": round(age_score * 100),
        "privacy_penalty": round(privacy_penalty * 100),
        "deductions": deductions,
        "verdict": verdict,
        "verdict_color": verdict_color,
    }


def calculate_trust_score(ml_result: dict, domain_info: dict | None) -> dict:
    """Combine ML, Safe Browsing, and domain-age signals into a 0-100 trust score.

    When domain_info is None (text-only tab) the score is purely ML-based (100% weight).
    When domain_info is provided (URL tab) weights are ML 70% + SafeBrowsing 20% + Age 10%.
    """
    label = ml_result["label"]
    confidence = ml_result["confidence"]

    content_score = confidence if label == "Real" else (1.0 - confidence)

    deductions = []

    if domain_info is not None:
        is_safe = domain_info["safe_browsing"]["is_safe"]
        age_days = domain_info.get("domain_age_days")
        safety_score = 1.0 if is_safe else 0.0
        age_score = min(age_days / 180, 1.0) if age_days is not None else 0.5
        privacy_penalty = 0.05 if domain_info.get("privacy_protected") else 0.0

        final = max(0.0, 0.70 * content_score + 0.20 * safety_score + 0.10 * age_score - privacy_penalty)

        if label == "Fake":
            d = _deduct(f"ML: класифицирана като фалшива ({confidence:.0%} увереност)", (1.0 - content_score) * 0.70)
            if d:
                deductions.append(d)
        if not is_safe:
            d = _deduct("Открити заплахи в Safe Browsing", (1.0 - safety_score) * 0.20)
            if d:
                deductions.append(d)
        if age_score < 1.0:
            lbl = "Непознат домейн (няма WHOIS дата)" if age_days is None else "Млад домейн (под 180 дни)"
            d = _deduct(lbl, (1.0 - age_score) * 0.10)
            if d:
                deductions.append(d)
        if privacy_penalty > 0:
            d = _deduct("Скрит регистрант (WHOIS поверителност)", privacy_penalty)
            if d:
                deductions.append(d)
    else:
        # Text-only: no URL to check — score is purely ML
        is_safe = True
        age_days = None
        safety_score = 0
        age_score = 0
        privacy_penalty = 0.0

        final = content_score

        if label == "Fake":
            d = _deduct(f"ML: класифицирана като фалшива ({confidence:.0%} увереност)", 1.0 - content_score)
            if d:
                deductions.append(d)

    if final >= 0.75:
        verdict = "Вероятно достоверна"
        verdict_color = "green"
    elif final >= 0.45:
        verdict = "Проверете допълнително"
        verdict_color = "yellow"
    else:
        verdict = "Вероятно фалшива"
        verdict_color = "red"

    return {
        "trust_score": round(final * 100),
        "content_score": round(content_score * 100),
        "safety_score": round(safety_score * 100),
        "age_score": round(age_score * 100),
        "privacy_penalty": round(privacy_penalty * 100),
        "deductions": deductions,
        "verdict": verdict,
        "verdict_color": verdict_color,
    }
