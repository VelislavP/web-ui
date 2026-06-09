import plotly.graph_objects as go
import streamlit as st
from urllib.parse import urlparse

from classifier import classify
from domain_checker import check_domain
from explainer import explain
from scraper import scrape_article
from trust_score import calculate_domain_score, calculate_trust_score

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="VerifyBG", layout="wide")

# Material Icons — open source (Apache 2.0), loaded via <link> which Streamlit allows
st.markdown(
    '<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">',
    unsafe_allow_html=True,
)

def _mi(name: str, size: int = 22, color: str = "white") -> str:
    """Render a Material Icons Round glyph as inline HTML."""
    return (
        f'<span class="material-icons-round" '
        f'style="font-size:{size}px;line-height:1;vertical-align:middle;color:{color};">'
        f'{name}</span>'
    )

_SCORE_COLORS = {"green": "#2ecc71", "yellow": "#f39c12", "red": "#e74c3c"}

_THREAT_LABELS = {
    "MALWARE":                         "Зловреден софтуер",
    "SOCIAL_ENGINEERING":              "Социално инженерство (фишинг)",
    "UNWANTED_SOFTWARE":               "Нежелан софтуер",
    "POTENTIALLY_HARMFUL_APPLICATION": "Потенциално вредно приложение",
}
_PLATFORM_LABELS = {
    "ANY_PLATFORM": "Всички платформи",
    "WINDOWS": "Windows", "LINUX": "Linux", "OSX": "macOS",
    "ANDROID": "Android", "IOS": "iOS", "CHROME": "Chrome",
}

# ── APP HEADER ────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="display:flex;align-items:center;gap:14px;padding-bottom:16px;">'
    '<div style="width:52px;height:52px;border-radius:14px;flex-shrink:0;'
    'background:linear-gradient(135deg,#11998e,#38ef7d);'
    f'display:flex;align-items:center;justify-content:center;">{_mi("policy", 28)}</div>'
    '<div>'
    '<div style="font-size:24px;font-weight:800;line-height:1.1;">VerifyBG</div>'
    '<div style="font-size:13px;color:#888;margin-top:2px;">Анализ на новини</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

tab_url, tab_text = st.tabs(["URL анализ", "Текстов анализ"])


# ── HTML / PLOTLY HELPERS ─────────────────────────────────────────────────────

def _gauge_chart(score: int, color: str) -> go.Figure:
    fig = go.Figure(go.Pie(
        values=[score, max(100 - score, 0)],
        hole=0.72,
        marker_colors=[color, "rgba(128,128,128,0.15)"],
        textinfo="none",
        showlegend=False,
        hoverinfo="skip",
        direction="clockwise",
        rotation=90,
    ))
    fig.update_layout(
        height=200, width=200,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(text=f"<b>{score}</b>",  x=0.5, y=0.55, showarrow=False,
                 font=dict(size=40, color=color)),
            dict(text="от 100",            x=0.5, y=0.35, showarrow=False,
                 font=dict(size=13, color="#888")),
        ],
    )
    return fig


def _signal_card(icon: str, icon_bg: str, label: str, value: str, pct: int, color: str) -> str:
    """icon is a Material Icons Round glyph name, e.g. 'psychology'."""
    card  = ("display:flex;align-items:center;gap:14px;padding:14px 16px;"
             "border-radius:12px;background:rgba(255,255,255,.05);"
             "border:1px solid rgba(255,255,255,.09);margin-bottom:10px;")
    icn   = (f"width:44px;height:44px;border-radius:10px;background:{icon_bg};"
             "display:flex;align-items:center;justify-content:center;flex-shrink:0;")
    label_s = "font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.5px;"
    value_s = f"font-size:15px;font-weight:600;color:{color};margin-top:3px;"
    bar_bg  = "height:6px;border-radius:3px;background:rgba(255,255,255,.1);overflow:hidden;"
    bar_f   = f"height:100%;border-radius:3px;background:{color};width:{pct}%;"
    pct_s   = f"font-size:15px;font-weight:700;color:{color};width:44px;text-align:right;flex-shrink:0;"
    return (
        f'<div style="{card}">'
        f'<div style="{icn}">{_mi(icon, 22)}</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="{label_s}">{label}</div>'
        f'<div style="{value_s}">{value}</div>'
        f'</div>'
        f'<div style="width:110px;flex-shrink:0;">'
        f'<div style="{bar_bg}"><div style="{bar_f}"></div></div>'
        f'</div>'
        f'<div style="{pct_s}">{pct}%</div>'
        f'</div>'
    )


def _lime_chips(features: list) -> str:
    common = ("padding:6px 13px;border-radius:20px;font-size:13px;font-weight:600;"
              "display:inline-flex;align-items:center;gap:5px;")
    styles = {
        "fake":    "background:rgba(231,76,60,.22);color:#e74c3c;border:1px solid rgba(231,76,60,.4);",
        "real":    "background:rgba(46,204,113,.18);color:#2ecc71;border:1px solid rgba(46,204,113,.35);",
        "neutral": "background:rgba(255,255,255,.07);color:#999;border:1px solid rgba(255,255,255,.12);",
    }
    items = []
    for word, weight in features:
        key = "fake" if weight > 0.05 else ("real" if weight < -0.05 else "neutral")
        items.append(f'<span style="{common}{styles[key]}">{word} {weight:+.2f}</span>')
    return f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:4px;">{"".join(items)}</div>'


def _dom_card(label: str, value: str, sub: str = "", val_color: str = "") -> str:
    cs = "padding:14px 16px;border-radius:10px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);"
    ls = "font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;"
    vs = f"font-size:15px;font-weight:700;{'color:' + val_color + ';' if val_color else ''}"
    ss = f'<div style="font-size:12px;color:#888;margin-top:4px;">{sub}</div>' if sub else ""
    return (f'<div style="{cs}"><div style="{ls}">{label}</div>'
            f'<div style="{vs}">{value}</div>{ss}</div>')


def _sect_hdr(icon: str, text: str) -> str:
    """icon is a Material Icons Round glyph name."""
    return (
        f'<div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px;'
        f'margin:22px 0 10px;display:flex;align-items:center;gap:8px;">'
        f'{_mi(icon, 16, "#888")} {text}</div>'
    )


def _ded_html(deductions: list) -> str:
    return "".join(
        f'<div style="font-size:13px;color:#e74c3c;margin:5px 0;display:flex;align-items:center;gap:6px;">'
        f'{_mi("warning", 16, "#e74c3c")} {d["reason"]}: <strong>−{d["points"]}%</strong></div>'
        for d in deductions
    )


# ── COMPONENT RENDERERS ───────────────────────────────────────────────────────

def _render_gauge(score: int, verdict: str, verdict_color: str, caption: str = "") -> None:
    color = _SCORE_COLORS[verdict_color]
    st.plotly_chart(
        _gauge_chart(score, color),
        use_container_width=False,
        config={"displayModeBar": False, "staticPlot": True},
    )
    if verdict_color == "green":
        st.success(verdict)
    elif verdict_color == "yellow":
        st.warning(verdict)
    else:
        st.error(verdict)
    if caption:
        st.caption(caption)


def _render_signal_section(trust: dict, ml_result: dict | None, domain_info: dict | None) -> None:
    cards = []

    # Weights: ML=70, SafeBrowsing=20, Domain=10
    if ml_result:
        fake        = ml_result["label"] == "Fake"
        conf        = ml_result["confidence"]
        color       = "#e74c3c" if fake else "#2ecc71"
        bg          = "rgba(231,76,60,.28)" if fake else "rgba(46,204,113,.25)"
        content_pct = trust.get("content_score", 0)
        content_pts = round(content_pct * 0.70)
        cards.append(_signal_card(
            "psychology", bg, "ML Съдържание · макс 70т",
            f"{'Фалшива' if fake else 'Достоверна'} ({conf:.0%}) · {content_pts} / 70т",
            content_pct, color,
        ))

    if domain_info:
        sb         = domain_info["safe_browsing"]
        is_safe    = sb["is_safe"]
        n          = len(sb.get("matches", []))
        color      = "#2ecc71" if is_safe else "#e74c3c"
        bg         = "rgba(46,204,113,.22)" if is_safe else "rgba(231,76,60,.25)"
        safety_pct = trust.get("safety_score", 100)
        safety_pts = round(safety_pct * 0.20)
        cards.append(_signal_card(
            "security", bg, "Safe Browsing · макс 20т",
            f"{'Чист' if is_safe else f'{n} заплаха/и'} · {safety_pts} / 20т",
            safety_pct, color,
        ))
        age_days = domain_info.get("domain_age_days")
        age_val  = f"{age_days:,} дни" if age_days else "Без данни"
        age_clr  = "#3d9df5"
    else:
        cards.append(_signal_card(
            "security", "rgba(128,128,128,.18)", "Safe Browsing · макс 20т", "Без URL · 0 / 20т", 0, "#888",
        ))
        age_val, age_clr = "Без URL", "#888"

    age_pct = trust.get("age_score", 50)
    age_pts = round(age_pct * 0.10)
    cards.append(_signal_card(
        "language", "rgba(61,157,245,.22)", "Домейн възраст · макс 10т",
        f"{age_val} · {age_pts} / 10т", age_pct, age_clr,
    ))

    st.markdown("".join(cards), unsafe_allow_html=True)

    deductions = trust.get("deductions", [])
    if deductions:
        st.markdown(_ded_html(deductions), unsafe_allow_html=True)


def _render_domain_section(domain_info: dict) -> None:
    age_days = domain_info.get("domain_age_days")
    if age_days:
        yrs     = age_days // 365
        age_val = f"{yrs} {'година' if yrs == 1 else 'години'}"
        age_sub = f"Рег. {domain_info.get('creation_date', '—')}"
    else:
        age_val, age_sub = "Непознат", ""

    privacy   = domain_info.get("privacy_protected", False)
    registrar = (domain_info.get("registrar") or "—")[:30]
    country   = domain_info.get("country", "")

    strip = (
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:4px; margin-bottom:16px;">'
        + _dom_card("САЙТ", domain_info["domain"], f"Държава: {country}" if country else "")
        + _dom_card("ВЪЗРАСТ", age_val, age_sub)
        + _dom_card("РЕГИСТРАТОР", registrar)
        + _dom_card("ПОВЕРИТЕЛНОСТ",
                    f'{_mi("warning", 14, "#e74c3c")} Скрит' if privacy else f'{_mi("check_circle", 14, "#2ecc71")} Публичен',
                    domain_info.get("org") or "",
                    "#e74c3c" if privacy else "#2ecc71")
        + "</div>"
    )
    st.markdown(strip, unsafe_allow_html=True)

    with st.expander("Пълен WHOIS + Safe Browsing отчет"):
        col_w, col_sb = st.columns(2)
        with col_w:
            st.markdown("**— WHOIS —**")
            for lbl, val in [
                ("Домейн",           f"`{domain_info['domain']}`"),
                ("Държава",          domain_info.get("country")),
                ("Организация",      domain_info.get("org")),
                ("Регистратор",      domain_info.get("registrar")),
                ("WHOIS сървър",     f"`{domain_info['whois_server']}`" if domain_info.get("whois_server") else None),
                ("Регистриран",      domain_info.get("creation_date")),
                ("Изтича",           domain_info.get("expiration_date")),
                ("Последна промяна", domain_info.get("updated_date")),
                ("Възраст",          f"{domain_info['domain_age_days']:,} дни" if domain_info.get("domain_age_days") else None),
                ("DNSSEC",           domain_info.get("dnssec")),
                ("Поверителност",    "Скрит регистрант" if privacy else "Публичен"),
            ]:
                if val:
                    st.markdown(f"**{lbl}:** {val}")
            if domain_info.get("name_servers"):
                st.markdown("**Name servers:**\n" + "\n".join(f"- `{ns}`" for ns in domain_info["name_servers"]))
            if domain_info.get("status"):
                raw = domain_info["status"]
                st.markdown("**Статус:**\n" + (
                    "\n".join(f"- `{s}`" for s in raw) if isinstance(raw, list) else f"`{raw}`"
                ))

        with col_sb:
            st.markdown("**— Google Safe Browsing v4 —**")
            sb = domain_info["safe_browsing"]
            if sb.get("stub"):
                st.caption("Stub — SAFE_BROWSING_API_KEY не е зададен")
            checked = sb.get("checked_types", [])
            if checked:
                st.markdown("**Проверени категории:**")
                for t in checked:
                    st.markdown(f"- {_THREAT_LABELS.get(t, t)}")
            st.markdown("---")
            matches = sb.get("matches", [])
            if sb["is_safe"]:
                st.success("Не са открити заплахи")
            else:
                st.error(f"Открити са {len(matches)} заплаха/и")
                for i, m in enumerate(matches, 1):
                    tt  = m.get("threatType", "—")
                    plt = m.get("platformType", "—")
                    st.markdown(f"**Заплаха #{i}**")
                    st.markdown(
                        f"| Поле | Стойност |\n|---|---|\n"
                        f"| Тип | {_THREAT_LABELS.get(tt, tt)} |\n"
                        f"| Платформа | {_PLATFORM_LABELS.get(plt, plt)} |\n"
                        f"| Вид запис | `{m.get('threatEntryType','—')}` |\n"
                        f"| URL | `{m.get('threat',{}).get('url','—')}` |\n"
                        f"| Кеш | {m.get('cacheDuration','—')} |"
                    )
    st.markdown('<div style="margin-bottom:32px;"></div>', unsafe_allow_html=True)


def _render_results(
    ml_result:      dict,
    explain_result: dict,
    trust:          dict,
    domain_info:    dict | None,
    article:        dict | None = None,
    url:            str  | None = None,
) -> None:
    # Expandable article header — clicking reveals the full text
    if article and url:
        host        = urlparse(url).netloc.lstrip("www.")
        title_short = (article.get("title") or "")[:80]
        words       = len(article.get("text", "").split())
        with st.expander(f"**{host}** · {title_short}… · {words:,} думи"):
            st.markdown(article["text"])

    st.divider()

    # AI explanation (shown before score so reasoning leads the verdict)
    if explain_result.get("explanation"):
        st.markdown(_sect_hdr("lightbulb", "AI Обяснение"), unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(explain_result["explanation"])
            st.caption("Stub · ще бъде заменен с Gemini API + LIME контекст")

    # LIME chips
    features = explain_result.get("top_features", [])
    if features:
        st.markdown(_sect_hdr("insights", "Ключови индикатори (LIME)"), unsafe_allow_html=True)
        st.markdown(_lime_chips(features), unsafe_allow_html=True)

    st.markdown(_sect_hdr("query_stats", "Оценка"), unsafe_allow_html=True)
    col_gauge, col_signals = st.columns([1, 2])
    with col_gauge:
        _render_gauge(
            trust["trust_score"], trust["verdict"], trust["verdict_color"],
            caption="" if domain_info else "Без домейн информация",
        )
    with col_signals:
        _render_signal_section(trust, ml_result, domain_info)

    # Domain strip
    if domain_info:
        st.markdown(_sect_hdr("language", "Домейн"), unsafe_allow_html=True)
        _render_domain_section(domain_info)


def _render_domain_only(domain_info: dict, ds: dict) -> None:
    st.divider()
    st.markdown(
        f'<div style="display:inline-flex;align-items:center;gap:6px;padding:5px 12px;'
        f'border-radius:20px;background:rgba(243,156,18,.15);border:1px solid rgba(243,156,18,.4);'
        f'font-size:12px;font-weight:600;color:#f39c12;margin-bottom:10px;">'
        f'{_mi("warning", 14, "#f39c12")} Само URL/домейн оценка · без AI анализ на съдържание'
        f'</div>',
        unsafe_allow_html=True,
    )
    col_gauge, col_signals = st.columns([1, 2])
    with col_gauge:
        _render_gauge(
            ds["domain_score"], ds["verdict"], ds["verdict_color"],
            caption="Резултатът отразява само домейн сигнали",
        )
    with col_signals:
        _render_signal_section(ds, None, domain_info)

    st.markdown(_sect_hdr("language", "Домейн"), unsafe_allow_html=True)
    _render_domain_section(domain_info)


# ── URL TAB ───────────────────────────────────────────────────────────────────
with tab_url:
    url_input   = st.text_input("Въведете URL на статия", placeholder="https://example.bg/news/article")
    analyze_url = st.button("Анализирай", key="btn_url")

    if analyze_url and url_input.strip():
        url = url_input.strip()
        _parsed = urlparse(url)
        if _parsed.scheme not in ("http", "https") or not _parsed.netloc:
            st.error("Въведете валиден URL, започващ с http:// или https://")
        else:
            domain_info = None
            with st.spinner("Проверка на домейна..."):
                try:
                    domain_info = check_domain(url)
                except Exception as exc:
                    st.warning(f"Не успях да проверя домейна: {exc}")

            article = None
            with st.spinner("Извличане на текст..."):
                try:
                    article = scrape_article(url)
                except Exception as exc:
                    st.error(f"Грешка при извличане: {exc}")

            if article is None:
                st.markdown(
                    f'<div style="display:flex;align-items:flex-start;gap:14px;padding:16px 20px;'
                    f'border-radius:12px;background:rgba(231,76,60,.12);'
                    f'border:1px solid rgba(231,76,60,.45);margin:12px 0;">'
                    f'<div style="flex-shrink:0;margin-top:1px;">{_mi("error_outline", 22, "#e74c3c")}</div>'
                    f'<div>'
                    f'<div style="font-size:14px;font-weight:700;color:#e74c3c;margin-bottom:4px;">'
                    f'Неуспешно извличане на текст</div>'
                    f'<div style="font-size:13px;color:#ccc;line-height:1.5;">'
                    f'Не успях да прочета съдържанието на страницата. '
                    f'Резултатите по-долу се базират <strong style="color:#f39c12;">само на URL и домейн данни</strong> '
                    f'— без AI анализ на съдържанието.<br>'
                    f'За пълна оценка отидете в <strong>Текстов анализ</strong> и поставете текста ръчно.'
                    f'</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if domain_info is not None:
                    try:
                        _render_domain_only(domain_info, calculate_domain_score(domain_info))
                    except Exception as exc:
                        st.error(f"Грешка при изчисляване: {exc}")
            else:
                with st.spinner("Анализиране..."):
                    ml_result = explain_result = trust = None

                    try:
                        ml_result = classify(article["text"])
                    except Exception as exc:
                        st.error(f"Грешка при класификация: {exc}")

                    if ml_result:
                        try:
                            explain_result = explain(article["text"], ml_result["label"], ml_result["confidence"])
                        except Exception as exc:
                            st.warning(f"Не успях да генерирам обяснение: {exc}")
                            explain_result = {"top_features": [], "explanation": "", "lime_html": None}

                        try:
                            trust = calculate_trust_score(ml_result, domain_info)
                        except Exception as exc:
                            st.error(f"Грешка при изчисляване на оценката: {exc}")

                        if trust and explain_result:
                            _render_results(ml_result, explain_result, trust, domain_info, article, url)


# ── TEXT TAB ──────────────────────────────────────────────────────────────────
with tab_text:
    text_input   = st.text_area("Поставете текст на статия", height=250,
                                placeholder="Поставете текста на статията тук...")
    analyze_text = st.button("Анализирай", key="btn_text")

    if analyze_text and text_input.strip():
        text = text_input.strip()
        with st.spinner("Анализиране..."):
            ml_result = explain_result = trust = None

            try:
                ml_result = classify(text)
            except Exception as exc:
                st.error(f"Грешка при класификация: {exc}")

            if ml_result:
                try:
                    explain_result = explain(text, ml_result["label"], ml_result["confidence"])
                except Exception as exc:
                    st.warning(f"Не успях да генерирам обяснение: {exc}")
                    explain_result = {"top_features": [], "explanation": "", "lime_html": None}

                try:
                    trust = calculate_trust_score(ml_result, None)
                except Exception as exc:
                    st.error(f"Грешка при изчисляване на оценката: {exc}")

                if trust and explain_result:
                    _render_results(ml_result, explain_result, trust, None)
