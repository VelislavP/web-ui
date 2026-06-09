import os
import requests
from datetime import datetime, timezone
from urllib.parse import urlparse

import whois
from dotenv import load_dotenv

load_dotenv()

_SAFE_BROWSING_KEY = os.environ.get("SAFE_BROWSING_API_KEY")
_SAFE_BROWSING_ENDPOINT = (
    "https://safebrowsing.googleapis.com/v4/threatMatches:find"
)
_CHECKED_TYPES = ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"]


def _check_safe_browsing(url: str) -> dict:
    if not _SAFE_BROWSING_KEY:
        return {
            "is_safe": True,
            "matches": [],
            "checked_types": _CHECKED_TYPES,
            "stub": True,
        }
    payload = {
        "client": {"clientId": "fake-news-bg", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": _CHECKED_TYPES,
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    resp = requests.post(
        _SAFE_BROWSING_ENDPOINT,
        params={"key": _SAFE_BROWSING_KEY},
        json=payload,
        timeout=5,
    )
    resp.raise_for_status()
    matches = resp.json().get("matches", [])
    # Each match: threatType, platformType, threatEntryType, threat.url, cacheDuration
    return {
        "is_safe": len(matches) == 0,
        "matches": matches,
        "checked_types": _CHECKED_TYPES,
        "stub": False,
    }


def check_domain(url: str) -> dict:
    """Return WHOIS info and Safe Browsing result for the domain in url."""
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path
    domain = domain.lstrip("www.")

    creation_date = None
    expiration_date = None
    updated_date = None
    domain_age_days = None
    registrar = None
    whois_server = None
    org = None
    country = None
    name_servers: list[str] = []
    status: list[str] = []
    dnssec = None
    privacy_protected = False
    raw_whois_text = None

    try:
        info = whois.whois(domain)

        def _first_date(val):
            if isinstance(val, list):
                val = val[0]
            if val is not None and val.tzinfo is None:
                val = val.replace(tzinfo=timezone.utc)
            return val

        raw_creation = _first_date(info.creation_date)
        if raw_creation is not None:
            creation_date = str(raw_creation.date())
            domain_age_days = (datetime.now(timezone.utc) - raw_creation).days

        raw_exp = _first_date(info.expiration_date)
        if raw_exp is not None:
            expiration_date = str(raw_exp.date())

        raw_upd = _first_date(info.updated_date)
        if raw_upd is not None:
            updated_date = str(raw_upd.date())

        registrar = info.registrar or None
        whois_server = info.whois_server or None
        org = info.org or None
        country = info.country or None
        dnssec = info.dnssec or None

        ns = info.name_servers
        if isinstance(ns, list):
            name_servers = sorted({s.lower() for s in ns if s})
        elif isinstance(ns, str):
            name_servers = [ns.lower()]

        st = info.status
        if isinstance(st, list):
            status = st
        elif isinstance(st, str):
            status = [st]

        # A domain is privacy-protected when the registrant contact contains
        # common privacy-proxy keywords instead of a real name/org.
        raw_whois_text = str(info)
        privacy_keywords = ["privacy", "redacted", "protected", "proxy", "withheld"]
        privacy_protected = any(kw in raw_whois_text.lower() for kw in privacy_keywords)

    except Exception:
        pass

    try:
        safe_browsing = _check_safe_browsing(url)
    except Exception:
        safe_browsing = {
            "is_safe": True,
            "matches": [],
            "checked_types": _CHECKED_TYPES,
            "stub": True,
        }

    return {
        "domain": domain,
        "creation_date": creation_date,
        "expiration_date": expiration_date,
        "updated_date": updated_date,
        "domain_age_days": domain_age_days,
        "registrar": registrar,
        "whois_server": whois_server,
        "org": org,
        "country": country,
        "name_servers": name_servers,
        "status": status,
        "dnssec": dnssec,
        "privacy_protected": privacy_protected,
        "safe_browsing": safe_browsing,
    }
