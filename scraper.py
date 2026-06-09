from newspaper import Article


def _parse(url: str, input_html: str | None = None) -> dict | None:
    article = Article(url, language="bg")
    if input_html is not None:
        article.download(input_html=input_html)
    else:
        article.download()
    article.parse()
    text = article.text
    if len(text) < 100:
        return None
    publish_date = None
    if article.publish_date is not None:
        publish_date = str(article.publish_date.date())
    return {
        "title": article.title or "",
        "text": text,
        "authors": article.authors or [],
        "publish_date": publish_date,
        "source_url": url,
    }


def _cloudscraper_html(url: str) -> str | None:
    """Fetch raw HTML via cloudscraper, which solves Cloudflare JS challenges."""
    try:
        import cloudscraper
        session = cloudscraper.create_scraper()
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def scrape_article(url: str) -> dict | None:
    # Attempt 1: standard newspaper download (fast, works for most sites)
    try:
        result = _parse(url)
        if result:
            return result
    except Exception:
        pass

    # Attempt 2: Cloudflare bypass — re-feed the HTML into newspaper
    html = _cloudscraper_html(url)
    if html:
        try:
            return _parse(url, input_html=html)
        except Exception:
            pass

    return None
