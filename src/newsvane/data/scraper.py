"""The live DATA source. Section = free label: I ask for the Sports page, so
every article I find there is Sports. I never trust the site's own section
meta-tag -- The Hindu calls its Sci/Tech page 'Technology', and my model has
no such class.

Fetching and parsing are deliberately separate. Fetching touches the network;
parsing is pure. Only the pure half can be tested honestly.
"""

import time
from datetime import datetime

import httpx
from lxml import html

from config.settings import settings

# The Hindu marks every real story with an article id + .ece. Anything without
# one is navigation, not news.
ARTICLE_HREF = "/article"

# Live-score pages match the article pattern but are scoreboards, not prose.
# Feeding "Full Time MEX 2 RSA 0" to a text classifier is feeding it noise.
JUNK_TOKENS = ("live-score", "/live/", "live-updates")


def _meta(tree: html.HtmlElement, prop: str) -> str | None:
    values = tree.xpath(f'//meta[@property="{prop}"]/@content')
    return values[0].strip() if values else None


def find_article_urls(page_html: str, limit: int) -> list[str]:
    """Pure: given a section listing page, pull out the story links."""
    tree = html.fromstring(page_html)
    urls: list[str] = []

    for href in tree.xpath("//a/@href"):
        if ARTICLE_HREF not in href or not href.endswith(".ece"):
            continue
        if any(token in href for token in JUNK_TOKENS):
            continue
        # A card links to the same story twice (image + headline). De-dupe by URL.
        if href not in urls:
            urls.append(href)
        if len(urls) >= limit:
            break

    return urls


def parse_article(page_html: str, topic: str) -> dict | None:
    """Pure: given one story page, lift the three fields the DATA contract owes."""
    tree = html.fromstring(page_html)

    title = _meta(tree, "og:title")
    description = _meta(tree, "og:description")
    published = _meta(tree, "article:published_time")

    # A story missing any of the three is unusable. I skip it rather than invent
    # a value -- a fabricated timestamp would silently corrupt the trend line.
    if not (title and description and published):
        return None

    return {
        # AG News trained the model on title + description joined. I serve it the
        # exact same shape, or the model sees a distribution it never learned.
        "text": f"{title}. {description}",
        "topic": topic,
        "timestamp": datetime.fromisoformat(published),
    }


def scrape_articles() -> list[dict]:
    """The live half of the DATA contract: [{text, topic, timestamp}, ...]."""
    articles: list[dict] = []

    with httpx.Client(
        headers={"User-Agent": settings.scraper_user_agent},
        follow_redirects=True,
        timeout=settings.scraper_timeout,
    ) as client:
        for topic, section_url in settings.scraper_sections.items():
            listing = client.get(section_url)
            listing.raise_for_status()

            for url in find_article_urls(listing.text, settings.scraper_limit):
                response = client.get(url)
                if response.status_code == 200:
                    article = parse_article(response.text, topic)
                    if article is not None:
                        articles.append(article)
                # I am a guest on someone else's server. One request, one pause.
                time.sleep(settings.scraper_delay)

    return articles