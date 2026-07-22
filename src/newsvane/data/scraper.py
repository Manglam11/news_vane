"""The live DATA source. Section = free label: I ask for the Sports page, so
every article I find there is Sports. I never trust the site's own section
meta-tag -- The Hindu calls its Sci/Tech page 'Technology', and my model has
no such class.

Two rules keep that promise honest, and both exist because the rows proved I
needed them:

  1. A link only counts if its own path sits UNDER the section path I asked for.
     The Hindu's section pages carry heavy cross-promotion -- more than half the
     story links on /business/ pointed at /news/national/ or /sci-tech/. Every
     one of those was being saved with the wrong label.

  2. A story older than a few days is archive, not news. This is what actually
     catches live blogs: a match thread carries its kickoff as published_time,
     so it fails on age whatever its URL happens to be called. Blocking slugs by
     name loses to whoever invents the next slug.

Fetching and parsing are deliberately separate. Fetching touches the network;
parsing is pure. Only the pure half can be tested honestly.

A rule I cannot check is a rule I am only hoping for, so the harvest audits
itself before anyone stores it -- see audit_harvest at the bottom of this file.
"""

import time
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx
from config.settings import settings
from lxml import html

# The Hindu marks every real story with an article id + .ece. Anything without
# one is navigation, not news.
ARTICLE_HREF = "/article"

# A cheap pre-filter, not the real defence. It costs nothing and saves me opening
# a scoreboard page; rule 2 is what actually holds the line.
JUNK_TOKENS = ("live-score", "/live/", "live-updates")


def _meta(tree: html.HtmlElement, prop: str) -> str | None:
    values = tree.xpath(f'//meta[@property="{prop}"]/@content')
    return values[0].strip() if values else None


def section_path(section_url: str) -> str:
    """The path a story must sit under to earn this section's label."""
    # Derived from the URL I already ask for. A second settings key would be a
    # copy of this fact, and copies drift apart.
    return urlparse(section_url).path


def find_article_urls(page_html: str, path_prefix: str, limit: int) -> list[str]:
    """Pure: given a section listing page, pull out the story links I trust."""
    tree = html.fromstring(page_html)
    urls: list[str] = []

    for href in tree.xpath("//a/@href"):
        if ARTICLE_HREF not in href or not href.endswith(".ece"):
            continue
        if any(token in href for token in JUNK_TOKENS):
            continue
        # Rule 1. A relative href has no host, so urlparse still gives me the path.
        if not urlparse(href).path.startswith(path_prefix):
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

    timestamp = datetime.fromisoformat(published)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    return {
        # AG News trained the model on title + description joined. I serve it the
        # exact same shape, or the model sees a distribution it never learned.
        "text": f"{title}. {description}",
        "topic": topic,
        "timestamp": timestamp,
    }


def is_fresh(article: dict, now: datetime, max_age_hours: float) -> bool:
    """Pure: rule 2. Is this story news, or is it archive?"""
    age_hours = (now - article["timestamp"]).total_seconds() / 3600
    return age_hours <= max_age_hours


def scrape_articles() -> list[dict]:
    """The live half of the DATA contract: [{text, topic, timestamp}, ...]."""
    articles: list[dict] = []
    now = datetime.now(UTC)

    with httpx.Client(
        headers={"User-Agent": settings.scraper_user_agent},
        follow_redirects=True,
        timeout=settings.scraper_timeout,
    ) as client:
        for topic, section_url in settings.scraper_sections.items():
            # One slow page must not cost me the sections I have not reached yet.
            # Before this, a single ReadTimeout ended the entire day's harvest.
            try:
                listing = client.get(section_url)
                listing.raise_for_status()
            except httpx.HTTPError:
                continue

            kept = 0
            # I look wide and save narrow. These are two different numbers and
            # collapsing them into one is what starved Sports for six days.
            for url in find_article_urls(
                listing.text,
                section_path(section_url),
                settings.scraper_candidate_limit,
            ):
                if kept >= settings.scraper_limit:
                    break

                try:
                    response = client.get(url)
                except httpx.HTTPError:
                    continue
                finally:
                    # I am a guest on someone else's server. One request, one pause.
                    time.sleep(settings.scraper_delay)

                if response.status_code != 200:
                    continue

                article = parse_article(response.text, topic)
                if article is None:
                    continue
                if not is_fresh(article, now, settings.scraper_max_age_hours):
                    continue

                articles.append(article)
                kept += 1

    return articles


def harvest_counts(articles: list[dict], sections: Iterable[str]) -> dict[str, int]:
    """Pure: rows per section, INCLUDING the sections that produced none.

    A total is a number that can hide a zero. `saved 58 new` closed a phase while
    Sports sat at nothing for six days, so every count I report is broken down by
    the dimension the failure lives in.
    """
    counts = dict.fromkeys(sections, 0)
    found = Counter(article["topic"] for article in articles)
    for topic, count in found.items():
        # An unexpected label is counted rather than dropped -- a breakdown that
        # silently omits a row is the same disease as a total that hides a zero.
        counts[topic] = counts.get(topic, 0) + count
    return counts


def audit_harvest(
    articles: list[dict],
    sections: Iterable[str],
    now: datetime,
    max_age_hours: float,
) -> tuple[list[str], list[str]]:
    """Pure: what is wrong with this harvest? Returns (poison, alarms).

    POISON is a row I must never store. A wrong label or a stale timestamp lands
    in the table permanently and bends every trend and drift number computed after
    it -- and without a source URL I cannot even find it again to delete it.

    An ALARM is a harvest that is missing, not wrong. Nothing to clean up, but the
    job must still go red: a section at zero is precisely the failure that ran
    green every morning for six days.
    """
    known = set(sections)
    poison: list[str] = []
    alarms: list[str] = []

    for article in articles:
        topic = article["topic"]
        if topic not in known:
            poison.append(f"label '{topic}' is not a class the model can predict")
        if not is_fresh(article, now, max_age_hours):
            age_hours = (now - article["timestamp"]).total_seconds() / 3600
            poison.append(f"{topic}: an article {age_hours:.0f}h old passed the freshness gate")

    counts = harvest_counts(articles, sections)
    for section in sections:
        if counts[section] == 0:
            alarms.append(f"{section}: 0 articles -- this section harvested nothing")

    return poison, alarms
