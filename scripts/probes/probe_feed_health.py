"""A dry run before two gates land. I measure what The Hindu's four section pages
actually offer me right now -- how many links point outside the section I asked
for, and how old the survivors are -- so I size both rules against real numbers
instead of a guess.

The first run capped out at 40 candidates on Sports, which means I measured the
cap and not the page. I now pull a deep pool on purpose: the live scraper's
limit of 15 is a SAVE limit, and reusing it as a LOOK limit is the bug I am here
to measure.

This probe writes nothing. It only looks. And unlike the scraper it is allowed to
lose a request without losing the run.

    uv run python -m scripts.probes.probe_feed_health          # all four
    uv run python -m scripts.probes.probe_feed_health Sports   # one section
"""

import sys
import time
from collections import Counter
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx
from config.settings import settings
from lxml import html

from newsvane.data.scraper import ARTICLE_HREF, JUNK_TOKENS, parse_article

# Deep enough to see past the junk. The question is not "what is in the top 15"
# but "how many good stories were sitting behind the junk all along".
RAW_LIMIT = 150

# Opening an article page costs a request and a polite pause, so I still cap it.
FETCH_LIMIT = 30


def _get(client: httpx.Client, url: str) -> httpx.Response | None:
    # A timeout is an answer too. I record it and keep walking rather than let
    # one slow page end the measurement.
    try:
        return client.get(url)
    except httpx.HTTPError as exc:
        print(f"    [skipped] {type(exc).__name__}: {urlparse(url).path}")
        return None


def raw_candidates(page_html: str, limit: int) -> list[str]:
    # Deliberately NOT find_article_urls(). That function already applies the
    # junk rule, and I am here to measure the ground, not the current rule's
    # opinion of the ground.
    tree = html.fromstring(page_html)
    urls: list[str] = []
    for href in tree.xpath("//a/@href"):
        if ARTICLE_HREF not in href or not href.endswith(".ece"):
            continue
        if href not in urls:
            urls.append(href)
        if len(urls) >= limit:
            break
    return urls


def age_bucket(hours: float) -> str:
    if hours < 6:
        return "<6h"
    if hours < 24:
        return "6-24h"
    if hours < 72:
        return "1-3d"
    return ">3d"


def measure_section(client: httpx.Client, topic: str, section_url: str, now: datetime) -> None:
    # The section path comes from the URL I already ask for. One source of truth --
    # a second settings key would be a copy waiting to drift.
    section_path = urlparse(section_url).path
    print(f"\n=== {topic}  (section path: {section_path}) ===")

    listing = _get(client, section_url)
    if listing is None or listing.status_code != 200:
        print("  listing unreachable -- no measurement for this section")
        return

    candidates = raw_candidates(listing.text, RAW_LIMIT)

    junk = [u for u in candidates if any(t in u for t in JUNK_TOKENS)]
    off_section = [
        u for u in candidates if u not in junk and not urlparse(u).path.startswith(section_path)
    ]
    survivors = [u for u in candidates if u not in junk and u not in off_section]

    # If this equals RAW_LIMIT I measured my own cap again, not the page.
    print(f"  raw candidates      : {len(candidates)}  (cap {RAW_LIMIT})")
    print(f"  cut by JUNK_TOKENS  : {len(junk)}")
    print(f"  cut by section path : {len(off_section)}")
    print(f"  survivors           : {len(survivors)}")

    ages: list[float] = []
    fresh_urls: list[str] = []
    unusable = 0
    lost = 0
    for url in survivors[:FETCH_LIMIT]:
        response = _get(client, url)
        if response is None:
            lost += 1
        elif response.status_code == 200:
            article = parse_article(response.text, topic)
            if article is None:
                unusable += 1
            else:
                published = article["timestamp"]
                if published.tzinfo is None:
                    published = published.replace(tzinfo=UTC)
                hours = (now - published).total_seconds() / 3600
                ages.append(hours)
                if hours < 72:
                    fresh_urls.append(url)
        time.sleep(settings.scraper_delay)

    attempted = min(len(survivors), FETCH_LIMIT)
    print(f"  opened              : {attempted}  (unusable meta: {unusable}, lost: {lost})")

    if ages:
        spread = Counter(age_bucket(h) for h in ages)
        line = "   ".join(f"{b}: {spread.get(b, 0)}" for b in ("<6h", "6-24h", "1-3d", ">3d"))
        print(f"  age of survivors    : {line}")
        print(f"  oldest survivor     : {max(ages):.1f}h")
        # This is the number both gates get sized against: what a day would
        # actually save if BOTH rules were live right now.
        print(f"  WOULD SAVE (<72h)   : {len(fresh_urls)}  (limit is {settings.scraper_limit})")
        for u in fresh_urls[:3]:
            print(f"    {urlparse(u).path}")
    else:
        print("  age of survivors    : NONE -- this section would land zero rows")


def main() -> None:
    now = datetime.now(UTC)
    wanted = sys.argv[1] if len(sys.argv) > 1 else None

    with httpx.Client(
        headers={"User-Agent": settings.scraper_user_agent},
        follow_redirects=True,
        timeout=settings.scraper_timeout,
    ) as client:
        for topic, section_url in settings.scraper_sections.items():
            if wanted and topic.lower() != wanted.lower():
                continue
            measure_section(client, topic, section_url, now)


if __name__ == "__main__":
    main()
