"""Is 'section = free label' actually true? Two faults surfaced on the sport page:
long-lived FIFA match blogs slip past JUNK_TOKENS, and the listing carries links
into OTHER sections entirely -- police, pharma, agriculture -- which my scraper
would happily label Sports.

Labels are the one thing I cannot afford to get wrong, so this probe rehearses the
two rules I intend to add -- require the link to live under the section path, and
reject match blogs -- and reports what each section would actually yield afterwards.
A rule that starves a section is worse than the bug it fixes, so I measure first.
"""

import httpx
from config.settings import settings
from lxml import html

from newsvane.data.scraper import ARTICLE_HREF, JUNK_TOKENS

# The candidate additions, kept here rather than in the scraper until they earn it.
CANDIDATE_JUNK = ("live-match-updates", "live-updates", "live-blog")


def section_path(section_url: str) -> str:
    """The path I asked for, e.g. '/sport/' -- a real story there lives under it."""
    return "/" + section_url.split("//", 1)[1].split("/", 1)[1]


def slug(href: str) -> str:
    parts = href.rstrip("/").split("/")
    return parts[-2] if len(parts) >= 2 else parts[-1]


def main() -> None:
    with httpx.Client(
        headers={"User-Agent": settings.scraper_user_agent},
        follow_redirects=True,
        timeout=settings.scraper_timeout,
    ) as client:
        for topic, section_url in settings.scraper_sections.items():
            listing = client.get(section_url)
            listing.raise_for_status()

            path = section_path(section_url)
            tree = html.fromstring(listing.text)

            kept: list[str] = []
            for href in tree.xpath("//a/@href"):
                if ARTICLE_HREF not in href or not href.endswith(".ece"):
                    continue
                if any(token in href for token in JUNK_TOKENS + CANDIDATE_JUNK):
                    continue
                # The new rule: the story's own URL must sit under the section I asked
                # for. Without this the label comes from which page I visited, not from
                # where the story actually lives.
                if path not in href:
                    continue
                if href not in kept:
                    kept.append(href)

            taken = kept[: settings.scraper_limit]
            print(f"\n=== {topic}  ({path}) ===")
            print(
                f"  would keep: {len(kept)}   would take: {len(taken)} / {settings.scraper_limit}"
            )
            for href in taken[:5]:
                print(f"      {slug(href)}")


if __name__ == "__main__":
    main()
