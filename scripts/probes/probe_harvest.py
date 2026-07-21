"""Is 'section = free label' actually true? Two faults surfaced on the sport page:
long-lived FIFA match blogs slip past JUNK_TOKENS, and the listing carries links
into OTHER sections entirely -- police, pharma, agriculture -- which my scraper
would happily label Sports.

Labels are the one thing I cannot afford to get wrong, so this probe rehearsed the
two rules I was considering and reported what each section would yield afterwards.
A rule that starves a section is worse than the bug it fixes, so I measured first.

The outcome, recorded here because a probe is evidence and evidence should say how
the argument ended:

  - The section-path rule SHIPPED. It was cutting 50-63% of candidates on three of
    the four sections -- far more than the 13% I had estimated by eye.
  - CANDIDATE_JUNK did NOT ship. Extending a slug blocklist only ever loses to the
    next slug ('live-updates', then 'live-match-updates', then whatever is invented
    next). I filter on the DEFECT instead: a match blog carries its kickoff as
    published_time, so a freshness gate catches every one of them by age. It is
    kept below only to show what I compared against.
"""

import httpx
from config.settings import settings
from lxml import html

from newsvane.data.scraper import ARTICLE_HREF, JUNK_TOKENS, section_path

# The rejected candidate, kept as the losing side of the comparison.
CANDIDATE_JUNK = ("live-match-updates", "live-updates", "live-blog")


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
