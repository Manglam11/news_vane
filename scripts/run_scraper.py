"""The daily job, called by a GitHub Action on a schedule.

It ran green every morning for six days while Sports saved nothing at all, so it
now has to earn the word "success": it reports per section, it checks the harvest
against the contract, and it exits non-zero when it cannot defend what it wrote.
"""

import sys
from datetime import UTC, datetime

from config.settings import settings

from newsvane.data import get_articles
from newsvane.data.scraper import audit_harvest, harvest_counts
from newsvane.storage.repository import save_articles


def main() -> None:
    now = datetime.now(UTC)
    articles = get_articles(source="scraper")

    counts = harvest_counts(articles, settings.scraper_sections)
    poison, alarms = audit_harvest(
        articles,
        settings.scraper_sections,
        now,
        settings.scraper_max_age_hours,
    )

    for topic, count in counts.items():
        print(f"  {topic:<9} {count:>3} scraped")

    if poison:
        for problem in poison:
            print(f"POISON  {problem}")
        # Nothing is written at all. A bad row is permanent and unfindable; a
        # missing day is a gap I can simply scrape again tomorrow.
        print("nothing written -- this harvest cannot defend its own labels")
        sys.exit(1)

    saved = save_articles(articles)
    # The gap between these two numbers is the whole point: on day one they match,
    # and from day two onward the difference is how much of the front page is stale.
    print(f"total {len(articles)} scraped, {saved} new")

    if alarms:
        for problem in alarms:
            print(f"ALARM   {problem}")
        sys.exit(1)


if __name__ == "__main__":
    main()
