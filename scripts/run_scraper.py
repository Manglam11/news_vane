"""The daily job, called by a GitHub Action on a schedule.

It ran green every morning for six days while Sports saved nothing at all, so it
now has to earn the word "success": it reports per section, it checks the harvest
against the contract, and it exits non-zero when it cannot defend what it wrote.

It also marks the model against every row it keeps and records how each story
reads. The section is ground truth for the topic, the model never sees it, and
the two answers land side by side in the same row -- which is what gives drift
something to measure that is not just my own quota. Mood has no ground truth at
all, so it is reported and stored, never marked.
"""

import sys
from datetime import UTC, datetime

from config.settings import settings

from newsvane.data import get_articles
from newsvane.data.scraper import audit_harvest, harvest_counts
from newsvane.models.batch import agreement, classify_articles, mood_by_section
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

    # Only rows that survived the poison gate are worth asking about. Classifying
    # before that check would spend the model on rows I am about to throw away.
    articles, failures = classify_articles(articles)

    # An unclassified row is MISSING, not WRONG -- and unlike a bad label it stays
    # findable forever as predicted_label IS NULL. Alarm, therefore, never poison.
    alarms += failures

    agreed, scored = agreement(articles)
    if scored:
        print(f"model agreed with {agreed}/{scored} sections ({agreed / scored:.1%})")

    # The mood reading gets printed for the same reason the counts do: a column
    # nobody ever looks at is a column that can quietly fill with zeros. If the
    # lexicon ever stops firing, this block is where I see it the next morning.
    for topic, (mood, read) in mood_by_section(articles, settings.scraper_sections).items():
        reading = f"{mood:+.3f}" if mood is not None else "   --  "
        print(f"  {topic:<9} mood {reading} over {read:>3} read")

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
