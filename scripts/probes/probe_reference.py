"""Derive the drift reference: the topic-mix the model was trained on.

Drift asks "has live news moved away from what the model learned?". That needs a
frozen yardstick -- the training split's own topic proportions. I derive them here
ONCE, from the same DATA door the model trained through, then freeze the numbers in
settings. I never read this at runtime: data/raw/ is git-ignored and gone at serve
and CI time, so a runtime read would crash exactly when drift matters most.
"""

from collections import Counter

from config.settings import settings

from newsvane.data.loader import get_articles

TOPICS = list(settings.scraper_sections)


def main() -> None:
    articles = get_articles("kaggle", "train")
    counts = Counter(a["topic"] for a in articles)
    total = sum(counts.values())

    print(f"Training articles: {total}")
    print("Reference topic-mix (paste these into settings.drift_reference):")
    for topic in TOPICS:
        proportion = counts[topic] / total
        print(f'    "{topic}": {proportion:.4f},')


if __name__ == "__main__":
    main()
