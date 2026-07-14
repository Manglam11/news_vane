"""The daily job. A GitHub Action will call this on a schedule in B5, so it must be
a plain module entry point that says loudly what it did and exits.
"""

from newsvane.data import get_articles
from newsvane.storage.repository import save_articles


def main() -> None:
    articles = get_articles(source="scraper")
    saved = save_articles(articles)

    # The gap between these two numbers is the whole point: on day one they match,
    # and from day two onward the difference is how much of the front page is stale.
    print(f"scraped {len(articles)} articles, saved {saved} new")


if __name__ == "__main__":
    main()