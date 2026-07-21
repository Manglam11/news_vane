"""I test the scraper's parsing, never its fetching. A test that calls the live
site fails when my wifi drops and changes when The Hindu edits a headline --
that is not a test, that is a weather report.

Two of these tests exist because a green suite was proving nothing. The scraper
ran clean for six days while saving more than half its rows under the wrong
label and none at all for Sports. Neither fault raised an exception, so only a
test that asserts the RULE can catch them.
"""

from datetime import UTC, datetime, timedelta

from newsvane.data.scraper import find_article_urls, is_fresh, parse_article, section_path

SECTION = "/news/international/"

LISTING_HTML = """
<html><body>
  <a href="https://x.com/news/international/one/article111.ece">One</a>
  <a href="https://x.com/news/international/one/article111.ece">One again</a>
  <a href="https://x.com/sport/football/live-score-mex-rsa/article222.ece">Score</a>
  <a href="https://x.com/news/international/">Section nav</a>
  <a href="https://x.com/sci-tech/technology/france-bill/article444.ece">Cross-promo</a>
  <a href="/news/international/three/article555.ece">Relative</a>
  <a href="https://x.com/news/international/two/article333.ece">Two</a>
</body></html>
"""

ARTICLE_HTML = """
<html><head>
  <meta property="og:title" content="Apple sues OpenAI" />
  <meta property="og:description" content="A trade secrets case." />
  <meta property="article:published_time" content="2026-07-11T04:31:58+05:30" />
  <meta property="article:section" content="Technology" />
</head><body></body></html>
"""

ARTICLE_HTML_NO_DATE = """
<html><head>
  <meta property="og:title" content="Apple sues OpenAI" />
  <meta property="og:description" content="A trade secrets case." />
</head><body></body></html>
"""

ARTICLE_HTML_NAIVE_DATE = """
<html><head>
  <meta property="og:title" content="Rupee steadies" />
  <meta property="og:description" content="Six paise up in early trade." />
  <meta property="article:published_time" content="2026-07-11T04:31:58" />
</head><body></body></html>
"""


def _article(hours_old: float, now: datetime) -> dict:
    return {
        "text": "Something happened. And then some detail.",
        "topic": "Sports",
        "timestamp": now - timedelta(hours=hours_old),
    }


def test_section_path_comes_from_the_url_i_ask_for():
    # One source of truth. A second settings key holding "/sport/" would be a copy
    # of this fact, and copies drift apart.
    assert section_path("https://www.thehindu.com/sport/") == "/sport/"


def test_listing_keeps_only_unique_real_articles():
    urls = find_article_urls(LISTING_HTML, SECTION, limit=10)

    # Gone: the duplicate card link, the live-score page, the nav link, and the
    # cross-promoted Sci/Tech story. Kept: both absolute stories and the relative one.
    assert urls == [
        "https://x.com/news/international/one/article111.ece",
        "/news/international/three/article555.ece",
        "https://x.com/news/international/two/article333.ece",
    ]


def test_listing_drops_a_story_from_another_section():
    # The one that mattered most. This link is a perfectly real article -- it is not
    # junk, it is not a duplicate, it just belongs to Sci/Tech. On the live site more
    # than half of every section page looked like this, and every one was being
    # saved as World, Business or Sci/Tech regardless of where it came from.
    urls = find_article_urls(LISTING_HTML, SECTION, limit=10)

    assert "https://x.com/sci-tech/technology/france-bill/article444.ece" not in urls


def test_listing_matches_a_relative_href_too():
    # The Hindu mixes absolute and relative hrefs on the same page. urlparse gives
    # me a path either way; a naive startswith() on the raw href would have thrown
    # away every relative link and quietly halved the harvest.
    urls = find_article_urls(LISTING_HTML, SECTION, limit=10)

    assert "/news/international/three/article555.ece" in urls


def test_listing_respects_the_limit():
    assert len(find_article_urls(LISTING_HTML, SECTION, limit=1)) == 1


def test_article_emits_the_data_contract():
    article = parse_article(ARTICLE_HTML, topic="Sci/Tech")

    assert article["text"] == "Apple sues OpenAI. A trade secrets case."
    # The page says its section is "Technology". I ignore that: the section I asked
    # for is the label, because that is the only label my model can predict.
    assert article["topic"] == "Sci/Tech"
    assert article["timestamp"] == datetime.fromisoformat("2026-07-11T04:31:58+05:30")


def test_article_without_a_timestamp_is_dropped():
    assert parse_article(ARTICLE_HTML_NO_DATE, topic="World") is None


def test_a_naive_timestamp_is_given_a_zone():
    # The freshness gate subtracts this from an aware "now". One naive timestamp
    # would raise TypeError mid-run and take the whole harvest down with it.
    article = parse_article(ARTICLE_HTML_NAIVE_DATE, topic="Business")

    assert article["timestamp"].tzinfo is not None


def test_a_recent_story_is_fresh():
    now = datetime.now(UTC)

    assert is_fresh(_article(hours_old=5, now=now), now, max_age_hours=72.0) is True


def test_an_archive_story_is_not_fresh():
    # The Sports jam, in one assertion. A live-blog thread carries its kickoff time
    # as published_time, so the oldest survivor on that page was 853 hours -- 35 days
    # -- old. It fails here on age, whatever its URL happens to be called.
    now = datetime.now(UTC)

    assert is_fresh(_article(hours_old=853, now=now), now, max_age_hours=72.0) is False


def test_the_freshness_boundary_is_inclusive():
    # A story exactly at the limit is still news. I pick a side deliberately rather
    # than leaving it to whichever comparison operator I typed first.
    now = datetime.now(UTC)

    assert is_fresh(_article(hours_old=72, now=now), now, max_age_hours=72.0) is True
