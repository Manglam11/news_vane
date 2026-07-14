"""I test the scraper's parsing, never its fetching. A test that calls the live
site fails when my wifi drops and changes when The Hindu edits a headline --
that is not a test, that is a weather report.
"""

from datetime import datetime

from newsvane.data.scraper import find_article_urls, parse_article

LISTING_HTML = """
<html><body>
  <a href="https://x.com/news/international/one/article111.ece">One</a>
  <a href="https://x.com/news/international/one/article111.ece">One again</a>
  <a href="https://x.com/sport/football/live-score-mex-rsa/article222.ece">Score</a>
  <a href="https://x.com/news/international/">Section nav</a>
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


def test_listing_keeps_only_unique_real_articles():
    urls = find_article_urls(LISTING_HTML, limit=10)

    # The duplicate card link, the live-score page and the nav link are all gone.
    assert urls == [
        "https://x.com/news/international/one/article111.ece",
        "https://x.com/news/international/two/article333.ece",
    ]


def test_listing_respects_the_limit():
    assert len(find_article_urls(LISTING_HTML, limit=1)) == 1


def test_article_emits_the_data_contract():
    article = parse_article(ARTICLE_HTML, topic="Sci/Tech")

    assert article["text"] == "Apple sues OpenAI. A trade secrets case."
    # The page says its section is "Technology". I ignore that: the section I asked
    # for is the label, because that is the only label my model can predict.
    assert article["topic"] == "Sci/Tech"
    assert article["timestamp"] == datetime.fromisoformat("2026-07-11T04:31:58+05:30")


def test_article_without_a_timestamp_is_dropped():
    assert parse_article(ARTICLE_HTML_NO_DATE, topic="World") is None