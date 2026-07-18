"""Throwaway probe. Before I write a scraper, I need proof of three things:
the section URL returns 200, the HTML is server-rendered (article links are
already in the raw bytes, no JavaScript needed), and I can spot a headline."""

import re

import httpx
from lxml import html

# Section name on the left is the AG News label the scraper will hand downstream.
SECTIONS = {
    "World": "https://www.thehindu.com/news/international/",
    "Sports": "https://www.thehindu.com/sport/",
    "Business": "https://www.thehindu.com/business/",
    "Sci/Tech": "https://www.thehindu.com/sci-tech/",
}

# A polite, real-looking identity. Many sites reject the default python UA outright.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}

# The Hindu's article URLs end in an article id + .ece. If I find these in the
# RAW html, the page is static and lxml is enough. If I find zero, it's JS-built
# and httpx+lxml will never see the content.
ARTICLE_RE = re.compile(r"/article\d+\.ece")


def probe(label: str, url: str) -> None:
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=20.0) as client:
        response = client.get(url)

    print(f"\n=== {label} :: {url}")
    print(f"status      : {response.status_code}")
    print(f"bytes       : {len(response.content):,}")

    if response.status_code != 200:
        print("!! non-200, this section is not usable as-is")
        return

    tree = html.fromstring(response.text)
    anchors = tree.xpath("//a[@href]")
    article_links = [a for a in anchors if ARTICLE_RE.search(a.get("href", ""))]

    # De-duplicate, because a card links to the same story from image + title.
    unique = {a.get("href"): (a.text_content() or "").strip() for a in article_links}

    print(f"article links: {len(unique)} unique (out of {len(anchors)} total anchors)")

    if not unique:
        print("!! zero article links in the raw HTML -> page is JS-rendered")
        return

    for href, text in list(unique.items())[:5]:
        print(f"  - {text[:70]!r}\n    {href}")


if __name__ == "__main__":
    for label, url in SECTIONS.items():
        probe(label, url)
