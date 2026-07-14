"""Second probe. The listing page gives me a URL and a headline — not enough.
AG News trained the model on TITLE + DESCRIPTION, so I must serve it the same
shape or I get train/serve skew. And Phase 4's radar is worthless without a real
publication timestamp. Both should live in the article page's meta tags."""

import httpx
from lxml import html

# One article per section, taken straight from the B1 probe output.
SAMPLES = {
    "World": "https://www.thehindu.com/news/international/us-immigration-agents-involved-in-another-fatal-shooting/article71219833.ece",
    "Business": "https://www.thehindu.com/business/petroleum-ministry-responds-to-the-hindu-editorial-higher-russian-oil-imports-part-of-deliberate-strategy/article71218350.ece",
    "Sci/Tech": "https://www.thehindu.com/sci-tech/technology/apple-sues-openai-for-stealing-trade-secrets/article71208364.ece",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


def probe(label: str, url: str) -> None:
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=20.0) as client:
        response = client.get(url)

    tree = html.fromstring(response.text)
    print(f"\n=== {label} :: status {response.status_code}")

    # Dump every meta tag that could plausibly carry a title, a summary or a date.
    # I want to SEE the real field names, not guess them.
    for meta in tree.xpath("//meta"):
        key = meta.get("property") or meta.get("name") or meta.get("itemprop")
        content = (meta.get("content") or "").strip()
        if not key or not content:
            continue
        if any(
            token in key.lower()
            for token in ("title", "description", "date", "time", "publish", "section")
        ):
            print(f"  {key:<28} : {content[:110]}")

    # JSON-LD is the machine-readable block search engines eat. If a datePublished
    # lives anywhere, it lives here.
    for script in tree.xpath('//script[@type="application/ld+json"]'):
        blob = (script.text or "")[:400].replace("\n", " ")
        print(f"  [ld+json] {blob}")


if __name__ == "__main__":
    for label, url in SAMPLES.items():
        probe(label, url)