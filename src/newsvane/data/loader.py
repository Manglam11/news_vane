import html
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from config.settings import settings
from newsvane.data.scraper import scrape_articles

# AG News stores the topic as an integer. The meaning lives in a separate classes.txt,
# so I verified this mapping against real headlines before trusting it.
LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

# The raw file's column names are not mine to live with. I translate them once, here,
# and the rest of the codebase never sees "Class Index" again.
COLUMNS = {"Class Index": "class_index", "Title": "title", "Description": "description"}


def clean_text(raw: str) -> str:
    # AG News was scraped from raw HTML in 2004 and never decoded. Three scars are left:
    # escaped newlines welded into words, HTML entities, and entities whose "&" was stripped.
    raw = raw.replace("\\", " ")
    raw = re.sub(r"#(\d+);", r"&#\1;", raw)
    raw = html.unescape(raw)
    return re.sub(r"\s+", " ", raw).strip()


def load_kaggle(split: str = "train") -> list[dict]:
    """The bootstrap source: a frozen CSV. Perfect for training, blind to today."""
    filename = settings.train_file if split == "train" else settings.test_file
    path: Path = settings.raw_data_dir / filename

    frame = pd.read_csv(path).rename(columns=COLUMNS)

    # The bootstrap file carries no publication date. I stamp ingestion time instead --
    # honest about what it is, and the field the scraper now fills for real.
    ingested_at = datetime.now(UTC)

    return [
        {
            "text": clean_text(f"{row.title} {row.description}"),
            "topic": LABELS[row.class_index],
            "timestamp": ingested_at,
        }
        for row in frame.itertuples(index=False)
    ]


# Every source is a function with the same signature and the same return shape. Adding a
# third source (another newspaper, an RSS feed) means adding a key here -- nothing else.
SOURCES = {
    "kaggle": load_kaggle,
    "scraper": lambda split="live": scrape_articles(),
}


def get_articles(source: str = "kaggle", split: str = "train") -> list[dict]:
    """The DATA box's one and only door: [{text, topic, timestamp}, ...].

    Downstream asks for articles. It is not told, and must never care, whether they
    came from a 2004 CSV or from a live newspaper five seconds ago.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}. Known: {sorted(SOURCES)}")

    return SOURCES[source](split)