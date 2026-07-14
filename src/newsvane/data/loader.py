import html
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from config.settings import settings

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


def get_articles(split: str = "train") -> list[dict]:
    filename = settings.train_file if split == "train" else settings.test_file
    path: Path = settings.raw_data_dir / filename

    frame = pd.read_csv(path).rename(columns=COLUMNS)

    # The bootstrap file carries no publication date. I stamp ingestion time instead —
    # honest about what it is, and the same field the scraper will fill for real in Phase 3.
    ingested_at = datetime.now(UTC)

    return [
        {
            "text": clean_text(f"{row.title} {row.description}"),
            "topic": LABELS[row.class_index],
            "timestamp": ingested_at,
        }
        for row in frame.itertuples(index=False)
    ]
