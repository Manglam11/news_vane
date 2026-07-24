"""Does the mood reader actually read mood? Measured before I store it.

I run this before adding a database column, because a column is permanent and a
probe is free. Two things I want to see with my own eyes: that VADER's grammar
rules really fire (negation and intensifiers should move the score, not just the
words), and where my own thresholds cut the line.
"""

from newsvane.models.sentiment import read_sentiment

HEADLINES = [
    "Markets surge to record high as investors celebrate strong earnings",
    "Death toll rises after devastating floods destroy thousands of homes",
    "The committee will meet on Thursday to review the proposal",
    "This is not a good result for the team",
    "This is a good result for the team",
    "This is a VERY good result for the team!!!",
]


def main() -> None:
    for headline in HEADLINES:
        mood, compound = read_sentiment(headline)
        print(f"{compound:+.4f}  {mood:<8}  {headline}")


if __name__ == "__main__":
    main()
