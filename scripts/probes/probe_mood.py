"""Did the mood actually reach the table? Read back, not assumed.

The job printing a mood proves the ENGINE ran. It says nothing about whether the
column landed, so this reads the rows back through the same repository function
the radar will use. An exit code describes a command; this describes an outcome.
"""

from datetime import UTC, datetime, timedelta

from newsvane.storage.repository import mood_by_day


def main() -> None:
    end = datetime.now(UTC)
    start = end - timedelta(days=7)

    rows = mood_by_day(start, end)
    if not rows:
        print("no row in the last 7 days carries a mood yet")
        return

    for row in rows:
        print(f"{row['day']:%b %d}  {row['topic']:<9} {row['mood']:+.4f}")


if __name__ == "__main__":
    main()
