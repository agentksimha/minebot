import feedparser
from datetime import datetime
from dateutil import parser

RSS_SOURCES = [
    "https://www.miningweekly.com/page/rss",
    "https://www.mining-technology.com/feed/",
    "https://www.mining.com/feed/",
]

def fetch_dgms_updates(limit: int = 5):

    for src in RSS_SOURCES:
        try:
            feed = feedparser.parse(src)

            if not feed.entries:
                continue

            updates = []

            for entry in feed.entries[:limit]:

                title = entry.get("title", "Untitled")
                link = entry.get("link", "")

                pub_date = entry.get("published", "Unknown")

                try:
                    pub_date = parser.parse(pub_date).date().isoformat()
                except:
                    pass

                updates.append({
                    "title": title,
                    "link": link,
                    "published": pub_date
                })

            if updates:
                return updates

        except Exception as e:
            print(f"⚠️ RSS source failed ({src}): {e}")

    return [{
        "title": "No updates available",
        "link": "",
        "published": ""
    }]




