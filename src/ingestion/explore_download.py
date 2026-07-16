## This script explores the PGN file to understand the structure and content of the games. 
## It counts the occurrences of header tags, identifies missing important fields, and provides examples of values for key
from collections import Counter, defaultdict
import chess.pgn

PGN_PATH = "data/raw/lichess_db_standard_rated_2026-02.pgn"
N_GAMES = 10_000

tag_counts = Counter()
value_examples = defaultdict(Counter)
missing_counts = Counter()

important_tags = [
    "Event",
    "Site",
    "White",
    "Black",
    "Result",
    "UTCDate",
    "UTCTime",
    "WhiteElo",
    "BlackElo",
    "WhiteRatingDiff",
    "BlackRatingDiff",
    "ECO",
    "Opening",
    "TimeControl",
    "Termination",
]

with open(PGN_PATH, encoding="utf-8", errors="replace") as pgn:
    for i in range(N_GAMES):
        game = chess.pgn.read_game(pgn)

        if game is None:
            break

        headers = game.headers

        for tag in headers:
            tag_counts[tag] += 1

        for tag in important_tags:
            value = headers.get(tag)

            if value is None or value == "":
                missing_counts[tag] += 1
            else:
                value_examples[tag][value] += 1

print(f"Games scanned: {i + 1}")

print("\nHeader tags found:")
for tag, count in tag_counts.most_common():
    print(f"{tag}: {count}")

print("\nMissing important fields:")
for tag in important_tags:
    print(f"{tag}: {missing_counts[tag]} missing")

print("\nExample values:")
for tag in important_tags:
    print(f"\n{tag}:")
    for value, count in value_examples[tag].most_common(10):
        print(f"  {value}: {count}")