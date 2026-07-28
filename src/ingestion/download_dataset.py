"""Download a monthly Lichess standard-rated games dump (compressed .zst)
from database.lichess.org, straight to data/raw/ -- meant to run on whatever
machine will actually do the parsing (e.g. a rented VM), not to push the
197GB+ decompressed file over a home connection.

Resumable via HTTP Range requests (the server advertises Accept-Ranges:
bytes) -- safe to re-run after an interrupted download; it picks up where it
left off instead of restarting.

After downloading, decompress.py turns the .zst into the .pgn the notebook
expects.
"""
import pathlib
import sys
import urllib.error
import urllib.request

RAW_DIR = pathlib.Path("data/raw")
BASE_URL = "https://database.lichess.org/standard/{filename}"
CHUNK_SIZE = 1 << 22  # 4 MB
DEFAULT_FILENAME = "lichess_db_standard_rated_2026-02.pgn.zst"


def download_with_resume(filename: str, dest_dir: pathlib.Path = RAW_DIR) -> pathlib.Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    resume_from = dest_path.stat().st_size if dest_path.exists() else 0

    req = urllib.request.Request(BASE_URL.format(filename=filename))
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")

    try:
        response_ctx = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            # Requested range starts at/past the server's total length --
            # already fully downloaded (e.g. re-running after a completed run).
            print(f"{dest_path} is already complete ({resume_from / 1e9:.2f} GB); nothing to do.")
            return dest_path
        raise

    with response_ctx as resp:
        resumed = resp.status == 206
        if resume_from and not resumed:
            # Server ignored the Range request (returned 200, full content) --
            # resuming would corrupt the file, so start over instead.
            print("Server did not honor resume request; restarting download from byte 0.")
            resume_from = 0

        content_length = resp.length  # bytes remaining in THIS response
        total = (resume_from + content_length) if content_length is not None else None
        mode = "ab" if resumed else "wb"

        downloaded = resume_from
        with dest_path.open(mode) as f:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
                else:
                    print(f"\r  {downloaded / 1e9:.2f} GB", end="", flush=True)
    print()
    return dest_path


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILENAME
    print(f"Downloading {filename} -> {RAW_DIR / filename}")
    path = download_with_resume(filename)
    print(f"Done: {path}  ({path.stat().st_size / 1e9:.2f} GB)")
