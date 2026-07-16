import pathlib
import zstandard

RAW_DIR = pathlib.Path("data/raw")
CHUNK_SIZE = 1 << 22  # 4 MB


def decompress_zst(src: pathlib.Path, dst: pathlib.Path) -> None:
    dctx = zstandard.ZstdDecompressor()
    total_bytes = 0
    with src.open("rb") as ifh, dst.open("wb") as ofh:
        with dctx.stream_reader(ifh) as reader:
            while True:
                chunk = reader.read(CHUNK_SIZE)
                if not chunk:
                    break
                ofh.write(chunk)
                total_bytes += len(chunk)
                print(f"\r  {total_bytes / 1_073_741_824:.2f} GB written", end="", flush=True)
    print()


if __name__ == "__main__":
    for zst_path in sorted(RAW_DIR.glob("*.zst")):
        out_path = zst_path.with_suffix("")  # strips .zst → .pgn
        if out_path.exists():
            print(f"Skipping {zst_path.name} — {out_path.name} already exists")
            continue
        print(f"Decompressing {zst_path.name} → {out_path.name}")
        decompress_zst(zst_path, out_path)
        print(f"Done: {out_path}")