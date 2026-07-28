"""Standalone entry point for the multiprocessing-based parsers.

Run as a plain subprocess -- NOT called directly from inside a running
Jupyter kernel. Doing the parallel parsing in-kernel was found to crash the
kernel outright: multiprocessing.Pool's worker processes, combined with the
Jupyter kernel's own ZMQ-based communication channels, exhausted a Windows
networking resource ("No buffer space available", a WSAENOBUFS error deep in
libzmq's socket polling) partway through a real 15-million-game run, tearing
down the kernel and every worker with it. Running this as an ordinary
standalone process means the heavy Pool never coexists with a live kernel,
which sidesteps that interaction entirely. See the notebook's Section 1
methodology note for the full story.

Usage:
  py run_parse.py random <pgn_path> <output_parquet_path> <n_games> <n_workers> <random_seed>
  py run_parse.py full   <pgn_path> <output_parquet_path> <n_workers>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.ingestion.pgn_parser import parallel_parse_pgn, random_sample_parse_pgn


def main():
    mode, pgn_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    if mode == 'random':
        n_games, n_workers, random_seed = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        df = random_sample_parse_pgn(pgn_path, n_games=n_games, n_workers=n_workers,
                                      random_seed=random_seed)
    elif mode == 'full':
        n_workers = int(sys.argv[4])
        df = parallel_parse_pgn(pgn_path, n_workers=n_workers)
    else:
        raise ValueError(f"Unknown mode '{mode}' -- expected 'random' or 'full'")

    df.to_parquet(output_path, index=False)
    print(f'Saved {len(df):,} games to {output_path}', flush=True)


if __name__ == '__main__':
    main()
