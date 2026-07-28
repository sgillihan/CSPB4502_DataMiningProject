"""Parallel PGN parsing.

Defined in a real, importable module rather than inline in a notebook cell
because multiprocessing.Pool on Windows uses the 'spawn' start method, which
re-imports worker functions by module + name in each child process. Functions
defined directly in a Jupyter notebook cell live in a dynamically-assembled
`__main__` that spawned children cannot correctly re-import -- that silently
hangs forever (verified directly: a trivial 10-item pool.map with the worker
defined in a notebook cell never returned, while the identical worker
imported from a module like this one completed immediately). See
notebooks/chess_full_analysis.ipynb Section 1 for how this is used.

The chunking approach (split the file into byte-range chunks aligned to safe
game boundaries, parse each chunk in its own process) was benchmarked
separately: on 6 physical / 12 logical local cores, 8 workers gave the best
throughput (4.5x over sequential) -- 12 workers was slower due to
hyperthreading contention. Re-benchmark N_WORKERS on whatever machine
actually runs the full parse; the optimal count depends on physical core
count, not just vCPUs reported by the OS.
"""
import os
import time

import chess.pgn
import pandas as pd

RECORD_COLUMNS = [
    'game_id', 'event', 'result', 'utc_date', 'white_elo', 'black_elo',
    'white_rating_diff', 'black_rating_diff', 'eco', 'opening',
    'time_control', 'termination', 'num_plies', 'moves',
]


def find_game_start_at_or_after(pgn_path, approx_offset):
    """Scan forward in binary mode from approx_offset to the next '[Event '
    line -- a safe, unambiguous game-boundary byte offset."""
    with open(pgn_path, 'rb') as f:
        f.seek(approx_offset)
        f.readline()  # discard partial line at the seek point
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return pos  # EOF
            if line.startswith(b'[Event '):
                return pos


def compute_chunk_boundaries(pgn_path, n_workers, region_start=0, region_end=None):
    """Byte offsets splitting [region_start, region_end) into n_workers
    game-boundary-aligned chunks.

    The final boundary is aligned the same way as every internal one. At true
    end-of-file this is a no-op (nothing exists past EOF to align to), but if
    region_end is an arbitrary cutoff short of EOF -- as when parsing a bounded
    region for a quick test -- leaving it unaligned would let the last chunk's
    reader truncate a game that starts before region_end but extends past it.
    Aligning it means the returned region may extend slightly past the
    requested region_end to complete that final game, rather than cut it off.
    """
    file_size = os.path.getsize(pgn_path)
    if region_end is None:
        region_end = file_size
    elif region_end < file_size:
        region_end = find_game_start_at_or_after(pgn_path, region_end)
    span = region_end - region_start
    approx_starts = [region_start + int(span * i / n_workers) for i in range(n_workers)]
    boundaries = [region_start]
    for approx in approx_starts[1:]:
        boundaries.append(find_game_start_at_or_after(pgn_path, approx))
    boundaries.append(region_end)
    return boundaries


class ChunkReader:
    """Binary-mode line reader bounded to [start, end) bytes -- exposes the
    readline()/tell() interface chess.pgn.read_game() needs, without loading
    the whole chunk into memory at once."""

    def __init__(self, fh, end_byte):
        self.fh = fh
        self.end_byte = end_byte

    def readline(self):
        if self.fh.tell() >= self.end_byte:
            return ''
        raw = self.fh.readline()
        return raw.decode('utf-8', errors='replace')

    def tell(self):
        return self.fh.tell()


def parse_chunk(args):
    """Parse one [start_byte, end_byte) chunk of the PGN file -- captures the
    same fields as the notebook's sequential parse_pgn(). game_id is left
    unset here; parallel_parse_pgn() assigns it sequentially in file order
    after combining all chunks."""
    pgn_path, start_byte, end_byte = args
    records = []
    with open(pgn_path, 'rb') as fh:
        fh.seek(start_byte)
        reader = ChunkReader(fh, end_byte)
        while True:
            game = chess.pgn.read_game(reader)
            if game is None:
                break
            h = game.headers
            board = game.board()
            moves = []
            for mv in game.mainline_moves():
                moves.append(board.san(mv))
                board.push(mv)
            records.append({
                'event':             h.get('Event', '?'),
                'result':            h.get('Result', '?'),
                'utc_date':          h.get('UTCDate', '?'),
                'white_elo':         h.get('WhiteElo', '?'),
                'black_elo':         h.get('BlackElo', '?'),
                'white_rating_diff': h.get('WhiteRatingDiff', '?'),
                'black_rating_diff': h.get('BlackRatingDiff', '?'),
                'eco':               h.get('ECO', '?'),
                'opening':           h.get('Opening', '?'),
                'time_control':      h.get('TimeControl', '?'),
                'termination':       h.get('Termination', '?'),
                'num_plies':         len(moves),
                'moves':             moves,
            })
    return records


def parallel_parse_pgn(pgn_path, n_workers, region_start=0, region_end=None):
    """Parse a PGN file using n_workers processes, each handling one
    byte-range chunk aligned to a safe game boundary. Returns a DataFrame
    with the same schema as the notebook's sequential parse_pgn(), with
    game_id assigned sequentially (0..N-1) in original file order -- not the
    order chunks happen to finish in, which multiprocessing doesn't guarantee.
    """
    import multiprocessing as mp

    t0 = time.time()
    boundaries = compute_chunk_boundaries(pgn_path, n_workers, region_start, region_end)
    tasks = [(pgn_path, boundaries[i], boundaries[i + 1]) for i in range(n_workers)]

    with mp.Pool(n_workers) as pool:
        chunk_results = pool.map(parse_chunk, tasks)  # pool.map preserves task order

    all_records = []
    for chunk in chunk_results:
        all_records.extend(chunk)
    for idx, rec in enumerate(all_records):
        rec['game_id'] = idx

    elapsed = time.time() - t0
    print(f'Parsed {len(all_records):,} games using {n_workers} workers in {elapsed:.1f}s '
          f'({len(all_records) / max(elapsed, 1e-9):,.0f} games/s)')

    return pd.DataFrame(all_records, columns=RECORD_COLUMNS)
