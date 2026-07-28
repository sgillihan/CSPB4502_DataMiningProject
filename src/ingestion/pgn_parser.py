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
import random
import sys
import time

import chess.pgn
import pandas as pd

RECORD_COLUMNS = [
    'game_id', 'event', 'result', 'utc_date', 'white_elo', 'black_elo',
    'white_rating_diff', 'black_rating_diff', 'eco', 'opening',
    'time_control', 'termination', 'num_plies', 'moves',
]


def _game_to_record(game):
    h = game.headers
    board = game.board()
    moves = []
    for mv in game.mainline_moves():
        # sys.intern: the SAN vocabulary is small (a few thousand distinct
        # tokens -- 'e4', 'Nf3', 'Qxd8+', ...) but board.san() mints a fresh
        # string object per call, so an unintern'd game corpus allocates one
        # object per ply per game even though the same tokens repeat millions
        # of times. Interning collapses that to one shared object per unique
        # token, which is where nearly all of the per-game moves-list memory
        # actually goes at multi-million-game scale (see the interning note
        # on build_full_game_sequences() in the notebook for measured numbers
        # on the equivalent tagged structure).
        moves.append(sys.intern(board.san(mv)))
        board.push(mv)
    return {
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
    }


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
            records.append(_game_to_record(game))
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
        async_result = pool.map_async(parse_chunk, tasks)  # pool.map preserves task order
        while not async_result.ready():
            async_result.wait(60)
            print(f'  ... still parsing ({(time.time() - t0) / 60:.1f} min elapsed)', flush=True)
        chunk_results = async_result.get()

    all_records = []
    for chunk in chunk_results:
        all_records.extend(chunk)
    for idx, rec in enumerate(all_records):
        rec['game_id'] = idx
        # Re-intern after the cross-process pickle/unpickle round trip: each
        # worker interned within its own process-local intern table, which
        # does NOT survive being sent back through multiprocessing.Pool, so
        # every move string arrives here as a fresh, non-shared object.
        # Re-interning now gives cross-worker sharing (e.g. two different
        # workers' games that both open 1.e4 share one 'e4' object here) --
        # this is the pass that actually matters for this DataFrame's memory
        # footprint.
        rec['moves'] = [sys.intern(m) for m in rec['moves']]

    elapsed = time.time() - t0
    print(f'Parsed {len(all_records):,} games using {n_workers} workers in {elapsed:.1f}s '
          f'({len(all_records) / max(elapsed, 1e-9):,.0f} games/s)')

    return pd.DataFrame(all_records, columns=RECORD_COLUMNS)


def parse_one_game_near(pgn_path, approx_offset, file_size):
    """Find the next safe game boundary at/after approx_offset and parse
    exactly one game starting there. Returns (start_byte, record), or None if
    there's no complete game left at/after that point (e.g. approx_offset
    landed in the file's trailing bytes)."""
    with open(pgn_path, 'rb') as fh:
        fh.seek(approx_offset)
        fh.readline()  # discard partial line at the seek point
        start_byte = None
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                return None  # ran off the end of the file with no boundary found
            if line.startswith(b'[Event '):
                start_byte = pos
                break
        fh.seek(start_byte)
        reader = ChunkReader(fh, file_size)
        game = chess.pgn.read_game(reader)
    if game is None:
        return None
    return start_byte, _game_to_record(game)


def sample_worker(args):
    """Draw target_count DISTINCT random games from the file, deduping
    against this worker's own draws via seen start-byte offsets. Independent
    workers can still occasionally land on the same game as each other --
    that cross-worker case is handled by random_sample_parse_pgn() after
    combining results, not here."""
    pgn_path, target_count, seed, file_size = args
    rng = random.Random(seed)
    seen = set()
    out = []
    attempts = 0
    max_attempts = target_count * 20  # generous safety valve, not expected to bind
    while len(out) < target_count and attempts < max_attempts:
        attempts += 1
        approx_offset = rng.randint(0, file_size - 1)
        result = parse_one_game_near(pgn_path, approx_offset, file_size)
        if result is None:
            continue
        start_byte, record = result
        if start_byte in seen:
            continue
        seen.add(start_byte)
        out.append((start_byte, record))
    return out


def random_sample_parse_pgn(pgn_path, n_games, n_workers, random_seed=42):
    """Parse a truly random cross-section of n_games scattered across the
    ENTIRE file, rather than the first n_games in file order -- avoids any
    bias from where in the month a game happened to be played (different
    player pools, rating drift, event timing, etc. can all vary over a month).

    Each worker independently picks random byte positions and takes whatever
    game starts there; duplicates (a game randomly hit more than once, within
    or across workers) are detected and replaced via a top-up round so the
    final sample has exactly n_games distinct games.

    One known, minor statistical quirk: picking uniformly-random BYTE
    positions and taking "the next game" means a game's chance of being
    selected is weighted very slightly by how much file space precedes it
    (roughly, how long the preceding game was) rather than being perfectly
    uniform per game index. For a dataset and sample size this large the
    effect is small, but it's a real approximation, not an exactly uniform
    sample by game count.

    game_id is assigned 0..n_games-1 in an arbitrary order -- these games
    aren't sequential in the source file, so there's no file order to
    preserve the way parallel_parse_pgn() does.
    """
    import multiprocessing as mp

    file_size = os.path.getsize(pgn_path)
    per_worker = -(-n_games // n_workers)  # ceil division
    tasks = [(pgn_path, per_worker, random_seed + i, file_size) for i in range(n_workers)]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        async_result = pool.map_async(sample_worker, tasks)
        while not async_result.ready():
            async_result.wait(60)
            print(f'  ... still sampling ({(time.time() - t0) / 60:.1f} min elapsed)', flush=True)
        results = async_result.get()

    combined = []
    seen_global = set()
    for worker_result in results:
        for start_byte, record in worker_result:
            if start_byte not in seen_global:
                seen_global.add(start_byte)
                combined.append(record)

    # Top-up: cross-worker collisions (two workers independently landing on
    # the same game) can leave combined a little short of n_games -- draw
    # more, single-process, until exactly n_games distinct games are held.
    if len(combined) < n_games:
        rng = random.Random(random_seed + 999)
        attempts = 0
        max_attempts = (n_games - len(combined)) * 50
        while len(combined) < n_games and attempts < max_attempts:
            attempts += 1
            approx_offset = rng.randint(0, file_size - 1)
            result = parse_one_game_near(pgn_path, approx_offset, file_size)
            if result is None:
                continue
            start_byte, record = result
            if start_byte in seen_global:
                continue
            seen_global.add(start_byte)
            combined.append(record)

    combined = combined[:n_games]  # trim any overshoot from per-worker ceil rounding
    for idx, rec in enumerate(combined):
        rec['game_id'] = idx
        # Re-intern post-pickle, for the same reason as parallel_parse_pgn --
        # worker-local interning doesn't survive the trip back to this
        # process, and this is the pass that gives this DataFrame's `moves`
        # column cross-game (and cross-worker) string sharing.
        rec['moves'] = [sys.intern(m) for m in rec['moves']]

    elapsed = time.time() - t0
    print(f'Randomly sampled {len(combined):,} distinct games using {n_workers} workers '
          f'in {elapsed:.1f}s ({len(combined) / max(elapsed, 1e-9):,.0f} games/s)')

    return pd.DataFrame(combined, columns=RECORD_COLUMNS)
