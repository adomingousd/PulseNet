"""Build data/processed/subject_<id>.npz from the raw PAMAP2 protocol files.

Usage (from the repo root, venv active):
    python scripts/prepare_pamap2.py
    python scripts/prepare_pamap2.py --raw-dir data/PAMAP2_Dataset/Protocol --subjects 101 102
    python scripts/prepare_pamap2.py --save-stream   # also write data/stream/subject_<id>_stream.npz

Each output file holds (see docs/data-contract.md):
    x         float32 [N, 19, 100]  input windows, z-scored with this subject's train stats
    y         float32 [N, 19, 10]   the next 10 steps (the forecast target)
    activity  int16   [N]           PAMAP2 activity ID of the window
    split     int8    [N]           0 = train, 1 = validation
    segment   int32   [N]           which clean segment the window came from
    start     int32   [N]           first step of the window inside its segment
    mean, std float32 [19]          this subject's train-split channel stats
    channels, subject, fs, input_len, horizon, stride, gap  (metadata)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_model import pamap2  # noqa: E402

STREAM_DIR = Path("data/stream")  # where --save-stream writes its files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("data"),
                   help="folder containing subject1xx.dat (searched recursively, Protocol preferred)")
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(101, 109)))
    p.add_argument("--save-stream", action="store_true",
                   help=f"also write each subject's physical-unit stream to {STREAM_DIR}")
    args = p.parse_args()

    summaries = []
    # The minutes column only appears with --save-stream, so the table is
    # unchanged on a normal run.
    minutes_header = f" {'minutes':>7}" if args.save_stream else ""
    print(f"{'subject':>7} {'train':>7} {'val':>6} {'acts':>5}{minutes_header}  file")
    for s in args.subjects:
        t0 = time.time()
        raw = pamap2.find_raw_file(args.raw_dir, s)
        # Keep the loaded file in a variable so the stream below can reuse it
        # instead of reading it from disk a second time.
        df = pamap2.read_subject(raw)
        data = pamap2.build_subject(df, s)
        path = pamap2.save_subject(data, args.out_dir)
        info = pamap2.summarize(data)
        info["source"] = str(raw)
        summaries.append(info)

        minutes_cell = ""
        if args.save_stream:
            stream = pamap2.build_stream(df, s)
            pamap2.save_stream(stream, STREAM_DIR)
            # Length of the stream when replayed: rows / 50 rows per second / 60 seconds.
            minutes = len(stream["rows"]) / pamap2.FS / 60
            minutes_cell = f" {minutes:>7.1f}"

        print(f"{s:>7} {info['train_samples']:>7} {info['val_samples']:>6} "
              f"{len(info['activities_train']):>5}{minutes_cell}  {path}  ({time.time() - t0:.1f}s)")
    pamap2.write_summary(summaries, args.out_dir)
    print(f"\nWrote {len(summaries)} subjects and summary.json to {args.out_dir}")


if __name__ == "__main__":
    main()
