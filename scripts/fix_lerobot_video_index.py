#!/usr/bin/env python
"""Backfill per-video chunk/file index columns into a LeRobot v3.0 dataset's episode
metadata so lerobot >= 0.5.0 can read it.

The TAMP IsaacDataRecorderLeRobot wrote each episode's video as a standalone file at
  videos/<key>/chunk-{ep//chunks_size}/file-{ep}.mp4
but did NOT write the `videos/<key>/chunk_index` / `videos/<key>/file_index` columns
that lerobot >= 0.5.0's get_video_file_path() requires. This adds them (derived from
episode_index), leaving the bulky data/ and videos/ files untouched via symlinks.

Usage:
    python fix_lerobot_video_index.py <SRC_SPLIT_DIR> <DST_SPLIT_DIR>
where SRC_SPLIT_DIR is e.g. .../problem_xxx/blocks_success (contains meta/ data/ videos/).
Run with: uv run --no-project --with pyarrow python scripts/fix_lerobot_video_index.py ...
"""

import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def video_keys(info: dict) -> list[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def patch_episodes_parquet(path: Path, vkeys: list[str], chunks_size: int, fps: int) -> None:
    table = pq.read_table(path)
    ep_idx = table.column("episode_index").to_pylist()
    lengths = table.column("length").to_pylist()  # frame count per episode
    chunk_vals = [e // chunks_size for e in ep_idx]
    file_vals = list(ep_idx)  # one video file per episode (file_index == episode_index)
    # to_timestamp = clip end. lerobot's writer sets to = from + get_video_duration_in_s(mp4),
    # and its aggregate adds the same probed duration as a concat offset; for these CFR clips
    # length/fps == the probed mp4 duration exactly (verified), so this matches what lerobot
    # expects. WITHOUT this column, aggregate_datasets raises KeyError on .../to_timestamp.
    to_ts = [length / fps for length in lengths]

    n = table.num_rows
    for key in vkeys:
        cols = {
            f"videos/{key}/chunk_index": pa.array(chunk_vals, type=pa.int64()),
            f"videos/{key}/file_index": pa.array(file_vals, type=pa.int64()),
            # One video file per episode, and the data `timestamp` column resets to 0.0 each
            # episode (verified), so the episode starts at t=0 within its video file.
            f"videos/{key}/from_timestamp": pa.array([0.0] * n, type=pa.float64()),
            f"videos/{key}/to_timestamp": pa.array(to_ts, type=pa.float64()),
        }
        for name, arr in cols.items():
            if name not in table.column_names:
                table = table.append_column(name, arr)
    pq.write_table(table, path)


def main(src: Path, dst: Path) -> None:
    info = json.loads((src / "meta" / "info.json").read_text())
    vkeys = video_keys(info)
    chunks_size = info["chunks_size"]
    fps = info["fps"]
    print(f"video keys ({len(vkeys)}): {vkeys}\nchunks_size={chunks_size} fps={fps}")

    dst.mkdir(parents=True, exist_ok=True)
    # copy meta/ (small) so we can patch it; symlink the heavy data/ and videos/
    if (dst / "meta").exists():
        shutil.rmtree(dst / "meta")
    shutil.copytree(src / "meta", dst / "meta")
    for heavy in ("data", "videos"):
        link = dst / heavy
        if link.is_symlink() or link.exists():
            link.unlink() if link.is_symlink() else shutil.rmtree(link)
        link.symlink_to((src / heavy).resolve())

    n = 0
    for ep_pq in sorted((dst / "meta" / "episodes").rglob("file-*.parquet")):
        patch_episodes_parquet(ep_pq, vkeys, chunks_size, fps)
        n += 1
    print(f"patched {n} episode parquet file(s) -> {dst}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
