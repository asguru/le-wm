#!/usr/bin/env python
"""Make a Blocksuite LeRobot v3.0 dataset loadable by lerobot >= 0.5.0 / stable-worldmodel.

The Blocksuite recorder writes a dataset whose data/ and meta/ are valid v3.0, but whose
VIDEO files are in a layout that does NOT match what lerobot's reader expects:

  declared video_path (meta/info.json):
      videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
  actual files on disk (per-episode, chunk/key swapped, ':' -> '_'):
      videos/chunk-{cc:03d}/{video_key_with_':'_as_'_'}/episode_{episode_index:06d}.mp4

lerobot's LeRobotDatasetMetadata.get_video_file_path(ep, key) does:
      chunk = ep["videos/{key}/chunk_index"]; file = ep["videos/{key}/file_index"]
      path  = video_path.format(video_key=key, chunk_index=chunk, file_index=file)
i.e. it (a) reads per-video chunk/file index columns from meta/episodes (ABSENT here), and
(b) formats the template with the RAW key (colon kept, no sanitization).

So this script builds a fixed root that:
  * symlinks data/ (already correct v3.0) unchanged,
  * copies meta/ and BACKFILLS videos/<key>/{chunk_index,file_index,from_timestamp,
    to_timestamp} into meta/episodes (file_index == episode_index, one file/episode;
    from=0, to=length/fps since each video file is one episode starting at t=0 — verified:
    the per-frame `timestamp` column resets to 0 each episode),
  * recreates videos/ as a tree of per-file symlinks at the reader-expected template paths
    (videos/<key-with-colon>/chunk-000/file-<ep>.mp4 -> the real episode_<ep>.mp4).

It does NOT transcode: the videos are AV1 (vs the TAMP H.264 used for throughput). For this
small dataset torchcodec decodes AV1 fine; transcode separately if decode becomes the
bottleneck (see scripts/transcode_av1_to_h264.sh for the pattern).

Usage:
    uv run --no-project --with pyarrow python scripts/fix_blocksuite_dataset.py <SRC> <DST>
"""

import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa


def video_keys(info: dict) -> list[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def find_actual_video(src: Path, vkey: str, ep: int, chunks_size: int) -> Path:
    """Locate the real per-episode video file for (vkey, ep) in the Blocksuite layout."""
    cc = ep // chunks_size
    san = vkey.replace(":", "_")
    cand = src / "videos" / f"chunk-{cc:03d}" / san / f"episode_{ep:06d}.mp4"
    if not cand.exists():
        raise FileNotFoundError(f"expected source video not found: {cand}")
    return cand


def link_videos(src: Path, dst: Path, info: dict) -> int:
    vkeys = video_keys(info)
    chunks_size = info["chunks_size"]
    video_tmpl = info["video_path"]  # videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
    n_eps = info["total_episodes"]
    n = 0
    for ep in range(n_eps):
        for vkey in vkeys:
            actual = find_actual_video(src, vkey, ep, chunks_size).resolve()
            rel = video_tmpl.format(video_key=vkey, chunk_index=ep // chunks_size, file_index=ep)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(actual)
            n += 1
    return n


def patch_episodes_parquet(path: Path, vkeys: list[str], chunks_size: int, fps: int) -> None:
    table = pq.read_table(path)
    ep_idx = table.column("episode_index").to_pylist()
    lengths = table.column("length").to_pylist()
    chunk_vals = [e // chunks_size for e in ep_idx]
    file_vals = list(ep_idx)                       # one video file per episode
    from_ts = [0.0] * table.num_rows               # each video file starts at t=0
    to_ts = [length / fps for length in lengths]   # clip end == episode duration
    for key in vkeys:
        cols = {
            f"videos/{key}/chunk_index": pa.array(chunk_vals, type=pa.int64()),
            f"videos/{key}/file_index": pa.array(file_vals, type=pa.int64()),
            f"videos/{key}/from_timestamp": pa.array(from_ts, type=pa.float64()),
            f"videos/{key}/to_timestamp": pa.array(to_ts, type=pa.float64()),
        }
        for name, arr in cols.items():
            if name not in table.column_names:
                table = table.append_column(name, arr)
    pq.write_table(table, path)


def main(src: Path, dst: Path) -> None:
    info = json.loads((src / "meta" / "info.json").read_text())
    vkeys = video_keys(info)
    print(f"video keys ({len(vkeys)}): {vkeys}")
    print(f"chunks_size={info['chunks_size']} fps={info['fps']} episodes={info['total_episodes']}")

    dst.mkdir(parents=True, exist_ok=True)
    # meta/ : copy (small) so we can patch episode parquet
    if (dst / "meta").exists():
        shutil.rmtree(dst / "meta")
    shutil.copytree(src / "meta", dst / "meta")
    # data/ : already valid v3.0 -> whole-dir symlink
    link = dst / "data"
    if link.is_symlink() or link.exists():
        link.unlink() if link.is_symlink() else shutil.rmtree(link)
    link.symlink_to((src / "data").resolve())
    # videos/ : rebuild as per-file symlinks at the reader-expected template paths
    if (dst / "videos").exists():
        shutil.rmtree(dst / "videos")
    nlinks = link_videos(src, dst, info)
    print(f"linked {nlinks} video files into reader-expected layout")

    n = 0
    for ep_pq in sorted((dst / "meta" / "episodes").rglob("file-*.parquet")):
        patch_episodes_parquet(ep_pq, vkeys, info["chunks_size"], info["fps"])
        n += 1
    print(f"patched {n} episode parquet file(s) -> {dst}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
