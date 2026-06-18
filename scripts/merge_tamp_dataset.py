#!/usr/bin/env python
"""Merge all problem_*/blocks_{success,failure} LeRobot datasets under an unmerged TAMP
dataset dir into ONE dataset, using lerobot's aggregate_datasets.

Each source is first schema-fixed (video-index backfill, see fix_lerobot_video_index) into a
lightweight staging root (symlinked data/+videos/, patched meta/). aggregate_datasets then reads
those and writes a single clean v3.0 dataset with global episode/chunk/file renumbering, merged
tasks/stats, and lerobot-native video chunking (so no post-fix needed on the output).

NOTE: aggregate COPIES + concatenates the video files, so the output is large (all 6 camera/depth
streams -> hundreds of GB) and the job runs for a while. To slim to just the training camera,
pass a 4th arg listing video keys to KEEP, e.g. "observation.images.front_camera".

Usage:
  python scripts/merge_tamp_dataset.py <UNMERGED_DIR> <STAGING_DIR> <MERGED_ROOT> \
      [blocks_success,blocks_failure] [keep_video_key1,keep_video_key2]
"""

import json
import sys
from pathlib import Path

from fix_lerobot_video_index import patch_episodes_parquet, video_keys
import shutil
from lerobot.datasets.aggregate import aggregate_datasets


def info(split_dir: Path) -> dict:
    return json.loads((split_dir / "meta" / "info.json").read_text())


def meta_signature(m: dict, keep_video: list[str] | None) -> str:
    """A signature capturing everything lerobot's validate_all_metadata compares:
    the full features dict (keys + dtype/shape/names), fps, and robot_type. We compute
    it on the features that SURVIVE the keep_video drop, since that's what aggregate
    actually sees on the staged datasets. Sources whose signatures differ would make
    aggregate_datasets raise ValueError mid-run, so we skip them cleanly instead."""
    feats = dict(m["features"])
    if keep_video:  # mirror prep_source: drop the video streams we won't stage
        for k in [k for k in video_keys(m) if k not in keep_video]:
            feats.pop(k, None)
    return json.dumps(
        {"features": feats, "fps": m.get("fps"), "robot_type": m.get("robot_type")},
        sort_keys=True,
    )


def prep_source(src: Path, dst: Path, keep_video: list[str] | None) -> None:
    """Schema-fix one split into `dst` (symlink data/+videos/, copy+patch meta/).
    If keep_video is given, drop all other video features from the staged info.json so
    aggregate only copies those streams."""
    meta = info(src)
    chunks_size = meta["chunks_size"]
    fps = meta["fps"]
    vkeys = video_keys(meta)
    if keep_video:
        vkeys = [k for k in vkeys if k in keep_video]

    dst.mkdir(parents=True, exist_ok=True)
    if (dst / "meta").exists():
        shutil.rmtree(dst / "meta")
    shutil.copytree(src / "meta", dst / "meta")
    for heavy in ("data", "videos"):
        link = dst / heavy
        if link.is_symlink() or link.exists():
            link.unlink() if link.is_symlink() else shutil.rmtree(link)
        link.symlink_to((src / heavy).resolve())

    if keep_video:  # rewrite staged info.json to keep only the chosen video streams
        dropped = [k for k in video_keys(meta) if k not in keep_video]
        for k in dropped:
            meta["features"].pop(k, None)
        (dst / "meta" / "info.json").write_text(json.dumps(meta, indent=2))

    for ep_pq in sorted((dst / "meta" / "episodes").rglob("file-*.parquet")):
        patch_episodes_parquet(ep_pq, vkeys, chunks_size, fps)


def main(unmerged: Path, staging: Path, merged: Path, splits: list[str], keep_video):
    problems = sorted(p for p in unmerged.iterdir() if p.is_dir() and p.name.startswith("problem_"))
    roots, labels, skipped = [], [], []
    ref_sig = None
    for prob in problems:
        for split in splits:
            sd = prob / split
            if not (sd / "meta" / "info.json").exists():
                continue
            m = info(sd)
            if m["total_episodes"] == 0:
                skipped.append((str(sd), "empty"))
                continue
            sig = meta_signature(m, keep_video)
            if ref_sig is None:
                ref_sig = sig
            if sig != ref_sig:
                skipped.append((str(sd), "feature/fps/robot_type-mismatch"))
                continue
            dst = staging / f"{prob.name}__{split}"
            prep_source(sd, dst, keep_video)
            roots.append(dst)
            labels.append(f"tamp/{prob.name}__{split}")

    print(f"sources to merge: {len(roots)}   skipped: {len(skipped)}")
    for s, why in skipped[:30]:
        print("  skip", why, s)

    merged.parent.mkdir(parents=True, exist_ok=True)
    aggregate_datasets(
        repo_ids=labels,
        aggr_repo_id="tamp/may28_topdown_flip_norand_3b",
        roots=roots,
        aggr_root=merged,
    )
    print("MERGED ->", merged)


if __name__ == "__main__":
    unmerged = Path(sys.argv[1]).resolve()
    staging = Path(sys.argv[2]).resolve()
    merged = Path(sys.argv[3]).resolve()
    splits = sys.argv[4].split(",") if len(sys.argv) > 4 else ["blocks_success", "blocks_failure"]
    keep_video = sys.argv[5].split(",") if len(sys.argv) > 5 else None
    main(unmerged, staging, merged, splits, keep_video)
