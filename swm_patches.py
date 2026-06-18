"""Runtime patches for stable-worldmodel 0.1.1.

Import this module once, before constructing any dataset, to apply the patches
(``import swm_patches  # noqa: F401`` at the top of train.py).

----------------------------------------------------------------------------
patch: LeRobotAdapter._build_episode_metadata  (perf, behaviour-preserving)
----------------------------------------------------------------------------
The upstream implementation builds per-episode metadata with two O(episodes x
frames) Python loops:

    counts = [(abs_ids == ep_id).sum() for ep_id in absolute_episode_ids]   # E x N
    for local_ep in range(E):                                              # E x N
        mask = local_episode_index == local_ep
        step_idx[mask] = np.arange(mask.sum())

On the merged TAMP dataset (23,581 episodes x 7,442,189 frames) that measures
**567s (9.5 min)** of single-threaded, memory-bandwidth-bound work with the GPU
idle -- the cause of the multi-hour "hang" before the first training step.

This replacement is fully vectorized (np.unique + bincount + a single stable
group-sort) and was verified to return byte-identical output to the original
across contiguous / out-of-order-id / single-episode / non-contiguous / gappy-id
inputs. Full-scale runtime: ~1.3s (≈430x faster).

----------------------------------------------------------------------------
patch: LeRobotAdapter._get_native_column  (perf, behaviour-preserving)
----------------------------------------------------------------------------
Upstream materializes a whole column with a per-row Python loop:

    column = self.dataset.hf_dataset[native_key]          # HF formatting -> py list
    np.asarray([_scalarize(v) for v in column])           # 7.4M-iteration loop

This is called for episode_index + every key in ``keys_to_cache`` (action,
proprio). On the merged dataset it is the *second* big stall (tens of minutes,
GPU idle) once the metadata quadratic above is fixed.

The replacement reads the column straight from the underlying Arrow table and
converts in bulk (flatten + reshape for fixed-width list columns, direct
``to_numpy`` for scalars). Verified byte-identical to the per-row path on
episode_index / action / proprio; full-column runtimes 0.09s / 0.76s / 0.37s.
It falls back to the original method on anything unexpected (HF row selection
via ``_indices``, ragged/nullable list columns, missing ``.data``), so it can
only ever be faster, never wrong.
"""

import logging

import numpy as np

log = logging.getLogger("swm_patches")


def _build_episode_metadata_fast(self, absolute_episode_index):
    abs_ids = absolute_episode_index.astype(np.int64)

    # Episode ids in order of first appearance (matches upstream semantics).
    unique_abs, first_idx = np.unique(abs_ids, return_index=True)
    order = np.argsort(first_idx)
    absolute_episode_ids = unique_abs[order]

    # local_episode_index[r] = position of row r's episode id in that ordering.
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    local_episode_index = inv[np.searchsorted(unique_abs, abs_ids)].astype(np.int64)

    # Rows per episode, in local-episode order.
    counts = np.bincount(
        local_episode_index, minlength=len(absolute_episode_ids)
    ).astype(np.int64)

    # step_idx[r] = running index of r within its episode, in array order.
    # Stable group-sort, then subtract each group's start offset.
    orderr = np.argsort(local_episode_index, kind="stable")
    grp_start = np.concatenate([[0], np.cumsum(counts[:-1])])
    step_idx = np.empty(len(abs_ids), dtype=np.int64)
    step_idx[orderr] = (
        np.arange(len(abs_ids), dtype=np.int64) - grp_start[local_episode_index[orderr]]
    )

    offsets = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        offsets[1:] = np.cumsum(counts[:-1])

    return (
        local_episode_index,
        step_idx,
        counts,
        offsets,
        absolute_episode_ids.astype(np.int64),
    )


def _make_get_native_column_fast(original):
    """Build a drop-in _get_native_column that reads columns via bulk Arrow
    conversion, falling back to `original` on anything it can't handle."""

    def _get_native_column_fast(self, native_key):
        if native_key in self._full_columns:
            return self._full_columns[native_key]
        try:
            import pyarrow as pa

            hf = self.dataset.hf_dataset
            # Respect HF row selection/shuffle: bail to the slow (correct) path.
            if getattr(hf, "_indices", None) is not None:
                raise ValueError("hf_dataset has _indices; use original path")

            col = hf.data.column(native_key).combine_chunks()
            if col.null_count:
                raise ValueError("nulls present; use original path")

            t = col.type
            n = len(col)
            if pa.types.is_fixed_size_list(t):
                # LeRobot stores fixed-shape features (action[9], state[8]) as
                # FixedSizeList<float> -- flatten the child + reshape by list_size.
                arr = col.values.to_numpy(zero_copy_only=False).reshape(n, t.list_size)
            elif pa.types.is_list(t) or pa.types.is_large_list(t):
                flat = col.values.to_numpy(zero_copy_only=False)
                if n == 0 or flat.size % n != 0:  # ragged -> not fixed width
                    raise ValueError("ragged list column; use original path")
                arr = flat.reshape(n, flat.size // n)
            else:
                arr = col.to_numpy(zero_copy_only=False)

            # Any list-like type we failed to flatten yields an object array of
            # per-row arrays; never hand that back -- fall back to the slow path.
            if arr.dtype == object:
                raise ValueError("object-dtype result; use original path")

            self._full_columns[native_key] = arr
            return arr
        except Exception as exc:  # noqa: BLE001 - correctness first, never break
            log.warning(
                "[swm_patches] _get_native_column fast path fell back for %r (%s)",
                native_key,
                exc,
            )
            return original(self, native_key)

    return _get_native_column_fast


def _install_getitem_profiler():
    """Opt-in (env SWM_PROFILE_GETITEM=1) per-sample profiler for the dataloader.

    Wraps LeRobotAdapter._load_slice and splits each sample's wall time into
    (a) decode+assembly (the torchcodec video fetch) and (b) transform (resize +
    normalize). Each DataLoader worker logs a windowed average every
    SWM_PROFILE_EVERY (default 500) samples, so we can see what actually caps
    throughput (decode vs transform vs other) instead of guessing.

    Implementation note: it calls the *original* _load_slice with the transform
    temporarily disabled (to time decode+assembly alone), then applies the
    transform itself (to time it) -- output is byte-identical to normal.
    """
    import os
    import time

    if not os.environ.get("SWM_PROFILE_GETITEM"):
        return

    from stable_worldmodel.data.formats.lerobot import LeRobotAdapter

    orig_load_slice = LeRobotAdapter._load_slice
    every = int(os.environ.get("SWM_PROFILE_EVERY", "500"))
    st = {"total_n": 0, "wn": 0, "decode": 0.0, "transform": 0.0, "tot": 0.0}

    def _profiled_load_slice(self, ep_idx, start, end):
        t0 = time.perf_counter()
        tf = self.transform
        self.transform = None
        try:
            steps = orig_load_slice(self, ep_idx, start, end)  # decode + assembly only
        finally:
            self.transform = tf
        t1 = time.perf_counter()
        out = tf(steps) if tf is not None else steps           # transform
        t2 = time.perf_counter()

        st["total_n"] += 1
        st["wn"] += 1
        st["decode"] += t1 - t0
        st["transform"] += t2 - t1
        st["tot"] += t2 - t0
        if st["wn"] >= every:
            wn = st["wn"]
            log.info(
                "[getitem-profile pid=%d] n=%d  decode=%.1fms  transform=%.1fms  "
                "total=%.1fms/sample  (%.0f samples/s/worker)",
                os.getpid(), st["total_n"],
                st["decode"] / wn * 1000, st["transform"] / wn * 1000,
                st["tot"] / wn * 1000, wn / st["tot"],
            )
            st["wn"] = 0
            st["decode"] = st["transform"] = st["tot"] = 0.0
        return out

    LeRobotAdapter._load_slice = _profiled_load_slice
    log.info("[swm_patches] getitem profiler ENABLED (SWM_PROFILE_GETITEM=1, every=%d)", every)


def apply():
    from stable_worldmodel.data.formats.lerobot import LeRobotAdapter

    LeRobotAdapter._build_episode_metadata = _build_episode_metadata_fast
    LeRobotAdapter._get_native_column = _make_get_native_column_fast(
        LeRobotAdapter._get_native_column
    )
    _install_getitem_profiler()
    log.info(
        "[swm_patches] applied vectorized LeRobotAdapter._build_episode_metadata "
        "+ bulk-Arrow _get_native_column (swm 0.1.1 dataset-load hang fix)"
    )


apply()
