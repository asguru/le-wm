# Data pipeline (Blocksuite-Drop) — the 4 files that actually get used

This documents the data-processing / loading code that the **current** le-wm training runs
depend on. As of the latest beaker experiments (`beaker/train_drop_landed_1gpu.yaml`,
`beaker/train_drop_1gpu.yaml`, `beaker/smoke_blocksuite.yaml`), training is on the
**Blocksuite-Drop** datasets, which are loaded **directly as AV1** — the AV1→H.264 transcode
(`scripts/transcode_av1_to_h264.sh`) and the TAMP merge path (`scripts/merge_tamp_dataset.py`,
`scripts/fix_lerobot_video_index.py`) fed the *earlier* full-TAMP runs and are **not** exercised
by the current datasets. See those scripts' own docstrings if you need the TAMP history.

The four files below are what a current run touches, in the order they come into play:

| Stage | File | Role |
|-------|------|------|
| 1. Build dataset (offline, once) | `scripts/fix_blocksuite_dataset.py` | Turn the raw recorder output into a loadable LeRobot v3.0 root |
| 2. Patch the loader (import-time) | `swm_patches.py` | Monkey-patch `stable-worldmodel`'s LeRobot adapter for speed |
| 3. Configure the load | `config/train/data/blocksuite_drop_landed.yaml` | Point `train.py` at the fixed root + declare which columns to load |
| 4. Adapt for training | `utils.py` | Normalizers, episode-wise split, per-row indexing used while building the DataLoaders |

---

## 1. `scripts/fix_blocksuite_dataset.py` — make a raw Blocksuite dataset loadable

**Run offline, once per dataset.** The Blocksuite recorder emits a dataset whose `data/` and
`meta/` are valid LeRobot v3.0, but which `lerobot >= 0.5.0` / `stable-worldmodel` **cannot read**
for two reasons:

1. **Video files are in the wrong layout.** The reader formats
   `videos/{video_key}/chunk-{chunk:03d}/file-{file:03d}.mp4` with the *raw* key (colon kept),
   but the recorder wrote per-episode files with chunk/key swapped and `:`→`_`:
   `videos/chunk-{cc:03d}/{key_with_'_'}/episode_{ep:06d}.mp4`.
2. **Missing metadata columns.** `meta/episodes/*.parquet` lacks the
   `videos/<key>/{chunk_index, file_index, from_timestamp, to_timestamp}` columns the reader's
   `get_video_file_path()` requires.

The script writes a new **reader-compatible root** without duplicating heavy data:

- `data/` → whole-directory **symlink** (already valid v3.0).
- `meta/` → **copied**, then `meta/episodes/*.parquet` **backfilled** with the four
  `videos/<key>/…` columns (`file_index == episode_index` since there's one video file per
  episode; `from_timestamp = 0`, `to_timestamp = length / fps`).
- `videos/` → rebuilt as a tree of **per-file symlinks** at the reader-expected template paths,
  each pointing at the real `episode_*.mp4`.

It does **not** transcode — videos stay AV1 (torchcodec/PyAV decode them fine at this dataset's
size).

**Usage:**
```bash
uv run --no-project --with pyarrow python scripts/fix_blocksuite_dataset.py <SRC_RAW> <DST_FIXED>
# e.g. DST = /weka/robots/aguru/datasets/blocksuite_drop_landed_fixed
```
The `<DST_FIXED>` you produce is exactly what the data config's `root:` points at.

---

## 2. `swm_patches.py` — speed patches for the LeRobot adapter (import-time)

`train.py` does `import swm_patches  # noqa: F401` at the top, which calls `apply()` and
monkey-patches `stable-worldmodel 0.1.1`'s `LeRobotAdapter`. All patches are
**behaviour-preserving** (verified byte-identical) and only ever make things faster or fall back
to the original path:

- **`_build_episode_metadata`** → fully vectorized (`np.unique`/`bincount`/stable group-sort)
  replacement for the upstream O(episodes × frames) Python loops. On the merged TAMP dataset this
  cut per-load metadata build from ~567 s to ~1.3 s (~430×). On the small Blocksuite set it's just
  a harmless speedup, but it removes a startup hang on any large dataset.
- **`_get_native_column`** → reads columns via bulk Arrow conversion instead of a per-row Python
  loop, with a safe fallback to the original method on anything unexpected (HF row selection,
  ragged/nullable columns, object-dtype results).
- **Opt-in getitem profiler** (`SWM_PROFILE_GETITEM=1`, window `SWM_PROFILE_EVERY`, default 500):
  wraps `_load_slice` to split each sample's time into decode+assembly vs. transform, so you can
  see what actually caps DataLoader throughput. Output is byte-identical to normal.

Nothing to run — importing the module applies the patches. Keep the import at the top of
`train.py`, **before** any dataset is constructed.

---

## 3. `config/train/data/blocksuite_drop_landed.yaml` — the load config

Selected on the command line with `data=blocksuite_drop_landed` (Hydra). This is the config the
**latest** run (`train_drop_landed_1gpu.yaml`) uses; `blocksuite_drop.yaml` is the same without the
extra probe/landed columns. It tells `swm.data.load_dataset(...)`:

- `root:` — the fixed root built in stage 1
  (`/weka/robots/aguru/datasets/blocksuite_drop_landed_fixed`).
- `primary_camera_key: observation.images.camera:drop_camera` — the drop camera (not TAMP's
  `front_camera`).
- `frameskip: 5` / `num_steps: 4` — 25 fps → 5 fps effective; each window spans 4 loaded steps.
- **`key_aliases`** — maps native columns with no default swm alias to clean batch keys so they can
  be loaded:
  - `action.policy → action` (there is no canonical `action`; the 9-dim policy action lives here),
  - `landed → landed` (per-timestep bool that flips True when the block lands — feeds the
    validation WM-error "block fallen vs not" histograms in `wm_error_logging.py`),
  - `observation.state.object_pose.block → block_pose` (7-dim x,y,z,quaternion — feeds the
    validation linear probe in `probe_logging.py`).
- `keys_to_load` / `keys_to_cache` — which columns to materialize / cache in RAM.

The `landed` and `block_pose` columns are deliberately **not** z-scored (see `train.py`, which
skips them when building normalizers) so the analysis callbacks get raw values.

**Runtime note:** the Blocksuite beaker jobs set `export LEROBOT_VIDEO_BACKEND=pyav` because these
videos are AV1 and PyAV bundles an AV1-capable ffmpeg.

---

## 4. `utils.py` — dataset/loader helpers used while building the DataLoaders

Imported by `train.py`; these run after `load_dataset` and shape how the data is normalized,
split, and indexed:

- **`get_img_preprocessor(source, target, img_size)`** — `ToImage` (ImageNet stats) + `Resize`
  compose applied to the `pixels` stream.
- **`ZScoreNormalizer`** + **`get_column_normalizer(dataset, source, target)`** — per-column
  z-score normalizer. `ZScoreNormalizer` is a **class, not a closure**, so it survives pickling
  when DataLoader workers are spawned (required by the Lance-backed dataset). `get_column_normalizer`
  computes mean/std over the (NaN-filtered) column and wraps it as a transform. Applied to
  `action` / `proprio`; **not** to `landed` / `block_pose` (kept raw — see the config note above).
- **`episode_split(dataset, train_split, seed)`** — splits by **whole episodes (videos)**, not
  random rows, using `dataset.clip_indices` (`(local_episode_index, start_step)` per clip). Val rows
  are returned sorted by `(episode, step)` so a `shuffle=False` DataLoader replays each val video in
  temporal order — a prerequisite for the sliding-window WM-error analysis
  (`wm_error_logging.WMErrorVideoCallback`). Returns `(train_rows, val_rows, per_row_ep,
  per_row_step)`.
- **`RowIndexDataset(base, rows)`** — a row-restricted view that attaches the underlying dataset
  `row_id` to every sample. This lets the validation error buffer be realigned to `(episode, step)`
  metadata regardless of DataLoader/DistributedSampler ordering, so the analysis is DDP-safe. Used
  for the **val** set; the train set uses a plain `torch.utils.data.Subset`.
- **`SaveCkptCallback(run_name, cfg, epoch_interval)`** — Lightning callback that saves the model
  via `save_pretrained` on an epoch interval (and at the final epoch), rank-zero only. (Not
  data-processing per se, but lives here and is wired up in `train.py`.)

---

## End-to-end, for the current Blocksuite runs

```
raw recorder output
   │  scripts/fix_blocksuite_dataset.py   (offline, once)
   ▼
*_fixed root  ──referenced by──►  config/train/data/blocksuite_drop_landed.yaml
                                          │  data=blocksuite_drop_landed
                                          ▼
   train.py:  import swm_patches  (adapter speed patches)
              swm.data.load_dataset(root, key_aliases, keys_to_load, …)
              utils.get_img_preprocessor / get_column_normalizer   (transforms)
              utils.episode_split + RowIndexDataset                (train/val DataLoaders)
              → training
```

**Not used by the current datasets** (TAMP-era only): `scripts/transcode_av1_to_h264.sh`,
`scripts/merge_tamp_dataset.py`, `scripts/fix_lerobot_video_index.py`.
