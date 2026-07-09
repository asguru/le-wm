"""World-model prediction-error video logging (validation only).

Goal: debug *where* the JEPA world model predicts well vs. poorly. During every validation
run we already compute, per sample, the latent-space prediction error
``(pred_emb - tgt_emb).pow(2).mean(dim=(1,2))`` -- this is exactly the per-datapoint value
behind the scalar ``val/pred_loss`` (we do NOT log a redundant scalar). This callback:

  1. collects those per-sample errors (keyed by dataset row id, so it is DDP-safe),
  2. tiles non-overlapping windows of ``window_size`` *consecutive* data points over each
     val episode and accumulates the error per window -> the {window -> error} map,
  3. logs wandb videos of the ``num_videos`` highest-error and lowest-error windows, to
     stable keys so wandb gives a step-slider to watch the hard/easy segments evolve.

Two separate wandb sections are logged each interval:
  * ``validate/wm_error/{high,low}/*``        -- absolute WM error (worst / best windows).
  * ``validate/wm_error_change/{most_reduced,least_reduced}/*`` -- change vs the *previous*
    eval interval (delta = prev_error - curr_error, per fixed window): the windows whose
    error dropped the most (learning) and the least / regressed. Needs the previous
    interval's per-window errors, kept in ``self._prev_win_err``.

All of (1)-(2) is pure numpy over already-computed errors: no extra model forward passes
and no per-window data loading. Only the finally-selected clips trigger frame reads.

The val DataLoader must yield each episode in temporal order (see utils.episode_split +
utils.RowIndexDataset) for the windows to be contiguous video.
"""

import logging

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from stable_pretraining import data as spt_data

log = logging.getLogger("wm_error_logging")


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def _imagenet_mean_std(device):
    stats = spt_data.dataset_stats.ImageNet
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device).view(-1, 1, 1)
    std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device).view(-1, 1, 1)
    return mean, std


def _unnormalize_to_uint8(clip):
    """clip: (T, C, H, W) ImageNet-normalized float -> (T, C, H, W) uint8 in [0, 255]."""
    mean, std = _imagenet_mean_std(clip.device)
    clip = clip.float() * std + mean
    clip = clip.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    return clip.cpu().numpy()


def _gather_records(buf, world_size):
    """buf: list of (row_id, err, landed, norm_err) cpu tensors. Returns
    (row_ids, errs, landed, norm_err) np arrays, all-gathered across ranks and deduplicated by
    row id. `landed` is the input-frame block-fallen flag (NaN if the dataset has no `landed`
    column); `norm_err` is the persistence-relative error (NaN if unavailable)."""
    def _col(i, default_nan=True):
        if buf and len(buf[0]) > i:
            return torch.cat([t[i] for t in buf]).to(torch.float64).numpy()
        return np.full(len(row_ids), np.nan) if default_nan else None

    if buf:
        row_ids = torch.cat([t[0] for t in buf]).to(torch.int64).numpy()
        errs = torch.cat([t[1] for t in buf]).to(torch.float64).numpy()
        landed = _col(2)
        norm_err = _col(3)
    else:
        row_ids = np.empty(0, dtype=np.int64)
        errs = np.empty(0, dtype=np.float64)
        landed = np.empty(0, dtype=np.float64)
        norm_err = np.empty(0, dtype=np.float64)

    if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, (row_ids, errs, landed, norm_err))
        row_ids = np.concatenate([g[0] for g in gathered])
        errs = np.concatenate([g[1] for g in gathered])
        landed = np.concatenate([g[2] for g in gathered])
        norm_err = np.concatenate([g[3] for g in gathered])

    # DistributedSampler can repeat rows to pad the last batch; keep first occurrence.
    _, uniq = np.unique(row_ids, return_index=True)
    return row_ids[uniq], errs[uniq], landed[uniq], norm_err[uniq]


def _stacked_decile_hist(values, fallen, xlabel, title):
    """Stacked bar over 10% percentile buckets (deciles) of `values`: green = block not yet
    fallen (bottom), red = block fallen (top). `values`/`fallen` are aligned per-sample arrays
    (`fallen` is 0/1). Returns a wandb.Image, or None if matplotlib is unavailable / no data."""
    values = np.asarray(values, dtype=np.float64)
    fallen = np.asarray(fallen) > 0.5
    n = len(values)
    if n == 0:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb
    except Exception as exc:  # noqa: BLE001
        log.warning("[wm_error] matplotlib/wandb unavailable for landed histogram (%s)", exc)
        return None

    order = np.argsort(values, kind="stable")
    decile = (np.arange(n) * 10) // n               # 0..9, equal-count buckets in sorted order
    fallen_sorted = fallen[order]
    green = np.array([(~fallen_sorted[decile == d]).sum() for d in range(10)])  # not fallen
    red = np.array([(fallen_sorted[decile == d]).sum() for d in range(10)])     # fallen

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(10)
    ax.bar(x, green, color="green", label="block not yet fallen")
    ax.bar(x, red, bottom=green, color="red", label="block fallen")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d*10}-{d*10+10}" for d in range(10)], rotation=45, fontsize=8)
    ax.set_xlabel(f"{xlabel} percentile (%)")
    ax.set_ylabel("# validation samples")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


class WMErrorVideoCallback(Callback):
    def __init__(
        self,
        dataset,
        val_rows,
        ep,
        step,
        *,
        enabled=True,
        window_size=16,
        stride=16,
        num_videos=10,
        fps=4,
        eval_every_n_epochs=None,
        frame_select="first",
    ):
        super().__init__()
        self.dataset = dataset
        self.enabled = enabled
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.num_videos = int(num_videos)
        self.fps = int(fps)
        self.eval_every_n_epochs = eval_every_n_epochs
        self.frame_select = frame_select

        # Previous active interval's per-window accumulated error ({window_index -> error}),
        # used to compute the per-window change (error reduction) between eval intervals.
        # Windows are fixed across intervals, so window_index is a stable key.
        self._prev_win_err = None
        # Previous interval's PER-SAMPLE error ({row_id -> error}), for the per-sample
        # rate-of-change landed histogram.
        self._prev_err_by_row = None

        # Precompute the windows once: group the (temporally ordered) val rows by episode,
        # then tile windows of `window_size` consecutive rows with the given stride. Windows
        # are fixed across epochs (only their accumulated error changes).
        self.windows = []          # list of np.ndarray[int] row ids, each length window_size
        self.window_meta = []      # list of (episode_id, start_step)
        self._build_windows(val_rows, ep, step)

        log.info(
            "[wm_error] %d windows (size=%d, stride=%d) over %d val rows",
            len(self.windows), self.window_size, self.stride, len(val_rows),
        )

    def _build_windows(self, val_rows, ep, step):
        val_rows = np.asarray(val_rows)
        if len(val_rows) == 0:
            return
        ep_of = ep[val_rows]
        # val_rows already sorted by (ep, step); split into per-episode contiguous groups.
        boundaries = np.nonzero(np.diff(ep_of) != 0)[0] + 1
        groups = np.split(np.arange(len(val_rows)), boundaries)
        for g in groups:
            rows = val_rows[g]
            steps = step[rows]
            for s in range(0, len(rows) - self.window_size + 1, self.stride):
                self.windows.append(rows[s : s + self.window_size])
                self.window_meta.append((int(ep_of[g[0]]), int(steps[s])))
        if not self.windows:
            log.warning("[wm_error] no full-length windows could be formed (episodes shorter than window_size=%d?)", self.window_size)

    # -- Lightning hooks -----------------------------------------------------------------

    def _active(self, trainer):
        if not self.enabled or trainer.sanity_checking:
            return False
        if self.eval_every_n_epochs:
            return (trainer.current_epoch % int(self.eval_every_n_epochs)) == 0
        return True

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module._wm_err_buf = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._active(trainer):
            return
        buf = getattr(pl_module, "_wm_err_buf", [])
        row_ids, errs, landed, norm_err = _gather_records(buf, trainer.world_size)

        if not trainer.is_global_zero:
            return
        # spt.Manager installs its own RegistryLogger/CSVLogger, so trainer.logger (== the
        # first logger) is NOT the WandbLogger — find it among trainer.loggers.
        wandb_logger = next(
            (lg for lg in (trainer.loggers or []) if isinstance(lg, WandbLogger)), None
        )
        if wandb_logger is None:
            return
        if len(self.windows) == 0 or len(row_ids) == 0:
            log.info("[wm_error] epoch %d: nothing to log (windows=%d, gathered_rows=%d, buf_chunks=%d)",
                     trainer.current_epoch, len(self.windows), len(row_ids), len(buf))
            return

        err_by_row = dict(zip(row_ids.tolist(), errs.tolist()))
        landed_by_row = dict(zip(row_ids.tolist(), landed.tolist()))
        # Persistence-relative normalized error (see train.py): computed once over the full
        # gathered val set on rank 0. We report the MEDIAN (not the mean): the per-sample ratio
        # has a heavy right tail (near-static / landed samples have tiny denominators -> huge
        # ratios), so a mean is dominated by outliers; the median is the robust typical value
        # and needs no cap. Split by the input-frame block state (`landed`).
        norm_arr = np.asarray(norm_err, dtype=np.float64)
        landed_arr = np.asarray(landed, dtype=np.float64)
        norm_curves = {}
        finite = np.isfinite(norm_arr)
        if finite.any():
            vals = norm_arr[finite]
            lnd = landed_arr[finite]
            norm_curves["validate/normalized_error"] = float(np.median(vals))
            falling = vals[lnd == 0.0]   # block not yet fallen
            fallen = vals[lnd == 1.0]    # block fallen + rolling
            if falling.size:
                norm_curves["validate/normalized_error/falling"] = float(np.median(falling))
            if fallen.size:
                norm_curves["validate/normalized_error/fallen_rolling"] = float(np.median(fallen))

        # Accumulate per-sample error per (precomputed, fixed) window. Skip windows missing
        # any row (shouldn't happen on a full val pass). curr: {window_index -> error}.
        curr = {}
        for i, w in enumerate(self.windows):
            vals = [err_by_row.get(int(r)) for r in w]
            if any(v is None for v in vals):
                continue
            curr[i] = float(np.sum(vals))
        if not curr:
            log.warning("[wm_error] no complete windows this epoch (gathered %d rows)", len(row_ids))
            return

        try:
            import wandb
        except Exception:  # noqa: BLE001
            return

        win_idx = np.array(list(curr.keys()))
        win_err = np.array([curr[i] for i in win_idx])

        # ---- Section A: absolute world-model error ----------------------------------------
        order = np.argsort(win_err)  # ascending: low error first
        n = min(self.num_videos, len(order))
        high_sel = [int(win_idx[k]) for k in order[::-1][:n]]
        low_sel = [int(win_idx[k]) for k in order[:n]]

        # ---- Section B: change in error vs the previous eval interval ---------------------
        # delta = prev - curr  (positive = error reduced / improved). Most-reduced =
        # biggest improvement; least-reduced = smallest delta (most negative = regressed).
        most_reduced, least_reduced, deltas = [], [], None
        if self._prev_win_err:
            shared = [i for i in curr if i in self._prev_win_err]
            if shared:
                d = np.array([self._prev_win_err[i] - curr[i] for i in shared])
                sh = np.array(shared)
                dorder = np.argsort(d)  # ascending: most-negative (regressed) first
                m = min(self.num_videos, len(dorder))
                most_reduced = [int(sh[k]) for k in dorder[::-1][:m]]
                least_reduced = [int(sh[k]) for k in dorder[:m]]
                deltas = d

        # Histograms first — they need no video encoder, so they survive an encode failure.
        media = {"validate/wm_error/window_error_hist": wandb.Histogram(win_err.tolist())}
        if deltas is not None:
            media["validate/wm_error_change/delta_hist"] = wandb.Histogram(deltas.tolist())

        # Normalized (persistence-relative) error curves: overall + per landed-class time series.
        media.update(norm_curves)
        if norm_curves:
            log.info("[wm_error] epoch %d normalized_error(median): all=%.3f falling=%s fallen_rolling=%s",
                     trainer.current_epoch, norm_curves.get("validate/normalized_error", float("nan")),
                     _fmt(norm_curves.get("validate/normalized_error/falling")),
                     _fmt(norm_curves.get("validate/normalized_error/fallen_rolling")))

        # ---- Per-sample landed-vs-not decile histograms (block fallen at the input frame) ----
        # Only for samples whose `landed` is known (not NaN). Bucket samples into deciles of a
        # per-sample metric and stack green (not yet fallen) / red (fallen) within each.
        known = [r for r in err_by_row if not np.isnan(landed_by_row.get(r, np.nan))]
        if known:
            err_vals = [err_by_row[r] for r in known]
            err_fallen = [landed_by_row[r] for r in known]
            img = _stacked_decile_hist(err_vals, err_fallen, "WM error", "Block fallen vs WM-error percentile")
            if img is not None:
                media["validate/wm_error/landed_by_error_pct"] = img
            # rate of change vs previous interval (per sample), if available
            if self._prev_err_by_row:
                ch = [r for r in known if r in self._prev_err_by_row]
                if ch:
                    ch_vals = [self._prev_err_by_row[r] - err_by_row[r] for r in ch]
                    ch_fallen = [landed_by_row[r] for r in ch]
                    img2 = _stacked_decile_hist(
                        ch_vals, ch_fallen, "WM error change", "Block fallen vs WM-error-change percentile")
                    if img2 is not None:
                        media["validate/wm_error_change/landed_by_change_pct"] = img2

        try:
            for rank, wi in enumerate(high_sel):
                media[f"validate/wm_error/high/{rank}"] = self._render_video(
                    wandb, wi, self._abs_caption(wi, curr[wi]))
            for rank, wi in enumerate(low_sel):
                media[f"validate/wm_error/low/{rank}"] = self._render_video(
                    wandb, wi, self._abs_caption(wi, curr[wi]))
            for rank, wi in enumerate(most_reduced):
                media[f"validate/wm_error_change/most_reduced/{rank}"] = self._render_video(
                    wandb, wi, self._delta_caption(wi, self._prev_win_err[wi], curr[wi]))
            for rank, wi in enumerate(least_reduced):
                media[f"validate/wm_error_change/least_reduced/{rank}"] = self._render_video(
                    wandb, wi, self._delta_caption(wi, self._prev_win_err[wi], curr[wi]))
        except Exception as exc:  # noqa: BLE001 - never crash training on a media-encode error
            log.warning("[wm_error] video encoding failed (%s); logging histograms only", exc)

        # NB: do NOT pass step=trainer.global_step here. spt/Lightning drive wandb's internal
        # _step ahead of trainer.global_step, so an explicit (smaller) step makes wandb reject
        # the media from the run history ("step must be monotonically increasing") — the video
        # files still upload (visible under Files) but no panel appears. Logging without a step
        # commits at wandb's current step (always accepted); we attach global_step as a field
        # for reference. Panels still progress across eval intervals (each logs at a higher step).
        media["validate/wm_error/global_step"] = trainer.global_step
        wandb_logger.experiment.log(media)
        log.info(
            "[wm_error] epoch %d: %d high / %d low abs-error videos%s "
            "(window err min=%.4g max=%.4g)",
            trainer.current_epoch, len(high_sel), len(low_sel),
            "" if deltas is None else
            f"; {len(most_reduced)} most / {len(least_reduced)} least error-reduction videos",
            win_err.min(), win_err.max(),
        )

        # Remember this interval's per-window AND per-sample errors for next interval's change.
        self._prev_win_err = curr
        self._prev_err_by_row = dict(err_by_row)

    # -- video construction --------------------------------------------------------------

    def _abs_caption(self, window_index, accum):
        ep_id, start_step = self.window_meta[window_index]
        return f"err={accum:.4g} ep={ep_id} step={start_step}-{start_step + self.window_size - 1}"

    def _delta_caption(self, window_index, prev, curr):
        ep_id, start_step = self.window_meta[window_index]
        return (
            f"Δerr={prev - curr:.4g} (prev={prev:.4g} -> curr={curr:.4g}) "
            f"ep={ep_id} step={start_step}-{start_step + self.window_size - 1}"
        )

    def _render_video(self, wandb, window_index, caption):
        rows = self.windows[window_index]
        frames = []
        for r in rows:
            px = self.dataset[int(r)]["pixels"]  # (T, C, H, W), ImageNet-normalized
            f = px[0] if self.frame_select == "first" else px[-1]
            frames.append(f if torch.is_tensor(f) else torch.as_tensor(f))
        clip = torch.stack(frames, dim=0)  # (window_size, C, H, W)
        arr = _unnormalize_to_uint8(clip)
        return wandb.Video(arr, fps=self.fps, format="mp4", caption=caption)
