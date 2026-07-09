"""Linear-probe decodability of block pose from world-model latents (validation only).

Question: how linearly decodable is the dropped block's pose from the JEPA latent? Each
validation epoch we fit a ridge linear map from the per-frame encoder latent ``emb[:, s]``
(D-dim, the "current state" s == last context frame) to two targets and report *held-out* R^2:

  * ``z``          -- the block's height, the drop axis        -> ``validate/probe/z/r2``
  * ``quaternion`` -- orientation (qw, qx, qy, qz)             -> ``validate/probe/quat/r2`` (+ per-comp)

The val EPISODES are split ONCE at construction (fixed for the whole run, seeded) into a
probe-fit set and a probe-eval set. Fixing the split makes R^2 (a) honest -- never fit and
scored on the same points -- and (b) directly comparable across epochs (same eval points every
time). Splitting by *episode* (not row) avoids leakage from temporally adjacent, near-identical
frames landing on both sides.

Latents+pose are buffered per-sample in ``lejepa_forward`` keyed by ``row_id`` (DDP-safe, same
pattern as WMErrorVideoCallback); we gather + dedup them on rank 0 here, then solve a tiny
(D x D) ridge system in numpy -- no extra model forward passes.

Caveats: (1) the quaternion has a sign/double-cover ambiguity (q and -q are the same rotation),
which can depress a *linear* probe's R^2 if the recorder flips sign within the data; treat the
quaternion R^2 as a lower bound. (2) A component with ~no variance across the eval set has an
undefined R^2 -- we log NaN and warn rather than divide by ~0.
"""

import logging

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger

log = logging.getLogger("probe_logging")

_EPS = 1e-8
Z_IDX = 2           # block_pose = [x, y, z, qw, qx, qy, qz]
QUAT_IDX = slice(3, 7)
QUAT_NAMES = ["qw", "qx", "qy", "qz"]


def _gather(buf, world_size):
    """buf: list of (row_id (B,), latent (B,D) fp16, pose (B,7) f32) cpu tensors.
    Returns (row_ids (N,), Z (N,D) f64, Y (N,7) f64), all-gathered across ranks and
    deduplicated by row id (DistributedSampler can repeat rows to pad the last batch)."""
    if buf:
        row_ids = torch.cat([t[0] for t in buf]).to(torch.int64).numpy()
        Z = torch.cat([t[1] for t in buf]).numpy()          # keep fp16 in transit
        Y = torch.cat([t[2] for t in buf]).to(torch.float32).numpy()
    else:
        row_ids = np.empty(0, dtype=np.int64)
        Z = np.empty((0, 0), dtype=np.float16)
        Y = np.empty((0, 7), dtype=np.float32)

    if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, (row_ids, Z, Y))
        row_ids = np.concatenate([g[0] for g in gathered])
        Z = np.concatenate([g[1] for g in gathered]) if any(g[1].size for g in gathered) else Z
        Y = np.concatenate([g[2] for g in gathered])

    _, uniq = np.unique(row_ids, return_index=True)
    return row_ids[uniq], Z[uniq].astype(np.float64), Y[uniq].astype(np.float64)


def _ridge_predict(Z_fit, Y_fit, Z_eval, lam):
    """Ridge on standardized features + centered targets, predict on the eval set.
    Z_*: (N, D); Y_fit: (N, K). Returns Y_pred_eval (N_eval, K).

    lam is SCALE-FREE: the penalty is lam * mean(diag(Zf^T Zf)) (~= lam * N for standardized
    features), so its meaning (fraction of per-feature signal energy) is invariant to fit-set
    size. This matters because the fit rows are temporally correlated frames from only a few
    dozen episodes -> few effectively-independent samples, so an absolute (tiny) lam degenerates
    to OLS on D=192 dims and overfits to a negative held-out R^2. lam~1 => shrinkage comparable
    to the signal; <1 lighter, >1 heavier."""
    mu = Z_fit.mean(0)
    sd = Z_fit.std(0) + _EPS
    Zf = (Z_fit - mu) / sd
    Ze = (Z_eval - mu) / sd
    ym = Y_fit.mean(0)
    Yc = Y_fit - ym
    D = Zf.shape[1]
    G = Zf.T @ Zf
    ridge = lam * (np.trace(G) / D)          # scale-free: ~= lam * N for standardized features
    W = np.linalg.solve(G + ridge * np.eye(D), Zf.T @ Yc)  # (D, K)
    return Ze @ W + ym


def _r2_per_component(Y_true, Y_pred):
    """Per-column R^2 = 1 - SS_res/SS_tot (SS_tot around the eval-set mean). NaN where a
    component is ~constant on the eval set (SS_tot ~ 0)."""
    Y_true = np.atleast_2d(Y_true.T).T if Y_true.ndim == 1 else Y_true
    Y_pred = np.atleast_2d(Y_pred.T).T if Y_pred.ndim == 1 else Y_pred
    ss_res = ((Y_true - Y_pred) ** 2).sum(0)
    ss_tot = ((Y_true - Y_true.mean(0)) ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(ss_tot > _EPS, 1.0 - ss_res / ss_tot, np.nan)
    return r2, ss_res, ss_tot


class ProbeCallback(Callback):
    def __init__(
        self,
        val_rows,
        ep,
        *,
        seed=0,
        enabled=True,
        eval_split=0.3,
        ridge_lambda=1.0,
        eval_every_n_epochs=None,
    ):
        super().__init__()
        self.enabled = enabled
        self.ridge_lambda = float(ridge_lambda)
        self.eval_every_n_epochs = eval_every_n_epochs

        # Fix the fit/eval split ONCE, by episode, for the whole run.
        val_rows = np.asarray(val_rows)
        ep_of = np.asarray(ep)[val_rows]
        episodes = np.unique(ep_of)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(episodes))
        n_eval = max(1, int(round(len(episodes) * float(eval_split)))) if len(episodes) > 1 else 0
        eval_eps = set(episodes[perm[:n_eval]].tolist())
        self.eval_row_ids = set(val_rows[np.isin(ep_of, list(eval_eps))].tolist())
        self.fit_row_ids = set(val_rows.tolist()) - self.eval_row_ids
        log.info(
            "[probe] fixed split: %d val episodes -> %d fit / %d eval episodes "
            "(%d fit / %d eval rows), ridge_lambda=%g",
            len(episodes), len(episodes) - len(eval_eps), len(eval_eps),
            len(self.fit_row_ids), len(self.eval_row_ids), self.ridge_lambda,
        )

    def _active(self, trainer):
        if not self.enabled or trainer.sanity_checking:
            return False
        if self.eval_every_n_epochs:
            return (trainer.current_epoch % int(self.eval_every_n_epochs)) == 0
        return True

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module._probe_buf = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._active(trainer):
            return
        buf = getattr(pl_module, "_probe_buf", [])
        row_ids, Z, Y = _gather(buf, trainer.world_size)

        if not trainer.is_global_zero:
            return
        wandb_logger = next(
            (lg for lg in (trainer.loggers or []) if isinstance(lg, WandbLogger)), None
        )
        if wandb_logger is None:
            return
        if len(row_ids) == 0 or Z.shape[1] == 0:
            log.info("[probe] epoch %d: nothing gathered (rows=%d)", trainer.current_epoch, len(row_ids))
            return

        # Canonicalize the quaternion double-cover: q and -q are the SAME rotation, but a linear
        # probe can't fit two opposite sign branches of one orientation. Put every sample on the
        # qw>=0 hemisphere (negate all 4 components where qw<0) so the target is single-valued.
        q = Y[:, QUAT_IDX]
        q[q[:, 0] < 0] *= -1.0

        # Partition gathered rows into the fixed fit / eval sets.
        is_fit = np.array([int(r) in self.fit_row_ids for r in row_ids])
        is_eval = np.array([int(r) in self.eval_row_ids for r in row_ids])
        Zf, Yf = Z[is_fit], Y[is_fit]
        Ze, Ye = Z[is_eval], Y[is_eval]
        if len(Zf) == 0 or len(Ze) == 0:
            log.warning("[probe] epoch %d: empty fit/eval partition (fit=%d eval=%d)",
                        trainer.current_epoch, len(Zf), len(Ze))
            return
        if len(Zf) <= Z.shape[1]:
            log.warning("[probe] epoch %d: only %d fit rows for D=%d features; ridge underdetermined "
                        "(R^2 still valid but noisy)", trainer.current_epoch, len(Zf), Z.shape[1])

        media = {"validate/probe/n_fit": len(Zf), "validate/probe/n_eval": len(Ze),
                 "validate/probe/global_step": trainer.global_step}

        # ---- z probe (scalar height / drop axis) ----
        z_pred = _ridge_predict(Zf, Yf[:, Z_IDX:Z_IDX + 1], Ze, self.ridge_lambda)
        z_r2, _, z_tot = _r2_per_component(Ye[:, Z_IDX:Z_IDX + 1], z_pred)
        media["validate/probe/z/r2"] = float(z_r2[0])

        # ---- quaternion probe (4-dim orientation) ----
        q_pred = _ridge_predict(Zf, Yf[:, QUAT_IDX], Ze, self.ridge_lambda)
        q_r2, q_res, q_tot = _r2_per_component(Ye[:, QUAT_IDX], q_pred)
        for name, r in zip(QUAT_NAMES, q_r2):
            media[f"validate/probe/quat/r2_{name}"] = float(r)
        # Combined (variance-weighted) quaternion R^2: 1 - sum SS_res / sum SS_tot.
        tot = float(q_tot.sum())
        media["validate/probe/quat/r2"] = float(1.0 - q_res.sum() / tot) if tot > _EPS else float("nan")

        wandb_logger.experiment.log(media)
        log.info(
            "[probe] epoch %d: z R^2=%.3f | quat R^2=%.3f (qw=%.3f qx=%.3f qy=%.3f qz=%.3f) "
            "[fit=%d eval=%d]",
            trainer.current_epoch, media["validate/probe/z/r2"], media["validate/probe/quat/r2"],
            *[media[f"validate/probe/quat/r2_{n}"] for n in QUAT_NAMES], len(Zf), len(Ze),
        )
