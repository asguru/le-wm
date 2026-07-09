import logging
import os
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

import swm_patches  # noqa: F401  # vectorizes LeRobotAdapter._build_episode_metadata (swm 0.1.1 hang fix)
from module import SIGReg
from utils import (
    episode_split,
    get_column_normalizer,
    get_img_preprocessor,
    RowIndexDataset,
    SaveCkptCallback,
)
from wm_error_logging import WMErrorVideoCallback
from probe_logging import ProbeCallback

log = logging.getLogger("train.setup")


@contextmanager
def step(name):
    """Log entry/exit + wall time for a setup step so a silent hang is locatable."""
    log.info("[setup] >>> %s ...", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log.info("[setup] <<< %s done in %.1fs", name, time.perf_counter() - t0)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    # Per-sample latent prediction error (the un-aggregated version of pred_loss, NOT a new
    # logged metric) — buffered for WMErrorVideoCallback to build the per-window error map.
    # Keyed by dataset row_id so it can be realigned to (episode, step) under DDP sharding.
    # NB: spt.Module.validation_step calls forward(batch, stage="validate") (not "val").
    if stage == "validate" and "row_id" in batch:
        err = (pred_emb - tgt_emb).pow(2).mean(dim=(1, 2)).detach().cpu()  # (B,)
        # Normalized (persistence-relative) error per sample:
        #   ||pred - s_{t+1}|| / ||s_t - s_{t+1}||  aggregated over the 1-step transitions.
        # The denominator is exactly the error of predicting "no state change" (s_t == ctx_emb
        # is the most-recent context frame), so =1 means no better than persistence, <<1 means
        # the world model captures the dynamics, >1 means worse. Requires ctx_emb/tgt_emb to
        # align 1:1 (true for num_preds=1). See wm_error_logging for the logged curves.
        if ctx_emb.shape[1] == tgt_emb.shape[1]:
            num = (pred_emb - tgt_emb).norm(dim=-1).sum(dim=1)  # (B,)
            den = (ctx_emb - tgt_emb).norm(dim=-1).sum(dim=1)   # (B,)
            norm_err = (num / (den + 1e-6)).detach().cpu()
        else:
            norm_err = torch.full_like(err, float("nan"))
        # `landed` of the input state (frame 0) per sample, for the val landed-vs-not
        # histograms + the per-class normalized-error curves; NaN when the dataset has no
        # `landed` column (older datasets).
        if "landed" in batch:
            landed = batch["landed"][:, 0].reshape(err.shape[0]).float().detach().cpu()
        else:
            landed = torch.full_like(err, float("nan"))
        if not hasattr(self, "_wm_err_buf"):
            self._wm_err_buf = []
        self._wm_err_buf.append((batch["row_id"].detach().cpu(), err, landed, norm_err))

        # Linear-probe buffer: latent of the "current state" s (last context frame, index
        # ctx_len-1) paired with the block pose at that same frame. ProbeCallback fits a ridge
        # map latent -> z / quaternion on a fixed val-episode split each epoch. Only when the
        # dataset carries block_pose (the landed build). fp16 latent keeps the buffer tiny.
        if "block_pose" in batch:
            s = ctx_len - 1
            lat = emb[:, s].detach().cpu().to(torch.float16)          # (B, D)
            pose = batch["block_pose"][:, s].detach().cpu().float()   # (B, 7)
            if not hasattr(self, "_probe_buf"):
                self._probe_buf = []
            self._probe_buf.append((batch["row_id"].detach().cpu(), lat, pose))
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    with step(f"load_dataset({dataset_name})"):
        dataset = swm.data.load_dataset(
            dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
        )
    log.info("[setup] dataset loaded: %d samples", len(dataset))
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            if col in ("landed", "block_pose", "block_velocity"):
                continue  # val-analysis columns (landed flag, probe targets): keep raw, do NOT z-score
            with step(f"get_column_normalizer({col})"):
                normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        with step("get_dim(action)"):
            cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    with step("episode_split + build DataLoaders"):
        rnd_gen = torch.Generator().manual_seed(cfg.seed)
        # Split by whole episodes (videos), not random rows, so the val set is contiguous
        # video — a prerequisite for the world-model-error sliding-window analysis below.
        train_rows, val_rows, row_ep, row_step = episode_split(
            dataset, cfg.train_split, cfg.seed
        )
        train_set = torch.utils.data.Subset(dataset, train_rows.tolist())
        # val carries row_id per sample (shuffle=False keeps episodes in temporal order).
        val_set = RowIndexDataset(dataset, val_rows)

        train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
        val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    log.info("[setup] train=%d val=%d samples", len(train_set), len(val_set))
    
    ##############################
    ##       model / optim      ##
    ##############################

    with step("instantiate model"):
        world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        with step("WandbLogger init"):
            logger = WandbLogger(**cfg.wandb.config)
        with step("wandb log_hyperparams"):
            logger.log_hyperparams(OmegaConf.to_container(cfg))

    log.info("[setup] run_dir=%s", run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    callbacks = [object_dump_callback]
    wm_err_cfg = OmegaConf.to_container(cfg.get("wm_error_logging", {}) or {}, resolve=True)
    if wm_err_cfg.get("enabled", False):
        callbacks.append(
            WMErrorVideoCallback(
                dataset=dataset,
                val_rows=val_rows,
                ep=row_ep,
                step=row_step,
                **{k: v for k, v in wm_err_cfg.items() if k != "enabled"},
                enabled=True,
            )
        )

    probe_cfg = OmegaConf.to_container(cfg.get("probe_logging", {}) or {}, resolve=True)
    if probe_cfg.get("enabled", False):
        callbacks.append(
            ProbeCallback(
                val_rows=val_rows,
                ep=row_ep,
                seed=cfg.seed,
                **{k: v for k, v in probe_cfg.items() if k != "enabled"},
                enabled=True,
            )
        )

    with step("build Trainer"):
        trainer = pl.Trainer(
            **cfg.trainer,
            callbacks=callbacks,
            num_sanity_val_steps=1,
            logger=logger,
            enable_checkpointing=True,
        )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    log.info("[setup] entering manager()/trainer.fit — first training step should follow")
    manager()
    log.info("[setup] manager() returned (training complete)")
    return


if __name__ == "__main__":
    run()
