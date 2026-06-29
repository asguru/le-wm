import logging

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

log = logging.getLogger("utils")

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def episode_split(dataset, train_split: float, seed: int):
    """Split a windowed dataset into train/val by *whole episodes* (videos).

    Unlike spt.data.random_split (which randomizes rows and shreds temporal contiguity),
    this assigns each complete episode to either train or val. The val rows are returned
    sorted by (episode, step) so a shuffle=False DataLoader yields each val video in
    temporal order — a prerequisite for the contiguous sliding-window error analysis in
    wm_error_logging.WMErrorVideoCallback.

    Per-row (episode, step) comes from ``dataset.clip_indices`` (the swm windowed-Dataset's
    list of ``(local_episode_index, start_step)`` per clip), which is the canonical per-row
    metadata aligned with ``dataset[i]``. NB: ``get_col_data("ep_idx"/"step_idx")`` is NOT
    used here — for the LeRobot adapter those return *per-frame* arrays (length = total
    frames), not per-clip, so they do not align with ``dataset[i]``.

    Returns:
        train_rows (np.ndarray[int]): dataset row indices for training.
        val_rows   (np.ndarray[int]): dataset row indices for validation, sorted by
                                      (episode, step).
        ep         (np.ndarray):      per-row episode id, length len(dataset).
        step       (np.ndarray):      per-row step index, length len(dataset).
    """
    clips = np.asarray(dataset.clip_indices)  # (N, 2): (local_episode_index, start_step)
    assert clips.ndim == 2 and clips.shape[1] == 2 and len(clips) == len(dataset), (
        f"dataset.clip_indices shape {clips.shape} not aligned with len(dataset) "
        f"{len(dataset)}; cannot derive per-row (episode, step)"
    )
    ep = clips[:, 0]
    step = clips[:, 1]

    unique_eps = np.unique(ep)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_eps)
    n_train = int(round(len(perm) * train_split))
    n_train = min(max(n_train, 1), len(perm) - 1)  # keep both splits non-empty
    train_eps = set(perm[:n_train].tolist())

    is_train = np.isin(ep, list(train_eps))
    train_rows = np.nonzero(is_train)[0]
    val_rows = np.nonzero(~is_train)[0]
    # order val rows so each episode plays forward in time
    val_rows = val_rows[np.lexsort((step[val_rows], ep[val_rows]))]

    log.info(
        "[episode_split] %d episodes -> train %d eps / %d rows, val %d eps / %d rows",
        len(unique_eps), n_train, len(train_rows),
        len(perm) - n_train, len(val_rows),
    )
    return train_rows, val_rows, ep, step


class RowIndexDataset(torch.utils.data.Dataset):
    """Map-style view over `base` restricted to `rows`, attaching the underlying dataset
    row id to every sample as `row_id`.

    The row id lets the validation error buffer be realigned to (episode, step) metadata
    regardless of DataLoader/DistributedSampler ordering (so the sliding-window analysis is
    DDP-safe). Default collate turns `row_id` into a (B,) int tensor, which the model's
    forward simply ignores.
    """

    def __init__(self, base, rows):
        self.base = base
        self.rows = [int(r) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        sample = self.base[row]
        sample["row_id"] = row
        return sample


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )