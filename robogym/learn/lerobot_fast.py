"""`lerobot-train` with a fast window query.

LeRobot 0.6's `DatasetReader._query_hf_dataset` gathers a policy's action / observation window with
`hf_dataset[key][indices]`, which formats the *whole rows* (decoding every image column) just to read one
column: an ACT sample with a 32-step chunk decodes 64 PNGs it never uses (12 ms instead of 0.1 ms per
sample here; on Colab's CPUs it capped training at 1-2 steps/s). `patch_reader()` indexes a cached
single-column view instead; the result is identical. On lerobot <= 0.4 there is no reader and nothing to
patch. Run as a script with the usual `lerobot-train` arguments (robogym.learn.lerobot_io does)."""
from __future__ import annotations


def patch_reader() -> bool:
    """Apply the patch (idempotent). Returns True when it is active."""
    try:
        from lerobot.datasets.dataset_reader import DatasetReader
    except ImportError:
        return False
    if getattr(DatasetReader, "_robogym_fast", False):
        return True
    import torch

    def _column(self, key):
        cache = self.__dict__.setdefault("_robogym_columns", {})
        ds = self.hf_dataset
        entry = cache.get(key)
        if entry is None or entry[0] is not ds:
            entry = (ds, ds.select_columns([key]))
            cache[key] = entry
        return entry[1]

    def _relative(self, q_idx):
        return q_idx if self._absolute_to_relative_idx is None else [self._absolute_to_relative_idx[i] for i in q_idx]

    def _query_hf_dataset(self, query_indices):
        result = {}
        for key, q_idx in query_indices.items():
            if key in self._meta.video_keys:
                continue
            result[key] = torch.stack(_column(self, key)[_relative(self, q_idx)][key])
        return result

    def _get_query_timestamps(self, current_ts, query_indices=None):
        out = {}
        for key in self._meta.video_keys:
            if query_indices is not None and key in query_indices:
                out[key] = torch.stack(_column(self, "timestamp")[_relative(self, query_indices[key])]["timestamp"]).tolist()
            else:
                out[key] = [current_ts]
        return out

    DatasetReader._query_hf_dataset = _query_hf_dataset
    DatasetReader._get_query_timestamps = _get_query_timestamps
    DatasetReader._robogym_fast = True
    return True


def main():
    active = patch_reader()
    print(f"[lerobot_fast] window-query patch: {'applied' if active else 'not needed (no DatasetReader in this lerobot)'}", flush=True)
    from lerobot.scripts.lerobot_train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
