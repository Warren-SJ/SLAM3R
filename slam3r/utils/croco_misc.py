"""
Minimal stub for training utilities removed in inference-only setup.

This module previously contained training helpers (logging, distributed helpers,
learning-rate scheduling, GradScaler wrappers, and checkpoint I/O). For an
inference-only repository, these utilities are intentionally omitted.

Any prior imports of `slam3r.utils.croco_misc` should not rely on its contents
anymore. Dataloader-related distributed helpers are now provided in
`slam3r.datasets.__init__` (get_world_size/get_rank).
"""

__all__ = ["placeholder_removed"]

def placeholder_removed():
    """No-op placeholder to keep backward compatibility with old imports."""
    return None
