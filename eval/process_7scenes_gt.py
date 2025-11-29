#!/usr/bin/env python3
"""Precompute ground-truth point clouds for the Microsoft 7-Scenes dataset.

This mirrors eval/process_gt.py (Replica) but iterates through every seq-* folder
under data/7Scenes. For each sequence we export two .npy files:
    * <scene>_<seq>_pcds.npy         — float32 array [N, H, W, 3] of world points
    * <scene>_<seq>_valid_masks.npy  — bool array   [N, H, W] tracking valid pixels
These files are consumed by eval/eval_recon.py when scoring reconstructions.
"""
import argparse
import os
import sys
import re
from typing import Iterable, List, Optional, Sequence, Tuple
from os import path as osp
SLAM3R_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, SLAM3R_DIR) # noqa: E402

from slam3r.datasets import Replica

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

from slam3r.datasets.seven_scenes_seq import SevenScenes_Seq

SEQ_PATTERN = re.compile(r"^seq-(\d+)$")


def _discover_sequences(scene_root: str, scenes_filter: Optional[Sequence[str]],
                        seq_names_filter: Optional[Sequence[str]]) -> List[Tuple[str, int, str]]:
    """Find (scene, seq_id, seq_name) tuples available on disk."""
    if not os.path.isdir(scene_root):
        raise FileNotFoundError(f"Dataset directory not found: {scene_root}")

    if scenes_filter:
        candidate_scenes: Iterable[str] = scenes_filter
    else:
        candidate_scenes = sorted(
            d for d in os.listdir(scene_root)
            if os.path.isdir(os.path.join(scene_root, d))
        )

    sequences: List[Tuple[str, int, str]] = []
    for scene in candidate_scenes:
        scene_dir = os.path.join(scene_root, scene)
        if not os.path.isdir(scene_dir):
            continue
        for entry in sorted(os.listdir(scene_dir)):
            match = SEQ_PATTERN.match(entry)
            if not match:
                continue
            if seq_names_filter and entry not in seq_names_filter:
                continue
            seq_dir = os.path.join(scene_dir, entry)
            if not os.path.isdir(seq_dir):
                continue
            seq_id = int(match.group(1))
            sequences.append((scene, seq_id, entry))
    return sequences


def _parse_targets(targets: Sequence[str]) -> List[Tuple[str, int, str]]:
    """Parse entries like "office:seq-01" into (scene, seq_id, seq_name)."""
    parsed: List[Tuple[str, int, str]] = []
    for item in targets:
        if ':' not in item:
            raise ValueError(f"Invalid --targets entry '{item}'. Expected scene:seq-XX")
        scene, seq_name = item.split(':', 1)
        scene = scene.strip()
        seq_name = seq_name.strip()
        match = SEQ_PATTERN.match(seq_name)
        if not match:
            raise ValueError(f"Invalid sequence name '{seq_name}'. Use the seq-XX folder name.")
        parsed.append((scene, int(match.group(1)), seq_name))
    return parsed


def export_sequence(scene: str, seq_id: int, seq_name: str, args: argparse.Namespace) -> None:
    scene_tag = f"{scene}_{seq_name}"
    save_dir = args.save_root
    os.makedirs(save_dir, exist_ok=True)

    pcd_path = os.path.join(save_dir, f"{scene_tag}_pcds.npy")
    mask_path = os.path.join(save_dir, f"{scene_tag}_valid_masks.npy")
    if not args.overwrite and os.path.isfile(pcd_path) and os.path.isfile(mask_path):
        print(f"[SKIP] {scene_tag}: files already exist. Use --overwrite to regenerate.")
        return

    dataset = SevenScenes_Seq(
        ROOT=args.scene_root,
        scene_id=scene,
        seq_id=seq_id,
        num_views=1,
        sample_freq=args.sample_freq,
        start_freq=args.start_freq,
        cycle=False,
        resolution=(args.resolution, args.resolution),
        need_pts3d=True,
    )

    num_frames = len(dataset)
    if num_frames == 0:
        print(f"[WARN] {scene_tag}: no frames discovered; skipping.")
        return

    sample_view = dataset[0][0]
    height, width, _ = sample_view['pts3d'].shape

    print(f"[INFO] Exporting {scene_tag}: {num_frames} frames at {width}x{height}")

    pcd_mmap = open_memmap(pcd_path, mode='w+', dtype=np.float32, shape=(num_frames, height, width, 3))
    mask_mmap = open_memmap(mask_path, mode='w+', dtype=np.bool_, shape=(num_frames, height, width))

    for idx in tqdm(range(num_frames), desc=f"{scene_tag}"):
        view = dataset[idx][0]
        pcd_mmap[idx] = view['pts3d']
        mask_mmap[idx] = view['valid_mask']

    del pcd_mmap
    del mask_mmap
    print(f"[DONE] Saved {pcd_path} and matching mask.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 7-Scenes ground-truth point clouds.")
    parser.add_argument('--scene_root', type=str, default='data/7Scenes',
                        help='Root directory containing scene folders (default: data/7Scenes).')
    parser.add_argument('--save_root', type=str, default='results/gt/7scenes',
                        help='Directory to store the generated numpy files (default: results/gt/7scenes).')
    parser.add_argument('--scenes', nargs='*',
                        help='Optional subset of scene folder names to process (e.g., office heads).')
    parser.add_argument('--seq_names', nargs='*',
                        help='Optional subset of sequence folder names to include (e.g., seq-01 seq-05).')
    parser.add_argument('--targets', nargs='*',
                        help='Explicit scene:seq-XX entries overriding --scenes/--seq_names filters.')
    parser.add_argument('--resolution', type=int, default=224,
                        help='Square resolution used to crop/resize inputs (default: 224).')
    parser.add_argument('--sample_freq', type=int, default=1,
                        help='Frame sampling stride within SevenScenes_Seq (default: 1).')
    parser.add_argument('--start_freq', type=int, default=1,
                        help='Sliding-window step used by the dataloader (default: 1).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Regenerate existing numpy files.')

    args = parser.parse_args()

    if args.targets:
        sequences = _parse_targets(args.targets)
    else:
        sequences = _discover_sequences(args.scene_root, args.scenes, args.seq_names)

    if not sequences:
        raise RuntimeError('No sequences matched the provided filters.')

    for scene, seq_id, seq_name in sequences:
        try:
            export_sequence(scene, seq_id, seq_name, args)
        except Exception as exc:  # keep going even if one sequence fails
            print(f"[ERROR] Failed to export {scene}_{seq_name}: {exc}")


if __name__ == '__main__':
    main()
