#!/bin/bash

# Automated reconstruction + evaluation for the Microsoft 7-Scenes dataset.
# This mirrors scripts/eval_replica.sh but walks every seq-* folder under each
# scene. Edit the parameter block below to tweak reconstruction behavior.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

shopt -s nullglob
export PYTHONUNBUFFERED="1"

######################################################################################
# Reconstruction parameters (see recon.py for detailed descriptions)
######################################################################################
KEYFRAME_STRIDE=10
UPDATE_BUFFER_INTV=3
MAX_NUM_REGISTER=10
WIN_R=5
NUM_SCENE_FRAME=10
INITIAL_WINSIZE=5
CONF_THRES_I2P=1.5

# Parameters below only affect saved recon/eval artifacts
NUM_POINTS_SAVE=1000000
CONF_THRES_L2W=10
GPU_ID=-1

# Dataset configuration
SCENE_ROOT="data/7Scenes"
GT_ROOT="results/gt/7scenes"
SCENES=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

if [[ ! -d "$SCENE_ROOT" ]]; then
  echo "Dataset directory not found: $SCENE_ROOT" >&2
  exit 1
fi

for SCENE_NAME in "${SCENES[@]}"; do
  SCENE_DIR="$SCENE_ROOT/$SCENE_NAME"
  if [[ ! -d "$SCENE_DIR" ]]; then
    echo "Skipping $SCENE_NAME (missing $SCENE_DIR)"
    continue
  fi
  SEQ_PATHS=("$SCENE_DIR"/seq-*)
  if [[ ${#SEQ_PATHS[@]} -eq 0 ]]; then
    echo "Skipping $SCENE_NAME (no seq-* folders in $SCENE_DIR)"
    continue
  fi

  for SEQ_PATH in "${SEQ_PATHS[@]}"; do
    [[ -d "$SEQ_PATH" ]] || continue
    SEQ_NAME="$(basename "$SEQ_PATH")"
    TEST_NAME="7Scenes_${SCENE_NAME}_${SEQ_NAME}"

    echo "-------- Start reconstructing $SCENE_NAME/$SEQ_NAME with test name $TEST_NAME --------"
    python recon.py \
      --test_name "$TEST_NAME" \
      --img_dir "$SEQ_PATH" \
      --gpu_id $GPU_ID \
      --keyframe_stride $KEYFRAME_STRIDE \
      --win_r $WIN_R \
      --num_scene_frame $NUM_SCENE_FRAME \
      --initial_winsize $INITIAL_WINSIZE \
      --conf_thres_l2w $CONF_THRES_L2W \
      --conf_thres_i2p $CONF_THRES_I2P \
      --num_points_save $NUM_POINTS_SAVE \
      --update_buffer_intv $UPDATE_BUFFER_INTV \
      --max_num_register $MAX_NUM_REGISTER \
      --save_for_eval

    GT_PCD="$GT_ROOT/${SCENE_NAME}_${SEQ_NAME}_pcds.npy"
    GT_MASK="${GT_PCD/_pcds/_valid_masks}"

    if [[ -f "$GT_PCD" && -f "$GT_MASK" ]]; then
      echo "-------- Start evaluating $SCENE_NAME/$SEQ_NAME with test name $TEST_NAME --------"
      python eval/eval_recon.py \
        --test_name="$TEST_NAME" \
        --gt_pcd="$GT_PCD"
    else
      echo "WARNING: Missing ground-truth pair ($GT_PCD / $GT_MASK); skipping evaluation for $SCENE_NAME/$SEQ_NAME."
    fi
  done

done
