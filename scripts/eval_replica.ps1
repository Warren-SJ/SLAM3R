# PowerShell version of scripts/eval_replica.sh for Windows users
# Run from anywhere; the script will cd to the repo root automatically.
# Requires Python available on PATH.

$ErrorActionPreference = "Stop"

# Move to repo root (parent of this scripts folder)
if ($PSScriptRoot) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
    Set-Location $RepoRoot
}

######################################################################################
# Set parameters for whole scene reconstruction below
# For definition of these parameters, please refer to recon.py
######################################################################################
$KEYFRAME_STRIDE = 20
$UPDATE_BUFFER_INTV = 3
$MAX_NUM_REGISTER = 10
$WIN_R = 5
$NUM_SCENE_FRAME = 10
$INITIAL_WINSIZE = 5
$CONF_THRES_I2P = 1.5

# The parameters below have nothing to do with the evaluation files saved
$NUM_POINTS_SAVE = 1000000
$CONF_THRES_L2W = 10
$GPU_ID = -1

$SCENE_NAMES = @("office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2")

foreach ($SCENE_NAME in $SCENE_NAMES) {
    $TEST_NAME = "Replica_$SCENE_NAME"

    Write-Host "--------Start reconstructing scene $SCENE_NAME with test name $TEST_NAME--------"

    $reconArgs = @(
        "recon.py",
        "--test_name", $TEST_NAME,
        "--img_dir", "data/Replica/$SCENE_NAME/results",
        "--gpu_id", $GPU_ID,
        "--keyframe_stride", $KEYFRAME_STRIDE,
        "--win_r", $WIN_R,
        "--num_scene_frame", $NUM_SCENE_FRAME,
        "--initial_winsize", $INITIAL_WINSIZE,
        "--conf_thres_l2w", $CONF_THRES_L2W,
        "--conf_thres_i2p", $CONF_THRES_I2P,
        "--num_points_save", $NUM_POINTS_SAVE,
        "--update_buffer_intv", $UPDATE_BUFFER_INTV,
        "--max_num_register", $MAX_NUM_REGISTER,
        "--save_for_eval"
    )

    py @reconArgs
    if ($LASTEXITCODE -ne 0) { throw "recon.py failed for $SCENE_NAME with exit code $LASTEXITCODE" }

    Write-Host "--------Start evaluating scene $SCENE_NAME with test name $TEST_NAME--------"

    $evalArgs = @(
        "evaluation/eval_recon.py",
        "--test_name=$TEST_NAME",
        "--gt_pcd=results/gt/replica/${SCENE_NAME}_pcds.npy"
    )

    py @evalArgs
    if ($LASTEXITCODE -ne 0) { throw "eval_recon.py failed for $SCENE_NAME with exit code $LASTEXITCODE" }
}

Write-Host "All scenes completed successfully."