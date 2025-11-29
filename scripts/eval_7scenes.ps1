# PowerShell automation for reconstructing + evaluating all Microsoft 7-Scenes sequences.
# Mirrors scripts/eval_replica.ps1 but iterates through every seq-* folder per scene.

param(
    [string[]]$Scenes = @("chess","fire","heads","office","pumpkin","redkitchen","stairs"),
    [string]$SceneRoot = "data/7Scenes",
    [string]$GtRoot = "results/gt/7scenes"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

if ($PSScriptRoot) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
    Set-Location $RepoRoot
}

######################################################################################
# Reconstruction parameters (see recon.py for detailed descriptions)
######################################################################################
$KEYFRAME_STRIDE = 10
$UPDATE_BUFFER_INTV = 3
$MAX_NUM_REGISTER = 10
$WIN_R = 5
$NUM_SCENE_FRAME = 10
$INITIAL_WINSIZE = 5
$CONF_THRES_I2P = 1.5

# Parameters below only affect saved recon/eval artifacts
$NUM_POINTS_SAVE = 1000000
$CONF_THRES_L2W = 10
$GPU_ID = -1

if (-not (Test-Path $SceneRoot -PathType Container)) {
    throw "Dataset directory not found: $SceneRoot"
}

foreach ($SceneName in $Scenes) {
    $SceneDir = Join-Path $SceneRoot $SceneName
    if (-not (Test-Path $SceneDir -PathType Container)) {
        Write-Warning "Skipping $SceneName (missing $SceneDir)"
        continue
    }

    $SeqDirs = Get-ChildItem -Path $SceneDir -Directory -Filter "seq-*" | Sort-Object Name
    if (-not $SeqDirs) {
        Write-Warning "Skipping $SceneName (no seq-* folders in $SceneDir)"
        continue
    }

    foreach ($SeqDir in $SeqDirs) {
        $SeqName = $SeqDir.Name
        $TestName = "7Scenes_{0}_{1}" -f $SceneName, $SeqName

        Write-Host "-------- Start reconstructing $SceneName/$SeqName with test name $TestName --------"
        $reconArgs = @(
            "recon.py",
            "--test_name", $TestName,
            "--img_dir", $SeqDir.FullName,
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
        if ($LASTEXITCODE -ne 0) { throw "recon.py failed for $SceneName/$SeqName with exit code $LASTEXITCODE" }

        $gtPcd = Join-Path $GtRoot ("{0}_{1}_pcds.npy" -f $SceneName, $SeqName)
        $gtMask = $gtPcd -replace '_pcds', '_valid_masks'

        if ((Test-Path $gtPcd -PathType Leaf) -and (Test-Path $gtMask -PathType Leaf)) {
            Write-Host "-------- Start evaluating $SceneName/$SeqName with test name $TestName --------"
            $evalArgs = @(
                "eval/eval_recon.py",
                "--test_name=$TestName",
                "--gt_pcd=$gtPcd"
            )
            py @evalArgs
            if ($LASTEXITCODE -ne 0) { throw "eval_recon.py failed for $SceneName/$SeqName with exit code $LASTEXITCODE" }
        }
        else {
            Write-Warning "Missing ground-truth pair ($gtPcd / $gtMask); skipping evaluation for $SceneName/$SeqName."
        }
    }
}

Write-Host "All requested 7-Scenes sequences completed."
