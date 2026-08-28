<#
.SYNOPSIS
  Res-TSSDNet vertical-slice driver (Phase 1 reproduction -> Phase 2 step 2.3
  parity gate). Windows PowerShell 5.1 compatible: no &&, no ||, no ternary,
  no null-coalescing, no -AsHashtable.

.DESCRIPTION
  Stages, in order:
    check    DATA_ROOT layout report            (no model needed)
    smoke    wrapper/checkpoint/sign smoke test (needs vendor + torch)
    repro    score 19LA dev + eval              (the long one)
    report   EER, gate verdict, tau freeze
    convert  ONNX fp32 -> int8 PTQ -> ExecuTorch
    parity   manifest, ONNX rescore, decision-flip gate
    all      everything above

.EXAMPLE
  .\scripts\pilot_tssdnet.ps1 -DataRoot D:\DF_Dataset -Stage check
  .\scripts\pilot_tssdnet.ps1 -DataRoot D:\DF_Dataset -Stage all -Device cpu
#>
[CmdletBinding()]
param(
    [string] $DataRoot = "D:\DF_Dataset",
    [ValidateSet("check","smoke","repro","report","convert","parity","all")]
    [string] $Stage = "all",
    [string] $Device = "cpu",
    [string] $Python = "python",
    [string] $Checkpoint = "",
    [switch] $Refreeze
)

$ErrorActionPreference = "Stop"

$RepoRoot    = Split-Path -Parent $PSScriptRoot
# Results moved under the repo on 2026-07-27 so they are version-controlled with
# the code that produced them.
$P1          = Join-Path $RepoRoot "results\phase_1_reproduction"
$P2          = Join-Path $RepoRoot "results\phase_2_conversion_app"
$Exports     = Join-Path $RepoRoot "exports"
$VendorDir   = Join-Path $RepoRoot "vendor\end-to-end-synthetic-speech-detection"

foreach ($d in @($P1, $P2, $Exports)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

$env:DATA_ROOT  = $DataRoot
$env:PYTHONPATH = $RepoRoot
Set-Location $RepoRoot

# Reproduction anchor. The author's README reports 1.6590% on Linux and 1.64%
# on Windows (soundfile FLAC decoding differed in 2021). A 15% relative gate
# spans 1.410-1.908, so BOTH numbers sit inside it -- a Windows run cannot be
# rejected for being a Windows run. Record the OS in the report either way.
$Published = "1.6590"
$PubNote   = "Linux pretrained Res-TSSDNet, ghua-ac README @36c9705"

function Say($m) { Write-Host ""; Write-Host "=== $m" -ForegroundColor Cyan }

function Invoke-Step {
    param([string] $Label, [string[]] $ArgList)
    Say $Label
    Write-Host "$Python $($ArgList -join ' ')" -ForegroundColor DarkGray
    & $Python @ArgList
    if ($LASTEXITCODE -ne 0) {
        throw "$Label FAILED (exit $LASTEXITCODE). Stopping -- later stages would build on a broken result."
    }
}

function Resolve-Checkpoint {
    if ($Checkpoint -ne "") { return $Checkpoint }
    $pre = Join-Path $VendorDir "pretrained"
    if (-not (Test-Path $pre)) {
        throw "no $pre -- clone the vendor repo first: git clone https://github.com/ghua-ac/end-to-end-synthetic-speech-detection `"$VendorDir`""
    }
    $hit = Get-ChildItem -Path $pre -Filter "Res_TSSDNet*.pth" | Select-Object -First 1
    if ($null -eq $hit) { throw "no Res_TSSDNet*.pth under $pre" }
    return $hit.FullName
}

$doAll = ($Stage -eq "all")

# ---------------------------------------------------------------- 0. check
if ($doAll -or $Stage -eq "check") {
    Say "0. DATA_ROOT layout check (DATA_ROOT=$DataRoot)"
    Write-Host "$Python -m deploybench.data --check --datasets asvspoof19_dev,asvspoof19_eval" -ForegroundColor DarkGray
    & $Python -m deploybench.data --check --datasets asvspoof19_dev,asvspoof19_eval
    Write-Host ""
    Write-Host "If any line above says MISS, PASTE THIS OUTPUT BACK to Claude before running any later stage." -ForegroundColor Yellow
    Write-Host "REGISTRY still holds layout guesses; do not hand-edit it."
    if ($Stage -eq "check") { exit 0 }
}

$Ckpt = Resolve-Checkpoint
Write-Host "checkpoint: $Ckpt" -ForegroundColor Green

# ---------------------------------------------------------------- 1. smoke
if ($doAll -or $Stage -eq "smoke") {
    Invoke-Step "1. smoke test (200 dev clips)" @(
        "scripts\smoke_tssdnet.py", "--device", $Device,
        "--checkpoint", $Ckpt, "--n", "200",
        "--out", (Join-Path $P1 "smoke_tssdnet.json"))
    if ($Stage -eq "smoke") { exit 0 }
}

# ---------------------------------------------------------------- 2. repro
$DevCsv  = Join-Path $P1 "scores_dev_tssdnet.csv"
$EvalCsv = Join-Path $P1 "scores_repro_tssdnet.csv"

if ($doAll -or $Stage -eq "repro") {
    Invoke-Step "2a. score 19LA DEV (tau calibration source, ~24.8k clips)" @(
        "score.py", "--model", "tssdnet", "--checkpoint", $Ckpt,
        "--dataset", "asvspoof19_dev", "--device", $Device,
        "--precision", "fp32", "--runtime", "pytorch", "--condition", "clean",
        "--batch", "32", "--progress", "2000", "--out", $DevCsv)

    Invoke-Step "2b. score 19LA EVAL (reproduction gate, ~71.2k clips)" @(
        "score.py", "--model", "tssdnet", "--checkpoint", $Ckpt,
        "--dataset", "asvspoof19_eval", "--device", $Device,
        "--precision", "fp32", "--runtime", "pytorch", "--condition", "clean",
        "--batch", "32", "--progress", "5000", "--out", $EvalCsv)
    if ($Stage -eq "repro") { exit 0 }
}

# --------------------------------------------------------------- 3. report
$TauJson = Join-Path $P1 "tau_tssdnet.json"
if ($doAll -or $Stage -eq "report") {
    $rargs = @("analysis\repro_report.py", "--model", "tssdnet",
               "--eval-scores", $EvalCsv, "--dev-scores", $DevCsv,
               "--published", $Published, "--published-note", $PubNote,
               "--gate", "0.15",
               "--out-report", (Join-Path $P1 "repro_report_tssdnet.md"),
               "--out-tau", $TauJson)
    if ($Refreeze) { $rargs += "--refreeze" }
    Invoke-Step "3. EER + gate verdict + tau freeze" $rargs
    if ($Stage -eq "report") { exit 0 }
}

# -------------------------------------------------------------- 4. convert
if ($doAll -or $Stage -eq "convert") {
    # D27: 500-clip BALANCED calibration set drawn from source DEV, never eval.
    $CalibList = Join-Path $RepoRoot "configs\calibration_500_tssdnet.txt"
    Invoke-Step "4a. calibration manifest (250 bona + 250 spoof, 19LA DEV)" @(
        "scripts\make_parity_manifest.py", "--dataset", "asvspoof19_dev",
        "--n-per-class", "250", "--seed", "20260725", "--out", $CalibList)

    Invoke-Step "4b. ONNX fp32 export (single-score adapter, D13)" @(
        "convert\to_onnx.py", "--model", "tssdnet", "--precision", "fp32",
        "--checkpoint", $Ckpt, "--out", $Exports)

    Say "4c. int8 PTQ (S8S8, per-channel, QDQ; D17/D27)"
    Write-Host "OPEN ITEM: --keep-io-fp32 is wired to extra_options{QuantizeBias:False}," -ForegroundColor Yellow
    Write-Host "which is about bias, not graph I/O. Read graph_input_type in the ptq log" -ForegroundColor Yellow
    Write-Host "(1=FLOAT) before calling the input fp32. See PILOT_tssdnet.md step 5." -ForegroundColor Yellow
    $CalibDir = Join-Path $DataRoot "LA\ASVspoof2019_LA_dev\flac"

    # RECIPE ESTABLISHED BY EXPERIMENT 2026-07-26 -- do not change casually.
    # Measured on the 1,000-clip parity manifest (fp32 EER 1.40%):
    #   minmax, no exclusions .......... EER 17.20%, 523 flips  (score scale COLLAPSED)
    #   percentile, no exclusions ...... EER 12%   (200-clip triage)
    #   percentile + first 5 convs fp32  EER  3.00%,  89 flips
    #   percentile + first 8 convs fp32  EER  2.20%,  60 flips  <-- chosen
    # Ruled out by direct experiment (all indistinguishable from baseline):
    # per-channel vs per-tensor, bias quantisation, excluding BatchNorm, and
    # int8-vs-uint8 activations. SYMMETRIC activations are actively harmful.
    # Why percentile: minmax takes literal extremes, and on a RAW-WAVEFORM net a
    # few loud calibration clips stretch the activation scale until ordinary
    # activations occupy a handful of the 256 levels.
    # Why the early convs: quantisation damage is concentrated where amplitude
    # dynamic range is largest; past K=8 there is no further gain.
    # NOTE percentile calibration is histogram-based -- ORT OOMs on 500 clips at
    # 96,000 samples, so the calibration subset here is 100, not D27's 500.
    # DEVIATION FROM D27 -- flagged, not silently absorbed.
    $CalibSmall = Join-Path $RepoRoot "configs\calibration_100_tssdnet.txt"
    if (-not (Test-Path $CalibSmall)) {
        & $Python "scripts\make_parity_manifest.py" --dataset asvspoof19_dev `
            --n-per-class 50 --seed 20260725 --out $CalibSmall
    }
    if (Test-Path $CalibDir) {
        $ExcludeConvs = & $Python -c "import onnx,sys;g=onnx.load(sys.argv[1]).graph;print(','.join([n.name for n in g.node if n.op_type=='Conv'][:8]))" (Join-Path $Exports "tssdnet_fp32.onnx")
        Invoke-Step "4c. int8 PTQ" @(
            "convert\ptq.py", "--onnx", (Join-Path $Exports "tssdnet_fp32.onnx"),
            "--calib-dir", $CalibDir, "--calib-manifest", $CalibSmall,
            "--input-samples", "96000",
            "--calib-method", "percentile", "--keep-io-fp32",
            "--exclude-nodes", $ExcludeConvs)
    } else {
        Write-Host "SKIPPED: $CalibDir not found -- settle the layout first (stage check)." -ForegroundColor Red
    }

    Invoke-Step "4d. ExecuTorch export (XNNPACK)" @(
        "convert\to_executorch.py", "--model", "tssdnet",
        "--checkpoint", $Ckpt, "--out", $Exports)

    Copy-Item (Join-Path $Exports "conversion_log_*.json") $P2 -Force -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $Exports "*_ptq_log.json") $P2 -Force -ErrorAction SilentlyContinue
    if ($Stage -eq "convert") { exit 0 }
}

# --------------------------------------------------------------- 5. parity
if ($doAll -or $Stage -eq "parity") {
    $Manifest = Join-Path $RepoRoot "configs\parity_manifest_tssdnet.txt"
    if (-not (Test-Path $Manifest)) {
        Invoke-Step "5a. parity manifest (500 bona + 500 spoof, 19LA EVAL)" @(
            "scripts\make_parity_manifest.py", "--dataset", "asvspoof19_eval",
            "--n-per-class", "500", "--seed", "20260725", "--out", $Manifest)
    } else {
        Say "5a. parity manifest already exists -- REUSING (it is frozen)"
    }

    $Ref = Join-Path $P2 "parity_ref_tssdnet_fp32_pytorch.csv"
    Invoke-Step "5b. reference scores (PyTorch fp32) over the manifest" @(
        "score.py", "--model", "tssdnet", "--checkpoint", $Ckpt,
        "--dataset", "asvspoof19_eval", "--manifest", $Manifest,
        "--device", "cpu", "--precision", "fp32", "--runtime", "pytorch",
        "--out", $Ref)

    $TauFlat = Join-Path $P2 "tau_tssdnet_flat.json"
    (Get-Content (Join-Path $P1 "tau_tssdnet.json") -Raw | ConvertFrom-Json).taus |
        ConvertTo-Json | Out-File -FilePath $TauFlat -Encoding utf8

    foreach ($prec in @("fp32", "int8")) {
        $onnx = Join-Path $Exports "tssdnet_$prec.onnx"
        if (-not (Test-Path $onnx)) {
            Write-Host "skip $prec -- $onnx not present" -ForegroundColor Yellow
            continue
        }
        $test = Join-Path $P2 "parity_test_tssdnet_${prec}_onnx.csv"
        Invoke-Step "5c. ONNX $prec scores over the manifest" @(
            "score_onnx.py", "--onnx", $onnx, "--model", "tssdnet",
            "--dataset", "asvspoof19_eval", "--manifest", $Manifest,
            "--precision", $prec, "--runtime", "onnx", "--out", $test)

        # fp32-vs-fp32 is a CONVERSION check (same maths, different runtime) and
        # legitimately gates at zero flips. int8-vs-fp32 compares DIFFERENT
        # precisions -- gating that at zero flips would demand lossless
        # quantisation, and it is not what Stage P3 means by parity. P3 parity is
        # the SAME artifact desktop-vs-DEVICE; see stage 6 below.
        if ($prec -eq "fp32") { $mode = "conversion" } else { $mode = "quantization" }
        Say "5d. $mode check: ONNX $prec vs PyTorch fp32"
        & $Python "tests\parity.py" --ref $Ref --test $test --taus $TauFlat --mode $mode |
            Tee-Object -FilePath (Join-Path $P2 "${mode}_report_tssdnet_$prec.json")
        if ($mode -eq "quantization") {
            Write-Host "DIAGNOSTIC ONLY -- these flips are the COST of int8 (an H6 result row)," -ForegroundColor Cyan
            Write-Host "not a parity verdict. Read eer_pct.delta_pp to see what discrimination survived." -ForegroundColor Cyan
        } elseif ($LASTEXITCODE -ne 0) {
            Write-Host "CONVERSION PARITY FAILED for $prec -- decision flips are non-zero." -ForegroundColor Red
            Write-Host "TSSDNet is the clean-converter control: this indicts OUR pipeline, not the ops." -ForegroundColor Red
        } else {
            Write-Host "CONVERSION PARITY PASS for $prec (zero decision flips)" -ForegroundColor Green
        }
    }

    Write-Host ""
    Write-Host "NOTE: Stage P3 parity (experiment_protocol.md) is DESKTOP vs ON-DEVICE" -ForegroundColor Yellow
    Write-Host "over this same manifest. Nothing above touched a phone, so P3 is NOT done." -ForegroundColor Yellow
    Write-Host "When an APK + handset exist:  python tests\parity.py --ref <desktop int8 csv> \" -ForegroundColor Yellow
    Write-Host "  --test <device int8 csv> --taus $TauFlat --mode device" -ForegroundColor Yellow
}

Say "done. Artifacts:"
Write-Host "  $P1"
Write-Host "  $P2"
Write-Host "  $Exports"
