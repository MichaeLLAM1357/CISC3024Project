# LIVE DEMO launcher for Windows.  It finds a Python that can actually import torch and runs
# demo_live.py with it, so nothing has to be typed twice on stage.
#
#   powershell -ExecutionPolicy Bypass -File run_demo.ps1 -Seed 1 -PerClass 4 -Open
#   .\run_demo.ps1 -Seed 1 -PerClass 4 -Open
#
# No absolute paths: the launcher works from any folder after unzipping.
param(
    [string]$Submission = "submission",
    [int]$Seed = 0,
    [int]$PerClass = 4,
    [string]$Out = "",
    [int]$Tile = 128,
    [string]$Device = "cpu",
    [switch]$Open
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition

# ---------------------------------------------------------------- interpreter detection
# Candidates in order: a virtualenv inside the project (optional, not in the zip), then `python`
# on PATH, then the Windows `py` launcher.  Each is run once to see whether torch imports; the
# first one that does wins.  If none does, say which ones were tried and why each failed.
$candidates = New-Object System.Collections.ArrayList
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPy) { [void]$candidates.Add(@($venvPy)) }

$cmd = Get-Command python -ErrorAction SilentlyContinue
if ($cmd -and $cmd.Source -ne $venvPy) { [void]$candidates.Add(@($cmd.Source)) }

$cmd = Get-Command py -ErrorAction SilentlyContinue
if ($cmd) { [void]$candidates.Add(@($cmd.Source, "-3")) }

$chosen = $null
$tried = @()
foreach ($cand in $candidates) {
    $exe = $cand[0]
    $pre = @()
    if ($cand.Count -gt 1) { $pre = $cand[1..($cand.Count - 1)] }
    $label = (@($exe) + $pre) -join " "

    & $exe @pre -c "import sys" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { $tried += "$label -> cannot run (exit $LASTEXITCODE)"; continue }

    & $exe @pre -c "import torch" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $chosen = $cand; break }
    $tried += "$label -> runs, but torch is not installed"
}

if (-not $chosen) {
    Write-Host ""
    Write-Host "No usable Python found.  The demo needs Python 3 with torch, numpy and pillow." -ForegroundColor Red
    foreach ($line in $tried) { Write-Host "  $line" }
    if ($tried.Count -eq 0) { Write-Host "  no 'python' and no 'py' on PATH at all" }
    Write-Host ""
    Write-Host "Install them, then run this again:"
    Write-Host "  py -3 -m pip install torch numpy pillow --index-url https://download.pytorch.org/whl/cpu"
    Write-Host ""
    Write-Host "Fallback for the talk: open demo_seed1.png, the grid this command produces."
    exit 1
}

$py = $chosen[0]
$pyPre = @()
if ($chosen.Count -gt 1) { $pyPre = $chosen[1..($chosen.Count - 1)] }
Write-Host "python: $py $pyPre" -ForegroundColor DarkGray

# ---------------------------------------------------------------- submission path
# Relative paths may be relative to the current folder or to this launcher's folder.
$sub = $Submission
if (-not [System.IO.Path]::IsPathRooted($sub)) {
    if (Test-Path (Join-Path (Get-Location) $sub)) {
        $sub = Join-Path (Get-Location) $sub
    } elseif (Test-Path (Join-Path $root $sub)) {
        $sub = Join-Path $root $sub
    } elseif ((Split-Path $sub -Leaf) -eq "submission" -and (Test-Path (Join-Path $root "submission"))) {
        # e.g. "CISC3024_UC325629/submission" while sitting inside the unzipped folder
        $sub = Join-Path $root "submission"
    }
}

$script = Join-Path $root "demo_live.py"
$argsList = @($script, "--submission", $sub, "--seed", "$Seed", "--per-class", "$PerClass",
              "--tile", "$Tile", "--device", $Device)
if ($Out -ne "") { $argsList += @("--out", $Out) }
if ($Open) { $argsList += "--open" }

& $py @pyPre @argsList
