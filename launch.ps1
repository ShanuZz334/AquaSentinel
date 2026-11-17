param(
    [switch]$NoBrowser = $false,
    [switch]$SkipDeps = $false
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  AQUASENTINEL: Underwater Vision-Language Mission System             " -ForegroundColor White
Write-Host "  Unified Auto-Build & Launch Pipeline (Ctrl+Shift+B)                 " -ForegroundColor DarkGray
Write-Host "======================================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $scriptDir) { $scriptDir = (Get-Location).Path }
Set-Location $scriptDir

# 1. Resolve Python Runtime Environment
$candidates = @(
    "C:\project\ALLBACKUP\praxis-research\.venv\Scripts\python.exe",
    (Join-Path $scriptDir "..\praxis-research\.venv\Scripts\python.exe"),
    (Join-Path $scriptDir ".venv\Scripts\python.exe"),
    "python"
)

$pythonExe = $null
foreach ($c in $candidates) {
    if (Test-Path $c) {
        $pythonExe = (Resolve-Path $c).Path
        break
    }
}
if (-not $pythonExe) {
    $pythonExe = "python"
}

Write-Host "[1/4] Python Runtime: $pythonExe" -ForegroundColor Cyan

# 2. Dependency Verification
if (-not $SkipDeps) {
    Write-Host "[2/4] Verifying and resolving dependencies..." -ForegroundColor Yellow
    $reqPath = Join-Path $scriptDir "requirements.txt"
    if (Test-Path $reqPath) {
        & $pythonExe -m pip install -r $reqPath --quiet
        if ($LASTEXITCODE -eq 0) {
            Write-Host "      All dependencies verified." -ForegroundColor Green
        } else {
            Write-Host "      Warning: pip exited with code $LASTEXITCODE. Continuing..." -ForegroundColor DarkYellow
        }
    }
} else {
    Write-Host "[2/4] Skipping dependency check (-SkipDeps)." -ForegroundColor DarkGray
}

# 3. Model Weights & Adapter Verification
Write-Host "[3/4] Verifying Marine Domain Adapter weights..." -ForegroundColor Yellow
$weightsPath = Join-Path $scriptDir "weights\aquasentinel_marine_adapter.pt"
if (-not (Test-Path $weightsPath)) {
    Write-Host "      Adapter weights not found. Fine-tuning domain adapter now..." -ForegroundColor Magenta
    & $pythonExe -c "import sys; sys.path.insert(0, r'$scriptDir'); from core.fine_tuner import train_marine_adapter; train_marine_adapter()"
} else {
    $wSize = [math]::Round((Get-Item $weightsPath).Length / 1MB, 2)
    Write-Host "      Found adapter checkpoint: $weightsPath ($wSize MB)" -ForegroundColor Green
}

# 4. Port Check & Stale Process Handling
$existing = Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue
if ($existing) {
    $pids = $existing | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($p in $pids) {
        if ($p -gt 0) {
            Write-Host "      Freeing port 7860 (terminating stale PID $p)..." -ForegroundColor Yellow
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 1
}

# 5. Open Browser in Background
if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        Start-Sleep -Seconds 3
        Start-Process "http://localhost:7860"
    } | Out-Null
}

# 6. Execute AquaSentinel Mission Server
Write-Host "[4/4] Starting AquaSentinel Server on http://localhost:7860 ..." -ForegroundColor Green
Write-Host "      Press Ctrl+C in this terminal anytime to stop the server." -ForegroundColor DarkGray
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray

& $pythonExe app.py
