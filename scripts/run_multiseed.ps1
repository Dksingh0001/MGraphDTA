param(
    [string]$ExperimentRoot = "results/davis/multiseed_retry_20261016",
    [int[]]$Seeds = @(42, 43, 44, 45, 46)
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project interpreter not found at $python. Create the documented .venv first."
}
if ($Seeds.Count -eq 0 -or ($Seeds | Select-Object -Unique).Count -ne $Seeds.Count) {
    throw "Provide one or more unique seed values."
}

Push-Location $projectRoot
try {
    $summaryCsv = Join-Path $ExperimentRoot "summary.csv"
    $summaryJson = Join-Path $ExperimentRoot "summary.json"
    $failureLog = Join-Path $ExperimentRoot "failures.json"
    $existingSummary = @($summaryCsv, $summaryJson, $failureLog) |
        Where-Object { Test-Path -LiteralPath $_ }
    if ($existingSummary.Count -gt 0) {
        throw "Experiment summary already exists; choose a new -ExperimentRoot to preserve prior results: $($existingSummary -join ', ')"
    }
    New-Item -ItemType Directory -Path $ExperimentRoot -Force | Out-Null

    $failedRuns = @()
    foreach ($seed in $Seeds) {
        $runDirectory = Join-Path $ExperimentRoot "seed_$seed"
        Write-Host "Starting Davis MGraphDTA run for seed $seed -> $runDirectory"
        & $python -m src.train `
            --epochs 150 `
            --steps-per-epoch 50 `
            --batch-size 32 `
            --learning-rate 0.0005 `
            --patience 30 `
            --validation-fraction 0.1 `
            --seed $seed `
            --device cuda `
            --no-deterministic `
            --run-dir $runDirectory
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            Write-Warning "Seed $seed exited with code $exitCode. Its partial artifacts will be retained and marked incomplete/invalid."
            $failedRuns += [pscustomobject]@{
                seed = $seed
                exit_code = $exitCode
                run_directory = $runDirectory
            }
        }
    }

    $failurePayload = @{
        experiment_root = $ExperimentRoot
        failures = @($failedRuns)
    } | ConvertTo-Json -Depth 4
    Set-Content -LiteralPath $failureLog -Value $failurePayload -Encoding utf8

    & $python -m scripts.aggregate_multiseed `
        --runs-dir $ExperimentRoot `
        --seeds $Seeds `
        --baseline results/davis/run_112_backup/evaluation_metrics_corrected.json
    if ($LASTEXITCODE -ne 0) {
        throw "Multi-seed aggregation failed with exit code $LASTEXITCODE. Failure details are in $failureLog."
    }

    if ($failedRuns.Count -gt 0) {
        $failedSeedIds = ($failedRuns | ForEach-Object { $_.seed }) -join ', '
        throw "Training failed for seed(s): $failedSeedIds. Partial outputs were retained; see $failureLog and summary.json. Retry only failed seeds with a new -ExperimentRoot."
    }
}
finally {
    Pop-Location
}
