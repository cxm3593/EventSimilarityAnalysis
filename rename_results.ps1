# Rename EventSimilarityAnalysis result folders to descriptive names.
# Generated 2026-08-19. Run from anywhere; edit $ResultsDir if you move the repo.
#
#   Dry run (default, changes nothing):   .\rename_results.ps1
#   Apply:                                .\rename_results.ps1 -Apply

param([switch]$Apply)

$ResultsDir = "C:\Users\cxm3593\Academic\Workspace\EventSimilarityAnalysis\results"

$map = @(
    @{ old = "run_mmd_20260819_140627"; new = "mmd_f1_rbf03_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_182543"; new = "mmd_f1_rbf15_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_142433"; new = "mmd_f1_rbf75_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_140742"; new = "mmd_f2_rbf03_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_182717"; new = "mmd_f2_rbf15_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_142546"; new = "mmd_f2_rbf75_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_140957"; new = "mmd_f3_rbf03_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_182917"; new = "mmd_f3_rbf15_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_142851"; new = "mmd_f3_rbf75_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_141320"; new = "mmd_f4_rbf03_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_183509"; new = "mmd_f4_rbf15_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_143145"; new = "mmd_f4_rbf75_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_141809"; new = "mmd_f5_rbf03_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_183957"; new = "mmd_f5_rbf15_w1000us_d999ms" },
    @{ old = "run_mmd_20260819_143613"; new = "mmd_f5_rbf75_w1000us_d999ms" },
    @{ old = "run_sliced_wasserstein_20260819_135412"; new = "swd_f1_p100_w1000us_d999ms" },
    @{ old = "run_sliced_wasserstein_20260819_135719"; new = "swd_f2_p100_w1000us_d999ms" },
    @{ old = "run_sliced_wasserstein_20260819_135909"; new = "swd_f3_p100_w1000us_d999ms" },
    @{ old = "run_sliced_wasserstein_20260819_140049"; new = "swd_f4_p100_w1000us_d999ms" },
    @{ old = "run_sliced_wasserstein_20260819_140340"; new = "swd_f5_p100_w1000us_d999ms" }
)

if (-not (Test-Path -LiteralPath $ResultsDir)) {
    Write-Error "Results directory not found: $ResultsDir"; exit 1
}

$renamed = 0; $skipped = 0; $missing = 0; $blocked = 0

foreach ($e in $map) {
    $src = Join-Path $ResultsDir $e.old
    $dst = Join-Path $ResultsDir $e.new

    if (Test-Path -LiteralPath $dst) {
        if (Test-Path -LiteralPath $src) {
            Write-Host "BLOCKED  $($e.old) -> $($e.new)  (target already exists)" -ForegroundColor Red
            $blocked++
        } else {
            Write-Host "done     $($e.new)  (already renamed)" -ForegroundColor DarkGray
            $skipped++
        }
        continue
    }
    if (-not (Test-Path -LiteralPath $src)) {
        Write-Host "MISSING  $($e.old)" -ForegroundColor Yellow
        $missing++
        continue
    }

    if ($Apply) {
        Rename-Item -LiteralPath $src -NewName $e.new -ErrorAction Stop
        Write-Host "renamed  $($e.old) -> $($e.new)" -ForegroundColor Green
    } else {
        Write-Host "would    $($e.old) -> $($e.new)"
    }
    $renamed++
}

Write-Host ""
if ($Apply) {
    Write-Host "Renamed $renamed, already done $skipped, missing $missing, blocked $blocked"
} else {
    Write-Host "Dry run: $renamed would be renamed, $skipped already done, $missing missing, $blocked blocked."
    Write-Host "Re-run with -Apply to perform the renames."
}
