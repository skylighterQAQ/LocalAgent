# Removes from Git only the files now matched by .gitignore.
# Local working-copy files remain intact. Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File .\scripts\cleanup-tracked-temp-files.ps1

$ErrorActionPreference = 'Stop'

$repoRoot = git rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) {
    throw 'Run this script inside a Git repository.'
}

Set-Location $repoRoot
$ignoredTracked = @(git ls-files -ci --exclude-standard)
if ($ignoredTracked.Count -eq 0) {
    Write-Host 'No ignored files are currently tracked.'
    exit 0
}

Write-Host 'Removing these ignored files from Git tracking (local files are kept):'
$ignoredTracked | ForEach-Object { Write-Host "  $_" }
$ignoredTracked | git rm --cached --pathspec-from-file=-

Write-Host "`nReview with: git status --short"
Write-Host 'Then commit and push the deletion from the repository index.'
