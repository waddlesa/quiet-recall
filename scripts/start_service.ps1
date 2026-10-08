param(
    [int]$Port = 9878,
    [ValidateSet('dense_bge', 'hybrid_rrf')]
    [string]$RetrievalMode = 'hybrid_rrf'
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found: $python"
}

Push-Location $projectRoot
try {
    & $python -m tools.v2_service --port $Port --retrieval-mode $RetrievalMode
}
finally {
    Pop-Location
}
