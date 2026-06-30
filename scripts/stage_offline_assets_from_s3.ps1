param(
    [string]$OutputDir = "offline_assets/code-jepa-small",
    [string]$Bucket = "",
    [string]$EndpointUrl = "",
    [string[]]$Stages = @("transform-v0", "transform-v1", "transform-v2"),
    [ValidateSet("all", "range")]
    [string]$ShardMode = "all",
    [int]$ShardStart = 0,
    [int]$ShardCount = 0,
    [string]$TokenizerName = "bpe16k",
    [switch]$SkipTokenizer,
    [switch]$IncludeBenchmarks,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*(?:export\s+)?([^#=\s]+)=(.*)$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"').Trim("'")
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

if (-not $Bucket) {
    $Bucket = $env:S3_BUCKET
}
if (-not $EndpointUrl) {
    $EndpointUrl = $env:S3_ENDPOINT_URL
}
if (-not $Bucket) {
    throw "Missing S3 bucket. Pass -Bucket or set S3_BUCKET in .env."
}
if ($ShardMode -eq "range" -and $ShardCount -le 0) {
    throw "-ShardMode range requires -ShardCount > 0."
}

function Invoke-S5cmd {
    param([string[]]$Arguments)
    $fullArgs = @()
    if ($EndpointUrl) {
        $fullArgs += @("--endpoint-url", $EndpointUrl)
    }
    $fullArgs += $Arguments
    Write-Host ("s5cmd " + ($fullArgs -join " "))
    if ($DryRun) {
        return
    }
    & s5cmd @fullArgs
    if ($LASTEXITCODE -ne 0) {
        throw "s5cmd failed with exit code $LASTEXITCODE"
    }
}

function New-Dir {
    param([string]$Path)
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

$tokenizerDir = Join-Path $OutputDir "tokenizers/codesearchnet-python/$TokenizerName"
$pretrainRoot = Join-Path $OutputDir "pretrain/codesearchnet-python"
$benchmarkRoot = Join-Path $OutputDir "benchmarks/codexglue"
New-Dir $pretrainRoot

if (-not $SkipTokenizer) {
    New-Dir $tokenizerDir
    Invoke-S5cmd @(
        "sync",
        "--size-only",
        "s3://$Bucket/tokenizers/codesearchnet-python/$TokenizerName/*",
        "$tokenizerDir/"
    )
}

Invoke-S5cmd @(
    "cp",
    "s3://$Bucket/data/codesearchnet-python/manifest.json",
    "$pretrainRoot/"
)

foreach ($stage in $Stages) {
    $stageRoot = Join-Path $pretrainRoot $stage
    $viewsDir = Join-Path $stageRoot "views"
    $triplesDir = Join-Path $stageRoot "triples"
    New-Dir $viewsDir
    New-Dir $triplesDir

    if ($ShardMode -eq "all") {
        Invoke-S5cmd @(
            "sync",
            "--size-only",
            "s3://$Bucket/data/codesearchnet-python/$stage/views/*",
            "$viewsDir/"
        )
        Invoke-S5cmd @(
            "sync",
            "--size-only",
            "s3://$Bucket/data/codesearchnet-python/$stage/triples/*",
            "$triplesDir/"
        )
    } else {
        for ($i = $ShardStart; $i -lt ($ShardStart + $ShardCount); $i++) {
            $name = "shard-{0:D5}.parquet" -f $i
            Invoke-S5cmd @(
                "cp",
                "s3://$Bucket/data/codesearchnet-python/$stage/views/$name",
                "$viewsDir/"
            )
            Invoke-S5cmd @(
                "cp",
                "s3://$Bucket/data/codesearchnet-python/$stage/triples/$name",
                "$triplesDir/"
            )
        }
    }
}

if ($IncludeBenchmarks) {
    New-Dir $benchmarkRoot
    $downloadArgs = @(
        "scripts/download_codexglue_benchmarks.py",
        "--output-root",
        $benchmarkRoot,
        "--benchmarks",
        "bigclonebench",
        "poj104",
        "--prepare-poj"
    )
    if ($DryRun) {
        $downloadArgs += "--dry-run"
    }
    Write-Host ("python " + ($downloadArgs -join " "))
    if (-not $DryRun) {
        & python @downloadArgs
        if ($LASTEXITCODE -ne 0) {
            throw "benchmark download failed with exit code $LASTEXITCODE"
        }
    }
}

$manifest = [ordered]@{
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    bucket = $Bucket
    endpoint_url = $EndpointUrl
    tokenizer = if ($SkipTokenizer) {
        "repo assets/tokenizers/codesearchnet-python/$TokenizerName"
    } else {
        "tokenizers/codesearchnet-python/$TokenizerName"
    }
    pretrain_root = "pretrain/codesearchnet-python"
    stages = $Stages
    shard_mode = $ShardMode
    shard_start = $ShardStart
    shard_count = $ShardCount
    include_benchmarks = [bool]$IncludeBenchmarks
    benchmark_root = "benchmarks/codexglue"
    note = "Copy this OutputDir to the offline cluster; no S3 access is needed after staging."
}

if (-not $DryRun) {
    $manifestPath = Join-Path $OutputDir "offline_manifest.json"
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -Path $manifestPath -Encoding UTF8
    Write-Host "Wrote $manifestPath"
}
