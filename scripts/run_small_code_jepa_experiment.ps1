param(
    [string[]]$Stages = @("transform-v0"),
    [int]$TrainShards = 2,
    [int]$EvalShards = 1,
    [int]$EvalOffset = 0,
    [int]$MaxLen = 256,
    [int]$BatchSize = 4,
    [int]$Steps = 20,
    [double]$DurationHours = 0.25,
    [int]$CodeSearchN = 256,
    [int]$ClonePairs = 512,
    [string]$ExperimentDir = "runs/small-code-jepa-ablation-smoke"
)

$ErrorActionPreference = "Stop"

Get-Content ".env" | ForEach-Object {
    if ($_ -match "^\s*export\s+([^=]+)=(.*)$") {
        [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}

if (-not $env:S3_ENDPOINT_URL -or -not $env:S3_BUCKET) {
    throw "Missing S3_ENDPOINT_URL or S3_BUCKET in .env"
}

$tokenizerDir = Join-Path $ExperimentDir "tokenizers/codesearchnet-python/bpe16k"
$trainRoot = Join-Path $ExperimentDir "data/train"
$evalRoot = Join-Path $ExperimentDir "data/eval"
$runDir = Join-Path $ExperimentDir "train"
$evalDir = Join-Path $ExperimentDir "eval"
New-Item -ItemType Directory -Force -Path $tokenizerDir, $trainRoot, $evalRoot, $runDir, $evalDir | Out-Null

s5cmd --endpoint-url $env:S3_ENDPOINT_URL cp `
    "s3://$env:S3_BUCKET/tokenizers/codesearchnet-python/bpe16k/*" `
    "$tokenizerDir/"

function Copy-StageShardSet {
    param(
        [string]$Stage,
        [string]$DestRoot,
        [int]$Start,
        [int]$Count
    )
    $stageRoot = Join-Path $DestRoot $Stage
    New-Item -ItemType Directory -Force `
        -Path (Join-Path $stageRoot "views"), (Join-Path $stageRoot "triples") | Out-Null
    for ($i = $Start; $i -lt ($Start + $Count); $i++) {
        $name = "shard-{0:D5}.parquet" -f $i
        s5cmd --endpoint-url $env:S3_ENDPOINT_URL cp `
            "s3://$env:S3_BUCKET/data/codesearchnet-python/$Stage/views/$name" `
            "$(Join-Path $stageRoot "views")/"
        s5cmd --endpoint-url $env:S3_ENDPOINT_URL cp `
            "s3://$env:S3_BUCKET/data/codesearchnet-python/$Stage/triples/$name" `
            "$(Join-Path $stageRoot "triples")/"
    }
}

foreach ($stage in $Stages) {
    Copy-StageShardSet -Stage $stage -DestRoot $trainRoot -Start 0 -Count $TrainShards
    Copy-StageShardSet -Stage $stage -DestRoot $evalRoot -Start $EvalOffset -Count $EvalShards
}

$trainDataRoots = $Stages | ForEach-Object { Join-Path $trainRoot $_ }
$evalDataRoots = $Stages | ForEach-Object { Join-Path $evalRoot $_ }

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe failed with exit code $LASTEXITCODE"
    }
}

$trainArgs = @("scripts/train_codebert_jepa_torch.py", "--data-roots")
$trainArgs += $trainDataRoots
$trainArgs += @(
    "--output-dir", $runDir,
    "--model-name", $tokenizerDir,
    "--init", "unixcoder_small_scratch",
    "--max-len", "$MaxLen",
    "--batch-size", "$BatchSize",
    "--steps", "$Steps",
    "--duration-hours", "$DurationHours",
    "--precision", "fp32",
    "--eval-every", "0",
    "--save-every", "$Steps",
    "--no-gradient-checkpointing"
)
Invoke-Checked -Exe "python" -Arguments $trainArgs

$checkpoint = Join-Path $runDir "latest.pt"

$codeSearchArgs = @(
    "scripts/evaluate_codesearchnet_retrieval.py",
    "--checkpoint", $checkpoint,
    "--model-name", $tokenizerDir,
    "--local-dataset-dir", "data/raw/codesearchnet/python",
    "--n", "$CodeSearchN",
    "--batch-size", "$BatchSize",
    "--max-len-query", "$MaxLen",
    "--max-len-code", "$MaxLen",
    "--skip-base"
)
& python @codeSearchArgs | Tee-Object -FilePath (Join-Path $evalDir "codesearch.json")
if ($LASTEXITCODE -ne 0) {
    throw "CodeSearch evaluation failed with exit code $LASTEXITCODE"
}

$cloneArgs = @("scripts/evaluate_clone_detection.py", "--checkpoint", $checkpoint)
$cloneArgs += @("--model-name", $tokenizerDir, "--data-roots")
$cloneArgs += $evalDataRoots
$cloneArgs += @(
    "--max-examples", "$ClonePairs",
    "--batch-size", "$BatchSize",
    "--max-len", "$MaxLen",
    "--output-json", (Join-Path $evalDir "clone-transformed-triples.json")
)
Invoke-Checked -Exe "python" -Arguments $cloneArgs
