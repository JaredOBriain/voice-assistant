# One-time setup for the Const-me/Whisper GPU trial. Run on the WINDOWS server
# PC — Const-me is Windows-only and the point of the trial is the GPU here.
#
#   powershell -ExecutionPolicy Bypass -File setup_constme.ps1
#
# Downloads the prebuilt CLI and a GGML model, then lists the GPUs it can see.
# Nothing here touches transcribe_server.py; the trial is measurement only.

$ErrorActionPreference = "Stop"
$dir = "C:\constme"

# Sizes are checked against the real ones so a truncated or HTML-error-page
# download is caught here rather than surfacing as a confusing CLI failure.
$cliUrl   = "https://github.com/Const-me/Whisper/releases/download/1.12.0/cli.zip"
$cliSize  = 377919
$modelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin"
$modelSize = 1533774781      # ~1.43 GB

New-Item -ItemType Directory -Force -Path $dir | Out-Null

# curl.exe, not Invoke-WebRequest: PowerShell 5.1 buffers the whole response in
# memory, which is a poor idea for a 1.4GB model. curl.exe ships with Windows
# 10 1803 and later.
function Get-Checked($url, $dest, $expected) {
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -eq $expected)) {
        Write-Host "already have $(Split-Path $dest -Leaf)" -ForegroundColor Green
        return
    }
    Write-Host "downloading $(Split-Path $dest -Leaf) ..."
    curl.exe -L --fail --progress-bar -o $dest $url
    $got = (Get-Item $dest).Length
    if ($got -ne $expected) {
        throw "$dest is $got bytes, expected $expected — download incomplete"
    }
    Write-Host "  ok ($got bytes)" -ForegroundColor Green
}

Get-Checked $cliUrl "$dir\cli.zip" $cliSize
Get-Checked $modelUrl "$dir\ggml-medium.en.bin" $modelSize

if (-not (Test-Path "$dir\main.exe")) {
    Write-Host "extracting cli.zip ..."
    Expand-Archive -Path "$dir\cli.zip" -DestinationPath $dir -Force
}

if (-not (Test-Path "$dir\main.exe")) {
    Write-Host "main.exe not found after extracting. Contents:" -ForegroundColor Yellow
    Get-ChildItem $dir -Recurse -Filter *.exe | Select-Object FullName
    throw "Locate main.exe and pass its path to bench_constme.py with --exe"
}

Write-Host "`n=== GPUs Const-me can see ===" -ForegroundColor Cyan
& "$dir\main.exe" -la

Write-Host @"

Check the RX 590 is listed above. If it falls back to the integrated GPU the
benchmark numbers mean nothing.

Next:
  `$env:WHISPER_AUTH_TOKEN = "<your token>"
  python bench_constme.py

(transcribe_server.py must be running locally — it is what gets raced.)
"@ -ForegroundColor Cyan
