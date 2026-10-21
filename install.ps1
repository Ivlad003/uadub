# uadub on Windows with NVIDIA: uv → uadub[cuda] → Ollama + model → models → doctor.
$ErrorActionPreference = "Stop"
$Repo = if ($env:UADUB_SOURCE) { $env:UADUB_SOURCE } else { "git+https://github.com/Ivlad003/uadub.git" }

function Say($t) { Write-Host "`n$t" -ForegroundColor Cyan }

Say "1/6 uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv --version

Say "2/6 NVIDIA"
$Extra = "cpu"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  $Extra = "cuda"
} else {
  Write-Host "nvidia-smi не знайдено: ставлю CPU-варіант (повільний). Для CUDA встановіть драйвер NVIDIA і запустіть скрипт знову."
}

Say "3/6 uadub[$Extra]"
uv tool install --force --python 3.12 "uadub[$Extra] @ $Repo"
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

Say "4/6 ffmpeg"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  Write-Host "ffmpeg не знайдено. Встановлюю через winget…"
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
  Write-Host "Відкрийте новий термінал після встановлення, щоб ffmpeg з'явився в PATH."
}

Say "5/6 Ollama"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
  $env:Path = "$env:LOCALAPPDATA\Programs\Ollama;$env:Path"
}
try { Invoke-RestMethod http://127.0.0.1:11434/api/tags | Out-Null } catch { Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden; Start-Sleep 3 }

Say "6/6 моделі"
$env:PYTHONUTF8 = "1"
uadub --prefetch
uadub --doctor
