# uadub on Windows with NVIDIA: uv → uadub[cuda] → Ollama + model → models → doctor.
$ErrorActionPreference = "Stop"
$Repo = if ($env:UADUB_SOURCE) { $env:UADUB_SOURCE } else { "git+https://github.com/Ivlad003/uadub.git" }

function Say($t) { Write-Host "`n$t" -ForegroundColor Cyan }

# Run a native command (first element is the executable) and stop on a non-zero exit code.
function Run($What, [string[]]$Cmd) {
  $exe = $Cmd[0]
  $rest = @($Cmd | Select-Object -Skip 1)
  & $exe @rest
  if ($LASTEXITCODE -ne 0) { throw "крок «$What» завершився з кодом $LASTEXITCODE" }
}

# Add the registry PATH (Machine + User) to the current session PATH, so tools installed by winget are visible.
function Refresh-Path {
  $env:Path = $env:Path + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}

function Ollama-Up {
  try { Invoke-RestMethod http://127.0.0.1:11434/api/tags | Out-Null; return $true } catch { return $false }
}

try {
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

  if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Потрібен git: winget install --id Git.Git -e"
  }

  Say "3/6 uadub[$Extra]"
  Run "uv tool install" @("uv", "tool", "install", "--force", "--python", "3.12", "uadub[$Extra] @ $Repo")
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

  Say "4/6 ffmpeg"
  if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
      throw "Потрібен winget (App Installer з Microsoft Store)"
    }
    Write-Host "ffmpeg не знайдено. Встановлюю через winget…"
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
      Write-Host "winget повернув код $LASTEXITCODE для Gyan.FFmpeg; якщо програму вже встановлено, це нормально — перевірте у звіті --doctor нижче"
    }
    Refresh-Path
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
      Write-Host "Відкрийте новий термінал після встановлення, щоб ffmpeg з'явився в PATH."
    }
  }

  Say "5/6 Ollama"
  if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
      throw "Потрібен winget (App Installer з Microsoft Store)"
    }
    winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
      Write-Host "winget повернув код $LASTEXITCODE для Ollama.Ollama; якщо програму вже встановлено, це нормально — перевірте у звіті --doctor нижче"
    }
    $env:Path = "$env:LOCALAPPDATA\Programs\Ollama;$env:Path"
    Refresh-Path
  }
  if (-not (Ollama-Up)) {
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    # Wait up to 30 s for the Ollama server to answer.
    for ($i = 0; $i -lt 30 -and -not (Ollama-Up); $i++) { Start-Sleep 1 }
    if (-not (Ollama-Up)) {
      Write-Host "Ollama не піднявся за 30 с — запустіть «ollama serve» вручну і повторіть uadub --prefetch"
    }
  }

  Say "6/6 моделі"
  $env:PYTHONUTF8 = "1"
  Run "uadub --prefetch" @("uadub", "--prefetch")
  uadub --doctor
} catch {
  Write-Host "`nВстановлення зупинилось: $($_.Exception.Message)`nВиправте причину вище і запустіть скрипт знову." -ForegroundColor Red
  exit 1
}
