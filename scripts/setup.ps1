<#
.SYNOPSIS
  Crea .venv e instala el proyecto en Windows (PowerShell 5.1+ o 7).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -TorchIndex cu126
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -TorchIndex cpu
#>
param(
  [string]$TorchIndex = "cu128",   # cu128 | cu126 | cpu | pypi
  [string]$Python = ""             # ej. "py -3.11"; vacío = autodetectar
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Venv = ".venv"

function Run($exe, $arguments) {
  & $exe @arguments
  if ($LASTEXITCODE -ne 0) { throw "Falló: $exe $($arguments -join ' ')" }
}

# 1) Intérprete: preferir 3.11 vía el lanzador `py` (soportado: 3.10 a 3.13).
$candidates = @()
if ($Python) { $candidates += ,($Python -split ' ') }
else { $candidates += ,@("py","-3.11"), @("py","-3.12"), @("py","-3.10"), @("py","-3.13"), @("python") }

$base = $null
foreach ($c in $candidates) {
  try {
    $exe = $c[0]; $rest = @(); if ($c.Count -gt 1) { $rest = $c[1..($c.Count-1)] }
    $ok = & $exe @rest -c "import sys; print(int((3,10) <= sys.version_info[:2] <= (3,13)))" 2>$null
    if ($LASTEXITCODE -eq 0 -and $ok -eq "1") { $base = @{ exe = $exe; rest = $rest }; break }
  } catch { }
}
if (-not $base) { throw "No se encontró Python 3.10-3.13. Instalar desde python.org (con el lanzador 'py')." }
Write-Host ">> Python: $(& $base.exe @($base.rest) -V)"

# 2) Entorno virtual.
if (-not (Test-Path $Venv)) {
  Write-Host ">> Creando $Venv"
  Run $base.exe (@($base.rest) + @("-m","venv",$Venv))
}
$Py = Join-Path $Venv "Scripts\python.exe"
Run $Py @("-m","pip","install","--upgrade","pip","wheel")

# 3) PyTorch primero (en Windows, `pip install torch` a secas trae la versión CPU).
if ($TorchIndex -eq "pypi") {
  Write-Host ">> Instalando torch desde PyPI"
  Run $Py @("-m","pip","install","torch")
} else {
  Write-Host ">> Instalando torch desde https://download.pytorch.org/whl/$TorchIndex"
  Run $Py @("-m","pip","install","torch","--index-url","https://download.pytorch.org/whl/$TorchIndex")
}

Write-Host ">> Instalando dependencias"
Run $Py @("-m","pip","install","-r","requirements-dev.txt")
Run $Py @("-m","pip","install","-e",".","--no-deps")

# 4) Congelar el entorno exacto.
& $Py -m pip freeze --exclude-editable | Out-File -Encoding ascii requirements.lock

# 5) Extras de comodidad.
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host ">> Creado .env (completar TABPFN_TOKEN)" }
if (Test-Path ".git") { & (Join-Path $Venv "Scripts\pre-commit.exe") install }

# 6) Verificación.
& $Py scripts\check_env.py

Write-Host ""
Write-Host ">> Listo. Activar con:  .\$Venv\Scripts\Activate.ps1"
