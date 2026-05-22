@echo off
:: One-time setup for YUME: portable Python + local .venv + all deps.
:: After this finishes, run run_oneclick_debug.bat (or bootstrap.py) to launch.
:: Idempotent — re-running is a no-op once %VENV%\.installed.ok is present.

setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

:: Drop any inherited venv vars before spawning a different interpreter,
:: otherwise that interpreter inherits a stale VIRTUAL_ENV and may hit an
:: SRE / stdlib module mismatch when loading sys/re/etc.
set VIRTUAL_ENV=
set PYTHONHOME=
set PYTHONPATH=

set "ROOT=%~dp0"
set "PYPORT=%ROOT%.pyport"
set "VENV=%ROOT%.venv"
set "CACHE=%ROOT%.cache"
set "PIP_CACHE_DIR=%ROOT%.pip-cache"
set "HF_HOME=%CACHE%\huggingface"
set "TRANSFORMERS_CACHE=%CACHE%\transformers"
set "TORCH_HOME=%CACHE%\torch"
set "XDG_CACHE_HOME=%CACHE%"
set "PYTHONUTF8=1"

if not exist "%CACHE%" md "%CACHE%"
if not exist "%PIP_CACHE_DIR%" md "%PIP_CACHE_DIR%"
if not exist "%HF_HOME%" md "%HF_HOME%"
if not exist "%TRANSFORMERS_CACHE%" md "%TRANSFORMERS_CACHE%"
if not exist "%TORCH_HOME%" md "%TORCH_HOME%"

set "PY_URL=https://www.python.org/ftp/python/3.12.6/python-3.12.6-embed-amd64.zip"
set "PY_ZIP=%CACHE%\python-3.12.6-embed-amd64.zip"
set "GETPIP=%CACHE%\get-pip.py"
:: Triton on Windows: woct0rdho/triton-windows ships pre-built wheels via the
:: triton-windows PyPI package (provides the `triton` import). Prefer that over
:: a local wheel; if a local wheel is dropped into the project root it still
:: takes precedence as an offline fallback.
set "TRITON_WHL=%ROOT%triton-3.0.0-cp312-cp312-win_amd64.whl"
:: Triton 3.1.x is the line paired with torch 2.5 upstream; the woct0rdho
:: package starts at 3.1.0 on PyPI (older 3.0.x wheels only exist as local
:: files on GitHub releases).
set "TRITON_PIP_SPEC=triton-windows>=3.1,<3.2"
set "REQ_IN=%ROOT%requirements-extra.txt"
set "REQ_WIN=%ROOT%requirements-extra.win.txt"

where powershell >nul 2>nul || (echo [ERROR] PowerShell not found & pause & exit /b 1)

echo.
echo === 1/4 Portable Python ===
if exist "%PYPORT%\python.exe" goto PP_OK

echo -- downloading portable python zip
if exist "%PY_ZIP%" goto PP_UNZIP
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest '%PY_URL%' -OutFile '%PY_ZIP%'" || (echo [ERROR] download failed & pause & exit /b 1)

:PP_UNZIP
if exist "%PYPORT%" rmdir /s /q "%PYPORT%"
md "%PYPORT%" || (echo [ERROR] mkdir .pyport failed & pause & exit /b 1)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%PY_ZIP%' -DestinationPath '%PYPORT%' -Force" || (echo [ERROR] unzip failed & pause & exit /b 1)

:PP_OK
if not exist "%PYPORT%\python.exe" (echo [ERROR] python.exe not found in .pyport & pause & exit /b 1)
if not exist "%PYPORT%\python312._pth" (echo [ERROR] python312._pth missing & pause & exit /b 1)

echo -- enabling site (python312._pth)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='%PYPORT:\=\\%\python312._pth'; (Get-Content $p) -replace '^\s*#?\s*import\s+site\s*$', 'import site' | Set-Content $p -Encoding ASCII" || (echo [ERROR] patch _pth failed & pause & exit /b 1)

echo -- ensure pip
if exist "%GETPIP%" goto PIP_RUN
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GETPIP%'" || (echo [ERROR] get-pip download failed & pause & exit /b 1)

:PIP_RUN
"%PYPORT%\python.exe" "%GETPIP%" --no-warn-script-location --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] pip install failed & pause & exit /b 1)
"%PYPORT%\python.exe" -m pip install --upgrade pip wheel setuptools --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] upgrade pip/wheel/setuptools failed & pause & exit /b 1)

echo.
echo === 2/4 Create/activate .venv ===
if exist "%VENV%\Scripts\python.exe" goto VENV_ACT

echo -- install virtualenv into portable python
"%PYPORT%\python.exe" -m pip install virtualenv --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] install virtualenv failed & pause & exit /b 1)
if exist "%VENV%" rmdir /s /q "%VENV%"
"%PYPORT%\python.exe" -m virtualenv "%VENV%" || (echo [ERROR] create venv failed & pause & exit /b 1)

:VENV_ACT
call "%VENV%\Scripts\activate.bat" || (echo [ERROR] activate venv failed & pause & exit /b 1)

echo.
echo === 3/4 Dependencies ===
set "MARKER=%VENV%\.installed.ok"
if exist "%MARKER%" (echo -- already installed (delete %MARKER% to force reinstall) & goto DONE)

python -c "import sys;print('venv python:', sys.version)" || (echo [ERROR] venv python error & pause & exit /b 1)
python -m pip install --upgrade pip wheel setuptools --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] upgrade pip in venv failed & pause & exit /b 1)

echo -- install torch cu121 wheels
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.0 torchvision==0.20.0 --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] torch install failed & pause & exit /b 1)

echo -- build tools
python -m pip install packaging ninja --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] packaging/ninja failed & pause & exit /b 1)

echo -- install Triton (Windows)
if exist "%TRITON_WHL%" (
  echo -- using local wheel "%TRITON_WHL%"
  python -m pip install "%TRITON_WHL%" --no-deps --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] local Triton install failed & pause & exit /b 1)
) else (
  echo -- no local wheel found, installing %TRITON_PIP_SPEC% from PyPI (woct0rdho/triton-windows)
  python -m pip install -U "%TRITON_PIP_SPEC%" --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] triton-windows install failed -- drop triton-3.0.0-cp312-cp312-win_amd64.whl in project root and re-run & pause & exit /b 1)
)

echo -- optional flash-attn (may fail on Windows, ignore)
python -m pip install flash-attn==2.7.0.post2 --no-build-isolation --cache-dir "%PIP_CACHE_DIR%"
if errorlevel 1 echo [WARN] flash-attn failed, ignored.

if not exist "%REQ_IN%" (echo [ERROR] requirements-extra.txt missing & pause & exit /b 1)
echo -- filter out triton/flash-attn/bitsandbytes then install
if exist "%REQ_WIN%" del "%REQ_WIN%"
findstr /V /R /C:"^triton" /C:"^flash-attn" /C:"^bitsandbytes" "%REQ_IN%" > "%REQ_WIN%"
python -m pip install -r "%REQ_WIN%" --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] extra deps failed & pause & exit /b 1)

echo -- configure source path for imports
set "PYTHONPATH=%ROOT%;%ROOT%wan;%ROOT%wan23;%ROOT%hyvideo;%ROOT%fastvideo;%PYTHONPATH%"

if exist "%ROOT%pyproject.toml" (
  echo -- found root pyproject.toml, trying: pip install -e .
  python -m pip install -e "%ROOT%" --cache-dir "%PIP_CACHE_DIR%"
  if errorlevel 1 echo [WARN] editable install of root project failed, continue with PYTHONPATH only
)

for %%D in ("%ROOT%wan" "%ROOT%wan23" "%ROOT%hyvideo" "%ROOT%fastvideo") do (
  if exist "%%~D" (
    if not exist "%%~D\__init__.py" (
      echo. > "%%~D\__init__.py"
      echo [INFO] created empty %%~D\__init__.py
    )
  )
)

echo ok>"%MARKER%"

:DONE
echo.
echo === 4/4 CUDA check ===
python -c "import torch;print('CUDA:', torch.cuda.is_available());print('Devices:', torch.cuda.device_count());print('Name:', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"

echo.
echo --- setup done ---
echo Next: run run_oneclick_debug.bat (full workflow) or "python bootstrap.py" inside .venv.
endlocal
exit /b 0
