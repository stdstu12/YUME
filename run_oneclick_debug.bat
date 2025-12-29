@echo on
setlocal EnableExtensions
chcp 65001 >nul

rem ==== paths (all local) ====
set "ROOT=%~dp0"
cd /d "%ROOT%" || (echo [ERROR] cd failed & pause & exit /b 1)

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

rem ==== constants ====
set "PY_URL=https://www.python.org/ftp/python/3.12.6/python-3.12.6-embed-amd64.zip"
set "PY_ZIP=%CACHE%\python-3.12.6-embed-amd64.zip"
set "GETPIP=%CACHE%\get-pip.py"
set "TRITON_WHL=%ROOT%triton-3.0.0-cp312-cp312-win_amd64.whl"
set "REQ_IN=%ROOT%requirements-extra.txt"
set "REQ_WIN=%ROOT%requirements-extra.win.txt"
set "APP=%ROOT%webapp_single_gpu.py"

where powershell >nul 2>nul || (echo [ERROR] PowerShell not found & pause & exit /b 1)

echo.
echo === 1) Portable Python ===
if exist "%PYPORT%\python.exe" goto PP_OK

echo -- downloading portable python zip
if exist "%PY_ZIP%" goto PP_UNZIP
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest '%PY_URL%' -OutFile '%PY_ZIP%'" || (echo [ERROR] download failed & pause & exit /b 1)

:PP_UNZIP
if exist "%PYPORT%" rmdir /s /q "%PYPORT%"
md "%PYPORT%" || (echo [ERROR] mkdir .pyport failed & pause & exit /b 1)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%PY_ZIP%' -DestinationPath '%PYPORT%' -Force" || (echo [ERROR] unzip failed & pause & exit /b 1)

:PP_OK
if exist "%PYPORT%\python.exe" goto PTH_CHECK
echo [ERROR] python.exe not found in .pyport
pause
exit /b 1

:PTH_CHECK
echo -- enabling site (python312._pth)
if exist "%PYPORT%\python312._pth" goto PTH_EDIT
echo [ERROR] python312._pth missing
pause
exit /b 1

:PTH_EDIT
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='%PYPORT:\=\\%\python312._pth'; (Get-Content $p) -replace '^\s*#?\s*import\s+site\s*$', 'import site' | Set-Content $p -Encoding ASCII" || (echo [ERROR] patch _pth failed & pause & exit /b 1)

echo -- ensure pip
if exist "%GETPIP%" goto PIP_RUN
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GETPIP%'" || (echo [ERROR] get-pip download failed & pause & exit /b 1)

:PIP_RUN
"%PYPORT%\python.exe" "%GETPIP%" --no-warn-script-location --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] pip install failed & pause & exit /b 1)
"%PYPORT%\python.exe" -m pip install --upgrade pip wheel setuptools --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] upgrade pip/wheel/setuptools failed & pause & exit /b 1)

echo.
echo === 2) Create/activate .venv ===
if exist "%VENV%\Scripts\python.exe" goto VENV_ACT

echo -- install virtualenv into portable python
"%PYPORT%\python.exe" -m pip install virtualenv --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] install virtualenv failed & pause & exit /b 1)
if exist "%VENV%" rmdir /s /q "%VENV%"
"%PYPORT%\python.exe" -m virtualenv "%VENV%" || (echo [ERROR] create venv failed & pause & exit /b 1)

:VENV_ACT
call "%VENV%\Scripts\activate.bat" || (echo [ERROR] activate venv failed & pause & exit /b 1)

rem keep all caches local
set "PIP_CACHE_DIR=%PIP_CACHE_DIR%"
set "HF_HOME=%HF_HOME%"
set "TRANSFORMERS_CACHE=%TRANSFORMERS_CACHE%"
set "TORCH_HOME=%TORCH_HOME%"
set "XDG_CACHE_HOME=%XDG_CACHE_HOME%"

echo.
echo === 3) First-time deps (skip if installed) ===
set "MARKER=%VENV%\.installed.ok"
if exist "%MARKER%" goto SKIP_DEPS

python -c "import sys;print('venv python:', sys.version)" || (echo [ERROR] venv python error & pause & exit /b 1)
python -m pip install --upgrade pip wheel setuptools --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] upgrade pip in venv failed & pause & exit /b 1)

echo -- install torch cu121 wheels
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.0 torchvision==0.20.0 --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] torch install failed & pause & exit /b 1)

echo -- build tools
python -m pip install packaging ninja --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] packaging/ninja failed & pause & exit /b 1)

echo -- install local Triton wheel (Windows cp312)
if not exist "%TRITON_WHL%" (
  echo [ERROR] missing local wheel: "%TRITON_WHL%"
  echo Put triton-3.0.0-cp312-cp312-win_amd64.whl into project root and re-run.
  pause & exit /b 1
)
python -m pip install "%TRITON_WHL%" --no-deps --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] local Triton install failed & pause & exit /b 1)

echo -- optional flash-attn (may fail on Windows, ignore)
python -m pip install flash-attn==2.7.0.post2 --no-build-isolation --cache-dir "%PIP_CACHE_DIR%"
if errorlevel 1 echo [WARN] flash-attn failed, ignored.

if not exist "%REQ_IN%" (echo [ERROR] requirements-extra.txt missing & pause & exit /b 1)
echo -- filter out triton/flash-attn/bitsandbytes then install
if exist "%REQ_WIN%" del "%REQ_WIN%"
findstr /V /R /C:"^triton" /C:"^flash-attn" /C:"^bitsandbytes" "%REQ_IN%" > "%REQ_WIN%"
python -m pip install -r "%REQ_WIN%" --cache-dir "%PIP_CACHE_DIR%" || (echo [ERROR] extra deps failed & pause & exit /b 1)

rem ===== fastvideo / project package handling =====
echo -- configure source path for imports
set "PYTHONPATH=%ROOT%;%ROOT%wan;%ROOT%wan23;%ROOT%hyvideo;%ROOT%fastvideo;%PYTHONPATH%"

rem optional: install the whole project if pyproject.toml exists in ROOT
if exist "%ROOT%pyproject.toml" (
  echo -- found root pyproject.toml, trying: pip install -e .
  python -m pip install -e "%ROOT%" --cache-dir "%PIP_CACHE_DIR%"
  if errorlevel 1 (
    echo [WARN] editable install of root project failed, continue with PYTHONPATH only
  )
)

rem ensure these dirs are python packages (create empty __init__.py if missing)
for %%D in ("%ROOT%wan" "%ROOT%wan23" "%ROOT%hyvideo" "%ROOT%fastvideo") do (
  if exist "%%~D" (
    if not exist "%%~D\__init__.py" (
      echo. > "%%~D\__init__.py"
      echo [INFO] created empty %%~D\__init__.py
    )
  )
)

echo ok>"%MARKER%"
echo -- deps done

:SKIP_DEPS
echo.
echo === 4) CUDA check ===
python -c "import torch;print('CUDA:', torch.cuda.is_available());print('Devices:', torch.cuda.device_count());print('Name:', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"

echo.
echo === 5) Launch Web ===
if not exist "%ROOT%bootstrap.py" (echo [ERROR] bootstrap.py not found & pause & exit /b 1)

rem 把根目录 + 三个源码目录放到 PYTHONPATH（wan23 现在是标准扁平结构）
set "PYTHONPATH=%ROOT%;%ROOT%wan;%ROOT%wan23;%ROOT%hyvideo;%ROOT%fastvideo;%PYTHONPATH%"

rem 兜底：若有缺失的 __init__.py 就创建（现在允许给外层 wan23 创建）
for %%D in ("%ROOT%wan" "%ROOT%wan23" "%ROOT%hyvideo" "%ROOT%fastvideo") do (
  if exist "%%~D" (
    if not exist "%%~D\__init__.py" (
      echo. > "%%~D\__init__.py"
      echo [INFO] created empty %%~D\__init__.py
    )
  )
)

python "%ROOT%bootstrap.py"
echo.
echo --- done ---
pause
exit /b 0