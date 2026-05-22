@echo off
setlocal enabledelayedexpansion

set PY_SCRIPT=%~dp0vbench_runner.py

cd /d "%~dp0..\.."

:: Default paths relative to this repo — override by passing them as arguments:
::   sample_5b_vbench.bat [VBENCH_JSON] [VBENCH_CROP] [NUM_FRAMES]
if not "%~1"=="" (set VBENCH_JSON=%~1) else (set VBENCH_JSON=%CD%\..\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\i2v-bench-info.json)
if not "%~2"=="" (set VBENCH_CROP=%~2) else (set VBENCH_CROP=%CD%\..\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\crop\1-1)
if not "%~3"=="" (set NUM_FRAMES=%~3) else (set NUM_FRAMES=161)

echo Output:    %CD%\outputs\vbench
echo VBench:    %VBENCH_JSON%
echo Crop:      %VBENCH_CROP%
echo Frames:    %NUM_FRAMES%

python "%PY_SCRIPT%" ^
    --vbench-json "%VBENCH_JSON%" ^
    --vbench-crop "%VBENCH_CROP%" ^
    --work-dir "%CD%" ^
    --num-frames %NUM_FRAMES%

exit /b %ERRORLEVEL%
