@echo off
setlocal enabledelayedexpansion

set TOKENIZERS_PARALLELISM=false
set TF_ENABLE_ONEDNN_OPTS=0
set LOCAL_RANK=0
set RANK=0
set WORLD_SIZE=1
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=29500

set EXAMPLE_DIR=C:\workspace\world\Infinite-World\assets\example_case

cd /d "%~dp0..\.."

:: Write default caption file once
echo A first-person view exploring an interactive game scene. > temp_caption_default.txt

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format ''yyyyMMdd_HHmmss''"') do set RUN_TS=%%T
for /f %%t in ('powershell -NoProfile -Command "[int64](Get-Date).Ticks"') do set OVERALL_START=%%t
set CASE_NUM=0

goto :main

:: -------------------------------------------------------
:: Subroutine: reads RC_DIR, RC_NAME, RC_CAPTION from env
:: -------------------------------------------------------
:run_case
    set /a CASE_NUM+=1
    echo.
    echo === !CASE_NUM!: !RC_NAME! ===
    echo output: ./outputs/%RUN_TS%/!RC_NAME!
    for /f %%t in ('powershell -NoProfile -Command "[int64](Get-Date).Ticks"') do set CASE_START=%%t

    python fastvideo/sample/sample.py ^
        --seed 42 ^
        --gradient_checkpointing ^
        --train_batch_size=1 ^
        --max_sample_steps=600000 ^
        --mixed_precision="bf16" ^
        --allow_tf32 ^
        --video_output_dir="./outputs/%RUN_TS%/!RC_NAME!" ^
        --jpg_dir="!RC_DIR!" ^
        --caption_path="!RC_CAPTION!" ^
        --test_data_dir="./val" ^
        --num_euler_timesteps 50 ^
        --rand_num_img 0.6 ^
        --t5_cpu

    powershell -NoProfile -Command "$e=([int64](Get-Date).Ticks-!CASE_START!)/1e7; $fps=[math]::Round(163/$e,2); Write-Host ('  time='+[math]::Round($e,1)+'s  inference='+$fps+'fps  video=16fps')"
    exit /b 0

:main

for /d %%D in (%EXAMPLE_DIR%\*) do (
    set RC_NAME=%%~nxD
    set RC_DIR=%%D
    set RC_CAPTION=temp_caption_default.txt
    if exist "%%D\prompt.txt" set RC_CAPTION=%%D\prompt.txt
    call :run_case
)

echo.
powershell -NoProfile -Command "$e=([int64](Get-Date).Ticks-%OVERALL_START%)/1e7; Write-Host ('Total: !CASE_NUM! cases')"

del temp_caption_default.txt 2>nul
exit /b 0
