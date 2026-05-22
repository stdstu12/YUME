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

for /f "tokens=*" %%T in ('powershell -NoProfile -Command "Get-Date -Format ''yyyyMMdd_HHmmss''" 2^>nul') do if not "%%T"=="" set RUN_TS=%%T
for /f %%t in ('powershell -NoProfile -Command "[int64](Get-Date).Ticks"') do set OVERALL_START=%%t
echo Output: %CD%\outputs\%RUN_TS%
set CASE_NUM=0

:: Count total cases
set TOTAL_CASES=0
for /d %%D in (%EXAMPLE_DIR%\*) do set /a TOTAL_CASES+=1
echo Found %TOTAL_CASES% cases  output: ./outputs/%RUN_TS%

goto :main

:: -------------------------------------------------------
:: Subroutine: reads RC_DIR, RC_NAME, RC_CAPTION from env
:: -------------------------------------------------------
:run_case
    set /a CASE_NUM+=1
    echo.
    echo === !CASE_NUM!/%TOTAL_CASES%: !RC_NAME! ===
    echo output: ./outputs/%RUN_TS%/!RC_NAME!
    echo img:    !RC_DIR!
    echo prompt: !RC_CAPTION!
    type "!RC_CAPTION!"
    echo.
    for /f %%t in ('powershell -NoProfile -Command "[int64](Get-Date).Ticks"') do set CASE_START=%%t

    python fastvideo/sample/sample_5b.py ^
        --seed 43 ^
        --gradient_checkpointing ^
        --train_batch_size=1 ^
        --max_sample_steps=1 ^
        --mixed_precision="bf16" ^
        --allow_tf32 ^
        --t5_cpu ^
        --video_output_dir="./outputs/%RUN_TS%/!RC_NAME!" ^
        --jpg_dir="!RC_DIR!" ^
        --caption_path="!RC_CAPTION!" ^
        --test_data_dir="./val" ^
        --num_euler_timesteps 8 ^
        --rand_num_img 0.6 ^
        --internvl_path "./InternVL3-2B-Instruct"

    powershell -NoProfile -Command "$e=([int64](Get-Date).Ticks-!CASE_START!)/1e7; $tot=([int64](Get-Date).Ticks-%OVERALL_START%)/1e7; $fps=[math]::Round(163/$e,2); $rem=(%TOTAL_CASES%-!CASE_NUM!)*($tot/!CASE_NUM!); Write-Host ('  time='+[math]::Round($e,1)+'s  fps='+$fps+'  ETA='+[math]::Round($rem,0)+'s')"
    exit /b 0

:main

for /d %%D in (%EXAMPLE_DIR%\*) do (
    set RC_NAME=%%~nxD
    set RC_DIR=%%D
    echo A %%~nxD scene. > temp_caption_%%~nxD.txt
    set RC_CAPTION=temp_caption_%%~nxD.txt
    if exist "%%D\prompt.txt" set RC_CAPTION=%%D\prompt.txt
    if exist "%%D\prompt.json" (
        powershell -NoProfile -Command "(Get-Content '%%D\prompt.json' -Raw | ConvertFrom-Json).prompt | Set-Content 'temp_prompt_%%~nxD.txt' -Encoding UTF8"
        set RC_CAPTION=temp_prompt_%%~nxD.txt
    )
    call :run_case
)

echo.
powershell -NoProfile -Command "$e=([int64](Get-Date).Ticks-%OVERALL_START%)/1e7; Write-Host ('Total: %TOTAL_CASES% cases in '+[math]::Round($e,1)+'s  avg='+[math]::Round($e/%TOTAL_CASES%,1)+'s/case')"

del temp_caption_*.txt 2>nul
del temp_prompt_*.txt 2>nul
exit /b 0
