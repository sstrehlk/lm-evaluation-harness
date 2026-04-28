@echo off
REM ============================================================
REM  run_mmlu_eval.bat  –  Windows launcher for mmlu_eval_logits.py
REM ============================================================
REM  Edit the variables below before running.
REM  Then double-click the file or run it from a cmd prompt.
REM ============================================================

REM --- Paths ---
set EXE=C:\sstrehlk\package_0416\Release\modeling_qwen3_5_logits.exe
set MODEL=C:\sstrehlk\models\qwen3.5-35b-a3e
set DATASET_CACHE=C:\sstrehlk\datasets
set WORK_DIR=%~dp0logits_tmp
set OUTPUT=%~dp0mmlu_results.json

REM --- Evaluation settings ---
set LIMIT=1000
set NUM_FEWSHOT=0
set THINK=0

REM --- Force UTF-8 output (avoids cp1252 crash on Greek/math chars in MMLU) ---
set PYTHONUTF8=1

REM --- Log file (timestamped) ---
for /f "tokens=1-6 delims=/:. " %%a in ("%date% %time: =0%") do (
    set TIMESTAMP=%%c%%a%%b_%%d%%e%%f
)
set LOG=%~dp0mmlu_eval_%TIMESTAMP%.log

REM --- Create work dir ---
if not exist "%WORK_DIR%" mkdir "%WORK_DIR%"

echo [*] Log file: %LOG%
echo [*] Starting evaluation...

REM --- Run (output to console AND log file) ---
python "%~dp0mmlu_eval_logits.py" ^
    --exe        "%EXE%"          ^
    --model      "%MODEL%"        ^
    --limit      %LIMIT%          ^
    --num-fewshot %NUM_FEWSHOT%   ^
    --dataset-cache "%DATASET_CACHE%" ^
    --work-dir   "%WORK_DIR%"     ^
    --output     "%OUTPUT%"       ^
    --think      %THINK%          ^
    2>&1 | tee "%LOG%"

echo.
echo [*] Done. Log saved to: %LOG%
pause
