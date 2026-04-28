@echo off
REM ============================================================
REM  run_mmlu_eval.bat  –  Windows launcher for mmlu_eval_logits.py
REM ============================================================
REM  Edit the variables below before running.
REM  Then double-click the file or run it from a cmd prompt.
REM ============================================================

REM --- Paths ---
set EXE=C:\Release\modeling_qwen3_5_logits.exe
set MODEL=C:\models\Qwen3.5-35B-A3B
set DATASET_CACHE=C:\datasets
set WORK_DIR=%~dp0logits_tmp
set OUTPUT=%~dp0mmlu_results.json

REM --- Evaluation settings ---
set LIMIT=50
set NUM_FEWSHOT=0
set THINK=0

REM --- Create work dir ---
if not exist "%WORK_DIR%" mkdir "%WORK_DIR%"

REM --- Run ---
python "%~dp0mmlu_eval_logits.py" ^
    --exe        "%EXE%"          ^
    --model      "%MODEL%"        ^
    --limit      %LIMIT%          ^
    --num-fewshot %NUM_FEWSHOT%   ^
    --dataset-cache "%DATASET_CACHE%" ^
    --work-dir   "%WORK_DIR%"     ^
    --output     "%OUTPUT%"       ^
    --think      %THINK%

pause
