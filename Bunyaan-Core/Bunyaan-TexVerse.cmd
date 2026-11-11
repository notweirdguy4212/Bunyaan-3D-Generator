@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" goto help
if /I "%~1"=="extract" goto extract
if /I "%~1"=="validate" goto validate
if /I "%~1"=="train" goto train
if /I "%~1"=="evaluate" goto evaluate
goto help
:extract
shift
python tools\extract_texverse.py %*
goto end
:validate
shift
python tools\validate_texverse.py %*
goto end
:train
shift
python training\train_texture.py %*
goto end
:evaluate
shift
python training\evaluate_texture.py %*
goto end
:help
echo Bunyaan TexVerse commands:
echo   Bunyaan-TexVerse.cmd extract --input GLB_FOLDER --license CC0
 echo  Bunyaan-TexVerse.cmd validate --data DATASET_FOLDER
 echo  Bunyaan-TexVerse.cmd train --data DATASET_FOLDER --out checkpoints\bunyaan-texture.pt
 echo  Bunyaan-TexVerse.cmd evaluate --checkpoint CHECKPOINT --data DATASET_FOLDER
:end
endlocal
