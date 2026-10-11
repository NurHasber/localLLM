@echo off
cd /d C:\AI\CosyVoice
set "PATH=C:\Users\NurHa\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin;%PATH%"
call C:\Users\NurHa\miniconda3\Scripts\activate.bat cosyvoice
python synthesize.py %*
