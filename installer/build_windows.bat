@echo off
rem VisionFlow Studio - Windows derleme ve kurulum paketi olusturma
cd /d "%~dp0\.."

echo PyInstaller paketi olusturuluyor...
pyinstaller --noconfirm visionflow.spec
if errorlevel 1 goto error

echo Inno Setup kurulum sihirbazi derleniyor...
ISCC.exe installer\visionflow.iss
if errorlevel 1 goto error

echo Basariyla tamamlandi.
exit /b 0

:error
echo Derleme sirasinda hata olustu!
exit /b 1
