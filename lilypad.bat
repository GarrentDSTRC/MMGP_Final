@echo off
echo Current path is %cd%
for /l %%i in (0,1,8) do (
 start "" processing-java --force --sketch=D:\MMGP_Final_EFD\MMGP_OL%%i --output=D:\MMGP_Fina_EFDl\MMGP_OL%%i\output --run
)
pause
