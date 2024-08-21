@echo off
echo Current path is %cd%
for /l %%i in (7,1,8) do (
 start "" processing-java --force --sketch=D:\timewarping2\MMGP_OL%%i --output=D:\timewarping2\MMGP_OL%%i\output --run
)
pause
