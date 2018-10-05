@echo off

python code\annotation.py

SET ROOT=data
SET FORMAT_IN=mp4
SET FORMAT_OUT=m4a
SET REMOVE_IN=

for %%f in (%ROOT%\*.%FORMAT_IN%) do (    
    ffmpeg -i %%~dpnf.%FORMAT_IN% -vn -c:a copy %%~dpnf.%FORMAT_OUT%
	if defined REMOVE_IN (	
		del %%~dpnf.%FORMAT_IN%
	)
)


