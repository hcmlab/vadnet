@echo off

set MODEL=models\model.ckpt-47072
set FILES=data\noise.wav data\speech.wav

bin\python.exe vad_extract.py --model %MODEL% --files %FILES%