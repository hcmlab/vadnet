@echo off

set MODEL=models\model.ckpt-593203
set FILES=data\noise.wav data\speech.wav
set N_BATCH=256

bin\python.exe vad_extract.py --model %MODEL% --files %FILES% --n_batch 256