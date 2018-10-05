@echo off

SET RETRAIN=True
SET ROOT=data
SET OUTPUT=nets

SET LEARNING_RATE=0.0001
SET N_EPOCHS=5
SET N_BATCH=512
SET SAMPLE_RATE=48000
SET FRAME=48000
SET STEP=24000

SET NETWORK=network.crnn.Conv7Rnn
SET CONV_STRIDE=2
SET CONV_POOL_STRIDE=2
SET CONV_POOL_SIZE=4
SET RNN_APPLY=False
SET RNN_LAYERS=2
SET RNN_UNITS=128

REM SET EVAL_AUDIO=eval.wav
REM SET EVAL_ANNO=eval.annotation
REM SET EVAL_THRES=0
REM SET LOG_FILENAME=True

python code\main.py --source source.audio_vad_files.AudioVadFiles --model model.model.Model --trainer trainer.adam.SceAdam --retrain %RETRAIN% --sample_rate=%SAMPLE_RATE% --n_frame %FRAME% --n_step %STEP% --files_root %ROOT% --files_filter *.info --files_audio_ext .m4a --files_anno_ext .voiceactivity.annotation --output_dir=%OUTPUT% --learning_rate %LEARNING_RATE% --n_batch %N_BATCH% --network %NETWORK% --conv_stride %CONV_STRIDE% --conv_pool_stride %CONV_POOL_STRIDE% --conv_pool_size %CONV_POOL_SIZE% --rnn_apply %RNN_APPLY% --n_rnn_layers %RNN_LAYERS% --n_rnn_units %RNN_UNITS% --n_epochs %N_EPOCHS% 
REM --eval_audio_file %EVAL_AUDIO% --eval_anno_file %EVAL_ANNO% --eval_blacklist_thres %EVAL_THRES% --log_filename %LOG_FILENAME%
