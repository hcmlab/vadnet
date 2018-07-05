@echo off

bin\xmlpipe -log ssi.log -confstr "audio:live=True;audio:live:mic=False;send:do=True" -config vad vad