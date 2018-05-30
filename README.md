# VadNet
VadNet is a robust real-time voice activity detector. It follows an End-to-end learning approach based on Deep Neural Networks.

<a href="https://www.youtube.com/watch?v=QL4HY_e21v0" target="_blank"><img src="https://raw.githubusercontent.com/hcmlab/vadnet/master/pics/vadnet.png"/></a>

# Platform

Windows

# Dependencies

Visual Studio 2015 Redistributable (https://www.microsoft.com/en-us/download/details.aspx?id=52685)

# Installation

`do_bin.cmd` - Installs embedded Python and downloads SSI interpreter. During the installation the script tries to detect if a GPU is available and possibly installs ``tensorflow-gpu``. This requires that a NVidia graphic card is detected and CUDA has been installed. Nevertheless, VadNet also runs fine on CPU only.

# Quick Guide

`do_vad.cmd` - Demo on pre-recorded files (requires 44.1k mono wav files)

`do_vad_live.cmd` - Live demo (requires a microphone and shares result through a socket connection)

`do_vad_extract.cmd` - Separates audio file into noisy and voiced parts (supports any audio format)

# Documentation

VadNet is implemented using the [Social Signal Interpretation (SSI)](http://openssi.net) framework. The processing pipeline is defined in ``vad.pipeline`` and can be configured by editing ``vad.pipeline-config``. Available options are:

```
audio:live = false                   # $(bool) use live input from a microphone
model:path=models\model.ckpt-357047  # path to model file
send:do = false                      # $(bool) send xml string
send:url = upd://localhost:1234      # socket address in format <protocol://host:port>
record:do = false                    # $(bool) capture screen and audio
record:path = capture                # capture path
```

If the option ``send:do`` is turned on, SSI will send a small XML string to the socket defined by ``send:url``. It includes the latest detection result and can be used by your application. For more information about SSI pipelines please consult the [documentation](https://rawgit.com/hcmlab/ssi/master/docs/index.html#xml) of SSI.

The Python script ``vad_extract.py`` can be used to separate noisy and voiced parts in an audio file. For each input file ``<name>.<ext>`` two new files ``<name>.speech.wav`` and ``<name>.noise.wav`` will be generated. The script should handle all common audio formats. You can run the script from the command line by calling  ``> bin\python.exe vad_extract.py <options>``:

```
usage: vad_extract.py [-h] [--model MODEL] [--files FILES [FILES ...]] [--n_batch N_BATCH]

optional arguments:
  -h, --help            		show this help message and exit
  --model MODEL         		path to model
  --files FILES [FILES ...]		list of files
  --n_batch N_BATCH     		number of batches
```

# Insights

The model we are using has been trained with [Tensorflow](https://www.tensorflow.org/). It takes as input the raw audio input and feeds it into a 3-layer Convolutional Network. The result of this filter operation is then further processed by a 2-layer Recurrent Network containing 64 RLU cells. The final bit is a fully-connected layer, which applies a softmax and maps the input to a tuple ``<noise, voice>`` in the range ``[0..1]``. 

Network architecture:

<img src="https://raw.githubusercontent.com/hcmlab/vadnet/master/pics/network.png"/>

We have trained the network on roughly 134 h of audio data (5.6 days) and run training for 25 epochs (381024 steps) using a batch size of 128.

Filter weights learned in the first CNN layer:

<img src="https://raw.githubusercontent.com/hcmlab/vadnet/master/pics/conv1-filter-weights.png"/>

Some selected activations of the last RNN layer for a wav sample:

<img src="https://raw.githubusercontent.com/hcmlab/vadnet/master/pics/rnn2-activiations-selected.png"/>

Activations for all cells in the last RNN layer for the same sample:

<img src="https://raw.githubusercontent.com/hcmlab/vadnet/master/pics/rnn2-activiations.png"/>

# Credits

* SSI -- Social Signal Interpretation Framework - http://openssi.net
* Tensorflow -- An open source machine learning framework for everyone  - https://www.tensorflow.org/

# License

VadNet is released under GPL (see LICENSE).

# Author

Johannes Wagner, Lab for Human Centered Multimedia, 2018