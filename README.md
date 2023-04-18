# NAM: neural amp modeler

This repository handles training, reamping, and exporting the weights of a model.
For playing trained models in real time in a standalone application or plugin, see the partner repo,
[NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

## How to use (Google Colab)

If you don't have a good computer for training ML models, you use Google Colab to train
in the cloud using the pre-made notebooks under `bin\train`.

For the very easiest experience, simply go to
[https://colab.research.google.com/github/sdatkinson/neural-amp-modeler/blob/main/bin/train/easy_colab.ipynb](https://colab.research.google.com/github/sdatkinson/neural-amp-modeler/blob/main/bin/train/easy_colab.ipynb) and follow the
steps!

For a little more visibility under the hood, you can use [colab.ipynb](https://colab.research.google.com/github/sdatkinson/neural-amp-modeler/blob/main/bin/train/colab.ipynb) instead.

**Pros:**

- No local installation required!
- Decent GPUs are available if you don't have one on your computer.

**Cons:**

- Uploading your data can take a long time.
- The session will time out after a few hours (for free accounts), so extended
  training runs aren't really feasible. Also, there's a usage limit so you can't hang
  out all day. I've tried to set you up with a good model that should train reasonably
  quickly!

## How to use (Local)

Alternatively, you can clone this repo to your computer and use it locally.

### Installation

Installation uses [Anaconda](https://www.anaconda.com/) for package management.

For computers with a CUDA-capable GPU (recommended):

```bash
conda env create -f environment_gpu.yml
```

Otherwise, for a CPU-only install (will train much more slowly):

```bash
conda env create -f environment_cpu.yml
```

Then activate the environment you've created with

```bash
conda activate nam
```

### Train models (GUI)
After installing, you can open a GUI trainer by running

```bash
nam
```

from the terminal.

### Train models (Python script)
For users looking to get more fine-grained control over the modeling process, 
NAM includes a training script that can be run from the terminal. In order to run it
#### Download audio files
Download the [v1_1_1.wav](https://drive.google.com/file/d/1v2xFXeQ9W2Ks05XrqsMCs2viQcKPAwBk/view?usp=share_link) and [overdrive.wav](https://drive.google.com/file/d/14w2utgL16NozmESzAJO_I0_VCt-5Wgpv/view?usp=share_link) to a folder of your choice 

### Version 1
#### Update data configuration
Edit `bin/train/data/single_pair.json` to point to relevant audio files 
```json
    "common": {
        "x_path": "C:\\path\\to\\v1_1_1.wav",
        "y_path": "C:\\path\\to\\overdrive.wav",
        "delay": 0
    }
```

#### Run training script
Open up a terminal. Activate your nam environment and call the training with
```bash
python bin/train/main.py \
bin/train/inputs/data/single_pair.json \
bin/train/inputs/models/demonet.json \
bin/train/inputs/learning/demo.json \
bin/train/outputs/MyAmp
```

`data/single_pair.json` contains the information about the data you're training
on   
`models/demonet.json` contains information about the model architecture that
is being trained. The example used here uses a `feather` configured `wavenet`.  
`learning/demo.json` contains information about the training run itself (e.g. number of epochs).

The configuration above runs a short (demo) training. For a real training you may prefer to run something like,

```bash
python bin/train/main.py \
bin/train/inputs/data/single_pair.json \
bin/train/inputs/models/wavenet.json \
bin/train/inputs/learning/default.json \
bin/train/outputs/MyAmp
```

As a side note, NAM uses [PyTorch Lightning](https://lightning.ai/pages/open-source/) 
under the hood as a modeling framework, and you can control many of the Pytorch Lightning configuration options from `bin/train/inputs/learning/default.json`

#### Export a model (to use with [the plugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin))
Exporting the trained model to a `.nam` file for use with the plugin can be done
with:

```bash
python bin/export.py \
path/to/config_model.json \
path/to/checkpoints/epoch=123_val_loss=0.000010.ckpt \
path/to/exported_models/MyAmp
```

Then, point the plugin at the exported `model.nam` file and you're good to go!

### Version 2 (running the easy colab version)

```bash
python bin/train/main.py -c
```

To set different epochs or architecture, you can use a command similar to the one below:

```bash
python bin/train/main.py -c -e 1000 -a feather
```

After run the command, it is required that you close the delay plot window to start the training process.

### Other utilities

#### Run a model on an input signal ("reamping")

Handy if you want to just check it out without needing to use the plugin:

```bash
python bin/run.py \
path/to/source.wav \
path/to/config_model.json \
path/to/checkpoints/epoch=123_val_loss=0.000010.ckpt \
path/to/output.wav
```

### Installation with pip (do not require Anaconda)

(This installation process may be better suited for developers)

Create a virtualenv and install the dependencies packages

For Mac:
```bash
python -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip3 install -r requirements.txt
```

While the default PyTorch installation on many Macs comes with GPU support enabled, it's not the case for Windows.

To check whether your machine has PyTorch GPU capability, you can access this page:

https://pytorch.org/get-started/locally/

So, for Windows to use PyTorch with GPU enabled, you may need to uninstall the existing default version
(if installed via requirements.txt) and replace it with the GPU version, like so:

Example for Windows (assuming that CUDA is already installed):
```bash
pip3 uninstall torch
pip3 install torch  --index-url https://download.pytorch.org/whl/cu117
```

Visit this page (https://pytorch.org/get-started/locally/) to find the pip command recommended for your machine
to install PyTorch with GPU support, and use it instead.

Before running the scripts, it's also necessary to set the PYTHONPATH variable as follows:

For Mac:
```bash
export PYTHONPATH=`pwd`
```

For Windows:
```bash
set PYTHONPATH=%cd%
```
