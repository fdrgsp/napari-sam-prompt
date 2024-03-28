# napari-sam-prompt

[![License BSD-3](https://img.shields.io/pypi/l/napari-sam-prompt.svg?color=green)](https://github.com/fdrgsp/napari-sam-prompt/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-sam-prompt.svg?color=green)](https://pypi.org/project/napari-sam-prompt)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-sam-prompt.svg?color=green)](https://python.org)
[![tests](https://github.com/fdrgsp/napari-sam-prompt/workflows/tests/badge.svg)](https://github.com/fdrgsp/napari-sam-prompt/actions)
[![codecov](https://codecov.io/gh/fdrgsp/napari-sam-prompt/branch/main/graph/badge.svg)](https://codecov.io/gh/fdrgsp/napari-sam-prompt)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-sam-prompt)](https://napari-hub.org/plugins/napari-sam-prompt)

A napari plugin that implements SAM prompts predictor


https://github.com/fdrgsp/napari-sam-prompt/assets/70725613/3bee3d2f-7197-4d97-a4c0-21dfed2fba1b


----------------------------------

## Installation

### Install napari-sam-prompt

```bash
pip install git+https://github.com/fdrgsp/napari-sam-prompt.git
```

The plugin also requires a Qt backend. For example run:

```bash
pip install pyqt6  # or any of {pyqt5, pyqt6, pyside2, pyside6}
```

### Install PyTorch and TorchVision

The plugin also requires `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both `PyTorch` and `TorchVision` dependencies.

### Install Segment Anything

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

A `SAM checkpoint model` is required to run the plugin. Download a model from [here](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints).

### To Run

```bash
python -m napari_sam_prompt
```

or, to directly specify `model checkpoint` and `model type`:

```bash
python -m napari_sam_prompt -mc "path/to/model/checkpoint.pth" -mt "model_type"
```

## License

Distributed under the terms of the [BSD-3] license,
"napari-sam-prompt" is free and open source software
