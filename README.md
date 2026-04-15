# MRI Liver Segmentation Demo

A lightweight Streamlit web application for liver segmentation on
abdominal MRI images, based on an improved U-Net architecture with
a grouped multi-directional attention mechanism.

## Overview

This project provides an interactive demo for liver segmentation from
single-channel MRI scans. Users can upload a medical image, choose
from several trained model checkpoints, and view the segmentation
result as a red overlay on the original image. The trained model is
small enough (under 1 MB) to run inference comfortably on CPU,
making it suitable for lightweight deployment scenarios.

This work was developed as an undergraduate graduation project. The
goal was to explore how attention mechanisms can be integrated into
the U-Net framework while keeping the model compact and deployable.

## Features

- Interactive web interface built with Streamlit
- Improved U-Net with grouped multi-directional attention modules
- Overlay visualization and downloadable result images
- Sample MRI images included for immediate testing

## Project Structure

```
.
├── start.py              # Streamlit application entry point
├── attunet.ipynb         # Training and experiment notebook
├── model/
│   ├── improved_unet.py  # Improved U-Net with grouped attention
│   ├── unet.py           # Baseline U-Net implementation
│   ├── test_weights_att.pth  # Pretrained weights (attention model)
│   └── test_weights.pth      # Pretrained weights (baseline)
├── sample/               # Sample MRI images for testing
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or later
- PyTorch 2.0 or later
- Streamlit 1.30 or later
- See `requirements.txt` for the full list of dependencies

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/FuChengchuan/att_unet.git
cd att_unet
pip install -r requirements.txt
```

## Usage

Launch the Streamlit application from the project root:

```bash
streamlit run start.py
```

The app will open automatically in your default browser at
`http://localhost:8501`. From the sidebar, you can choose which
segmentation model to use from the available checkpoints. After
selecting a model, upload an MRI image (PNG or JPEG) and click
"Run Segmentation" to view the predicted liver region as a red
overlay on the original image. The result can be downloaded as a
PNG file.

For a quick test, you can use any of the sample images provided in
the `sample/` directory.

## Method

The segmentation network is based on the U-Net encoder-decoder
framework, with the following modifications:

The deeper encoder and decoder stages replace standard convolutional
blocks with a **Grouped Multi-Directional Attention** module. This
module splits the feature map along the channel dimension into four
groups and applies learned attention along the spatial xy plane, the
channel-x plane, the channel-y plane, and a depthwise convolution
respectively. The four branches are then concatenated and projected,
allowing the network to capture both spatial and channel-wise
context with very few parameters.

Skip connections use a simplified fusion module that upsamples the
decoder feature, concatenates it with the corresponding encoder
feature, and applies a single 3×3 convolution.

Input images are converted to single-channel grayscale and normalized
to [0, 1]. The output probability map is thresholded at 0.5, followed
by morphological closing and opening as post-processing, to produce
the final binary mask.

## Training

The training pipeline is provided in `attunet.ipynb`. 

The notebook contains the dataset class, training loop, validation
loop, inference and post-processing code, and Dice computation.

## Dataset

The model was trained on a publicly available liver MRI dataset.
Due to size considerations, the full dataset is not included in this
repository. A small set of sample images is provided in the `sample/`
directory so that the demo can be tested immediately after cloning.


## Notes

- This demo is configured to run on CPU by default. To enable GPU
  inference, modify the `map_location` argument in the `load_model`
  function in `start.py`.
- The pretrained weights are intended for demonstration purposes.
  Performance on images outside the training distribution may vary.
- This project is an undergraduate graduation work and is shared
  for reference and evaluation purposes.


## Contact

<Fu Chengchuan>
<1191732972f@gmail.com>
