# Super Resolution

Term project for ECE 371Q -- Digital Image Processing

## Project Video

[![Super Resolution](https://img.youtube.com/vi/7ivT_hD1gQU/0.jpg)](https://www.youtube.com/watch?v=7ivT_hD1gQU)

## SRGAN

We implemented SRGAN as described in [<em>Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network</em>](https://arxiv.org/pdf/1609.04802.pdf)

![image](https://user-images.githubusercontent.com/84476225/198865448-44935565-bc64-4849-9c99-64f45be3d54b.png)

## Model

[SRGAN Trained Model Files](https://drive.google.com/drive/folders/1JY2nZuanTdqid_lJ65mRdZfsasl__KQ3?usp=sharing) -- Trained for 4000 Epochs (2000 MSE Only, 2000 w/ Perceptual Loss) on the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) HR training dataset

For testing, download gen.tar and place it in this repository's folder. Then, run `test.py` to generate SR images from the LR images in `test/` (output is in `test_output/`). Alternatively, you can run `run.py` to run it on an individual image, video, or webcam. For video and webcam streams it is recommended to be using an NVIDIA GPU or it will be very slow.

## Data

Put data in `data/`. Expected structure is:
```
.
├── class1_directory
│   ├── img1_file
│   ├── img2_file
│   └── ...
├── class2_directory
└── ...
```
