# Super Resolution

Term project for ECE 371Q -- Digital Image Processing

## SRGAN

We implemented SRGAN as described in [<em>Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network</em>](https://arxiv.org/pdf/1609.04802.pdf)

![image](https://user-images.githubusercontent.com/84476225/198865448-44935565-bc64-4849-9c99-64f45be3d54b.png)

## Model

[SRGAN Trained Model Files](https://drive.google.com/drive/folders/1JY2nZuanTdqid_lJ65mRdZfsasl__KQ3?usp=sharing) -- Trained for 4000 Epochs (2000 MSE Only, 2000 w/ Perceptual Loss)

For testing, download gen.tar and place it in this repository's folder. Then, run `test.py` to generate SR images from the LR images in `test/` (output is in `test_output/`). Alternatively, you can run `run.py` to run it on an individual image, video, or webcam. For video and webcam streams it is recommended to be using a GPU or it will take a very long time.

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
