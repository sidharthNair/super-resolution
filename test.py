import numpy as np
import os
import imageio
import torch
import albumentations as A
import albumentations.pytorch as AT

from PIL import Image
from torchvision.utils import save_image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from config import *
from models import Generator

transform = A.Compose([
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    AT.ToTensorV2(),
])

def test_generator(generator, prefix='', folder='test'):
    files = os.listdir(folder)
    if not os.path.exists(f'{folder}_output/'):
        os.makedirs(f'{folder}_output/')

    generator.eval()
    for file in files:
        lr = np.asarray(Image.open(f'{folder}/' + file))
        lr = transform(image=lr)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            hr = generator(lr)
            save_image(hr * 0.5 + 0.5, f'{folder}_output/{prefix}upscaled_' + file)
    generator.train()

def create_gif(contains='', folder='test_output', skip=0, start=0, num_frames=None):
    files = os.listdir(folder)
    files = [file for file in files if contains in file]
    sorted_files = sorted(files, key = lambda x: int(x.split("_")[0]))
    imgs = [np.array(Image.open(os.path.join(folder, sorted_files[i]))) for i in range(start, min(len(sorted_files), start + num_frames * (skip + 1)), skip + 1)]
    imageio.mimsave(f'test_output/generated{contains}.gif', imgs, duration=0.2)

def compare(hr_image_path, sr_image_path):
    hr_image = np.asarray(Image.open(hr_image_path))
    sr_image = np.asarray(Image.open(sr_image_path))
    ssim = structural_similarity(hr_image, sr_image, channel_axis=2)
    psnr = peak_signal_noise_ratio(hr_image, sr_image)
    mse = mean_squared_error(hr_image, sr_image)
    print('SSIM:', ssim)
    print('PSNR:', psnr)
    print('MSE:', mse)
    return ssim, psnr, mse

def main():
    generator = Generator(in_channels=CHANNELS, scaling_factor=SCALE_FACTOR).to(DEVICE)
    checkpoint = torch.load(GEN_FILE, map_location=DEVICE)
    generator.load_state_dict(checkpoint["state_dict"])
    test_generator(generator)

if __name__ == "__main__":
    main()
