import cv2
import numpy as np
from PIL import Image
import os
import torch

import albumentations as A
import albumentations.pytorch as AT

from PIL import Image
from torchvision.utils import save_image
from config import *
from models import Generator

from datetime import datetime
import shutil

transform = A.Compose([
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    AT.ToTensorV2(),
])


# Run image to demo
def run_image(generator, imagePath, imageName):
    # Load image
    #image = cv2.imread(os.path.join(imagePath, imageName))
    image = np.asarray(Image.open(os.path.join(imagePath, imageName)))
    # Upscale image using Super Resolution
    generator.eval()
    image = transform(image=image)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        upscaleImage = generator(image)
        save_image(upscaleImage * 0.5 + 0.5, imagePath+"/up_"+imageName)
        print("Generating Upscaled: " + imagePath+"/up_"+imageName)
    generator.train()

# Run video to demo
def run_video(generator, videoPath, videoName):
    # Extract frames from video and store in temp folder
    print(os.path.join(videoPath, videoName))
    cap = cv2.VideoCapture(os.path.join(videoPath, videoName))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(fps)
    print(width)
    print(height)

    skip = 3

    frameFolder = "./temp/"
    if os.path.exists(frameFolder):
        shutil.rmtree(frameFolder)
    if not os.path.exists(frameFolder):
        os.mkdir(frameFolder)
    
    index = 0
    frameNum = 0

    print("Extracting frames ...")
    while True:
        hasNext, frame = cap.read()
        if hasNext:
            frameName = "frame_"+str(index).zfill(3)+".png"
            cv2.imwrite(frameFolder+frameName, frame)
            index += 1

            frameNum += skip
            cap.set(1, frameNum)
        else:
            break
    print(str(index) + " frames extracted ...")
    
    # Run Super Resolution on individual frames
    for i in range(index):
        run_image(generator, frameFolder,"frame_"+str(i).zfill(3)+".png")


    

    # Convert upscaled images into upscaled video
    imgs = [img for img in os.listdir(frameFolder) if img.startswith("up_") and img.endswith(".png")]
    imgs.sort()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    img = cv2.imread(os.path.join(frameFolder, imgs[0]))
    width = img.shape[1]
    height = img.shape[0]
    print(width)
    print(height)
    upVideo = cv2.VideoWriter(videoPath+"/up_"+videoName, fourcc, fps//skip, (int(width), int(height)))
    print("Generating video ...")
    for img in imgs:
        print("Attaching " + img)
        upVideo.write(cv2.imread(os.path.join(frameFolder, img)))

    #cv2.destroyAllWindows()
    upVideo.release()
    print("Output video stored at: "+videoPath+"/up_"+videoName)


def main():
    start = datetime.now()
    # Load generator
    generator = Generator(in_channels=CHANNELS, scaling_factor=SCALE_FACTOR).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_GEN, map_location=DEVICE)
    generator.load_state_dict(checkpoint["state_dict"])

    print("Do you want to run the image (1) or video (2)?")
    print("Enter '1' or '2': ")
    userInput = input()
    #userInput = '2'
    if userInput == '1':
        run_image(generator, "./demo/","baboon.png")
    elif userInput == '2':
        run_video(generator, "./demo/", "videoTest.mp4")
    end = datetime.now()
    print("Elapsed Time: ", (end-start).total_seconds(), "s")

if __name__ == "__main__":
    main()