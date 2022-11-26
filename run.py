import cv2
import numpy as np
from PIL import Image
import models
import os

def upscale(generator, image):
    return image

def run_image(generator, imagePath, imageName):
    image = cv2.imread(os.path.join(imagePath, imageName))
    upscaleImage = upscale(generator, image)
    cv2.imwrite(imagePath+"/up_"+imageName, upscaleImage)

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

    if not os.path.exists("temp/"):
        os.mkdir("temp/")
    index = 0
    frameNum = 0
    while True:
        hasNext, frame = cap.read()
        if hasNext:
            frameName = "frame_"+str(index)+".png"
            cv2.imwrite("/temp/"+frameName, frame)
            index += 1

            frameNum += fps
            cap.set(1, frameNum)
        else:
            break
    
    
    for i in range(index):
        run_image(generator, "/temp/","frame_"+str(i)+".png")

    frameFolder = "/temp/"
    imgs = [img for img in os.listdir(frameFolder) if img.startswith("up_") and img.endswith(".png")]
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    upVideo = cv2.VideoWriter(videoPath+"/up_"+videoName, fourcc, 1, (int(width), int(height)))

    for img in imgs:
        upVideo.write(cv2.imread(os.path.join(frameFolder, img)))

    cv2.destroyAllWindows()
    upVideo.release()




if __name__ == "__main__":
    gen = models.Generator()
    print("Do you want to run the image (1) or video (2)?")
    print("Enter '1' or '2': ")
    userInput = input()
    #userInput = '2'
    if userInput == '1':
        run_image(gen, "./demo/","imageTest.png")
    elif userInput == '2':
        run_video(gen, "./demo/", "videoTest.mp4")