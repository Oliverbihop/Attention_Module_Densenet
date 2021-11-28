import numpy as np 
import glob
import cv2
image_list = []
n=0
for item in glob.glob("data/*.jpg"):
    image_list.append(item)
for i,item in enumerate(image_list):
  image=cv2.imread(item)
  image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
  cv2.imwrite("data/frame"+str(i)+".jpg",image)