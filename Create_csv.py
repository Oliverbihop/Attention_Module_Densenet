# Import some necessary library
import csv
import os
from imutils import paths
import random

# list all the images in folder ./d/ataset
img_folder_ck = './Data-Angle' 

img_name_ck_list = list(paths.list_images(img_folder_ck))

random.shuffle(img_name_ck_list)

# divide the dataset into 2 part: training and testing
# Number of training images/ testing iamges ~ 7/3
num_img = len(img_name_ck_list)
num_train = int(num_img*70/100) # chia thành tập train

#num_train = int(num_img)
# write image path and corresponding labels to 2 file data_train.csv and data_test.csv
with open('data_test.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(0, num_train):

        img_path = img_name_ck_list[i]
         # write the training section
        if 'class0' in img_path:
            cls_gt = 0
            writer.writerow([img_path, cls_gt])
        if 'class1' in img_path:
            cls_gt = 1
            writer.writerow([img_path, cls_gt])
        if 'class2' in img_path:
            cls_gt = 2
            writer.writerow([img_path, cls_gt])
        if 'class3' in img_path:
            cls_gt = 3
            writer.writerow([img_path, cls_gt])
        if 'class4' in img_path:
            cls_gt = 4
            writer.writerow([img_path, cls_gt])
        if 'class5' in img_path:
            cls_gt = 5
            writer.writerow([img_path, cls_gt])
        if 'class6' in img_path:
            cls_gt = 6
            writer.writerow([img_path, cls_gt])
        


with open('data_train_s.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(num_train, num_img):

        img_path = img_name_ck_list[i]
         # write the training section
        if 'class0' in img_path:
            cls_gt = 0
            writer.writerow([img_path, cls_gt])
        if 'class1' in img_path:
            cls_gt = 1
            writer.writerow([img_path, cls_gt])
        if 'class2' in img_path:
            cls_gt = 2
            writer.writerow([img_path, cls_gt])
        if 'class3' in img_path:
            cls_gt = 3
            writer.writerow([img_path, cls_gt])
        if 'class4' in img_path:
            cls_gt = 4
            writer.writerow([img_path, cls_gt])
        if 'class5' in img_path:
            cls_gt = 5
            writer.writerow([img_path, cls_gt])
        if 'class6' in img_path:
            cls_gt = 6
            writer.writerow([img_path, cls_gt])


print("end")

