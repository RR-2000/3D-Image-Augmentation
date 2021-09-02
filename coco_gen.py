import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import json
from pycocotools import mask
from skimage import measure, img_as_ubyte



image_path_annon = './data/Image/' # Image Directory
mask_path_annon =  './data/Mask/' # Binary Mark Directory
path_to_gen_data = './Generated Data/'

out_path = './data/image_format/' # 2D image slices will be saved here
if not os.path.exists(out_path):
    os.makedir(out_path)

df = pd.read_csv("./meta.csv") # Metadata CSV path


#Initializing Image and Annotation Lists
images_test = []
images_train= []
images_val = []
images_test_bn = []
images_train_bn = []
images_val_bn = []
images_test_ml = []
images_train_ml = []
images_val_ml = []
annotations = []

count = 1 # For progress Print
for i in df.itertuples(index=False):
    print("Processing Annoted Data: ", count, "/", len(df), end = "\r")
    count = count + 1

    file_name = str(i[5]) # File name attribute
    mask_name = str(i[6]) # Mask name attribute
    class_meta = i[7] # Category attribute
    split = i[11] # Test/Train/Val split attribute


    img = np.load(image_path_annon + file_name + '.npy')
    imageio.imwrite(out_path + file_name[15:] + '.jpg', img_as_ubyte(img/max(np.max(img), -1*np.min(img) ) ) )

    # Binary Mask to COCO
    img = np.load((mask_path_annon + mask_name) + '.npy')
    ground_truth_binary_mask = img
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    s_id = int( file_name[15:19] + file_name[22:25] + file_name[31:]) # For unique numerical ID
    image = {
            "id": s_id,
            "width": img.shape[1],
            "height": img.shape[0],
            "file_name": "{}.jpg".format(file_name[15:])
    }

    # Decide final class for annotation
    annon_class = 0
    if class_meta >= 3:
    	annon_class = 2
    elif class_meta > 0:
    	annon_class = 1


    annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": s_id,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": annon_class,
            "id": s_id
        }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)


    if annon_class > 0:
        annotations.append(annotation)

    if split == 'Train':
        images_train.append(image)
    elif split == 'Test':
        images_test.append(image)
    else:
    	images_val.append(image)


df = pd.read_csv("./gen_meta_train.csv") # Metadata CSV path


count = 1 # For progress Print
for i in df.itertuples(index=False):
    print("Processing Annoted Data: ", count, "/", len(df), end = "\r")
    count = count + 1

    file_name = str(i[0]) # File name attribute
    mask_name = str(i[1]) # Mask name attribute
    annon_class = 1 # Category attribute
    split = i[11] # Test/Train/Val split attribute


    img = np.load(path_to_gen_data + file_name)
    imageio.imwrite(out_path + file_name[0:-4] + '.jpg', img_as_ubyte(img/max(np.max(img), -1*np.min(img) ) ) )

    # Binary Mask to COCO
    img = np.load(path_to_gen_data + mask_name)
    ground_truth_binary_mask = img
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    s_id = int('2' +file_name[5] + file_name[:4] + file_name[7; -10]) # For unique numerical ID

    image = {
            "id": s_id,
            "width": img.shape[1],
            "height": img.shape[0],
            "file_name": "{}.jpg".format(file_name[:-4])
    }

    annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": s_id,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": annon_class,
            "id": s_id
        }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)


    annotations.append(annotation)
    images_train.append(image)

df = pd.read_csv("./gen_meta_test.csv") # Metadata CSV path


count = 1 # For progress Print
for i in df.itertuples(index=False):
    print("Processing Annoted Data: ", count, "/", len(df), end = "\r")
    count = count + 1

    file_name = str(i[0]) # File name attribute
    mask_name = str(i[1]) # Mask name attribute
    annon_class = 1 # Category attribute
    split = i[11] # Test/Train/Val split attribute


    img = np.load(path_to_gen_data + file_name)
    imageio.imwrite(out_path + file_name[0:-4] + '.jpg', img_as_ubyte(img/max(np.max(img), -1*np.min(img) ) ) )

    # Binary Mask to COCO
    img = np.load(path_to_gen_data + mask_name)
    ground_truth_binary_mask = img
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    s_id = int('2' +file_name[5] + file_name[:4] + file_name[7; -10]) # For unique numerical ID

    image = {
            "id": s_id,
            "width": img.shape[1],
            "height": img.shape[0],
            "file_name": "{}.jpg".format(file_name[:-4])
    }

    annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": s_id,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": annon_class,
            "id": s_id
        }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)


    annotations.append(annotation)
    images_test.append(image)

df = pd.read_csv("./gen_meta_val.csv") # Metadata CSV path


count = 1 # For progress Print
for i in df.itertuples(index=False):
    print("Processing Annoted Data: ", count, "/", len(df), end = "\r")
    count = count + 1

    file_name = str(i[0]) # File name attribute
    mask_name = str(i[1]) # Mask name attribute
    annon_class = 1 # Category attribute
    split = i[11] # Test/Train/Val split attribute


    img = np.load(path_to_gen_data + file_name)
    imageio.imwrite(out_path + file_name[0:-4] + '.jpg', img_as_ubyte(img/max(np.max(img), -1*np.min(img) ) ) )

    # Binary Mask to COCO
    img = np.load(path_to_gen_data + mask_name)
    ground_truth_binary_mask = img
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    s_id = int('2' +file_name[5] + file_name[:4] + file_name[7; -10]) # For unique numerical ID

    image = {
            "id": s_id,
            "width": img.shape[1],
            "height": img.shape[0],
            "file_name": "{}.jpg".format(file_name[:-4])
    }

    annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": s_id,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": annon_class,
            "id": s_id
        }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)


    annotations.append(annotation)
    images_val.append(image)

# Balancing Malignant and Benign slices for train and val sets

# min_cnt = min(len(images_train_ml), len(images_train_bn))
# images_train.extend(images_train_ml[:min_cnt])
# images_train.extend(images_train_bn[:min_cnt])
#
# min_cnt = min(len(images_val_ml), len(images_val_bn))
# images_val.extend(images_val_ml[:min_cnt])
# images_val.extend(images_val_bn[:min_cnt])


# Dumping Image and annotation data into 3 json files for Train, Validation and Test
json_object1 = json.dumps(annotations, indent=4)

with open("annotations.json", "w") as outfile:
    outfile.write(json_object1)

json_object2 = json.dumps(images_train, indent=4)

json_object3 = json.dumps(images_test, indent=4)
json_object4 = json.dumps(images_val, indent=4)

with open("images_train.json","w") as outfile:
    outfile.write(json_object2)

with open("images_test.json","w") as outfile:
    outfile.write(json_object3)

with open("images_val.json","w") as outfile:
    outfile.write(json_object4)

f2data = ""
with open('annotations.json') as outfile:
    f2data= '\n' + outfile.read()

with open('images_test.json','a+') as outfile:
    outfile.write(f2data)

with open('images_train.json','a+') as outfile:
    outfile.write(f2data)

with open('images_val.json','a+') as outfile:
    outfile.write(f2data)
