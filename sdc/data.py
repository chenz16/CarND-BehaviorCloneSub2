'''
This module is for preprocessing data, which consists of four functions:
1) get_data():
a) Output a dataframe containing the train image path and lable value
a) This function specifies the directory where the data log and images are saved
b) Image path and labled values are read into a dataframe
c) Only center camera image data (CenterImage) and steering (SteeringAngle) is kept
d) Raw image data is processed through removing speed below 20mph, mirroring the raw image data

2) vgg_processor(img_io, image_size):
a) Output an numpy array for an image
b) read image from img_io, which could be image path or ByteIO
c) resize image to target size
d) mirror image by using image array transformation
e) use imagnet function to preposs image (RGB->BGR, substact predefined mean)

3) x_reader(path, image_size):
a) output image array (4 dimension) for image batch with input of image batch path


4) data_generator (log, data_size, image_size=(80,80,3), batch_size = 256):
a) output image array (4 dimension) and labels
b) it is a infinite batch data generator
c) the generator is consumed by model.fit_generator and model.predict_generator in
   model.py


'''

from os import path
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.image as img
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import imagenet_utils

# collect data

def get_data():
    # define train data path
    xycols = ["CenterImage", "SteeringAngle"]
    train_data = [("../data_given/data/driving_log.csv", "../data_given/data/IMG")]
    headers = ["CenterImage", "LeftImage", "RightImage",
                "SteeringAngle", "Throttle", "Break", "Speed"]

    # collect data
    logs =[] # as a list
    for log_path, image_folder in train_data:
        log = pd.read_csv(log_path, header=None, names=headers)
        for col in ["CenterImage", "LeftImage", "RightImage"]:
            log[col] = log[col].str.rsplit("/", n=1).str[-1].apply(lambda p: path.join(image_folder, p))
            logs.append(log)
            # combine list to DataFrame
    logs = pd.concat(logs, axis=0, ignore_index=True)

    logs = logs.ix[1:, :] # remove the first row, which is actually the header
    logs.index = logs.index -1 # change the index accordingly

    # data type conversion for column of SteeringAngle and Speed
    logs["SteeringAngle"] = pd.to_numeric(logs["SteeringAngle"], errors='coerce')
    logs["Speed"]         = pd.to_numeric(logs["Speed"], errors='coerce')


    logs = logs[logs['Speed']>20] # Remove noise
    logs3 = logs[['CenterImage', "SteeringAngle"]] # only keep the infor needed for this project
    mirror = logs3.copy()

    # mirror data to get more train data
    mirror["CenterImage"] = mirror["CenterImage"] + "_mirror" #
    mirror["SteeringAngle"] = - mirror["SteeringAngle"]
    log = pd.concat([logs3, mirror], axis=0, ignore_index=True)

    return log

def vgg_processor(img_io, image_size):
	"""Load image, reshape to image_size,
	convert to BGR channel and normalize pixels by subtracting predefined means.
	"""
	h, w, nch = image_size
	if type(img_io) == str and img_io.endswith("_mirror"):
		img_io = img_io[:-7]
		ismirror = True
	else:
		ismirror = False
	img = load_img(img_io, target_size=(h, w))
	# convert to uint array
	img_arr = img_to_array(img)
	img_batch = np.expand_dims(img_arr, axis=0)
	# normalize
	x = imagenet_utils.preprocess_input(img_batch)[0]
	if ismirror:
		x = x[:, ::-1, :]
	return x


def x_reader(path, image_size):
    '''
    read image batch
    '''
    xlist=[]
    path = path.tolist()
    for img_io in path:
        x = vgg_processor(img_io, image_size)
        xlist.append(x)
    return np.array(xlist)

def data_generator (log, data_size, image_size=(80,80,3), batch_size = 256):
    '''
    infinity data generator
    '''
    while 1:
        for offset in range(0, data_size, batch_size):
            path = log["CenterImage"][offset:offset+batch_size]
            y    = log["SteeringAngle"][offset:offset+batch_size]
            y    = np.array(y)
            x    = x_reader(path, image_size)
            yield (x, y)
