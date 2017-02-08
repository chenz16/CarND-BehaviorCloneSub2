''''
7 steps to define the model:
1) retrieve training data
2) Extact features from VGG model
3) modify the last 3 layers
4) train model
5) Check training results
6) save model
7) save weights
'''

import numpy as np
import json
from keras.applications import VGG16
from keras.layers import AveragePooling2D, Conv2D
from keras.layers import Input, Flatten, Dense, Lambda, merge
from keras.layers import Dropout, BatchNormalization, ELU
from keras.optimizers import Adam
from keras.models import Model, Sequential, model_from_json
from keras.regularizers import l2
from keras import backend as K
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import data

# 1) retrieve training data from stored directory

# return data frame, "CenterImage" column stores image address (string)
# "SteeringAngle" column stores steering angle (np.float32)
log = data.get_data()

# shuffle data
data_train = shuffle(log)

# split data for training and validation.
# Note: Since the project does not require to test on the second track
# this submissioni ignores verification on test data
data_train, data_val = train_test_split(data_train, test_size=0.2, random_state=200)

train_size = len(data_train)
val_size   = len(data_val)
print('the size of train data', train_size)
print('the size of validation data', val_size)

image_size = (80,80,3)
batch_size = 256

# Use data generator to read data
# Data generator is consumed by model.fit_generator
train_generator = data.data_generator(data_train, train_size, image_size=image_size, batch_size = batch_size)
val_generator   = data.data_generator(data_val,val_size, image_size=image_size, batch_size = batch_size)


# 2) Extact features from VGG model
input_image = Input(shape = image_size) # input shape, base on VGG16
base_model = VGG16(input_tensor=input_image, include_top=False) # extract vgg 16 model


print('\n Total number of VGG16 layers:', len(base_model.layers))

# reset trainable layers
for layer in base_model.layers[:-3]:
    layer.trainable = False
    #print(layer.name, layer.trainable )

W_regularizer = l2(0.01)
x = base_model.get_layer("block5_conv3").output

# 3) modify the last 3 layers
x = AveragePooling2D((2, 2))(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(4096, activation="elu", W_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
x = Dense(1, activation="linear")(x)

model = Model(input=input_image, output=x) # Keras model structure

print("\nmodel summary:")
print(model.summary())


# print sampled weights for the first hidden layer and the last layer
for index, layer in enumerate (model.layers):
    if index==1:
        print(len(layer.get_weights()))
        print ('weight sample of first layer before training\n', layer.get_weights()[0][1][1][1][1:10])
    if index== (len(model.layers)-1):
        print('weight sample of last layer before training\n', layer.get_weights()[0][1:10,:])

# 4) train model
# preload trained weights if file exists
Folder2save = "../models/"
weight_file = Folder2save + 'model.h5'
try:
    model.load_weights(weight_file)
    print("Weights preloaded from" + weight_file,
            "  Weights are being trained now......")
except:
    print ('No weights preloaded. Weights are being trained now........')

# compile model
model.compile(loss= "mean_squared_error", optimizer='Adam', metrics=['accuracy'])

#train model
nb_epoch = 10
history = model.fit_generator(train_generator, verbose=1,
                            samples_per_epoch = train_size,
							nb_epoch = nb_epoch,
							validation_data = val_generator,
							nb_val_samples = val_size)

# print sampled weights for the first hidden layer and the last layer
for index, layer in enumerate (model.layers):
    if index==1:
        print ('weight sample of first layer after training\n', layer.get_weights()[0][1][1][1][1:10])
    if index== (len(model.layers)-1):
        print('weight sample of last layer after training\n', layer.get_weights()[0][1:10,:])

# 5) validate model
y_p = model.predict_generator(val_generator, val_size)
y_true = data_val['SteeringAngle']
y_true = np.array(y_true)
assert(y_p.shape == y_true.shape, 'Dimension of prediction and true value of validation data does not match')
mse = np.mean ((y_p- y_true) * (y_p- y_true))
print("Mean-square-error of validation data set:", mse)

evaluation = model.evaluate_generator(val_generator, val_size)
print("Loss of validation data set:", evaluation[0])
print("Accuracy of validation data set:", evaluation[1])

# 6) save model
json_string = model.to_json()
#json.dump(json_string,  open(Folder2save + "model.json", "w"))
json.dump(json_string,  open("../models4/model.json", "w"))
print("Successfully saved model to" + Folder2save + "model.json")

# 7) save weights
model.save_weights(Folder2save + 'model.h5', overwrite=True)
print("Successfully saved weights to" + Folder2save + "model.h5")
