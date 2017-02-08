'''
This module is for predicting steering with input of current image
Input: image array (3 dimension)
Output: function self.predict to predict the steering given image input
'''

import numpy as np
import json
import tensorflow as tf
from keras.models import model_from_json

class prediction():
    def __init__(self):
        # load data
        model_file = '../models/model.json'
        weight_file = '../models/model.h5'
        self.model = model_from_json(json.load(open(model_file)))
        self.model.load_weights(weight_file)

    def predict(self, image):
        # predict steering
        image = np.expand_dims(image, axis=0) # reshape
        st = self.model.predict(image)
        return st[0][0]
