'''
setup_campaign_tf2.py -- campaign data
This model was derived from Nicholas Carlini nn_robust_attacks model 
'''

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

# get IMG data from campaign_data dir
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 720, 720, 1)
        return data

# get IMG labels from campaign_data dir
def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

# Note: campaign 2, 3, & 7 have images with same names, must hanlde them seperatly
class MNIST:
    def __init__(self):
        if not os.path.exists("campaign_data"):
            print("Please import needed campaign data")
	    return
	'''
	# dont need since data is loaded internally from campaign_data Dir
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:
	'''

                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

	# want 25K data sampels of 50K from each campaign
        train_data_1 = extract_data("campaign_data/campaign2_train_images.tar.gz", 25000)
        train_labels_1 = extract_labels("campaign_data/campaign2_train_labels.tar.gz", 25000)
        train_data_2 = extract_data("campaign_data/campaign3_train_images.tar.gz", 25000)
        train_labels_2 = extract_labels("campaign_data/campaign3_train_labels.tar.gz", 25000)
	self.test_data = extract_data("campaign_data/campaign7_images.tar.gz", 25000)
        self.test_labels = extract_labels("data/campaign7_labels.tar.gz", 25000)
        
	# 8% of training batch size
        VALIDATION_SIZE = 2000
        
        self.validation_data_1 = train_data_1[:VALIDATION_SIZE, :, :, :]
        self.validation_labels_1 = train_labels_1[:VALIDATION_SIZE]
        self.validation_data_2 = train_data_2[:VALIDATION_SIZE, :, :, :]
        self.validation_labels_2 = train_labels_2[:VALIDATION_SIZE]
	
	self.train_data_1 = train_data_1[VALIDATION_SIZE:, :, :, :]
        self.train_labels_1 = train_labels_1[VALIDATION_SIZE:]
	self.train_data_2 = train_data_2[VALIDATION_SIZE:, :, :, :]
        self.train_labels_2 = train_labels_2[VALIDATION_SIZE:]

class MNISTModel:
    def __init__(self, restore, session=None):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
	
        model = Sequential()

        model.add(Conv2D(32, (3, 3),
                         input_shape=(28, 28, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)
