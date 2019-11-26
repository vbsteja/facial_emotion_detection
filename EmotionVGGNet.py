from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import ELU,Softmax
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import Adam
#from config import emotion_config as config
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from keras.utils import to_categorical

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints",
help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
help="epoch to restart training at")
args = vars(ap.parse_args())

class EmotionVGGNet:

	@staticmethod
	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height,width,depth)
		chanDim = -1

		if K.image_data_format == "channels_first":
			inputShape = (depth,height,width)
			chanDim = 1
		model = Sequential([
			Conv2D(32,(3,3), padding="same",
			kernel_initializer="he_normal",input_shape=inputShape),
			ELU(),
			BatchNormalization(axis=chanDim),
			Conv2D(32,(3,3), kernel_initializer="he_normal",
				padding="same"),
			ELU(),
			BatchNormalization(axis=chanDim),
			MaxPooling2D(pool_size=(2,2)),
			Dropout(0,25),
			Conv2D(64,(3,3), padding="same",
			kernel_initializer="he_normal",input_shape=inputShape),
			ELU(),
			BatchNormalization(axis=chanDim),
			Conv2D(64,(3,3), kernel_initializer="he_normal",
				padding="same"),
			ELU(),
			BatchNormalization(axis=chanDim),
			MaxPooling2D(pool_size=(2,2)),
			Dropout(0,25),
			Conv2D(64,(3,3), padding="same",
			kernel_initializer="he_normal",input_shape=inputShape),
			ELU(),
			BatchNormalization(axis=chanDim),
			Conv2D(64,(3,3), kernel_initializer="he_normal",
				padding="same"),
			ELU(),
			BatchNormalization(axis=chanDim),
			MaxPooling2D(pool_size=(2,2)),
			Dropout(0,25),
			Flatten(),
			Dense(64,kernel_initializer="he_normal"),
			ELU(),
			BatchNormalization(),
			Dropout(0.25),
			Dense(64,kernel_initializer="he_normal"),
			ELU(),
			BatchNormalization(),
			Dropout(0.25),
			Dense(classes,kernel_initializer="he_normal"),
			Softmax()])
		return model


emotion_dict = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
BASE_DIR = "dataset/kaggle_fer/fer2013/"
data = pd.read_csv(BASE_DIR+"fer2013.csv")
data.head()

train_pixels = [np.array([int(i) for i in j.split()]) for j in data[data.Usage == "Training"]["pixels"].values]
train_pixels = np.array([i.reshape((48,48)) for i in train_pixels])
train_pixels = np.expand_dims(train_pixels,-1)
train_labels = np.array([int(j) for j in data[data.Usage == 'Training']["emotion"].values])
train_labels = np.array(to_categorical(train_labels,num_classes = 7))

test_pixels = [np.array([int(i) for i in j.split()]) for j in data[data.Usage == "PrivateTest"]["pixels"].values]
test_pixels = np.array([i.reshape((48,48)) for i in test_pixels])
test_pixels = np.expand_dims(test_pixels,-1)
test_labels = np.array([int(j) for j in data[data.Usage == 'PrivateTest']["emotion"].values])
test_labels = np.array(to_categorical(test_labels,num_classes = 7))


validation_pixels = [np.array([int(i) for i in j.split()]) for j in data[data.Usage == "PublicTest"]["pixels"].values]
validation_pixels = np.array([i.reshape((48,48)) for i in validation_pixels])
validation_pixels = np.expand_dims(validation_pixels,-1)
validation_labels = np.array([int(j) for j in data[data.Usage == 'PublicTest']["emotion"].values])
validation_labels = np.array(to_categorical(validation_labels,num_classes = 7))

train = ImageDataGenerator(rotation_range=10, zoom_range=0.1,horizontal_flip=True, rescale=1 / 255.0, fill_mode="nearest")
test = ImageDataGenerator(rescale=1 / 255.0)
validation = ImageDataGenerator(rescale=1 / 255.0)

#train.fit(train_pixels)
#test.fit(test_pixels)
#validation.fit(validation_pixels)

if args["model"] is None:
	print("[INFO] compiling model...")
	model = EmotionVGGNet.build(width=48, height=48, depth=1,classes=len(emotion_dict))
	opt = Adam(lr=1e-3)
	model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

else:
	print("[INFO] loading {}...".format(args["model"]))
	model = load_model(args["model"])
	# update the learning rate
	print("[INFO] old learning rate: {}".format(
	K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 1e-3)
	print("[INFO] new learning rate: {}".format(
	K.get_value(model.optimizer.lr)))


checkpointer = ModelCheckpoint(filepath='weights/EmotionVGGNet.hdf5', verbose=1, save_best_only=True)


model.fit_generator(
	train.flow(train_pixels,train_labels,batch_size=32),
	validation_data = validation.flow(validation_pixels,validation_labels),
	epochs = 15,
	callbacks = [checkpointer],
	verbose=1)

model.save("EmotionVGGNet.hdf5")
model = load_model("EmotionVGGNet.hdf5")

loss, acc = model.evaluate_generatoor(
	test.flow(test_pixels,test_labels,color_mode="grayscale"))

print("[INFO] accuracy : {:.2f}".format(acc))

