#!/usr/bin/env python

import sys
import cv2
import keras
import h5py
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import load_model
import math
import numpy as np
import os
import random
import datetime
import numpy as np

from image_reader import ReadImageFiles

TRAIN_SPLIT = 0.9
IMAGE_SCALE = 1.0 / 8.0

GRAY = False
AugFlip = False     
NOISE = True


NUM_FILE_READ_THREADS = 10

def main():
  if len(sys.argv) < 2:
    print('USAGE: ./model.py path/to/images')
    sys.exit(1)

  image_path = sys.argv[1]

  start_time = datetime.datetime.now()
  X, Y = ReadImageFiles(image_path, IMAGE_SCALE, NUM_FILE_READ_THREADS, shuffle=False, gray=GRAY)
  N, image_height, image_width, image_channels = X.shape
  end_time = datetime.datetime.now()
  print('Time to read files: ' + str(end_time - start_time))

  train_count = int(math.floor(N * TRAIN_SPLIT))
  test_count = N - train_count
  print('Loaded %d images from %s (%d train, %d test)' %
      (N, image_path, train_count, test_count))

  if AugFlip==True:
    X_flip = X[0:train_count].copy()
    Y_flip = Y[0:train_count].copy()

    for index, img in enumerate(X_flip):
      img = cv2.flip(img, 1)
      X_flip[index] = img
      Y_flip[index] *= -1.0


    # Create the arrays that are going to store our training and test data.
    X_train = np.concatenate( (X[0:train_count], X_flip), axis=0)
    Y_train = np.concatenate( (Y[0:train_count], Y_flip), axis=0)
  elif NOISE==True:
    center = 1
    X_blur = X[0:train_count].copy()
    Y_blur = Y[0:train_count].copy()
    X_shape = X[0].shape

    for index, img in enumerate(X_blur):
        n = np.random.normal(0, math.sqrt(center), X_shape)
        X_blur[index] = img + n

    # Create the arrays that are going to store our training and test data.
    X_train = np.concatenate( (X[0:train_count], X_blur), axis=0)
    Y_train = np.concatenate( (Y[0:train_count], Y_blur), axis=0)
  else:
    X_train = X[0:train_count]
    Y_train = Y[0:train_count]

  X_test = X[train_count: ]
  Y_test = Y[train_count: ]

  # Scale inputs to [0, 1]
  X_train /= 255
  X_test /= 255

  Y_train = (Y_train + 100.0) / 200.0
  Y_test = (Y_test + 100.0) / 200.0


  # Construct our model
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, image_channels)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(1))

  model.compile(loss=keras.losses.mean_squared_error,
      optimizer=keras.optimizers.Adadelta())
  model.fit(X_train, Y_train, batch_size=32, epochs=15, validation_split=0.1,
      verbose=1)

  model.save('learned_model.h5')

  print(model.evaluate(X_test, Y_test))
  #Y_predicted = model.predict(X_test, verbose=1)
  
if __name__ == '__main__':
  main()
