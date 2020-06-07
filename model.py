import csv
import os
import cv2
import numpy as np
import sklearn
import PIL.Image as pimg

# Setup Keras
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Discard the header row
lines = lines[1:]

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def append_data(images, angles, batchSample, steerOffset=0.2):
    # Steering angle offsets for Centre, Left & Right images, respectively
    offset = steerOffset*np.array([0, 1, -1])
    for i in range(len(offset)):
        name = 'data/IMG/' + batchSample[i].split('/')[-1]
        img = pimg.open(name)
        image = np.asarray(img)
        angle = float(batchSample[3]) + offset[i]
        images.append(image)
        angles.append(angle)
        # Now flip the image left-to-right
        flippedImg = np.fliplr(image)
        images.append(flippedImg)
        angles.append(-angle)

def generator(lines, batch_size=32):
    num_samples = len(lines)
    steer_angle_offset = 0.2
    while 1: # Loop forever so the generator never terminates
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                append_data(images, angles, batch_sample, steer_angle_offset)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=np.ceil(len(validation_samples)/batch_size),
            epochs=5, verbose=1)

model.save('model.h5')

'''
augmented_images, augmented_angles = [], []
for image,angles in zip(images, angles):
    augmented_images.append(image)
    augmented_angles.append(angle)
    augmented_images.append(cv2.flip(image,1))
    augmented_angles.append(angle*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_angles)
'''