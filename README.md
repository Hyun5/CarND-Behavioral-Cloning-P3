# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

[//]: # (Image References)

[image0]: ./examples/NVIDIA.JPG "NVIDIA Architecture"
[image1]: ./examples/center_drive.png "Center Drive"
[image2]: ./examples/original_image.png "Normal Image"
[image3]: ./examples/flipped_image.png "Flipped Image"


Overview
---
This repository contains the final submit for the Udacity Behavioral Cloning Project.

In this project, I used the deep neural networks and convolutional neural networks to clone driving behavior. I traind, validated and tested a model using Keras. The model output a steering angle to an autonomous vehicle.

Udacity have provided a simulator where I can steer a car around a track for data collection. I used image data and steering angles to train a neural network and then used this model to drive the car autonomously around the track.

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 a video recording of the vehicle driving autonomously around the track
* READ.md summarizing the results


The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My model is based on the information of the Udacity lecture and the NVIDA architecture. The model contains dropout layers in order to reduce overfitting. I added 2 times of 50% Dropout in the model.
![alt text][image0]

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers.

```
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
```
#### 3. Creation of the Training Set & Training Process
To capture good driving behavior, I first recorded two laps on track one using center lane driving using the Udacity provided simulator. All the image files and the log are stored in data directory.
```
python drive.py
```
Here is an example image of center lane driving:
![alt text][image1]

Run the model.py to proceed the trained convolutiona neural network.
```
python model.py
```
To augment the data set, I flipped images and changed angles to opposite. For example, here is an image that has then been flipped:
```
# flip the image left-to-right
flippedImg = np.fliplr(image)
images.append(flippedImg)
angles.append(-angle)
```
![alt text][image2]
![alt text][image3]

Hyperparameters
```
# Hyperparameteres
batch_size=32
EPOCH=5
```
Training and validation result
```
201/201 [==============================] - 266s 1s/step - loss: 0.0206 - val_loss: 0.0170
Epoch 2/5
201/201 [==============================] - 34s 167ms/step - loss: 0.0177 - val_loss: 0.0175
Epoch 3/5
201/201 [==============================] - 37s 183ms/step - loss: 0.0167 - val_loss: 0.0163
Epoch 4/5
201/201 [==============================] - 36s 180ms/step - loss: 0.0160 - val_loss: 0.0157
Epoch 5/5
201/201 [==============================] - 36s 180ms/step - loss: 0.0152 - val_loss: 0.0155
```

#### 4. Model Testing
After training finished, using the Udacity provided simulator, I tested the perpormance with my model.h5. 
The car was able to be driven autonomously around the track by executing; 
```
python drive.py model.h5
```
The car was smoothly driven around track. [See Video File Here.](https://github.com/Hyun5/CarND-Behavioral-Cloning-P3/blob/master/examples/video.mp4)


#### 5. Discussions
1. There can be more data augument technics I did not implemeted, i.e random image selecting between the left, right and center images, converting BGR to RBG, translating the image, ... Those would help to increase the accuracy. 
2. Will try 2nd track.
