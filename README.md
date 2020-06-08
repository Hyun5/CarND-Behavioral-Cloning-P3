# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

[//]: # (Image References)

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

This README file describes how to output the video in the "Details About Files In This Directory" section.


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

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:
![alt text][image1]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:
```
# flip the image left-to-right
flippedImg = np.fliplr(image)
images.append(flippedImg)
angles.append(-angle)
```
![alt text][image2]
![alt text][image3]

#### 4. Model Testing

After the model was trained, the test drive was done with Autonomous mode using the simulator. The car could drive smothly around track.
![Test Video File](https://github.com/Hyun5/CarND-Behavioral-Cloning-P3/blob/master/examples/video.mp4)




