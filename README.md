## Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia_model]: ./examples/nvidia.png "NVIDIA Model"
[center_1]: ./examples/center_1.jpg "Center 1 Image"
[center_2]: ./examples/center_2.jpg "Center 2 Image"
[left_2]: ./examples/left_2.jpg "Left 2 Image"
[right_2]: ./examples/right_2.jpg "Right 2 Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

## Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* [model.h5](./model.h5) containing a trained convolution neural network 
* [writeup_report.md](./writeup_report.md) or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my [drive.py](./drive.py), the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](./model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 1x1, 3x3, 5x5 filter sizes and depths between 3 and 64 ([model.py](./model.py) lines 71-76) 

The model includes ELU layers to introduce nonlinearity (code lines 72-76), and the data is normalized in the model using a Keras lambda layer (code line 67). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting ([model.py](./model.py) lines 83-89). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 97). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with the learning rate 0.0001. ([model.py](./model.py) line 95).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I used left and right camera imges to simulated the effect of car wandering off to the side, and recovering. Using left camera images, I added a small ange of 0.25 and using right camera images, I subtracted a small angle of 0.25 from the right camera.

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try new techniques step by step. 

My first step was to use a convolution neural network model similar to the LeNet model I thought this model might be appropriate because It is not too complex for the initial model.

Step by step, I cropped the images in order to remove noise portion of the images and used [the NVIDIA model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). By applying those techniques, the driving performance had been improved. But, the self driving car in autonomouse mode of the simulator came to get out of the track in the sharp curve on the road.

After using images from the left and right cameras and adjusting the angle for the images, the vehicle was able to drive autonomously around the track without leaving the road. 

Finally, I found that setting the epsilon parameter of Adam optimizer to 0.001 improved the training model.

To combat the overfitting, I modified the model so that I added some Dropout layers into my network and set the number of the epochs to 5.

#### 2. Final Model Architecture

The final model architecture ([model.py](./model.py) lines 67-92) consisted of a convolution neural network with the following layers and layer sizes. The final model is based on [the NVIDIA model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), which has been used by NVIDIA for the end-to-end self driving test.


|Layer (type)    |             Output Shape   |      Param #   |
|-|-|-|
|lambda_2 (Lambda)       |     (None, 160, 320, 3)   |    0       |  
|cropping2d_2 (Cropping2D)   | (None, 70, 320, 3)  |      0   |      
|conv2d_7 (Conv2D)   |         (None, 70, 320, 3)   |     12     |   
|conv2d_8 (Conv2D)      |      (None, 33, 158, 24)   |    1824    |  
|conv2d_9 (Conv2D)      |      (None, 15, 77, 36)  |      21636   |  
|conv2d_10 (Conv2D)     |      (None, 6, 37, 48)   |      43248   |  
|conv2d_11 (Conv2D)      |     (None, 4, 35, 64)   |      27712   |  
|conv2d_12 (Conv2D)      |     (None, 2, 33, 64)   |      36928   |  
|flatten_2 (Flatten)     |     (None, 4224)        |      0       |  
|dense_6 (Dense)         |     (None, 1164)        |      4917900 |  
|dropout_5 (Dropout)     |     (None, 1164)        |      0       |  
|dense_7 (Dense)         |     (None, 100)         |      116500  |  
|dropout_6 (Dropout)     |     (None, 100)         |     0        | 
|dense_8 (Dense)         |     (None, 50)          |      5050    |  
|dropout_7 (Dropout)     |     (None, 50)          |      0       |  
|dense_9 (Dense)         |     (None, 10)          |      510     |  
|dropout_8 (Dropout)     |     (None, 10)          |      0       |  
|dense_10 (Dense)        |     (None, 1)           |      11      |  
|Total params: 5,171,331 <br>Trainable params: 5,171,331<br> Non-trainable params: 0|
||


Here is a visualization of the architecture.
![alt text][nvidia_model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_1]

In order to simulate the effect of car wandering off to the side, I used left and right camera imges. I added a small angle 0.25(14.32 degrees) to the left camera and subtract a small angle 0.25 from the right camera.

##### Ajdust steering angle value from -0.05055147 to 0.19944853<br>
![alt text][left_2]
<br>
##### Use the original value
![alt text][center_2]
<br>
##### Ajdust steering angle value from 1 to 0.75 <br>
![alt text][right_2]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and multiply steerning angle by -1 to get correct angle values. 

#### 4. Testing
The vehicle was able to drive autonomously around the track without leaving the road using trained model named model.h5 as follows

python drive.py model.h5

Click on the image to watch the video or [click here](https://youtu.be/B81hCfodRMA). You will be directed to YouTube.

[![Demo Video](https://img.youtube.com/vi/B81hCfodRMA/0.jpg)](https://youtu.be/B81hCfodRMA)
