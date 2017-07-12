
# **Behavioral Cloning**
---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
---
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 41-49)

The model includes RELU layers to introduce nonlinearity (code lines 41-59), and the data is normalized in the model using a Keras lambda layer (code line 39).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 52 and 55).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 113-119). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 66).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving counter-clockwise.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the [NVIDIA Architecture](https://arxiv.org/abs/1604.07316).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by adding dropout and augmenting more data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially at keen curves. To improve the driving behavior in these cases, I recorded more data at curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 37-61) consisted of a convolution neural network with the following layers and layer sizes:

Layer Description  
Input 66x200x3 YUV image  
lambda outputs 66x200x3
Convolution 5x5 2x2 stride, valid padding, outputs 31x98x24  
RELU    
Convolution 5x5 2x2 stride, valid padding, outputs 14x47x36  
RELU    
Convolution 5x5 2x2 stride, valid padding, outputs 5x22x48  
RELU    
Convolution 3x3 1x1 stride, valid padding, outputs 3x20x64  
RELU  
Convolution 3x3 1x1 stride, valid padding, outputs 1x18x64  
RELU  
Flatten, outputs 1152  
Dropout  
Fully connected, outputs 100  
RELU   
Dropout  
Fully connected, outputs 50  
RELU    
Fully connected, outputs 10  
RELU    
Fully connected, outputs 1  


Here is a visualization of the architecture

![model_architecture](images/model.png)



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first started with sample data: recording several laps on track one using center lane driving. Here is an example image of center lane driving:

![center](images/center_2016_12_01_13_32_43_558.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to keep driving center. These images show what a recovery looks like starting from the left side of the road. :

![recoverfromleft](images/center_2017_04_15_14_22_58_364.jpg)

Then, I recorded driving clockwise and counter-clockwise at curves so that the vehicle would learn to curve more smoothly.

To obtain more data, I used multiple cameras: center, left and right cameras. To correct camera positions, I added 0.06 to left steering data and subtract 0.06 from right steering data.  Here is example images of multiple cameras:

![centerlane](images/dataexample.png)
![centerlane](images/dataexample2.png)
![centerlane](images/dataexample3.png)
![centerlane](images/dataexample4.png)

To augment the data set, I also flipped images and angles thinking that this would combat the overfitting.

![flip](images/flip.png)

After the collection process, I had 55800 number of data points. Here is an histogram of steering angles:

![hist](images/histogram.png)

I then preprocessed this data by converting RGB color to YUV, cropping center region of images and resize images so that image sizes are same as the input size of the model (model.py lines 73-77). For example, here is an image that has then been preprocessed:

![preprocessed](images/preprocessed.png)

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the following graph. I used an Adam optimizer so that manually training the learning rate wasn't necessary.

![modeltraining](images/trainhistory.png)
____________________________________________________________________________________________________
