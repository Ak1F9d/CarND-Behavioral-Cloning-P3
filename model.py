# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
##load data
#filepath
data_folder='windows_sim/data_sample/data/'
logcsv_file=data_folder+'driving_log.csv'
# read  the path of three cameras images and steering angles data from csv file
camera_filename = np.loadtxt(logcsv_file,delimiter=",", usecols=(0,1,2),dtype='S1000', skiprows =1).reshape(-1,order='F')
steering_data=np.loadtxt(logcsv_file,delimiter=",", usecols=(3,),skiprows =1)
print('numimages=%s' % len(steering_data))
#split filename
vfunc = np.vectorize(os.path.split)
camera_filename=vfunc(camera_filename.astype('str'))[-1]
#offset for left and right cameras
correction=0.06
steering_data=np.concatenate((steering_data, steering_data+correction, steering_data-correction ))
steering_data=np.concatenate((steering_data, -steering_data))
#for flipping augmentation
numimage_unflip=len(camera_filename)
camera_filename=np.concatenate((camera_filename, camera_filename))
Is_to_be_flipped=np.concatenate((np.ones(numimage_unflip), -np.ones(numimage_unflip)))
steering_data=np.concatenate((steering_data[:,None],Is_to_be_flipped[:,None]), axis=1)
#show histogram of steering angles
plt.hist(steering_data[:,0], bins=30)
plt.show()

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda
from keras import optimizers

##modeling
#Create the Sequential model
model = Sequential()
# set up lambda layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66,200,3)))
#1st 2dconv layer
model.add(Convolution2D(24, 5, 5, activation='elu',input_shape=(66, 200, 3),  subsample=(2,2)))
#2nd 2dconv layer
model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2,2)))
#3rd 2dconv layer
model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2,2)))
#4th 2dconv layer
model.add(Convolution2D(64, 3, 3, activation='elu'))
#5th 2dconv layer
model.add(Convolution2D(64, 3, 3, activation='elu'))
#flatten and dropout
model.add(Flatten())
model.add(Dropout(0.5))
#1st fully connected layer and dropout
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
#2nd fully connected layer
model.add(Dense(50, activation='elu'))
#3rd fully connected layer
model.add(Dense(10, activation='elu'))
#final fully connected layer
model.add(Dense(1))

#optimizer setting
#adam=optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)
#model compilation
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#summerize model
model.summary()

##Three functions to train
import cv2
#crop, convert to YUV and resize images
def preprocess(img_in,imsize=(66,200)):
    img=img_in[54:-40,:,:] .astype('uint8')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img_out=cv2.resize(img,imsize[::-1])
    return img_out
#display training history
def traininghistory(history_object):
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
#generator function
EPOCHS=3
batch_size=200
import sklearn
from sklearn.model_selection import train_test_split
def generator(samples, batch_size=32):
    num_samples = len(samples[0])
    while 1: # Loop forever so the generator never terminates
        samples_image, samples_angle=sklearn.utils.shuffle(samples[0], samples[1])
        for offset in range(0, num_samples, batch_size):
            batch_samples = zip(samples_image[offset:offset+batch_size], samples_angle[offset:offset+batch_size,:] )
            images = []
            angles = []

            for batch_sample_i,batch_sample_a in batch_samples:
                img=cv2.imread(data_folder+'IMG/'+batch_sample_i)
                img_pp=preprocess(img)
                images.append(img_pp[:,::int(batch_sample_a[1]),:])
                angles.append(batch_sample_a[0])
            yield sklearn.utils.shuffle(np.array(images), np.array(angles))

## compile and train the model using the generator function
X_train, X_valid, Y_train, Y_valid = train_test_split(camera_filename,steering_data, test_size=0.2)
train_samples=[X_train,Y_train]
validation_samples=[X_valid, Y_valid]
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

history_object=model.fit_generator(train_generator, samples_per_epoch= len(train_samples[0]), validation_data=validation_generator,
                    verbose=1, nb_val_samples=len(validation_samples), nb_epoch=EPOCHS)
traininghistory(history_object)

model.save('model.h5')

from keras.utils.visualize_util import plot
plot(model,to_file='model.png', show_shapes=True)
