###This is a pipeline for training the model to drive a car
import os
import math
import csv
import cv2
import numpy as np
import sklearn
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, Cropping2D, Conv2D, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.model_selection import train_test_split


###################################################################################################
###DEFINE KERAS MODEL -- using NVIDIA model here
model=Sequential()
model.add(Lambda(lambda x: x/255.0-0.5,input_shape=(160,320,3))) #normalization lambda layer
model.add(Cropping2D(cropping=((70,25),(0,0)))) #cropping
model.add(Conv2D(24,(5,5),activation="relu", strides=(2,2)))
model.add(Conv2D(36,(5,5),activation="relu",strides=(2,2)))
model.add(Conv2D(48,(5,5),activation="relu",strides=(2,2)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#############################################################################
###IMPORT DATA
#CSV-- read in samples
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
n_samples=len(samples)-1
samples = samples[1:]       #get rid of header line
print("Samples imported: ",n_samples)
for i in range(8036,len(samples)):
    if len(samples[i][0])>50:
        samples[i][0] = samples[i][0][54:]

#Split into test and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)       
print("#Training: ",len(train_samples))
print("#Validation: ",len(validation_samples))
##############################################################################
###USE GENERATOR
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            if offset+batch_size>len(samples):
                break
                
            batch_samples = samples[offset:(offset+batch_size)]

            images = np.zeros([batch_size*2, 160, 320,3])
            i=0 
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                flipped_image = np.fliplr(center_image)
                flipped_angle = -1*(center_angle)
                angles = np.append(angles,center_angle)
                images[i] = center_image
                angles = np.append(angles, flipped_angle)
                images[i+1] = flipped_image
                i+=2
            
           
            yield sklearn.utils.shuffle(images, angles)

##############################################################################
###TRAIN THE MODEL
batch_size=33
generator_step = int(batch_size/3)
train_steps = math.floor(len(train_samples)/batch_size)
valid_steps = math.floor(len(validation_samples)/batch_size)
train_generator = generator(train_samples, generator_step)
validation_generator = generator(validation_samples, generator_step)
          
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= train_steps, validation_data=validation_generator, validation_steps=valid_steps, epochs=5, verbose = 1)


model.save("model5.h5")
print("Model saved.")
##Model codes:
#model.h5: provided data- flipped images & left & right camera, 0.1 correction offset
#model2.h5: 0.2 correction offset
#model3.h5: doesn't use right & left cameras-- WAY better (loss 0.01)
#model4.h5: my data.. (loss 0.0177)
#model5.h5: added dropout (loss 0.0127)
###############################################################################
#run the model with: python drive.py model1.h5
#run model & record with: python drive.py model1.h5 run1
#write video with: python video.py run1