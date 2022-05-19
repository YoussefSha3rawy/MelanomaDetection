#!/usr/bin/env python
# coding: utf-8

# # **Imports**

# In[3]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import PIL
from imutils import paths
from glob import glob
import matplotlib.pyplot as plt
import shutil


# # **Dataset Loading**

# In[2]:


DATASET_PATH="/content/drive/MyDrive/ISIC 2019 Dataset"
TRAINING_PATH=DATASET_PATH+'/Training'
VALIDATION_PATH=DATASET_PATH+'/Validation'
TEST_PATH=DATASET_PATH+'/Test'
GROUND_TRUTH_PATH=DATASET_PATH+'/ISIC_2019_Training_GroundTruth.csv'
data = pd.read_csv(GROUND_TRUTH_PATH, index_col='image')

BENIGN_TRAINING_PATH = TRAINING_PATH+'/0'
MEL_TRAINING_PATH = TRAINING_PATH+'/1'

BENIGN_TRAINING_PATHS = sorted(paths.list_images(BENIGN_TRAINING_PATH))
print("Working with {0} non-melanoma images".format(len(BENIGN_TRAINING_PATHS)))

MEL_TRAINING_PATHS = sorted(paths.list_images(MEL_TRAINING_PATH))
print("Working with {0} melanoma images".format(len(MEL_TRAINING_PATHS)))


# ## **Dataset Testing**

# In[6]:


plt.figure(figsize=(20, 10))
for i in range(5):
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(PIL.Image.open(BENIGN_TRAINING_PATHS[i]))
    plt.title("Non-Melanoma")
    plt.axis("off")
    ax = plt.subplot(2, 5, i + 1+5)
    plt.imshow(PIL.Image.open(MEL_TRAINING_PATHS[i]))
    plt.title("Melanoma")
    plt.axis("off")


# # **ResNet152V2**

# In[8]:


from tensorflow.keras.applications.resnet_v2 import preprocess_input
base = tf.keras.applications.ResNet152V2(input_shape=[224,224,3], weights='imagenet', include_top=False)

base.trainable = False

# Create new model on top
inputs = keras.Input(shape=(224, 224, 3))
x=base(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(2, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

from keras import optimizers


adam = optimizers.Adam()
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_set = validation_datagen.flow_from_directory(VALIDATION_PATH,(224,224),
                                                 class_mode = 'categorical')
train_set = train_datagen.flow_from_directory(TRAINING_PATH,(224,224),
                                                 class_mode = 'categorical')


# In[11]:


from datetime import datetime
from keras.callbacks import ModelCheckpoint



checkpoint = ModelCheckpoint(filepath=DATASET_PATH+'/model/ResNet152V2', 
                               verbose=1, save_best_only=True)
callbacks = [checkpoint]

start = datetime.now()

model_history=model.fit(
  train_set,
  validation_data=validation_set,
  epochs=50,
  steps_per_epoch= 50,
  validation_steps=32,
    callbacks=callbacks ,verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# In[12]:


base.trainable = True
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  
    loss='binary_crossentropy',
    metrics=[keras.metrics.BinaryAccuracy()],
)

start = datetime.now()

model_history=model.fit(
  train_set,
  validation_data=validation_set,
  epochs=50,
  steps_per_epoch= 300,
  validation_steps=32,
    callbacks=callbacks ,verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)


# In[15]:


model = keras.models.load_model(DATASET_PATH+'/model/ResNet152V2')
# model.load_weights('mymodel.h5',by_name=True)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_set = train_datagen.flow_from_directory(TEST_PATH,(224,224), shuffle=False,
                                                 class_mode = 'categorical')
result = model.predict(test_set)


# In[16]:


res=np.argmax(result, axis=1)
names=test_set.filenames
tp=0
tn=0
fp=0
fn=0
for i in range(len(names)):
  gt = names[i][0]
  filename = names[i][2:-4]
  if res[i]==0:
    if gt=='0':
      tn+=1
    else:
      fn+=1
  else:
    if gt=='0':
      fp+=1
    else:
      tp+=1
sensitivity = round(tp*100/(tp+fn),2)
specifity = round(tn*100/(tn+fp),2)
accuracy = round((tp+tn)*100/(tn+tp+fn+fp),2)

print(tp)
print(tn)
print(fp)
print(fn)
print('sensitivity: ',sensitivity,'%')
print('specifity: ',specifity,'%')
print('accuracy: ',accuracy,'%')


# In[17]:


_# Plot training & validation loss values
plt.plot(model_history.history['binary_accuracy'])
plt.plot(model_history.history['val_binary_accuracy'])
plt.title('CNN Model accuracy values')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

