# Import Modules
from keras.models  import load_model, Model
import numpy as np
import tensorflow as tf
from keras import optimizers
from tensorflow.keras import backend as K
import pickle

# Input Parameters
learning_rate = 10**(-5)
img_shape = (100,100,3)
des_layer = -8
only_drama_thresh = 1000
comedy_and_drama_thresh = 1000
train_perc = 0.90

# Uncomment if using 6 genres
#num_imgs = 68054 # For 6 genre dataset
#data = np.memmap('Data/data_6.dat', dtype='float32', mode='r', shape=(num_imgs,img_shape[0],img_shape[1],img_shape[2])) # For 6 genre datset

# Uncomment if using 18 genres
num_imgs = 80762 # For 18 genre dataset
data = np.memmap('Data/data_18.dat', dtype='float32', mode='r', shape=(num_imgs,img_shape[0],img_shape[1],img_shape[2]))

# Load model
model=load_model('Data/vgg16_1.h5')
# Create new model, removing the last layer
new_model = Model(model.inputs, model.layers[des_layer].output)
# Compile model
new_model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9), metrics=['mean_squared_error'])
del model

# Make prediction for each image
print('Making predictions for each image...')
print('')
feature_array = new_model.predict(data)

print('Below is the shape of the feature vector array:')
print(feature_array.shape)

# Save the feature_array
pickle_out = open('Data/feature_array.pickle','wb')
pickle.dump(feature_array, pickle_out)

print('feature vectors saved')

