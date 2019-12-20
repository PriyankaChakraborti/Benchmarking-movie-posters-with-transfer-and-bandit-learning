
# Import modules
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from utilities import model_net, plot_learning_curve

patience_num = 10
epoch = 50
BS = 32
img_shape = (112,112,3)
verbose_bool = 2

learning_rate = 10**(-6) # Was 10**(-6)
proportion = 0.90 # Rest goes into validation

print('')
print('Loading data...')
########################################################

# Uncomment if using 6 genres
#genres = ['Action','Comedy','Drama','Horror','Romance','Thriller'] # For 6 genre dataset
#full_csv_df = pd.read_csv('Data/full_poster_6.csv') # For 6 genre dataset
#num_imgs = 68054 # For 6 genre dataset
#data = np.memmap('Data/data_6.dat', dtype='float32', mode='r', shape=(num_imgs,img_shape[0],img_shape[1],img_shape[2])) # For 6 genre datset

# Uncomment if using 18 genres
genres = ['Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Family','Fantasy','History',         'Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'] # For 18 genre dataset
full_csv_df = pd.read_csv('Data/full_poster_18.csv') # For 18 genre dataset
num_imgs = 80762 # For 18 genre dataset
data = np.memmap('Data/data_18.dat', dtype='float32', mode='r', shape=(num_imgs,img_shape[0],img_shape[1],img_shape[2]))

# Initialize chosen model
model=model_net(learning_rate, img_shape)

######## Create Validation Set ##########
print('')
print('Creating train and validation sets...')
num_in_valid = int(num_imgs*(1-proportion))
num_in_train = int(num_imgs*proportion)+1

valid_data = np.zeros((num_in_valid,img_shape[0],img_shape[1],img_shape[2]))
counter = 0
for row in list(range(num_in_valid)):
    valid_data[counter,:,:,:] = data[row,:,:,:]
    counter += 1

train_data = np.zeros((num_in_train,img_shape[0],img_shape[1],img_shape[2]))
counter = 0
for row in list(range(num_in_valid,data.shape[0])):
    train_data[counter,:,:,:] = data[row,:,:,:]
    counter += 1

del data

print('The shape of the train data is: ' + str(train_data.shape))
print('The shape of the validation data is: ' + str(valid_data.shape))
print('')
################################################################

train_datagen = ImageDataGenerator(rescale = 1.)
#train_datagen = ImageDataGenerator(
#    horizontal_flip = True, # Boolean. Randomly flip inputs horizontally.
#    vertical_flip = False,
#    fill_mode = "nearest", # Points outside the boundaries of the input are filled according to the given mode
    #zoom_range = 0.5, # Range for random zoom
#    width_shift_range = 0.1, # fraction of total width
#   height_shift_range=0.1, # fraction of total height
#   rotation_range=5, # Degree range for random rotations
#)
train_generator = train_datagen.flow(train_data, train_data, batch_size=BS, shuffle=True)

valid_datagen = ImageDataGenerator(rescale = 1.)
valid_generator = train_datagen.flow(valid_data, valid_data, batch_size=BS,  shuffle=True)

# Create checkpoint for early stopping and to save model
checkpoint = ModelCheckpoint("Data/vgg16_1.h5", monitor='val_mean_squared_error', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_mean_squared_error', min_delta=0, patience=patience_num, verbose=1, mode='auto')

# Train the model
print('Training the model...')
history = model.fit_generator(train_datagen.flow(train_data, train_data, batch_size=BS, shuffle=True), validation_data=train_datagen.flow(valid_data, valid_data, batch_size=BS,  shuffle=True), steps_per_epoch=train_data.shape[0]//BS, validation_steps=valid_data.shape[0]//BS, epochs=epoch, callbacks=[early,checkpoint], verbose=verbose_bool)

#history = model.fit_generator(train_generator, validation_data=valid_generator, steps_per_epoch=1000//BS, validation_steps=1000//BS, epochs=epoch, verbose=verbose_bool)

model.save('Data/vgg16_1.h5')
del train_data
del valid_data

# Create and save the learning curves
plot_learning_curve(history.history['loss'],history.history['val_loss'])

print('All plots saved')
