

import numpy as np
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Concatenate, convolutional, Conv2DTranspose, Conv2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import pandas as pd
from keras.applications.resnet50 import ResNet50
from keras.utils.data_utils import Sequence
from keras.layers import Lambda, Reshape, BatchNormalization
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras.losses import mean_absolute_error, mean_squared_error
from keras import regularizers

import os
import matplotlib.pyplot as plt

import glob



from imblearn.over_sampling import RandomOverSampler
from imblearn.keras import balanced_batch_generator

K = tf.keras.backend



# Once-hot encodes labels
def one_hot_encode(labels):
    # Instantiate one hot encoder
    enc_lab=OneHotEncoder()
    
    # Fit the encoder with its labels
    enc_lab.fit(labels.reshape(-1,1))
    
    return enc_lab.transform(labels.reshape(-1,1)).toarray()

class BalancedDataGenerator(Sequence):
    """ImageDataGenerator + RandomOversampling"""
    def __init__(self, x, y, datagen, batch_size=32):
        self.datagen = datagen
        self.batch_size = batch_size
        self._shape = x.shape
        datagen.fit(x)
        
        self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y, sampler=RandomOverSampler(), batch_size=self.batch_size, keep_sparse=True)

    def __len__(self):
        return self._shape[0] // self.batch_size

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()



# first one hot encode the pandas datframe by category values
genres= ['Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Family','Fantasy','History','Horror','Musical','Mystery','Romance','Sci-Fi','SuperHero','Thriller','War','Western']



# in case we decide to augment the image
def transform_image(data,labels):
    # Note we return the exact same labels as we sent in
    '''Parameters
       labels:np array
       data:np nd array
       One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.

    '''
    
    stack_data=[]
    # Reshape data from flattened image back into original image
    for i in range(data.shape[0]):
        item=data[i,:].reshape(28,28)
        stack_data.append(item)
        
    available_transformations=['s&p','speckle','poisson','gauss']
    dup_Arrays = np.empty((data.shape[0],data.shape[1]))
    for index,image in enumerate(stack_data):

        # Apply a single transformation or a combination of transformations
        num_trans_to_apply=re.randint(1,len(available_transformations))
        j=0
        
        while j<num_trans_to_apply:
            new_img=noisy(re.choice(available_transformations),image)
            image=new_img
            j+=1
            
        #flatten this new image
        image=image.reshape(1,784)
        
        # Concatenate to original np arrays of data and labels
        dup_Arrays[index,:] = image
        
    return dup_Arrays,labels


# ## please filter before preprocessing so that anything deleted from df is deleted from folder


def _preprocess_image(model_name,bool_trans=True):
    if model_name=='vgg-16':
        target_size=(224,224)
    elif model_name=='inception-v3' or model_name=='inception_resnet':
        target_size=(299,299)
    
    image_list=[]
    label_list
    for filename in os.listdir('/Users/priyanka/Box Sync/CSE496-final/posters_*'):
        for picture in glob.glob(filename+'/*.jpg'):
            temp_name=picture.split('/')[-1].split('.')[0].split('_')
            df=mega_gdf.loc[mega_gdf['year']==temp_name.pop() and mega_gdf['title']==temp_name.pop()]
            label_list.append(df[genres])
            temp_image=image.load(filename,target_size)
            #convert to array including RGB color channel
            temp_image=image.image_to_array(temp_image)
            image_list.append(temp_image)
    if not len(image_list)==mega_gdf.shape[0]:
        raise AssertionError()
    return np.asarray(image_list),np.asarray(label_list)        







def FocalLoss(y_true, y_pred):
    gamma=2
    alpha=0.25
    y_true = tf.cast(y_true,dtype=tf.float32)
    y_pred = tf.cast(y_pred,dtype=tf.float32)
    
    cross_entropy_loss = K.binary_crossentropy(y_true, y_pred, from_logits=False)
    p_t = ((y_true * y_pred) + ((1.0 - y_true) * (1.0 - y_pred)))
    modulating_factor = 1.0
    if gamma:
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_weight_factor = 1.0
        if alpha is not None:
            alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))
            focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * cross_entropy_loss)

            return K.mean(focal_cross_entropy_loss, axis=-1)
        
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        threshold = 0.5
        y_pred = K.cast(K.greater(y_pred, threshold), K.floatx())
        
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def focal_loss_fixed(y_true, y_pred):
    """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
    gamma=2.
    alpha=4.
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(reduced_fl)

def KerasFocalLoss(y_true, y_pred):
    
    gamma = 2.
    y_pred = tf.cast(y_pred, tf.float32)
    
    max_val = K.clip(-y_pred, 0, 1)
    loss = y_pred - y_pred * y_true + max_val + K.log(K.exp(-max_val) + K.exp(-y_pred - max_val))
    invprobs = tf.log_sigmoid(-y_pred * (y_true * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss
    
    return K.mean(K.sum(loss, axis=1))
        
def hamming_dist(y_true,y_pred):
    #average(y_true*(1-y_pred)+(1-y_true)*y_pred)
    
    y_true = tf.cast(y_true,dtype=tf.float32)
    y_pred = tf.cast(y_pred,dtype=tf.float32)
    
    return K.mean(K.sum(K.abs(y_true-y_pred),axis=1))

#create model
def model_net(learning_rate, img_shape):

    layers_freeze=3

    #include top discards the final dense layers
    model = VGG16(weights = "imagenet", include_top=False, input_shape = img_shape)
    """
        Layer (type)                 Output Shape              Param #   
        =================================================================
        input_1 (InputLayer)         (None, 256, 256, 3)       0         
        _________________________________________________________________
        block1_conv1 (Conv2D)        (None, 256, 256, 64)      1792      
        _________________________________________________________________
        block1_conv2 (Conv2D)        (None, 256, 256, 64)      36928     
        _________________________________________________________________
        block1_pool (MaxPooling2D)   (None, 128, 128, 64)      0         
        _________________________________________________________________
        block2_conv1 (Conv2D)        (None, 128, 128, 128)     73856     
        _________________________________________________________________
        block2_conv2 (Conv2D)        (None, 128, 128, 128)     147584    
        _________________________________________________________________
        block2_pool (MaxPooling2D)   (None, 64, 64, 128)       0         
        _________________________________________________________________
        block3_conv1 (Conv2D)        (None, 64, 64, 256)       295168    
        _________________________________________________________________
        block3_conv2 (Conv2D)        (None, 64, 64, 256)       590080    
        _________________________________________________________________
        block3_conv3 (Conv2D)        (None, 64, 64, 256)       590080    
        _________________________________________________________________
        block3_conv4 (Conv2D)        (None, 64, 64, 256)       590080    
        _________________________________________________________________
        block3_pool (MaxPooling2D)   (None, 32, 32, 256)       0         
        _________________________________________________________________
        block4_conv1 (Conv2D)        (None, 32, 32, 512)       1180160   
        _________________________________________________________________
        block4_conv2 (Conv2D)        (None, 32, 32, 512)       2359808   
        _________________________________________________________________
        block4_conv3 (Conv2D)        (None, 32, 32, 512)       2359808   
        _________________________________________________________________
        block4_conv4 (Conv2D)        (None, 32, 32, 512)       2359808   
        _________________________________________________________________
        block4_pool (MaxPooling2D)   (None, 16, 16, 512)       0         
        _________________________________________________________________
        block5_conv1 (Conv2D)        (None, 16, 16, 512)       2359808   
        _________________________________________________________________
        block5_conv2 (Conv2D)        (None, 16, 16, 512)       2359808   
        _________________________________________________________________
        block5_conv3 (Conv2D)        (None, 16, 16, 512)       2359808   
        _________________________________________________________________
        block5_conv4 (Conv2D)        (None, 16, 16, 512)       2359808   
        _________________________________________________________________
        block5_pool (MaxPooling2D)   (None, 8, 8, 512)         0         
        =================================================================
        Total params: 20,024,384.0
        Trainable params: 20,024,384.0
        Non-trainable params: 0.0
        """
    # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in model.layers[:layers_freeze]:
        layer.trainable=False
    #Adding custom Layers 
    x = model.output
    x = Flatten()(x)
    # Output has 32768 neurons
    
    x = Dropout(0.5)(x)
    #x = Dense(4900, activation="relu")(x)
    x = Dense(9800, activation="relu")(x)

    #x = Dropout(0.5)(x)
    #x = Dense(100, activation="relu")(x)

    #x = Dropout(0.5)(x)
    #x = Dense(49, activation="relu")(x)

    #x = Reshape((7,7,1))(x)
    #x = Conv2DTranspose(512, 3, 3, activation='relu', border_mode='same', subsample=(2,2))(x)
    # Size is now 14x14
    #x = BatchNormalization()(x)
    
    #x = Dense(196, activation="relu")(x)
    #x = Reshape((14,14,25))(x)
    x = Reshape((14,14,50))(x)
    
    x = Conv2DTranspose(256, 3, 3, activation='relu', border_mode='same', subsample=(2,2))(x)
    # Size is now 28x28
    x = BatchNormalization()(x)
    x = Conv2DTranspose(128, 3, 3, activation='relu', border_mode='same', subsample=(2,2))(x)
    # Size is now 56x56
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, 3, 3, activation='relu', border_mode='same', subsample=(2,2))(x)
    # Size is now 112x112
    x = BatchNormalization()(x)
    output = Conv2DTranspose(3, 1, 1, activation='relu', border_mode='same', subsample=(1,1))(x)
    #output = Conv2D(3, 1, 1, activation='relu', border_mode='same', subsample=(1,1))(x)

    model=Model(input=model.input,output=output)

    print(model.summary()) # Prints model to use

    opt=optimizers.Adam(lr=learning_rate, beta_1=0.9)
    #opt = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error']) # Use for ML
    #model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9), metrics=[hamming_dist,'accuracy']) # Use for MC

    return model

    


# Function to plot learning curve
def plot_learning_curve(loss_train,loss_val):
    
    plt.title('Mean Squared Error')
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.xlabel('epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend(['train', 'validation'], loc='upper right')
    #plt.ylim([0.05,0.15])
    
    plt.savefig('Data/learningCurves.png')

