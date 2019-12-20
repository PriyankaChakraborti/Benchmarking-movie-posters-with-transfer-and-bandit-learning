import matplotlib.pyplot as plt
from keras.models  import load_model, Model
from keras import optimizers
import numpy as np

# Choose index
des_layer = -9
learning_rate = 10**(-7)
chsn_index = 20

# Load model
model=load_model('Data/vgg16_1.h5')

# Load data
img_shape = (112,112,3)
num_imgs = 68054 # For 6 genre dataset
data = np.memmap('Data/data_6.dat', dtype='float32', mode='r', shape=(num_imgs,img_shape[0],img_shape[1],img_shape[2]))

# Create original image
orig_img = data[chsn_index]

# Create predicted image
pred_img = np.squeeze(model.predict(np.expand_dims(orig_img, axis=0)))

model = Model(model.inputs, model.layers[des_layer].output)
# Compile model
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9), metrics=['binary_crossentropy'])

code_layer = model.predict(np.expand_dims(orig_img, axis=0))
print(code_layer.shape)

# Create figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# Display the original Image
ax1.set_title('Original Image')
ax1.imshow(orig_img)

# Display the predicted image
ax2.set_title('Predicted Image')
ax2.imshow(pred_img)

ax3.set_title('Code Layer Output')
ax3.scatter(list(range(code_layer.shape[1])), code_layer)

fig.set_size_inches(20, 10)
plt.savefig('Data/autoencoder_performance.png')


print('')
print('All plots saved!')