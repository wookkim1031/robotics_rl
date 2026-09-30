"""
- convolution function (for hidden layers)
- non-linear transformation function (for hidden layers)
- pooling function (for hidden layers)
- linear function (for fully-connected layer)
- euclidean radial basis (RBF) function (for the output)
- cost function (to compute overall error)
- learning procedure (i.e., backpropagation)
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, AveragePooling2D
from keras import backend as K

import matplotlib.pyplot as plt
import altair as alt

import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data() # unpack data into training and test
print(f'x_train shape:{x_train.shape}')
print(f'x_test shape:{x_test.shape}')

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))


img_rows, img_cols, channels = 28, 28, 1 # weight, height, colors

# Reshaping since Keras expects (num-sample, width, height, RGB-channels)
if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else: 
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)

# Normalization
# Grayscale values range from 0 (white) to 255 (black). Neural Networks tend to converge faster when data is normali
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 
x_test /= 255

# One-hot encoding target vector
# Move from binary to multi-class classification,
# for example 5 into [0 0 0 0 0 1 0 0 0 0]
print(f'y_train shape *before* one-hot encoding: {y_train.shape}')
print(f'y_test shape *before* one-hot encoding: {y_test.shape}')

num_classes = 10 # Numbers 0-9 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(f'y_train shape *after* one-hot encoding: {y_train.shape}')
print(f'y_test shape *after* one-hot encoding: {y_test.shape}')

# Define a network as a linear stack of layers
model = Sequential() 

# Add 1st convolutional layer with:
    # - features maps: 6
    # - kernel shape: 5 x 5 
    # - activation function post-convolution: hyperbolic tanget (tanh)
model.add(Conv2D(filters=6,kernel_size=(5,5),
                activation='tanh',
                input_shape=input_shape))

# Add 1st pooling layer with kernel shape: 2 x 2 
model.add(AveragePooling2D(pool_size=(2, 2)))

# Add 2st convolutional layer with:
    # - features maps: 16
    # - kernel shape: 5 x 5 
    # - activation function post-convolution: hyperbolic tanget (tanh)
model.add(Conv2D(filters=16,kernel_size=(5,5),
                activation='tanh'))

# Add 2st pooling layer with kernel shape: 2 x 2 
model.add(AveragePooling2D(pool_size=(2, 2)))

# Flatten the feature maps           
model.add(Flatten())

# Add 1st fully-connected layer with sigmoid activation function
model.add(Dense(120, activation='sigmoid'))

# Add 2st fully-connected layer with sigmoid activation function
model.add(Dense(84, activation='sigmoid'))

# Add output layer with softmax activation with 10 output classes         
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# LeNet-5 Application: MNIST classification
# Compile model with:
    # - cost function: categorical cross-entropy
    # - optimizer: Adadelta (variation of backpropagation)
    # - metrics recorded: accuracy
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

epochs = 12 # number of passes of the entire dataset
batch_size = 128 # model iterations before a gradient upgrade

history =model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_data=(x_test, y_test))

loss = history.history['loss']
accuracy = history.history['accuracy']

df2 = pd.DataFrame({"accuracy":accuracy, "loss":loss, "time-step": np.arange(0, len(accuracy))})

base = alt.Chart(df2).mark_line(color="blue").encode(x="time-step", y="accuracy")
loss = alt.Chart(df2).mark_line(color="red").encode(x="time-step", y="loss")
(base  + loss).properties(title='Chart 4')

score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss score: {score[0]}')
print(f'Test accuracy score:{ score[1]}')