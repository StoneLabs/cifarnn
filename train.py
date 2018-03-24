from cifarpy.cifar import cifar_ready, cifar_load_train, cifar_load_test, cifar_load_labels, cifar_download
from keras.layers import Input, Convolution2D, Dropout, MaxPool2D, Flatten, Dense, Dropout, Conv2D
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import os
os.chdir("cifarpy")

# If the data is not ready (i.e. not downloaded)
if not cifar_ready():
    cifar_download() # Download the data

# Load training/test data and label names
X_train, y_train = cifar_load_train()
X_test, y_test = cifar_load_test()
labels = cifar_load_labels()    

y_train = [[i==o for i in range(0, len(labels))] for o in y_train]
y_test = [[i==o for i in range(0, len(labels))] for o in y_test]


os.chdir("..")

# Print some numbers...
print("Loaded training data with shape %s and %i classes" % (X_train.shape, len(np.unique(y_train))))
print("Loaded test data with shape %s and %i classes" % (X_test.shape, len(np.unique(y_train))))

print("The labels are defined as following:")
for i in range(0, len(labels)):
    print(" - %i: %s" % (i, labels[i]))

X_train = X_train.reshape((X_train.shape[0], 32, 32, 3))
X_test = X_test.reshape((X_test.shape[0], 32, 32, 3))

# Build model
inp     = Input(shape=(32, 32, 3)) 

conv_1  = Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu')(inp)
pool_1  = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_1)

flat_1  = Flatten()(pool_1)
dnse_1  = Dense(25, activation='relu')(flat_1)
dnse_2  = Dense(25, activation='relu')(dnse_1)
out     = Dense(10, activation='softmax')(dnse_2)

print("Creating model...")
model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers
model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

plot_model(model, show_shapes=True, show_layer_names=False)

print("Training model...")
model.fit(X_train, y_train, # Train the model using the training set...
          epochs=100, verbose=1, batch_size=128,
          validation_split=0.1) # ...holding out 10% of the data for validation