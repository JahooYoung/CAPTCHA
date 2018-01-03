import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.utils import plot_model


LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "keras_model/captcha_model.hdf5"
MODEL_LABELS_FILENAME = "keras_model/model_labels.dat"

DATA_TYPE = 'rotate_mellow_resize'
DATA_SIZE = 10000
IMAGE_WIDTH = 24
IMAGE_HEIGHT = 32


print('loading data ... ', end = '')
# initialize the data and labels
with open('data/_train_package_%s_%d' % (DATA_TYPE, DATA_SIZE), 'rb') as f:
    data = pickle.load(f)
with open('data/_train_ans_%s_%d' % (DATA_TYPE, DATA_SIZE), 'rb') as f:
    labels = pickle.load(f)

# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype = "float")
data = np.reshape(data, (-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1))
labels = np.array(labels)

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)
print('complete')


print('building network ... ', end = '')
# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(20, (5, 5), padding = "same", input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 1), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(50, (5, 5), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(512, activation = "relu"))

model.add(Dropout(0.5))

# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(36, activation = "softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

print('complete')

# plot_model(model, to_file = 'keras_model/model.png')
# exit()


# Train the neural network
model.fit(X_train, Y_train, validation_data = (X_test, Y_test), batch_size = 32, epochs = 6, verbose = 1)

# Save the trained model to disk
model.save(MODEL_FILENAME)
