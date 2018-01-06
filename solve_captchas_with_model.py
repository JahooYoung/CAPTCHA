from keras.models import load_model
from imutils import paths
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from preprocessor import process_for_predict


MODEL_FILENAME = "keras_model/captcha_model.hdf5"
MODEL_LABELS_FILENAME = "keras_model/model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "data/train/"
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 24

# Load up the model labels
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(
    captcha_image_files, size=(10,), replace=False)

# loop over the image paths
for image_file in captcha_image_files:
    # Load the image and preprocess it
    image_data = process_for_predict(image_file)
    image_data = np.array(image_data, dtype="float")
    image_data = np.reshape(image_data, (-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1))

    # Make prediction
    prediction = model.predict(image_data)
    # Convert the one-hot-encoded prediction back to a normal letter
    letter = lb.inverse_transform(prediction)
    print('Prediction: ', end='')
    for i in letter:
        if i >= 10:
            print(chr(i + 55), end='')
        else:
            print(i, end='')
    print('')

    image = Image.open(image_file).convert('1')
    plt.imshow(np.array(image.getdata()).reshape(image.size[1], image.size[0]))
    plt.axis('off')
    plt.show()
