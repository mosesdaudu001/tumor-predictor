from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adamax
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
import numpy as np
import os

img_size = (224, 224)
channels = 3
color_mode = 'rgb'
img_shape = (img_size[0], img_size[1], channels)

def create_model():
    base_model = tf.keras.applications.efficientnet.EfficientNetB5(include_top= False,
                                                                weights= "imagenet",
                                                                input_shape= img_shape,
                                                                pooling= 'max')

    model = Sequential([
        base_model,
        BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
        Dense(256,
            kernel_regularizer= regularizers.l2(l= 0.016),
            activity_regularizer= regularizers.l1(0.006),
            bias_regularizer= regularizers.l1(0.006),
            activation= 'relu'),

        Dropout(rate= 0.45,
                seed= 123),

        Dense(44, activation= 'softmax')
    ])

    model.load_weights("model.h5")
    model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

    return model

class_list = [
    ['Astrocitoma T1', 0],
    ['Astrocitoma T1C+', 1],
    ['Astrocitoma T2', 2],
    ['Carcinoma T1', 3],
    ['Carcinoma T1C+', 4],
    ['Carcinoma T2', 5],
    ['Ependimoma T1', 6],
    ['Ependimoma T1C+', 7],
    ['Ependimoma T2', 8],
    ['Ganglioglioma T1', 9],
    ['Ganglioglioma T1C+', 10],
    ['Ganglioglioma T2', 11],
    ['Germinoma T1', 12],
    ['Germinoma T1C+', 13],
    ['Germinoma T2', 14],
    ['Glioblastoma T1', 15],
    ['Glioblastoma T1C+', 16],
    ['Glioblastoma T2', 17],
    ['Granuloma T1', 18],
    ['Granuloma T1C+', 19],
    ['Granuloma T2', 20],
    ['Meduloblastoma T1', 21],
    ['Meduloblastoma T1C+', 22],
    ['Meduloblastoma T2', 23],
    ['Meningioma T1', 24],
    ['Meningioma T1C+', 25],
    ['Meningioma T2', 26],
    ['Neurocitoma T1', 27],
    ['Neurocitoma T1C+', 28],
    ['Neurocitoma T2', 29],
    ['Oligodendroglioma T1', 30],
    ['Oligodendroglioma T1C+', 31],
    ['Oligodendroglioma T2', 32],
    ['Papiloma T1', 33],
    ['Papiloma T1C+', 34],
    ['Papiloma T2', 35],
    ['Schwannoma T1', 36],
    ['Schwannoma T1C+', 37],
    ['Schwannoma T2', 38],
    ['Tuberculoma T1', 39],
    ['Tuberculoma T1C+', 40],
    ['Tuberculoma T2', 41],
    ['_NORMAL T1', 42],
    ['_NORMAL T2', 43]
    ]


# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=img_size, color_mode=color_mode)
    image_array = img_to_array(img)
    preprocessed_image = preprocess_input(image_array)
    x = np.expand_dims(preprocessed_image, axis=0)

    return x

def predict(img):
    model = create_model()
    class_prediction=model.predict(img) 
    classes_x=np.argmax(class_prediction)
    pred = class_list[classes_x][0]

    return str(pred)
