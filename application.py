from flask import Flask, render_template, request
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
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=50,
                            kernel_size=3, # can also be (3, 3)
                            activation="relu",
                            input_shape=(224, 224, 3)), # first layer specifies input shape (height, width, colour channels)
    tf.keras.layers.Conv2D(50, 3, activation="relu"),
    tf.keras.layers.Conv2D(50, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2, # pool_size can also be (2, 2)
                                padding="valid"), # padding can also be 'same'
    tf.keras.layers.Conv2D(50, 3, activation="relu"),
    tf.keras.layers.Conv2D(50, 3, activation="relu"),
    tf.keras.layers.Conv2D(50, 3, activation="relu"), # activation='relu' == tf.keras.layers.Activations(tf.nn.relu)
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Conv2D(50, 3, activation="relu"),
    tf.keras.layers.Conv2D(50, 3, activation="relu"),
    tf.keras.layers.Conv2D(50, 3, activation="relu"), # activation='relu' == tf.keras.layers.Activations(tf.nn.relu)
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(44, activation="softmax") #
    ])

    model.load_weights("model_weights_2.h5")
    model.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

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


app = Flask(__name__)

target_img = os.path.join(os.getcwd() , 'static/images')

@app.route('/')
def index_view():
    return render_template('index.html')

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXT

# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=img_size, color_mode=color_mode)
    image_array = img_to_array(img)
    preprocessed_image = preprocess_input(image_array)
    x = np.expand_dims(preprocessed_image, axis=0)

    return x

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path) #prepressing method

            model = create_model()
            class_prediction=model.predict(img) 
            classes_x=np.argmax(class_prediction)
            pred = class_list[classes_x][0]

            return render_template('predict.html', fruit = pred,prob=class_prediction, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"


if __name__ == '__main__':
    app.run(debug=False,use_reloader=False, port=8000)
