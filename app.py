from __future__ import division, print_function

import os
import numpy as np
from PIL import Image as pil_image

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.layers import GlobalAveragePooling2D,Dropout,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import efficientnet.tfkeras as efn

app = Flask(__name__)

def model_predict(img_path):

    lesion_classes_dict = {
        0: 'Melanocytic nevi',
        1: 'Melanoma',
        2: 'Benign keratosis-like lesions ',
        3: 'Basal cell carcinoma',
        4: 'Actinic keratoses',
        5: 'Vascular lesions',
        6: 'Dermatofibroma'
    }

    model = efn.EfficientNetB3(weights='noisy-student', include_top=False,
                               input_shape=(90, 120, 3))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    predictions = Dense(7, activation="softmax")(x)
    model = Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=['accuracy'])

    model.save_weights("medel.h5")

    # RESIZING THE IMAGE
    resized_image = np.asarray(pil_image.open(img_path).resize((120, 90)))
    image_array = np.asarray(resized_image.tolist())
    test_image = image_array.reshape(1, 90, 120, 3)

    prediction_class = model.predict(test_image)
    prediction_class = np.argmax(prediction_class, axis=1)

    return lesion_classes_dict[prediction_class[0]]

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        return model_predict(file_path)

    return None


if __name__ == '__main__':
    app.run(debug=True)

