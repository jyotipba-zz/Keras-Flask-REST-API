# Flask
from flask import Flask, jsonify, request

# utilities
import numpy as np
from PIL import Image
from io import BytesIO

# tensorflow
import tensorflow as tf

def load_model():
    '''function to load model from disc'''

    global model
    # load model weight
    model = tf.keras.models.load_model('cat_vs_dog.h5')
    # This is very important
    #model.keras_model._make_predict_function()
    #return model

# Declare a flask app
app = Flask(__name__)
model = None
#model = load_model()

@app.route("/predict", methods=["POST"])
def predict():

    # initialize the data dictionary that will be returned from the  view
    data = {'success': False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == 'POST':
        if request.files.get('image'):
            # read the image in PIL format
            img = request.files['image'].read()
            img = Image.open(BytesIO(img))

            # preprocessing
            img = img.resize(size=(150,150))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            # make prediction
            prediction_result = model.predict(img, batch_size=1)
            data["predictions"] = []

            if prediction_result [0]>0.5:
                data['predictions'] = ['dog']
            else:
                data['predictions'] = ['cat']
            # indicate that the request was a success
            data['success'] = True
        # return the data dictionary as a JSON response
    return jsonify(data)

if __name__ == '__main__':
    print("Loading Keras model and Flask starting server... please wait until server has fully started")
    load_model()
    app.run(host='0.0.0.0', debug=True)
