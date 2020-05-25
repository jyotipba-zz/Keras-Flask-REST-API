"""
Web server script that exposes endpoints and pushes images to Redis for classification by model server.
Polls Redis for response from model server.
Adapted from https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
"""

import base64
import io
import json
import os
import time
import uuid

# flask
from flask import Flask, jsonify, request

# tensorflow
import tensorflow as tf
import numpy as np
from PIL import Image

import redis

# initializing Flask application, Redis server
app = Flask(__name__)
db = redis.Redis(host=os.environ.get("REDIS_HOST"))

def preprocess_image(image, target_size):
    # preprocessing
    img = image.resize(size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # Return the processed image
    return img


@app.route("/")
def index():
    return "Hello, this is  Keras flask rest API"


@app.route("/predict", methods=['POST'])
def predict():
    data = {"success": False}

    if request.method == "POST":
        if request.files.get('image'):
            # read the image in PIL format
            img = request.files['image'].read()
            image = Image.open(io.BytesIO(img))
            image = preprocess_image(image,(150, 150))

            # Ensure our NumPy array is C-contiguous as well, otherwise we won't be able to serialize it
            image = image.copy(order="C")

            # Generate an ID for the classification then add the classification ID + image to the queue
            k = str(uuid.uuid4())
            image = base64.b64encode(image).decode("utf-8")
            d = {"id": k, "image": image}
            db.rpush(os.environ.get("IMAGE_QUEUE"), json.dumps(d))

            # Keep looping for CLIENT_MAX_TRIES times
            num_tries = 0
            while num_tries < int(os.environ.get('CLIENT_MAX_TRIES')):
                num_tries += 1

                # Attempt to grab the output predictions
                output = db.get(k)

                # Check to see if our model has classified the input image
                if output is not None:
                    # Add the output predictions to our data dictionary so we can return it to the client
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)

                    # Delete the result from the database and break from the polling loop
                    db.delete(k)
                    break

                # Sleep for a small amount to give the model a chance to classify the input image
                time.sleep(float(os.environ.get('CLIENT_SLEEP')))

                # indicate that the request was a success
                data["success"] = True
    # return the data dictionary as a JSON response
    return jsonify(data)
if __name__ == "__main__":
    app.run()
