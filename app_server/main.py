"""
Model server script that polls Redis for images to classify
Adapted from https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
"""
import base64
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

import redis


def load_model():
    '''function to load model from disc'''

    global model
    # load model weight
    model = tf.keras.models.load_model('cat_vs_dog.h5')
    # This is very important
    #model.keras_model._make_predict_function()
    #return model

model = None
load_model()

# Connect to Redis server
db = redis.Redis(host=os.environ.get("REDIS_HOST"))

def base64_decode_image(img, dtype, shape):
    ''' deserializing image prior to passing them through  model.

    '''
    # If this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        img = bytes(img, encoding="utf-8")

    # Convert the string to a NumPy array using the supplied data
    # type and target shape
    img = np.frombuffer(base64.decodestring(img), dtype=dtype)
    img = img.reshape(shape)

    # Return the decoded image
    return img


def classify_process():
    '''this thread will continually poll for new images and then classify them
    '''

    while True:
        # Pop off multiple images from Redis queue atomically
        with db.pipeline() as pipe:
            pipe.lrange(os.environ.get("IMAGE_QUEUE"), 0, int(os.environ.get("BATCH_SIZE")) - 1)
            pipe.ltrim(os.environ.get("IMAGE_QUEUE"), int(os.environ.get("BATCH_SIZE")), -1)
            queue, _ = pipe.execute()

        imageIDs = []
        batch = None
        for q in queue:
            # Deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"],
                                        os.environ.get("IMAGE_DTYPE"),
                                        (1, int(os.environ.get("IMAGE_HEIGHT")),
                                         int(os.environ.get("IMAGE_WIDTH")),
                                         int(os.environ.get("IMAGE_CHANS")))
                                        )

            # Check to see if the batch list is None
            if batch is None:
                batch = image

            # Otherwise, stack the data
            else:
                batch = np.vstack([batch, image])

            # Update the list of image IDs
            imageIDs.append(q["id"])

        # Check to see if we need to process the batch
        if len(imageIDs) > 0:
            # Classify the batch
            print("* Batch size: {}".format(batch.shape))
            results = model.predict(batch)
            #results = imagenet_utils.decode_predictions(preds)

            # Loop over the image IDs and their corresponding set of results from our model
            for (imageID, result) in zip(imageIDs, results):
                # Initialize the list of output predictions
                output = []
                if result[0]>0.5:
                    r = {'label': 'dog'}
                    output.append(r)
                else:
                    r = {'label': 'cat'}
                    output.append(r)

                # Store the output predictions in the database, using image ID as the key so we can fetch the results
                db.set(imageID, json.dumps(output))

        # Sleep for a small amount
        time.sleep(float(os.environ.get("SERVER_SLEEP")))

if __name__ == "__main__":
    classify_process()
