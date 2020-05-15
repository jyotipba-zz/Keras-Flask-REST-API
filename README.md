# Keras-Flask-Gunicorn-REST-API
A simple REST API to serve Keras deep learning model.  API consists of one route to ask for a prediction.
Route parses the input from the request, calls the instantiated model on it and sends the output back to the user.
The deep learning model in this case involves classifying photos as either containing a dog or cat. This model was trained 
using transfer learning using VGG16 pre-trained model.
