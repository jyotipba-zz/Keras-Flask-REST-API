import requests
import os

addr = 'http://localhost:5000'
test_url = os.path.join(addr, 'predict')
image_path ='cat.jpeg'
files = {'image': open(image_path, 'rb')}

# load the input image and construct the payload for the request
#image = open('outside_2.png', "rb").read()
#payload = {"image": image}

# send http request with image and receive response
response = requests.post(test_url, files=files)
#decode response
print(response.json())
