import os
from flask import Flask,request
import cv2
import numpy as np
from ultralytics import YOLO
import requests

app=Flask(__name__)


@app.route('/api')
def get():
    # str(request.args['Male'])
    # url = 'https://i0.wp.com/theperfectcurry.com/wp-content/uploads/2021/09/PXL_20210830_005055409.PORTRAIT.jpg'
    url=request.args['url']
    # Fetch the image from the URL

    response = requests.get(url)
    save_folder = 'images'  # Adjust the save folder path as needed
    save_name = 'testimage.jpg'
    if response.status_code == 200:
        # Create the save folder if it does not exist
        os.makedirs(save_folder, exist_ok=True)

        # Determine the image file name
        if save_name is None:
            save_name = os.path.basename(url)

        # Construct the full save path
        save_path = os.path.join(save_folder, save_name)

        # Write the image content to the file
        with open(save_path, 'wb') as file:
            file.write(response.content)

    # image_data = response.content

    # Convert the image data to a format OpenCV can read
    # image = Image.open(BytesIO(image_data))
    # cap = np.array(image)
    cap = cv2.imread("images/testimage.jpg")
    model = YOLO("250ps.pt")

    results = model(cap)
    result = results[0]
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    return str(classes)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('love')
    app.run()
