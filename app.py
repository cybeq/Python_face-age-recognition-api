import logging

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/image', methods=['POST'])
def detect_faces():
    # Get the image from the request
    img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)

    # Get the image dimensions and construct a blob from the image
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network to detect faces
    model.setInput(blob)
    detections = model.forward()

    # Log the number of faces detected
    num_faces = detections.shape[2]
    logging.info(f'Detected {num_faces} faces in image')

    # Draw green rectangles around the faces
    for i in range(num_faces):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x_end, y_end) = box.astype('int')
            cv2.rectangle(img, (x, y), (x_end, y_end), (0, 255, 0), 2)

    # Convert the processed image back to bytes and return it
    _, img_bytes = cv2.imencode('.png', img)
    return img_bytes.tobytes(), 200, {'Content-Type': 'image/png'}

if __name__ == '__main__':
    app.run(debug=True)
