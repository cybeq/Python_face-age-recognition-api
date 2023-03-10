import logging

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
@app.route('/')
def home():
    return render_template('index.html')

AGE_BRACKETS = [
    "(0-4)", "(5-10)", "(11-14)", "(15-20)", "(20-28)",
    "(29-34)", "(35-45)", "(46-60)", "(61-100)"
]

age_net = cv2.dnn.readNetFromCaffe('age_deploy.protoxt.txt', 'age_net.caffemodel')
def predict_age(face):
   
    (h, w) = face.shape[:2]

 
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    preds = age_net.forward()

   
    i = np.argmax(preds)
    age_bracket = AGE_BRACKETS[i]


    return age_bracket

@app.route('/image', methods=['POST'])
def detect_faces():

    img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)

    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))


    model.setInput(blob)
    detections = model.forward()


    num_faces = detections.shape[2]
    logging.info(f'Detected {num_faces} faces in image')


    for i in range(num_faces):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x_end, y_end) = box.astype('int')
            face = img[y:y_end, x:x_end]
            age_bracket = predict_age(face)
            cv2.rectangle(img, (x, y), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(img, age_bracket, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    _, img_bytes = cv2.imencode('.png', img)
    return img_bytes.tobytes(), 200, {'Content-Type': 'image/png'}

if __name__ == '__main__':
      app.run(host='0.0.0.0')
