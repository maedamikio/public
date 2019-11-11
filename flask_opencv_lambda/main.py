import time

import cv2
import numpy as np

from flask import Flask, make_response, render_template, request
app = Flask(__name__)


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        start = time.time()

        count = int(request.form.get("count", 1))
        image = request.files.get("image")
        data = image.read()

        for _ in range(count):
            img = np.fromstring(data, np.uint8)
            img = cv2.imdecode(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        data = cv2.imencode(".jpeg", img)[1].tostring()

        response = make_response()
        response.data = data
        response.mimetype = "image/jpeg"

        end = time.time()

        print("opencv version: {}, elapsed second: {}".format(cv2.__version__, end - start))

        return response

    return render_template("index.html")
