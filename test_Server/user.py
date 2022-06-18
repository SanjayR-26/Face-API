from flask import Flask, render_template, Response
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import pickle
import pymongo
from pymongo import MongoClient
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


app = Flask(__name__)
camera = cv2.VideoCapture(0)

def push(frame):
    import requests
    import numpy as np

    BASE = "http://127.0.0.1:8000/findFace"

    data = {
        "frame": frame
    }

    response = requests.post(BASE, json=data)
    print(response.content)

def frames():
    while True:
        success,frame = camera.read()
 
        imgS = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)    

        if facesCurFrame:
          push(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True, port=5001)