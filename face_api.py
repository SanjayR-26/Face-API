import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Response
from fastapi.params import Body
import face_recognition
import cv2
import numpy as np
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
from pymongo import MongoClient
import datetime
import base64
import io
from starlette.responses import StreamingResponse
from PIL import Image,ImageDraw


app = FastAPI()
# cluster = MongoClient("mongodb+srv://testface:testface@cluster0.xzclj.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")


class EncodeImageData(BaseModel):
    faceName : str
    stringimg : Optional[str] = None 
    arrImg : Optional[list] = None 

class findLocImgData(BaseModel):
    stringimg : Optional[str] = None 
    arrImg : Optional[list] = None

class findfaceland(BaseModel):
    stringimg : Optional[str] = None 
    arrImg : Optional[list] = None


def findEncodings(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(img)
    encodeList = face_recognition.face_encodings(img, facesCurFrame)[0]
    return encodeList

def faceLocations(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(img)
    return facesCurFrame

def faceLandmarks(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    facialLandmarks = face_recognition.face_landmarks(img)
    return facialLandmarks


@app.post("/encode/image/")
async def Encodeimage(image: UploadFile = File(...),faceName: str = Form(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    encodelist= findEncodings(img)
    listEncode = encodelist.tolist()
    return JSONResponse({"Name":faceName, "Encodings": listEncode})

@app.post("/encode/imgdata/")
async def EncodeImgData(datastring:EncodeImageData):
    if datastring.stringimg is not None:
        base64_img_bytes = datastring.stringimg.encode('utf-8')
        base64bytes = base64.b64decode(base64_img_bytes)
        nparr = np.fromstring(base64bytes, dtype="uint8")
        strimg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        stringencodelist = findEncodings(strimg)
    if datastring.arrImg is not None:
        nparr = np.asarray(datastring.arrImg, dtype=np.uint8, order=None)
        stringencodelist = findEncodings(nparr)
    elif datastring.stringimg is None and datastring.arrImg is None:
        return {"Error": "Image data not given"}

    listEncodestring = stringencodelist.tolist()
    return JSONResponse({"Name":datastring.faceName, "Encodings": listEncodestring})
    
@app.post("/findface/image/")
async def findFaceLocation(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    points = faceLocations(img)
    for i in range(len(points)):
        top, right, bottom, left  = points[i][0],  points[i][1],  points[i][2],  points[i][3]
        cv2.rectangle(img,(left, top),(right, bottom),(0,255,0),2)
    res, im_png = cv2.imencode(".png", img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

@app.post("/findface/imgdata/")
async def findFaceLocationimgdata(dataimg:findLocImgData):
    if dataimg.stringimg is not None:
        base64_img_bytes = dataimg.stringimg.encode('utf-8')
        base64bytes = base64.b64decode(base64_img_bytes)
        img = np.fromstring(base64bytes, dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        points = faceLocations(img)
        top, right, bottom, left  = points[0][0],  points[0][1],  points[0][2],  points[0][3]
        cv2.rectangle(img,(left, top),(right, bottom),(0,255,0),2)
        res, im_png = cv2.imencode(".png", img)
    if dataimg.arrImg is not None:
        img = np.asarray(dataimg.arrImg, dtype=np.uint8, order=None)
        points = faceLocations(img)
        top, right, bottom, left  = points[0][0],  points[0][1],  points[0][2],  points[0][3]
        cv2.rectangle(img,(left, top),(right, bottom),(0,255,0),2)
        res, im_png = cv2.imencode(".png", img)
    elif dataimg.stringimg is None and dataimg.arrImg is None:
        return {"Error": "Image data not given"}
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

@app.post("/encode/image/compare")
async def Encodeimage(image1: UploadFile = File(...),image2: UploadFile = File(...)):
    img1 = await image1.read()
    img2 = await image2.read()
    nparr1 = np.fromstring(img1, np.uint8)
    nparr2 = np.fromstring(img2, np.uint8)
    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
    encodelist1 = findEncodings(img1)
    encodelist2 = findEncodings(img2)
    matches =  face_recognition.face_distance([encodelist1],encodelist2)
    matchIndex = np.argmin(matches)
    if matches[matchIndex]< 0.50:
        return True
    return False

@app.post("/findlandmarks/image")
async def findLandmarks(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    facialLandmarks= faceLandmarks(img)
    return facialLandmarks

@app.post("/findlandmarks/imgdata")
async def EncodeImgData(datalandmarks:findfaceland):
    if datalandmarks.stringimg is not None:
        base64_img_bytes = datalandmarks.stringimg.encode('utf-8')
        base64bytes = base64.b64decode(base64_img_bytes)
        nparr = np.fromstring(base64bytes, dtype="uint8")
        strimg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        facialLandmarks = faceLandmarks(strimg)
    if datalandmarks.arrImg is not None:
        nparr = np.asarray(datalandmarks.arrImg, dtype=np.uint8, order=None)
        facialLandmarks = faceLandmarks(nparr)
    elif datalandmarks.stringimg is None and datalandmarks.arrImg is None:
        return {"Error": "Image data not given"}
    return facialLandmarks

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)


















