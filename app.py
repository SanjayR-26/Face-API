from flask import Flask
from flask_restful import Api, Resource, reqparse
import werkzeug
import numpy as np
import face_recognition

app = Flask(__name__)
api = Api(app)

encodings = {
    1: {"name": "Sanjay", "faceData":"None" }
}

encode_post_args = reqparse.RequestParser()
encode_post_args.add_argument('img', type=np.ndarray(500,500, 3), required=True)

class Encode(Resource):
    def post(self):
        args = encode_post_args.parse_args()
        x = args["img"]
        # enc = face_recognition.face_encodings(x)[0]

        return x

api.add_resource(Encode, "/encode")

if __name__ == "__main__":
    app.run(debug=True)  