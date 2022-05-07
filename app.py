from flask import Flask
from imutils import paths  # imutils includes opencv functions
import face_recognition
import pickle
import cv2
import os

app = Flask(__name__)


# to create the embeddings
def create_embedding(image_url):
    """
    pass the raw image while saving to the DB, save these encodings,
    later use them to recognize
    """

    im = face_recognition.load_image_file(image_url)
    embed = face_recognition.face_encodings(im)[0]
    return embed


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# Create arrays of known face encodings and their names


@app.route("/attendance")
def mark_attendance():
    """ ####### TO DO #########
    1) get all the embeddings  #array
    2) get all the ids [in the same order] #array

    eg - 

    known_face_encodings = [

        image_1_face_encoding,
        image_2_face_encoding,
        image_5_face_encoding,
        image_4_face_encoding,
        image_7_face_encoding,
        image_3_face_encoding,

    ]
    known_face_names = [
        "Elly",
        "Aayush",
        "rolli",
        "Abhishek",
        "Rishabh",
        "Yash"
    ]

    """

    known_encodings = None
    known_ids = None

    frame = cv2.imread("./Images/input.png")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations, face_ids_present = get_present_ids(
        known_encodings, known_ids, frame)
    frame = get_mapped_ss(frame, face_locations, face_ids_present)

    return face_ids_present


def get_present_ids(known_encodings, known_ids, frame):
    """
    encodings - complete database encoding
    ids - mapping to corresponding encodings
    image - screenshot to mark the attendence
    """

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_ids_present = []  # store present students
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, 0.56)
        id = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            id = known_ids[first_match_index]

        face_ids_present.append(id)

    return face_locations, face_ids_present


def get_mapped_ss(frame, face_locations, face_ids_present):
    for (top, right, bottom, left), name in zip(face_locations, face_ids_present):

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 5),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom + 10),
                    font, 0.4, (255, 255, 255), 1)

    cv2.imwrite(f"output.jpg", frame)
    return frame


app.run(port=5000)
