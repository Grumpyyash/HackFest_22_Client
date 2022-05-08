from flask import Flask, request, jsonify
# from requests import request
from imutils import paths  # imutils includes opencv functions
import face_recognition
import pickle
import cv2
import os
from firebase_admin import credentials, firestore, initialize_app

import dlib
import cv2
import numpy as np
import argparse

import glob

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2

app = Flask(__name__)

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()
todo_ref = db.collection('todos')


# ====== CONSTANTS ======

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.30
EYE_AR_CONSEC_FRAMES = 45



# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# ========================

# to create the embeddings
def create_embedding(image_url):
    """
    pass the raw image while saving to the DB, save these encodings,
    later use them to recognize
    """

    im = face_recognition.load_image_file(image_url)
    embed = face_recognition.face_encodings(im)[0]
    return embed


def ref3DModel():
    modelPoints = [[0.0, 0.0, 0.0],
                   [0.0, -330.0, -65.0],
                   [-255.0, 170.0, -135.0],
                   [225.0, 170.0, -135.0],
                   [-150.0, -150.0, -125.0],
                   [150.0, -150.0, -125.0]]
    return np.array(modelPoints, dtype=np.float64)


def ref2DImagePoints(shape):
    imagePoints = [[shape.part(30).x, shape.part(30).y],
                   [shape.part(8).x, shape.part(8).y],
                   [shape.part(36).x, shape.part(36).y],
                   [shape.part(45).x, shape.part(45).y],
                   [shape.part(48).x, shape.part(48).y],
                   [shape.part(54).x, shape.part(54).y]]
    return np.array(imagePoints, dtype=np.float64)


def get_calibrateCameraMatrix():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0: 7, 0: 6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('data/images/*.jpg')

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx


def get_attention_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    return detector, predictor


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# ===== Attendance =====


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

# ====================


detector, predictor = get_attention_models()
mtx = get_calibrateCameraMatrix()


@app.route('/add', methods=['POST'])
def create():
    """
        create() : Add document to Firestore collection with request body
        Ensure you pass a custom ID as part of json body in post request
        e.g. json={'id': '1', 'title': 'Write a blog post'}
    """
    try:
        id = request.json['id']
        todo_ref.document(id).set(request.json)
        return jsonify({"success": True}), 200
    except Exception as e:
        return f"An Error Occured: {e}"


@app.route('/list', methods=['GET'])
def read():
    """
        read() : Fetches documents from Firestore collection as JSON
        todo : Return document that matches query ID
        all_todos : Return all documents
    """
    try:
        # Check if ID was passed to URL query
        todo_id = request.args.get('id')
        if todo_id:
            todo = todo_ref.document(todo_id).get()
            return jsonify(todo.to_dict()), 200
        else:
            all_todos = [doc.to_dict() for doc in todo_ref.stream()]
            return jsonify(all_todos), 200
    except Exception as e:
        return f"An Error Occured: {e}"


@app.route('/update', methods=['POST', 'PUT'])
def update():
    """
        update() : Update document in Firestore collection with request body
        Ensure you pass a custom ID as part of json body in post request
        e.g. json={'id': '1', 'title': 'Write a blog post today'}
    """
    try:
        id = request.json['id']
        todo_ref.document(id).update(request.json)
        return jsonify({"success": True}), 200
    except Exception as e:
        return f"An Error Occured: {e}"


@app.route('/delete', methods=['GET', 'DELETE'])
def delete():
    """
        delete() : Delete a document from Firestore collection
    """
    try:
        # Check for ID in URL query
        todo_id = request.args.get('id')
        todo_ref.document(todo_id).delete()
        return jsonify({"success": True}), 200
    except Exception as e:
        return f"An Error Occured: {e}"


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


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


@app.route("/facealignment", methods=["POST"])
def face_alignment():
    """
    Input:
        Link - class recording
    """

    try:
        print("url is", request.json['url'])
        video_url = request.json['url']
        cap = cv2.VideoCapture(video_url)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps

        print(f"Duration of the recording- {duration} sec")
        print(f"Total Frames count: {frame_count}")
        print(f"Frames per sec: {fps}")

        frame_cnt = 0
        store = []
        start_duration = 0

        LST_GAZE = "Face Not Found"
        while True:
            GAZE = "Face Not Found"

            ret, img = cap.read()

            # if no frame found
            if not ret:
                break

            faces = detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
            face3Dmodel = ref3DModel()

            for face in faces:
                shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)
                refImgPts = ref2DImagePoints(shape)
                mdists = np.zeros((4, 1), dtype=np.float64)

                success, rotationVector, translationVector = cv2.solvePnP(
                    face3Dmodel, refImgPts, mtx, mdists)

                noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
                noseEndPoint2D, jacobian = cv2.projectPoints(
                    noseEndPoints3D, rotationVector, translationVector, mtx, mdists)

                rmat, jac = cv2.Rodrigues(rotationVector)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                if angles[1] < -15:
                    GAZE = "Looking: Left"
                elif angles[1] > 15:
                    GAZE = "Looking: Right"
                else:
                    GAZE = "Forward"

                if GAZE != LST_GAZE:

                    if LST_GAZE != "Forward":
                        store.append(
                            [LST_GAZE, frame_cnt/fps - start_duration])

                    start_duration = frame_cnt/fps
                    # print(GAZE, "starting time(sec):", start_duration ,  "sec")
                LST_GAZE = GAZE

            frame_cnt += 1
        return jsonify({"drowsiness_alerts": store})
    except Exception as e:
        return f"An Error Occured: {e}"


@app.route("/drowsiness", methods=["POST"])
def get_drowsiness():

    try:
        print("url is", request.json['url'])
        video_url = request.json['url']
        cap = cv2.VideoCapture(video_url)

        # initialize the frame counters and the total number of blinks
        COUNTER = 0
        ALARM_ON = False

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps

        print(f"Duration of the recording- {duration} sec")
        print(f"Total Frames count: {frame_count}")
        print(f"Frames per sec: {fps}")

        store = []
        frame_cnt = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = imutils.resize(frame, width=500)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale image
            rects = detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        # if the alarm is not on, turn it on
                        if not ALARM_ON:
                            ALARM_ON = True

                            # check to see if an alarm file was supplied,
                            # and if so, start a thread to have the alarm
                            # sound played in the background

                            store.append(["Drowsiness", frame_cnt/fps])

                            # print("Drowsiness", "time(sec):", frame_cnt/fps,  "sec")

                    # otherwise, the eye aspect ratio is not below the blink
                    # threshold
                else:

                    # reset the eye frame counter
                    COUNTER = 0
                    ALARM_ON = False
            frame_cnt += 1

        return jsonify({"drowsiness_alerts": store})
    except Exception as e:
        return f"An Error Occured: {e}"


# Create arrays of known face encodings and their names


app.run(port=5000)
