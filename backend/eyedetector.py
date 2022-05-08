import dlib
import cv2
import numpy as np
import argparse

import glob

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


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


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

cap = cv2.VideoCapture('video.webm')


print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps

print(f"duration is {duration}, fps: {fps} and frame_count:{frame_count}")
cnt = 0

LST_GAZE = "Face Not Found"
while True:
    GAZE = "Face Not Found"

    ret, img = cap.read()
    if not ret:
        break

    faces = detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
    face3Dmodel = ref3DModel()

    for face in faces:
        shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)

        # draw(img,shape)

        refImgPts = ref2DImagePoints(shape)

        height, width, channels = img.shape

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
            print(GAZE, "starting time(sec):", cnt/fps,  "sec")

        LST_GAZE = GAZE
        # if GAZE != "Forward":
        #     print(GAZE, "time(sec):", cnt/fps,  "sec")
    cnt += 1
