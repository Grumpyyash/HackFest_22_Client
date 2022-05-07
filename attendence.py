from flask import Flask
from imutils import paths  # imutils includes opencv functions
import face_recognition
import pickle
import cv2
import os


# #get paths of each file in folder named Images
# #Images here that contains data(folders of various people)
# imagePath = list(paths.list_images('Images'))
# kEncodings = []
# kNames = []

# # Create a woorksheet
# book=Workbook()
# sheet=book.active

# Load images.

image_1 = face_recognition.load_image_file("./Images/Elly.jpg")
image_1_face_encoding = face_recognition.face_encodings(image_1)[0]

image_2 = face_recognition.load_image_file("./Images/Aayush.jpg")
image_2_face_encoding = face_recognition.face_encodings(image_2)[0]

image_4 = face_recognition.load_image_file("./Images/Abhishek.jpg")
image_4_face_encoding = face_recognition.face_encodings(image_4)[0]

image_5 = face_recognition.load_image_file("./Images/rolli.jpg")
image_5_face_encoding = face_recognition.face_encodings(image_5)[0]

image_7 = face_recognition.load_image_file("./Images/Rishabh.jpg")
image_7_face_encoding = face_recognition.face_encodings(image_7)[0]

image_3 = face_recognition.load_image_file("./Images/Yash.jpg")
image_3_face_encoding = face_recognition.face_encodings(image_3)[0]


# Create arrays of known face encodings and their names
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

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# Grab a single frame of video
frame = cv2.imread("./Images/input.png")

# # Resize frame of video to 1/4 size for faster face recognition processing
# small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Find all the faces and face encodings in the current frame of video
face_locations = face_recognition.face_locations(rgb_small_frame)
face_encodings = face_recognition.face_encodings(
    rgb_small_frame, face_locations)

face_names = []
for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(
        known_face_encodings, face_encoding, 0.56)
    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    print(matches)
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
        print(name)
        #    # Assign attendance
        #    if int(name) in range(1, 61):
        #         sheet.cell(row=int(name), column=int(
        #             today)).value = "Present"
        #     else:
        #         pass
    face_names.append(name)

# Display the results
print(face_names, face_locations)
for (top, right, bottom, left), name in zip(face_locations, face_names):

    print(top, right, bottom, left, name)

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 5),
                  (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom + 10),
                font, 0.4, (255, 255, 255), 1)

cv2.imwrite(f"output.jpg", frame)
# Display the resulting image
# cv2.imshow('Video', frame)

# # Save Woorksheet as present month
# book.save(str(month)+'.xlsx')


app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


# app.run(port=5000)
