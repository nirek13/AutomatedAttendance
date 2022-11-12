
import sys
from cv2 import imread
import numpy as np
import cv2
import qrcode
import face_recognition 
import face_recognition as fr

import os

### FACIAL RECOGNITION

MODE = "QR" # FACIAL or QR


def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces
    :return: dict of (name, image encoded)
    """
    encoded = {}

            
    for dirpath, dnames, fnames in os.walk("./FacialAttendance"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("FacialAttendance/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("FacialAttendance/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding

def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are
    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
    while True:

        cv2.imshow('Video', img)
        if not cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names 

### QR CODE READER

def QRReader(img):
    im = cv2.imread(img)
    det = cv2.QRCodeDetector()
    retval, points, straight_qrcode = det.detectAndDecode(im)
    return retval

def get_people(mode = MODE):
    """
    looks through the faces folder and qr folder  and encodes all
    the faces + qr codes
    :return: dict of (name, image encoded)
    """

    if mode == "QR":
        
        encoded = {}
    
        for dirpath, dnames, fnames in os.walk("./QRAttendance"):
            for f in fnames:
                if f.endswith(".jpg") or f.endswith(".png"):
                    face = fr.load_image_file("QRAttendance/" + f)
                    encoding = fr.face_encodings(face)[0]
                    encoded[f.split(".")[0]] = encoding

            return encoded
    else :
        
        encoded = {}
        
        for dirpath, dnames, fnames in os.walk("./FacialAttendance"):
            for f in fnames:
                if f.endswith(".jpg") or f.endswith(".png"):
                    face = fr.load_image_file("FacialAttendance/" + f)
                    encoding = fr.face_encodings(face)[0]
                    encoded[f.split(".")[0]] = encoding

        return encoded

def QRCreator(classList):
    """Takes in an input of people seperated by commas"""

    classNames = []
    qrcodes = []
    y = 0
    print(len(classList))

    for i in range(len(classList)):
        
        if classList[i] != ",":
            classNames.append([])
            classNames[y] += classList[i]
        else:

            y += 1

    for name in classList:
        qrcodes.append(qrcode.make(name))
        img = qrcode.make(name)
        img.save(f"QRAttendance/{name}.png" , "PNG")
        os.system(f"open QRAttendance/{name}.png", "PNG" )


   
while True: 

    if MODE == "QR":
        if get_people(MODE) == {}:
            print("No QR codes found")
            print("Please enter the names of the people in the class seperated by commas")
            classList = input()
            QRCreator(classList)
        else:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read() 
            cv2.imwrite("cur.jpg", frame)
            person = QRReader("cur.jpg")
            print(person)
    else:
        if get_people("FACIAL") == {}:
            print("No faces found")
            print("Please enter the names of the people in the class seperated by commas")
            classList = input()
            QRCreator(classList)
        else:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read() 
            cv2.imwrite("cur.jpg", frame)
            person = classify_face("cur.jpg")
            print(person)

            





print(people)
cap.release()
cv2.destroyAllWindows()
