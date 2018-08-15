import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from multiprocessing.dummy import Pool
import glob
import cv2 as cv
import os

from keras import backend as K
K.set_image_data_format('channels_first')

from fr_utils import *
from inception_blocks_v2 import *

PADDING = 50
ready_to_detect_identity = True

# converts an image to its encoding
def img_path_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    return img_to_encoding(img1, model)

# iterates through every image in the images folder and retrieves the name of the image.
# I use this name as the key in our database dictionary, which stores the encoding of that image. 
def prepare_database():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, FRmodel)

    print(database.keys())

    return database

# A function to calculate the triplet loss.
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula.

    anchor -- the encodings for the anchor images, of shape (None, 128)
    positive -- the encodings for the positive images, of shape (None, 128)
    negative -- the encodings for the negative images, of shape (None, 128)
    
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist) ,alpha)

    loss = tf.maximum(tf.reduce_mean(basic_loss), 0.0)

    return loss

# function to recognize the face of a person using the webcam in OpenCV.
def webcam_face_recognizer(database):
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.

    """
    global ready_to_detect_identity

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # haar cascade model.
    
    # loop runs until the exc. key is pressed.
    while vc.isOpened():
        _, frame = vc.read()
        img = frame

        # We do not want to detect a new identity while the program is in the process of identifying another person
        if ready_to_detect_identity:
            img = process_frame(img, frame, face_cascade)     
        
        key = cv2.waitKey(100)
        cv2.imshow("preview", img)

        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")

# function to process a given frame.
def process_frame(img, frame, face_cascade):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    global ready_to_detect_identity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING

        img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)

        identity = find_identity(frame, x1, y1, x2, y2) # finds the person's identity

        # adding the person's name to the top left corner of our bounding box.
        img = cv2.putText(frame, identity, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

        if identity is not None:
            identities.append(identity)

        if identities != []:
            cv2.imwrite('example.png',img)

        ready_to_detect_identity = False

    return img


def find_identity(frame, x1, y1, x2, y2):
    """
    Determine whether the face contained within the bounding box exists in our database

    """
    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    
    return recognize(part_image, database, FRmodel)


def recognize(image, database, model):

    encoding = img_to_encoding(image, model)
    
    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist <= min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.52:
        return None
    else:
        print(str(identity))    
        return str(identity)        

# driver program. 
if __name__ == '__main__':


    FRmodel = faceRecoModel(input_shape=(3, 96, 96)) # inception model

    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    
    # loading pre-trained weights.
    load_weights_from_FaceNet(FRmodel) 

    # get a list of images in database.
    database = prepare_database()
    # recognize the face.
    webcam_face_recognizer(database)
    


