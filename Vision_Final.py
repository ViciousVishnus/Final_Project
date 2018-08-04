

from flask import Flask
from flask_ask import session, Ask, statement, question, request
app= Flask(__name__)
ask = Ask(app, '/')

@app.route('/')
def homepage():
    return "Image Recognition"

import pickle

@ask.launch
def start_skill():
    welcome_message= 'Hello, could I take a picture so I can recognize you? If you are not in our database just say store me to be added'
    return question(welcome_message)
from camera import take_picture
import numpy as np
from dlib_models import download_model, download_predictor, load_dlib_models
from dlib_models import models
import matplotlib.pyplot as plt

def euclidean_distance(arr1,arr2):
    """
    Returns the euclidean distance between two matrices.
    """
    import numpy as np
    distance = np.sqrt(np.sum((arr1 - arr2)**2))
    return distance


def pickle_the_pickle(faces_dictionary):
    """
    Takes in a dictionary of face labels and associated descriptors and populates a pickle.

    Parameters:
    faces_dictionary: a dictionary where key, value is face label, associated descriptors
    """
    with open("faces.pkl", mode="wb") as opened_file:
        pickle.dump(faces_dictionary, opened_file)

def import_pickle():
    """
    Returns:
    faces: a dictionary with key, value of face and associated descriptors.
    """
    faces={}
    with open("faces.pkl", mode="rb") as opened_file:
        faces=pickle.load(opened_file)
    return faces

@ask.intent("YesIntent")
def final_func():
    img_array = take_picture()
    # load the models that dlib has to detect faces.
    load_dlib_models()
    face_detect2 = models["face detect"]
    face_rec_model2 = models["face rec"]
    shape_predictor2 = models["shape predict"]

    # Take in the (H,W,3) img_array and return the number of face detections in the photo
    detections = list(face_detect2(img_array))

    # for each detected face, create a descriptor
    descriptors = []
    for image in range(len(detections)):
        shape = shape_predictor2(img_array, detections[image])
        descriptor = np.array(face_rec_model2.compute_face_descriptor(img_array, shape))
        descriptors.append(descriptor)
    if type(detections) != list:
        detections = list(detections)
    if len(detections) == 1:
        rect = detections[0]
        fig, ax = plt.subplots()
        x1, x2, y1, y2 = rect.left(), rect.right(), rect.top(), rect.bottom()
        ax.imshow(img_array[y1:y2, x1:x2, :])


    else:
        fig, ax = plt.subplots(ncols=len(detections))
        # https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly
        for i, rect in enumerate(detections):
            x1, x2, y1, y2 = rect.left(), rect.right(), rect.top(), rect.bottom()
            ax[i].imshow(img_array[y1:y2, x1:x2, :])
            ax[i].axis('off')
    names=[]
    for descriptor in descriptors:
        with open("faces.pkl", mode="rb") as opened_file:
            faces = pickle.load(opened_file)
        distances = []

        for name in enumerate(faces.keys()):
            mean = np.mean(faces[name[1]], axis=0)  # takes element wise mean
            distances.append(euclidean_distance(mean, descriptor))

        # populate a list with probable names within the threshold

        threshold = .4
        probable_faces=""
        for i, distance in enumerate(distances):
            if distance < threshold:
                probable_faces=str((list(faces.keys()))[i])

        # todo: figure out the threshold value. will we be able to return just one?
        if probable_faces is "":
            probable_faces="you are not in our database"
        names.append(probable_faces)
    names_string=" and ".join(names)
    if descriptors is []:
        names_string="We don't see you"
    return statement(names_string)
@ask.intent('StoringIntent')
def name():
    names="What is your name? say my name is then your name"
    return question(names)
@ask.intent('NameIntent')
def storeintodatabase(name):
    img_array = take_picture()
    # load the models that dlib has to detect faces.
    load_dlib_models()
    face_detect2 = models["face detect"]
    face_rec_model2 = models["face rec"]
    shape_predictor2 = models["shape predict"]

    # Take in the (H,W,3) img_array and return the number of face detections in the photo
    detections = list(face_detect2(img_array))

    # for each detected face, create a descriptor
    descriptors = []
    for image in range(len(detections)):
        shape = shape_predictor2(img_array, detections[image])
        descriptor = np.array(face_rec_model2.compute_face_descriptor(img_array, shape))
        descriptors.append(descriptor)
    if type(detections) != list:
        detections = list(detections)
    if len(detections) == 1:
        rect = detections[0]
        fig, ax = plt.subplots()
        x1, x2, y1, y2 = rect.left(), rect.right(), rect.top(), rect.bottom()
        ax.imshow(img_array[y1:y2, x1:x2, :])


    else:
        fig, ax = plt.subplots(ncols=len(detections))
        # https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly
        for i, rect in enumerate(detections):
            x1, x2, y1, y2 = rect.left(), rect.right(), rect.top(), rect.bottom()
            ax[i].imshow(img_array[y1:y2, x1:x2, :])
            ax[i].axis('off')
    faces = import_pickle()
    if name in faces.keys():
        faces[name].append(descriptor)
    else:
        faces.update([(name,descriptor)])
    pickle_the_pickle(faces)
    return statement("Now you are in our database!!")


@ask.intent('AMAZON.CancelIntent')
@ask.intent('AMAZON.StopIntent')
@ask.intent('AMAZON.NoIntent')
def stop_alexa():
    quit = "Alright, never mind."
    return statement(quit)

if __name__ == '__main__':
    app.run(debug=True)