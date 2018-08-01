
# coding: utf-8

# In[1]:


#By Eden, Ryan, Michael, Vishnu


# In[2]:
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
    welcome_message= 'Hello, would you like me to translate for you?'
    return question(welcome_message)

@ask.intent("YesIntent")
def final_func():
    from camera import take_picture
    img_array = take_picture()
    import numpy as np

    # download the models from dlib
    from dlib_models import download_model, download_predictor, load_dlib_models

    from dlib_models import models

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
    if descriptors is []:
        return_statement="We don't see you"
        return statement(return_statement)
    import matplotlib.pyplot as plt
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
    names = []
    for descriptor in descriptors:
        import numpy as np
        with open("faces.pkl", mode="rb") as opened_file:
            faces = pickle.load(opened_file)
        distances = []

        for name in list(faces.keys()):
            mean = np.mean(faces[name], axis=0)  # takes element wise mean
            distances.append(euclidean_distance(mean, descriptor))

        print(distances)
        # populate a list with probable names within the threshold
        probable_faces = []
        threshold = .4
        for i, distance in enumerate(distances):
            if distance < threshold:
                probable_faces.append(list(faces.keys())[i])

        # todo: figure out the threshold value. will we be able to return just one?
        if probable_faces is []:
            probable_faces.append("idk lol")
        names=probable_faces[0]
        names.append(name)
    return statement(names)

@ask.intent('AMAZON.CancelIntent')
@ask.intent('AMAZON.StopIntent')
@ask.intent('AMAZON.NoIntent')
def stop_alexa():
    quit = "Alright, never mind."
    return statement(quit)

if __name__ == '__main__':
    app.run(debug=True)