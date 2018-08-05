height = int(input("For the function to measure distances, you need to input your height (in inches) so we can use it for scale:\n"))-4
#Substract 4 because the height only goes from the forehead to the ankles
from pathlib import Path
import sys
sys.path.insert(0, 'C:/Users/micha/')


import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
import cv2
import matplotlib.pyplot as plt
from camera import take_picture
import numpy as np
from PIL import Image, ImageDraw
import math
cfg = load_config("demo/pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)
print("FINISHED IMPORTING EVERTHING")
def __init__():
    pass
__init__()
def new_pic(name):
    '''
    Takes new picture and saves it as a PNG
    
    Parameters
    ----------
    name : String representing the name of the file
    
    '''
    A = take_picture()
    im = Image.fromarray(A)
    im.save(name+".PNG")
def print_locations(pose):
    '''
    Prints each body part and the corresponding coords
    
    Parameters
    ----------
    pose: 14*3 Numpy Array for the 14 Body Parts and their coordinates
    
    '''
    print("Right ankle " + str(pose[0]))#
    print("Right knee " + str(pose[1]))#
    print("Right waist " + str(pose[2]))
    print("Left waist " + str(pose[3]))
    print("Left knee " +str(pose[4]))#
    print("Left ankle " + str(pose[5]))#
    print("Right wrist " + str(pose[6]))#
    print("Right elbow " + str(pose[7]))#
    print("Right shoulder " + str(pose[8]))
    print("Left shoulder " + str(pose[9]))
    print("Left elbow " + str(pose[10]))#
    print("Left wrist " + str(pose[11]))#
    print("Neck " + str(pose[12]))
    print("Forehead " + str(pose[13]))#
def set_scale(pose):
    '''
    Returns the ratio between pixels
    
    Parameters
    ----------
    pose: 14*3 Numpy Array for the 14 Body Parts and their coordinates
    
    '''
    return height/abs(pose[13][1]-pose[0][1])
def arm_span(pose,sc):
    '''
    Measures distance from one wrist to another
    
    Parameters
    ----------
    pose: 14*3 Numpy Array for the 14 Body Parts and their coordinates
    sc: Scale
    
    Returns
    -------
    Double representing the wingspan from one wrist to another
    '''
    return sc * (math.sqrt((pose[6][0]-pose[7][0])**2+(pose[6][1]-pose[7][1])**2)#Right wrist to right elbow
                    +math.sqrt((pose[7][0]-pose[8][0])**2+(pose[7][1]-pose[8][1])**2)#Right elbow to right shoulder
                    +math.sqrt((pose[9][0]-pose[8][0])**2+(pose[9][1]-pose[8][1])**2)#Right shoulder to left shoulder
                    +math.sqrt((pose[9][0]-pose[10][0])**2+(pose[9][1]-pose[10][1])**2)#Left shoulder to left elbow
                    +math.sqrt((pose[10][0]-pose[11][0])**2+(pose[10][1]-pose[11][1])**2))#Left elbow to left wrist
def arm_span_est(pose,sc):
    '''
    Estimates distance from one fingertip to another
    
    Parameters
    ----------
    pose: 14*3 Numpy Array for the 14 Body Parts and their coordinates
    sc: Scale
    
    Returns
    -------
    Double representing the estimate of the wingspan from one fingertip to another
    '''
    print("(Not as accurate as wrist to wrist)")
    return arm_span(pose,sc)*1.25
    #multiply by 1.25 to account for the fact that the hands are not included in the measurement
    #used the number 1.25 because hands are about 20% of the wingspan
def kick_height(pose,sc):
    '''
    Measures distance from one knee to another
    
    Parameters
    ----------
    pose: 14*3 Numpy Array for the 14 Body Parts and their coordinates
    sc: Scale
    
    Returns
    -------
    Double representing the vertical distance from one ankle to the other
    '''
    return sc * abs(pose[0][1] - pose[5][1])
def vid_pics(sec=1,where = "temp2/vid_pic"):
    '''
    Records video and saves pictures in the where folder
    
    Parameters
    ----------
    sec: amount of time (when sec=1 the time is about 1 second but not exactly)
    where: save locations
    '''
    cap = cv2.VideoCapture(0)
    for i in range(10*sec):
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                cv2.imshow('video', frame)
            im = Image.fromarray(frame)
            b, g, r = im.split()
            im = Image.merge("RGB", (r, g, b))
            im.save(where+str(i)+".PNG")
        if cv2.waitKey(10) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
def vid_view(sec=10):
    '''
    Displays video feed but does not save it anywhere. This allows the user to position themselves appropriately
    
    Parameters
    ----------
    sec: amount of time (when sec=1 the time is about 1 second but not exactly)
    
    '''
    cap = cv2.VideoCapture(0)
    for i in range(10*sec):
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                cv2.imshow('video', frame)
        if cv2.waitKey(10) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
def arms(new = False,where="your_file"):
    '''
    Does arm_span and arm_span_est
    '''
    scale = 0 #Number of inches per pixel
    if(new):
        new_pic(where+".PNG")
    image = imread(where+".PNG", mode='RGB')

    image_batch = data_to_input(image)

    # Compute prediction with the CNN
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

    # Extract maximum scoring location from the heatmap, assume 1 person
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

    scale = set_scale(pose)
    print(arm_span(pose,scale))
    print(arm_span_est(pose,scale))
    disp_pic(where=where)
# Read image from file
#new_pic()
def kick_vid(new = False):
    '''
    Takes a video, takes 40 frames from the video, maps the body in each, and finds the maximum height
    '''
    shoe_size = int(input("What is your shoe size (US):\n"))
    scale = 0 #Number of inches per pixel
    if(new):
        vid_view(sec=10)
        print("GO")
        vid_pics(sec=4)
    ans = []
    for i in range(40):
        image = imread("temp/vid_pic"+str(i)+".PNG", mode='RGB')

        image_batch = data_to_input(image)

        # Compute prediction with the CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    
        if(i==0):
            scale = set_scale(pose)
        ans.append(kick_height(pose,scale))
    cv2.destroyAllWindows()
    ans = np.array(ans)
    print(ans)
    print("We measured " + str(max(ans)) + " using the ankles as our points")
    print("The true value is approximately " + str(max(ans)+(4/3*shoe_size)))
    print(np.argmax(ans))

    # Visualise
    disp_pic(where="temp/vid_pic"+str(np.argmax(ans)))
def disp_pic(new = False,where="your_file4"):
    '''
    Displays image
    '''
    if(new):
        new_pic("QWERTY12345")
        where="QWERTY12345"
    image = imread(where+".PNG", mode='RGB')
    
    image_batch = data_to_input(image)

        # Compute prediction with the CNN
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

        # Extract maximum scoring location from the heatmap, assume 1 person
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    
    visualize.show_heatmaps(cfg, image, scmap, pose)
    visualize.waitforbuttonpress()
def trainer_vid(wh):
    '''
    Prepares data for GAIT
    '''
    vid_view(sec=10)
    print("GO")
    return vid_pics(sec=2,where=wh)
def get_person_data(person):
    '''
    Takes in the data for GAIT and resizes the array in the form most suitable for the train
    '''
    Final_Array = np.zeros(shape=(15,20,9,2))
    for i in range(15):
        trainer_vid("walk_data/"+person+"/vid"+str(i)+"/pic")
        if(input("Video index: "+str(i)+" has been taken. Type yeet to leave:\n")=="yeet"):
            return
    for i in range(15):
        for j in range(20):
            image = imread("walk_data/"+person+"/vid"+str(i)+"/pic"+str(j)+".PNG", mode='RGB')

            image_batch = data_to_input(image)

            # Compute prediction with the CNN
            outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
            scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

            # Extract maximum scoring location from the heatmap, assume 1 person
            pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
            
            temp = np.zeros(shape=(9,2))
            list_indexes = [0,1,4,5,6,7,10,11,13]
            for k in range(9):
                temp[k] = pose[list_indexes[k]][0:2]
            Final_Array[i][j] = temp
        
        print(str(i)+" is finished")
    print(Final_Array.shape)
    Answer = np.zeros(shape=(9,15,20,2))
    for i in range(9):
        Answer[i] = Final_Array[:,:,i,:]
    print(Answer.shape)
    return Answer
def get_test_data():
    '''
    Takes in the data for GAIT and resizes the array in the form most suitable for the train
    '''
    Final_Array = np.zeros(shape=(20,9,2))
    trainer_vid("test/pic")
    for j in range(20):
        image = imread("test/pic"+str(j)+".PNG", mode='RGB')

        image_batch = data_to_input(image)

        # Compute prediction with the CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

        temp = np.zeros(shape=(9,2))
        list_indexes = [0,1,4,5,6,7,10,11,13]
        for k in range(9):
            temp[k] = pose[list_indexes[k]][0:2]
        Final_Array[j] = temp
    print(Final_Array.shape)
    temp2 = np.zeros(shape=(2,20,9))
    temp2[0] = Final_Array[:,:,0]
    temp2[1] = Final_Array[:,:,1]
    Answer = np.zeros(shape=(9,2,20))
    for i in range(9):
        Answer[i] = temp2[:,:,i]
    print(Answer.shape)
    return Answer
#for i in range(1,2):
#    np.save("Unknown_Walker_e",get_test_data())
#    if(input("Yeet?")=="yeet"):
#        break
#np.save("",get_person_data(""))
#disp_pic()
#arms()
kick_vid(new=False)