### importing required libraries
from json.encoder import INFINITY
from socket import timeout
import torch
import cv2
import time
import numpy as np
import win32gui, win32con
from math import sqrt
import PySimpleGUI as sg
from playsound import playsound
from plyer import notification
from threading import Thread

detection_threshold = 0.65 # Cutoff enemy certainty percentage for aiming

windowTitle = "vid"

IN = 1

modelFile = "yolov5s.pt" # This is the AI model the program will use, multiple are included, (W) = working, (NW) = not working.

### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    #print(f"[INFO] Detecting. . . ")
    results = model(frame)
    
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame, classes):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    numPersons = 0
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    for i in range(n):
        row = cord[i]
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
        text = classes[int(labels[i])]
        
        if text == 'person':
            # Draw bbox for this detection    
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox  
            cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1) ## Text BG
            cv2.putText(frame, text + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            numPersons += 1

    return frame, numPersons

### ---------------------------------------------- Notify function -----------------------------------------------------
def notifyMe(title, message):

    notification.notify(
        title = title,
        message = message,
        app_icon = "warning.ico", 
        timeout = 10,
    )

### ---------------------------------------------- Main function -----------------------------------------------------
def main(IN=None):

    canNotify = True

    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    model = torch.hub.load('./yolov5', 'custom', source ='local', path=modelFile, force_reload=True)
    classes = model.names

    cap = cv2.VideoCapture(IN)

    while True:
        persons = 0

        ret, img = cap.read()
        
        if ret:
            results = detectx(img, model = model)
            currframe, persons = plot_boxes(results, img, classes = classes) 
            cv2.imshow("vid", currframe)

            if persons >= 1:
                if canNotify == True:
                    notifyMe("Warning", "Person detected")
                    canNotify = False

            elif persons <= 0:
                canNotify = True
                

        if cv2.waitKey(5) and 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

### -------------------  calling the main function-------------------------------

main(IN)


