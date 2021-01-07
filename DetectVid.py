from pathlib import Path
from os import path
import cv2
import numpy as np

root_folder = Path(path.dirname(path.realpath(__file__)))

input_folder = root_folder / "input"
net = cv2.dnn.readNet(str(root_folder / 'yolov3.weights'), str(root_folder / 'yolov3.cfg'))
classes_file = open(str(root_folder / 'classes.txt'), 'r')
classes = classes_file.read().splitlines()
classes_file.close()

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors= np.random.uniform(0,255,size=(len(classes),3))

def getPredictions(outs):
    predictions = {
                    "boxes" : [],
                    "confidences": [],
                    "class_ids": []
                  }

    # Showing info on screen and get confidence score of algorithm in detecting an object in blob
    font = cv2.FONT_HERSHEY_PLAIN
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                #rectangle co-ordinates
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                predictions["boxes"].append([x,y,w,h])
                predictions["confidences"].append(float(confidence))
                predictions["class_ids"].append(class_id)

    return predictions

def drawPredictions(predictions, img):
    # Create a new image
    new_img = img.copy()

    boxes = predictions["boxes"]
    confidences = predictions["confidences"]
    class_ids = predictions["class_ids"]

    font = cv2.FONT_HERSHEY_PLAIN
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence= confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(new_img, (x,y), (x+w,y+h), color, 2)
            cv2.putText(new_img, f"{label} {str(round(confidence,2))}", (x,y+30), font, 1, (255,255,255), 2)
    
    return new_img

# Open video feed
capture = cv2.VideoCapture(str(input_folder / "cars.mp4"))
while True:
    # Capture the frame
    retval, frame = capture.read()
    ################################################
    # Your code here
    ################################################
