import cv2
import numpy as np
import glob
import random
import urllib.request
import time
from playsound import playsound

URL = "http://192.168.43.1:8080/shot.jpg"
input_video_path = 'video.mp4'

cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_1000.weights", "yolov3_testing.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

print("Set video source:")
print("1. camera")
print("2. video")

choice = input()
now = time.time() - 2
detected = False


if int(choice)==1:
    print("print camera view url:")
    URL = input()
else:
    print("print video path:")
    input_video_path = input()

while(True):
    if int(choice)==1:
        img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()),dtype=np.uint8)
        img = cv2.imdecode(img_arr,-1)
    else:
        while time.time() - now > 0:
            now += 1/fps
            ret, img = cap.read()
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    higher = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.001 and confidence > higher:
                higher = confidence
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                box = [x, y, w, h]


    font = cv2.FONT_HERSHEY_PLAIN
    if higher > 0:
        if detected is False:
            playsound("nice-work.wav");
            detected = True
        x, y, w, h = box
        label = "zebra"
        color = (255,0,0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + str(int(higher*100)) + "%", (x, y + 30), font, 1, color, 2)
    else:
        detected = False

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break;

cv2.destroyAllWindows()