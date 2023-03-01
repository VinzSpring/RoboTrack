
from miio import DreameVacuum
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
import openai
import time
import json

secrets = json.load(open("secrets.json", "r"))

openai.api_key = secrets["openai_api_key"]

robo_data = secrets["mi_vacuum_credentials"]

def get_webcam():
    return cv2.VideoCapture(0)

def get_frame(cam): # yields frame
    while True:
        _, frame = cam.read()
        yield frame

def get_objects(frame): # just get all the objects in the frame and their center coordinates
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    objects = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            objects.append(label)
            # color = colors[i]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
    coordinates = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            coordinates.append([x, y, w, h])
    return objects, coordinates

def get_objects_from_webcam(cam):
    for frame in get_frame(cam):
        yield get_objects(frame)

history = ["action: IDLE"]


def get_gpt_query(objects, coordinates, objective="do whatever it takes to drive straight at the bottle"):
    # generates a text query for GPT-3
    # asking it to decide wether to issue the command IDLE | TURN_LEFT | TURN_RIGHT | GO_FORWARD based on the objects
    # detected in the frame
    query = "The robot's objective is to " + objective + ". "
    query += "This is a operational protocol for a robot. "
    query += "History: "
    for obj in history:
        query += obj + ", "
    query = query[:-2]
    query += "; "
    query += "Current objects: "
    for obj, coord in zip(objects, coordinates):
        query += obj + " at bounding box" + str(coord) + ", "
        history.append("object: " + obj + " at bounding box" + str(coord))
    query = query[:-2]
    query += "; "
    query += "Based on this information, what should the robot do? "
    query += "Possible answers: IDLE | TURN_LEFT | TURN_RIGHT | GO_FORWARD\nanswer: "
    return query

def get_gpt_response(query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=query,
        temperature=0,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.2,
        presence_penalty=0.0,
        stop=["\n"]
    )
    return response.choices[0].text


def get_gpt_command(objects, coordinates):
    query = get_gpt_query(objects, coordinates)
    response = get_gpt_response(query)
    return response.strip()

def get_gpt_command_from_webcam(cam, max_gpt_query_time_seconds=1):
    t = time.time()
    no_objects = 0
    for objects, coordinates in get_objects_from_webcam(cam):
        if not objects and no_objects < 10:
            no_objects += 1
            continue
        elif not objects and no_objects >= 10:
            no_objects = 0
            yield "TURN_LEFT"
            continue

        if time.time() - t < max_gpt_query_time_seconds:
            continue
        t = time.time()        
        res = get_gpt_command(objects, coordinates)
        history.append("action: " + res)
        print(res, objects, coordinates)
        yield res


vac = DreameVacuum(robo_data["IP"], robo_data["TOKEN"])
# vac.home()

# vac.start()
vac.stop()

for command in get_gpt_command_from_webcam(get_webcam(), 5):
    print(command)

    if "IDLE" in command:
        vac.stop()
    elif "TURN_LEFT" in command:
        vac.rotate(45)
    elif "TURN_RIGHT" in command:
        vac.rotate(-45)
    elif "GO_FORWARD" in command:
        vac.forward(100)
    else:
        vac.stop()

    print("")

    # vac.forward(100)
    # time.sleep(10)
