import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
import random

def load_data():

    DATADIR = "./data"
    CATEGORIES = ["Takeoff", "TurnLeft", "TurnRight", "Stay"]
    IMG_SIZE =128

    temp_data = []
    x_train, y_train, x_test, y_test = [], [], [], []

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                temp_data.append([new_array, class_num])
            except Exception as e:
                print(e)

    random.shuffle(temp_data)
    index = int(len(temp_data) * 0.8)
    train, test = temp_data[:index], temp_data[index:]

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in test:
        x_test.append(feature)
        y_test.append(label)

    return (x_train, y_train), (x_test, y_test)