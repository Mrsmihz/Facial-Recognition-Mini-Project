import glob

import cv2
import numpy as np
import tqdm as t
import sklearn.svm as svm
from joblib import load, dump
import matplotlib.pyplot as plt
import os
from random import randint
import random

# กำหนด IMAGE SIZE
IMAGE_SIZE = (64, 128)

# เรียกใช้ HOG โดยใช้ค่า Default
HOG = cv2.HOGDescriptor()

# path for datasets
TEST_PATH = "./datasets/test/"
TRAIN_PATH = "./datasets/train/"
MODEL_PATH = './handcraft_model.sav'

# get class sub-directory name
TRAIN_CLASSES = [name for name in os.listdir(TRAIN_PATH) if name != '.DS_Store']
TRAIN_CLASSES.sort()

TEST_CLASSES = [name for name in os.listdir(TEST_PATH) if name != '.DS_Store']
TEST_CLASSES.sort()


def fill(img, h, w):
    img = cv2.resize(img, (w, h), cv2.INTER_AREA)
    return img


def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w * ratio
    if ratio > 0:
        img = img[:, :int(w - to_shift), :]
    if ratio < 0:
        img = img[:, int(-1 * to_shift):, :]
    img = fill(img, h, w)
    return img


def vertical_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :, :]
    img = fill(img, h, w)
    return img


def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1]*value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2]*value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def zoom(img, value):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    return img


def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:, :, :][img[:, :, :] > 255] = 255
    img[:, :, :][img[:, :, :] < 0] = 0
    img = img.astype(np.uint8)
    return img


def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, m, (w, h))
    return img


# Train Model
def train(epoch=1):
    sample_size = 0
    features = []
    labels = []
    for i in range(epoch):
        for _classname in t.tqdm(range(len(TRAIN_CLASSES))):
            # อ่านไฟล์
            filenames = glob.glob(TRAIN_PATH + TRAIN_CLASSES[_classname] + "/*.pgm")
            images = [cv2.imread(img) for img in filenames]
            sample_size += len(images)
            for img in images:
                # Resize Image
                img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                # 37 - 42 เป็นกระบวนการทำ Image processing
                if randint(0, 1):
                    img = cv2.rotate(img, cv2.ROTATE_180)
                if randint(0, 1):
                    img = cv2.flip(img, 0)
                if randint(0, 1):
                    img = cv2.flip(img, 1)
                if randint(0, 1):
                    img = cv2.flip(img, -1)
                if randint(0, 1):
                    img = horizontal_shift(img, random.randrange(0, 1))
                if randint(0, 1):
                    img = vertical_shift(img, random.randrange(0, 1))
                if randint(0, 1):
                    img = brightness(img, random.randrange(0, 1), randint(1, 3))
                if randint(0, 1):
                    img = zoom(img, 0.1)
                if randint(0, 1):
                    img = channel_shift(img, random.randrange(0, 100))
                if randint(0, 1):
                    img = rotation(img, randint(0, 360))
                if randint(0, 1):
                    img = cv2.GaussianBlur(img, (5, 5), 0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = closing(img)
                # img = opening(img)
                # img = erosion(img)
                # img = dilation(img)
                # img = canny(img)
                hist = HOG.compute(img)
                features.append(np.array(hist))
                labels.append(_classname)
    # Classifier
    features = np.reshape(np.array(features), (sample_size, -1))
    #print(features.shape)
    classifier = svm.SVC(kernel='rbf', C=2, gamma=0.1)
    classifier.fit(features, labels)
    dump(classifier, 'handcraft_model.sav')


def test():
    classifier = load(MODEL_PATH)
    correct = 0
    fail = 0
    for _classname in t.tqdm(range(len(TEST_CLASSES))):
        filenames = glob.glob(TEST_PATH + TEST_CLASSES[_classname] + "/*.pgm")
        # อ่านไฟล์
        images = [cv2.imread(img) for img in filenames]
        for img in images:
            # Resize
            img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            # 66 - 71 กระบวนการทำ Image Processing
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = opening(img)
            img = closing(img)
            # img = erosion(img)
            # img = dilation(img)
            # img = canny(img)
            feature = np.array(HOG.compute(img)).T
            out = classifier.predict(feature)
            if int(out.item()) == _classname:
                correct += 1
            else:
                fail += 1
        # คำนวณหา %
        total_correct = (correct / (correct + fail))*100
        total_fail = (fail / (correct + fail))*100
        print(f"{TEST_CLASSES[_classname]} \n Success Rate: {total_correct:.2f} \n Failure Rate: {total_fail:.2f}")
        correct = 0
        fail = 0


def canny(img):
    """Canny Function"""
    canny_low = 15
    canny_high = 150
    img = cv2.Canny(img, canny_low, canny_high)
    return img


def opening(img):
    """Opening Function"""
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img


def closing(img):
    """Closing Function"""
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def erosion(img):
    """Erosion Function"""
    kernel = np.ones((5, 5), np.uint8)
    erosed = cv2.erode(img, None, kernel, iterations=1)
    return erosed


def dilation(img):
    """Dilation Function"""
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(img, None, kernel, iterations=1)
    return dilated


train(epoch=100)
test()
