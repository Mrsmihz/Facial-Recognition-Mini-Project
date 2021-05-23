import glob

import cv2
import numpy as np
import tqdm as t
import sklearn.svm as svm
from joblib import load, dump
import os

# กำหนด IMAGE SIZE
IMAGE_SIZE = (64, 128)

# เรียกใช้ HOG โดยใช้ค่า Default
HOG = cv2.HOGDescriptor()

# กำหนดชื่อ CLASS
CLASSES = [name for name in os.listdir('./datasets/handcraft_train/') if name != '.DS_Store']
CLASSES.sort()
TEST_PATH = "./datasets/test/"
TRAIN_PATH = "./datasets/handcraft_train/"

# Train Model
def train():
    sample_size = 0
    features = []
    labels = []
    for _classname in t.tqdm(range(len(CLASSES))):

        # อ่านไฟล์
        filenames = glob.glob(TRAIN_PATH+CLASSES[_classname]+"/*.*")
        images = [cv2.imread(img) for img in filenames]
        sample_size += len(images)
        for img in images:
            # Resize Image
            img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            # 37 - 42 เป็นกระบวนการทำ Image processing
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = opening(img)
            # img = closing(img)
            # img = erosion(img)
            # img = dilation(img)
            # img = canny(img)
            hist = HOG.compute(img)
            features.append(np.array(hist))
            labels.append(_classname)
    features = np.reshape(np.array(features), (sample_size, -1))

    #Classifier
    classifier = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo')
    classifier.fit(features, labels)
    dump(classifier, 'handcraft_model.sav')


def test():
    classifier = load('handcraft_model.sav')
    correct = 0
    fail = 0
    for _classname in t.tqdm(range(len(CLASSES))):
        filenames = glob.glob(TEST_PATH+CLASSES[_classname]+"/*.*")
        # อ่านไฟล์
        images = [cv2.imread(img) for img in filenames]
        for img in images:
            # Resize
            img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            # 66 - 71 กระบวนการทำ Image Processing
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = opening(img)
            # img = closing(img)
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
        print(f"{CLASSES[_classname]} \n Success Rate: {total_correct:.2f} \n Failure Rate: {total_fail:.2f}")
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


train()
test()
